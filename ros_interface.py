#!/usr/bin/env python3
"""
ROS 2 Anchor‑Local Grasp Detection Node

⚠ 直接替换原 graspnet_node.py 即可：
  * 订阅：
      /camera/d435/color/image_raw     (sensor_msgs/Image, rgb8)
      /camera/d435/aligned_depth_to_color/image_raw (16UC1)
  * 服务： get_grasps (grasp_msgs/srv/GetGrasp)   — 接口保持不变
  * 可视化：/grasp_markers (visualization_msgs/Marker)

工作流程：RGB‑D → AnchorGraspNet → PointMultiGraspNet → collision_detect → NMS → Marker[]
"""
import os
import sys
import time
import argparse
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from cv_bridge import CvBridge

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Pose
from tf_transformations import quaternion_from_matrix

# ───── Anchor‑Local 算法依赖 ─────
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.extend([
    os.path.join(ROOT_DIR, 'models'),
    os.path.join(ROOT_DIR, 'dataset'),
    os.path.join(ROOT_DIR, 'dataset', 'config'),
])
from dataset.config import get_camera_intrinsic
from dataset.evaluation import (
    anchor_output_process, detect_2d_grasp, feature_fusion, data_process,
    detect_6d_grasp_multi, collision_detect,
)
from models.anchornet import AnchorGraspNet
from models.localgraspnet import PointMultiGraspNet
from graspnetAPI import GraspGroup  # 复用 graspnetAPI 的容器类

# ───── 参数解析 ─────
def parse_args():
    p = argparse.ArgumentParser("anchor_grasp_ros2_node")
    # —— 模型 ——
    p.add_argument('--checkpoint_path', required=True,
                   help='Path to checkpoint .pth or .tar')

    # —— AnchorNet / LocalNet 超参 ——
    p.add_argument('--ratio', type=int, default=8)
    p.add_argument('--anchor_k', type=int, default=6)
    p.add_argument('--anchor_num', type=int, default=7,
                   help='划分 γ/β 网格数量 (用于 LocalNet 分类长度 == anchor_num**2)')
    p.add_argument('--anchor_w', type=float, default=50.0)
    p.add_argument('--anchor_z', type=float, default=20.0)
    p.add_argument('--grid_size', type=int, default=8)
    p.add_argument('--sigma', type=int, default=10)
    p.add_argument('--heatmap_thres', type=float, default=0.01)
    p.add_argument('--local_k', type=int, default=10)

    # —— 点云采样 ——
    p.add_argument('--all_points_num', type=int, default=25600)
    p.add_argument('--center_num', type=int, default=48)
    p.add_argument('--group_num', type=int, default=512)

    # —— 图像输入分辨率 ——
    p.add_argument('--input_w', type=int, default=640)
    p.add_argument('--input_h', type=int, default=360)

    # —— 设备 & 随机种子 ——
    p.add_argument('--random_seed', type=int, default=123)
    return p.parse_args()

# ───── PointCloud ⇄ Torch 辅助类 ─────
class PointCloudHelper:
    """与原脚本保持一致，用相机内参把深度图转换为点云 & xyz maps"""

    def __init__(self, all_points_num: int, down_shape: Tuple[int, int] = (80, 45)):
        self.all_points_num = all_points_num
        self.down_shape = down_shape  # (W_ds, H_ds)

        K = get_camera_intrinsic()
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        # —— 原始分辨率映射表 (1280×720) ——
        ymap, xmap = np.meshgrid(np.arange(720), np.arange(1280))
        self.points_x = torch.from_numpy((xmap - cx) / fx).float()
        self.points_y = torch.from_numpy((ymap - cy) / fy).float()

        # —— Downscale xyz map ——
        h_ds = down_shape[1]
        w_ds = down_shape[0]
        ymap_ds, xmap_ds = np.meshgrid(np.arange(h_ds), np.arange(w_ds))
        factor = 1280 / w_ds
        self.points_x_ds = torch.from_numpy((xmap_ds - cx / factor) / (fx / factor)).float()
        self.points_y_ds = torch.from_numpy((ymap_ds - cy / factor) / (fy / factor)).float()

    def to_scene_points(self, rgb: torch.Tensor, depth: torch.Tensor):
        """rgb: (1,3,H,W) 0‑1 float32; depth: (1,H,W) 原始 uint16→float32 mm"""
        b = 1  # batch 一般==1
        device = depth.device
        feature_len = 6  # xyz + rgb
        pts_all = -torch.ones((b, self.all_points_num, feature_len), device=device)

        # Z in meters
        z = depth / 1000.0  # (1,H,W)
        x = self.points_x.to(device) * z
        y = self.points_y.to(device) * z
        mask = depth > 0

        pts = torch.stack([x[0], y[0], z[0]], dim=-1)[mask[0]]  # (M,3)
        colors = rgb[0].permute(1, 2, 0)[mask[0]]               # (M,3)
        M = pts.shape[0]
        if M >= self.all_points_num:
            sel = torch.randperm(M, device=device)[:self.all_points_num]
            pts = pts[sel]
            colors = colors[sel]
        pts_all[0, :pts.shape[0], :3] = pts
        pts_all[0, :pts.shape[0], 3:] = colors
        return pts_all  # (1,N,6)

    def to_xyz_maps(self, depth: torch.Tensor):
        depth_ds = F.interpolate(depth.unsqueeze(1), size=self.down_shape[::-1], mode='nearest').squeeze(1)
        z = depth_ds / 1000.0
        device = depth.device
        x = self.points_x_ds.to(device) * z
        y = self.points_y_ds.to(device) * z
        xyz = torch.stack([x, y, z], dim=1)  # (1,3,H_ds,W_ds)
        return xyz

# ───── ROS2 Node ─────
class AnchorGraspNode(Node):

    def __init__(self, cfg):
        super().__init__('anchor_grasp_node')
        self.cfg = cfg
        self.bridge = CvBridge()
        self.rgb: Optional[torch.Tensor] = None  # (1,3,H,W)
        self.depth: Optional[torch.Tensor] = None  # (1,H,W)

        # QoS
        qos_sensor = QoSProfile(depth=10)
        qos_sensor.reliability = QoSReliabilityPolicy.BEST_EFFORT
        qos_sensor.durability = QoSDurabilityPolicy.VOLATILE

        # —— 订阅 ——
        self.create_subscription(Image, '/camera/d435/color/image_raw', self._on_rgb, qos_sensor)
        self.create_subscription(Image, '/camera/d435/aligned_depth_to_color/image_raw', self._on_depth, qos_sensor)

        # —— Marker 发布 ——
        self.marker_pub = self.create_publisher(Marker, '/grasp_markers', qos_sensor)

        # —— Service ——
        from grasp_msgs.srv import GetGrasp
        self.srv = self.create_service(GetGrasp, 'get_grasps', self._on_get_grasps)

        # —— PointCloud helper ——
        self.pc_helper = PointCloudHelper(cfg.all_points_num)

        # —— 加载模型 ——
        torch.cuda.manual_seed(cfg.random_seed)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.anchornet = AnchorGraspNet(in_dim=4, ratio=cfg.ratio, anchor_k=cfg.anchor_k).to(device)
        self.localnet = PointMultiGraspNet(info_size=3, k_cls=cfg.anchor_num ** 2).to(device)
        self._load_checkpoint(cfg.checkpoint_path)

        # 预生成 anchors
        basic = torch.linspace(-1, 1, cfg.anchor_num + 1, device=device)
        basic_anchors = (basic[1:] + basic[:-1]) / 2
        self.anchors = {'gamma': basic_anchors, 'beta': basic_anchors}
        self.get_logger().info('AnchorGraspNode ready')

    # ───────── Callbacks ─────────
    def _on_rgb(self, msg: Image):
        try:
            cv_rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')  # H×W×3 uint8
            arr = (torch.from_numpy(cv_rgb).permute(2, 1, 0).float() / 255.0).unsqueeze(0)
            self.rgb = arr.to(self.device)
        except Exception as e:
            self.get_logger().error(f'RGB decode error: {e}')

    def _on_depth(self, msg: Image):
        try:
            cv_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            if cv_depth.dtype == np.float32 or cv_depth.dtype == np.float64:
                cv_depth = (cv_depth * 1000.0).astype(np.uint16)  # m→mm
            arr = torch.from_numpy(cv_depth).float().T.unsqueeze(0)  # (1,H,W)
            self.depth = arr.to(self.device)
        except Exception as e:
            self.get_logger().error(f'Depth decode error: {e}')

    # ───────── Service entry ─────────
    def _on_get_grasps(self, request, response):
        if request.input != 'start_grasp':
            response.success = False
            response.output.markers = []
            return response

        # 等待图像就绪
        start = time.time()
        while (self.rgb is None or self.depth is None):
            rclpy.spin_once(self, timeout_sec=0.05)
            if time.time() - start > 3.0:
                self.get_logger().error('Timeout waiting for RGB‑D')
                response.success = False
                response.output.markers = []
                return response

        try:
            markers = self._infer_and_build_markers()
            if markers:
                response.success = True
                response.output.markers = markers
            else:
                response.success = False
                response.output.markers = []
        except Exception as e:
            self.get_logger().error(f'Inference failed: {e}')
            response.success = False
            response.output.markers = []
        return response

    # ───────── Inference pipeline ─────────
    def _infer_and_build_markers(self):
        cfg = self.cfg
        # —— 预处理 ——
        rgb_ds = F.interpolate(self.rgb, size=(cfg.input_w, cfg.input_h))  # (1,3,h,w)
        depth_ds = F.interpolate(self.depth.unsqueeze(1), size=(cfg.input_w, cfg.input_h), mode='nearest').squeeze(1)
        depth_norm = torch.clip((depth_ds / 1000.0 - (depth_ds / 1000.0).mean()), -1, 1)
        x = torch.cat([depth_norm.unsqueeze(1), rgb_ds], dim=1)  # (1,4,h,w)

        # —— AnchorNet 输出 ——
        with torch.no_grad():
            pred_2d, perpoint_feat = self.anchornet(x)
        loc_map, cls_mask, theta_offset, h_offset, w_offset = anchor_output_process(*pred_2d, sigma=cfg.sigma)

        rect_gg = detect_2d_grasp(loc_map, cls_mask, theta_offset, h_offset, w_offset,
                                  ratio=cfg.ratio, anchor_k=cfg.anchor_k, anchor_w=cfg.anchor_w,
                                  anchor_z=cfg.anchor_z, mask_thre=cfg.heatmap_thres,
                                  center_num=cfg.center_num, grid_size=cfg.grid_size,
                                  grasp_nms=cfg.grid_size, reduce='max')
        if rect_gg.size == 0:
            self.get_logger().info('No 2‑D grasp found')
            return []

        # —— 特征融合 & patch 生成 ——
        pts_all = self.pc_helper.to_scene_points(self.rgb, self.depth)
        xyzs = self.pc_helper.to_xyz_maps(self.depth)
        points_all_fused = feature_fusion(pts_all[..., :3], perpoint_feat, xyzs)
        pc_group, valid_centers = data_process(points_all_fused, self.depth, [rect_gg],
                                               cfg.center_num, cfg.group_num,
                                               (cfg.input_w, cfg.input_h),
                                               min_points=32, is_training=False)
        grasp_info = torch.tensor(np.vstack([rect_gg.thetas, rect_gg.widths, rect_gg.depths]).T,
                                  dtype=torch.float32, device=self.device)

        # —— LocalNet ——
        with torch.no_grad():
            _, local_pred, local_off = self.localnet(pc_group, grasp_info)
        _, pred_rect_gg = detect_6d_grasp_multi(rect_gg, local_pred, local_off,
                                                valid_centers, (cfg.input_w, cfg.input_h),
                                                self.anchors, k=cfg.local_k)

        # —— Collision & NMS ——
        pred_grasp6d = pred_rect_gg.to_6d_grasp_group(depth=0.02)
        pred_gg, _ = collision_detect(points_all_fused.squeeze(), pred_grasp6d, mode='graspnet')
        pred_gg = pred_gg.nms()

        self.get_logger().info(f'Found {len(pred_gg)} grasps')
        markers = [self._grasp_to_marker(g, i) for i, g in enumerate(pred_gg)]
        for m in markers:
            self.marker_pub.publish(m)
        return markers

    # ───────── Checkpoint ─────────
    def _load_checkpoint(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.anchornet.load_state_dict(ckpt['anchor'])
        self.localnet.load_state_dict(ckpt['local'])
        # anchors γ/β
        self.anchors = {'gamma': ckpt['gamma'], 'beta': ckpt['beta']}
        self.get_logger().info(f'Loaded checkpoint "{ckpt_path}"')
        self.anchornet.eval()
        self.localnet.eval()

    # ───────── Marker builder ─────────
    def _grasp_to_marker(self, grasp: GraspGroup, idx: int):
        # translation
        px, py, pz = grasp.grasp_array[13:16]
        pose = Pose()
        pose.position.x = float(px)
        pose.position.y = float(py)
        pose.position.z = float(pz)
        # orientation: 取 –R 来翻转抓取方向，使握爪+Z 朝下
        R = -grasp.rotation_matrix
        homo = np.eye(4)
        homo[:3, :3] = R
        qx, qy, qz, qw = quaternion_from_matrix(homo)
        pose.orientation.x = float(qx)
        pose.orientation.y = float(qy)
        pose.orientation.z = float(qz)
        pose.orientation.w = float(qw)

        m = Marker()
        m.header.frame_id = 'd435_color_optical_frame'
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = 'grasp_markers'
        m.id = idx
        m.type = Marker.CUBE
        m.action = Marker.ADD
        m.scale.x = m.scale.y = m.scale.z = 0.05
        m.color.r = m.color.g = m.color.b = m.color.a = 1.0
        m.pose = pose
        return m

# ───── main ─────
def main():
    cfg = parse_args()
    rclpy.init()
    node = AnchorGraspNode(cfg)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

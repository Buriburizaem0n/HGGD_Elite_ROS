#!/usr/bin/env python3
"""
ROS 2 Anchor‑Local Grasp Detection Node — 纯 Pose → Marker 版

• 订阅：
    /camera/d435/color/image_raw                      (sensor_msgs/Image, rgb8)
    /camera/d435/aligned_depth_to_color/image_raw      (sensor_msgs/Image, 16UC1 或 32FC1)
• 服务：get_grasps (grasp_msgs/srv/GetGrasp)   — 与旧接口完全一致
• 发布：/grasp_markers (visualization_msgs/Marker)

与上一版相比：
  1. **完全去掉 GraspGroup 依赖**，直接把 LocalNet 输出的 6‑D pose (3 × 3 R + t) 转 Marker。
  2. 移除 collision detect / NMS，只做算法原始输出可视化，方便离线评估。
"""
import os
import sys
import time
import argparse
from typing import List, Tuple, Optional

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
from grasp_msgs.srv import GetGrasp

# ───── Anchor‑Local 依赖 ─────
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.extend([
    os.path.join(ROOT_DIR, 'models'),
    os.path.join(ROOT_DIR, 'dataset'),
    os.path.join(ROOT_DIR, 'dataset', 'config'),
])
from dataset.config import get_camera_intrinsic
from dataset.evaluation import (
    anchor_output_process, detect_2d_grasp, detect_6d_grasp_multi,
)
from dataset.pc_dataset_tools import (
        feature_fusion, data_process
)
from models.anchornet import AnchorGraspNet
from models.localgraspnet import PointMultiGraspNet

# ───── argparse ─────
print('Loading AnchorPoseNode...')
def parse_args():
    p = argparse.ArgumentParser('anchor_grasp_ros2_node_pose')
    p.add_argument('--checkpoint_path', required=True)
    # Anchor & Local
    p.add_argument('--ratio', type=int, default=8)
    p.add_argument('--anchor_k', type=int, default=6)
    p.add_argument('--anchor_num', type=int, default=13)
    p.add_argument('--anchor_w', type=float, default=50.0)
    p.add_argument('--anchor_z', type=float, default=20.0)
    p.add_argument('--grid_size', type=int, default=8)
    p.add_argument('--sigma', type=int, default=10)
    p.add_argument('--heatmap_thres', type=float, default=0.01)
    p.add_argument('--local_k', type=int, default=10)
    # Point cloud
    p.add_argument('--all_points_num', type=int, default=40000)
    p.add_argument('--center_num', type=int, default=1024)
    p.add_argument('--group_num', type=int, default=32)
    # Image size fed to AnchorNet
    p.add_argument('--input_w', type=int, default=320)
    p.add_argument('--input_h', type=int, default=180)
    # Misc
    p.add_argument('--random_seed', type=int, default=123)
    return p.parse_args()

# ───── PointCloud Helper ─────

class PointCloudHelper:
    """深度 → 点云 / xyz map (与原算法保持同维度)"""

    def __init__(self, n_points: int, ds_shape: Tuple[int, int] = (80, 45)):
        self.n_points = n_points
        self.ds_shape = ds_shape  # (W_ds, H_ds)
        K = get_camera_intrinsic()
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        # full‑res map (1280×720)
        ymap, xmap = np.meshgrid(np.arange(720), np.arange(1280))
        self.x_full = torch.from_numpy((xmap - cx) / fx).float()
        self.y_full = torch.from_numpy((ymap - cy) / fy).float()
        # downscale map
        h_ds = ds_shape[1]; w_ds = ds_shape[0]
        ymap_ds, xmap_ds = np.meshgrid(np.arange(h_ds), np.arange(w_ds))
        factor = 1280 / w_ds
        self.x_ds = torch.from_numpy((xmap_ds - cx / factor) / (fx / factor)).float()
        self.y_ds = torch.from_numpy((ymap_ds - cy / factor) / (fy / factor)).float()

    def to_point_cloud(self, rgb: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        """返回 (1, N, 6) xyzrgb"""
        dev = depth.device
        z = depth / 1000.0
        x = self.x_full.to(dev) * z
        y = self.y_full.to(dev) * z
        mask = depth > 0
        xyz = torch.stack([x[0], y[0], z[0]], -1)[mask[0]]  # (M,3)
        rgb_pts = rgb[0].permute(1, 2, 0)[mask[0]]          # (M,3)
        if xyz.shape[0] >= self.n_points:
            sel = torch.randperm(xyz.shape[0], device=dev)[:self.n_points]
            xyz = xyz[sel]; rgb_pts = rgb_pts[sel]
        pc = torch.zeros((1, self.n_points, 6), device=dev) - 1
        pc[0, :xyz.shape[0], :3] = xyz
        pc[0, :xyz.shape[0], 3:] = rgb_pts
        return pc

    def to_xyz_map(self, depth: torch.Tensor) -> torch.Tensor:
        depth_ds = F.interpolate(depth.unsqueeze(1), size=self.ds_shape[::-1], mode='nearest').squeeze(1)
        z = depth_ds / 1000.0
        dev = depth.device
        x = self.x_ds.to(dev) * z
        y = self.y_ds.to(dev) * z
        return torch.stack([x, y, z], 1)  # (1,3,H_ds,W_ds)

# ───── Node ─────

class AnchorPoseNode(Node):

    def __init__(self, cfg):
        super().__init__('anchor_pose_node')
        self.cfg = cfg
        self.bridge = CvBridge()
        self.rgb: Optional[torch.Tensor] = None
        self.depth: Optional[torch.Tensor] = None
        # QoS
        qos = QoSProfile(depth=10)
        qos.reliability = QoSReliabilityPolicy.BEST_EFFORT
        qos.durability = QoSDurabilityPolicy.VOLATILE
        self.create_subscription(Image, '/camera/d435/color/image_raw', self._cb_rgb, qos)
        self.create_subscription(Image, '/camera/d435/aligned_depth_to_color/image_raw', self._cb_depth, qos)
        self.marker_pub = self.create_publisher(Marker, '/grasp_markers', qos)
        self.create_service(GetGrasp, 'get_grasps', self._srv_grasp)
        # Helper & model
        self.pc_helper = PointCloudHelper(cfg.all_points_num)
        torch.manual_seed(cfg.random_seed)
        self.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.anchor_net = AnchorGraspNet(in_dim=4, ratio=cfg.ratio, anchor_k=cfg.anchor_k).to(self.dev)
        self.local_net = PointMultiGraspNet(info_size=3, k_cls=cfg.anchor_num**2).to(self.dev)
        self._load_ckpt(cfg.checkpoint_path)
        # anchor bins for detect_6d
        basic = torch.linspace(-1, 1, cfg.anchor_num + 1, device=self.dev)
        self.anchors = {'gamma': (basic[1:] + basic[:-1]) / 2,
                        'beta':  (basic[1:] + basic[:-1]) / 2}
        self.get_logger().info('AnchorPoseNode ready.')

    # ───── Callbacks ─────
    def _cb_rgb(self, msg):
        try:
            arr = self.bridge.imgmsg_to_cv2(msg, 'rgb8')
            self.rgb = (torch.from_numpy(arr).permute(2, 1, 0).float() / 255.0).unsqueeze(0).to(self.dev)
        except Exception as e:
            self.get_logger().error(f'RGB error: {e}')

    def _cb_depth(self, msg):
        try:
            arr = self.bridge.imgmsg_to_cv2(msg, 'passthrough')
            if arr.dtype != np.uint16:
                arr = (arr * 1000.0).astype(np.uint16)
            self.depth = torch.from_numpy(arr).float().T.unsqueeze(0).to(self.dev)
        except Exception as e:
            self.get_logger().error(f'Depth error: {e}')

    # ───── Service ─────
    def _srv_grasp(self, req, res):
        if req.input != 'start_grasp':
            res.success = False; res.output.markers = []; return res
        if not self._ensure_frames():
            res.success = False; res.output.markers = []; return res
        try:
            markers = self._infer_and_publish()
            res.success = bool(markers)
            res.output.markers = markers
        except Exception as e:
            self.get_logger().error(f'Inference fail: {e}')
            res.success = False; res.output.markers = []
        return res

    # ───── Helpers ─────
    def _ensure_frames(self):
        t0 = time.time()
        while (self.rgb is None or self.depth is None) and (time.time() - t0 < 3.0):
            rclpy.spin_once(self, timeout_sec=0.05)
        return self.rgb is not None and self.depth is not None

    def _load_ckpt(self, path):
        ckpt = torch.load(path, map_location=self.dev)
        self.anchor_net.load_state_dict(ckpt['anchor'])
        self.local_net.load_state_dict(ckpt['local'])
        self.anchors['gamma'] = ckpt['gamma']; self.anchors['beta'] = ckpt['beta']
        self.anchor_net.eval(); self.local_net.eval()
        self.get_logger().info(f'Loaded ckpt: {os.path.basename(path)}')

    # ───── Inference ─────
    def _infer_and_publish(self) -> List[Marker]:
        cfg = self.cfg
        # →预处理
        rgb_ds = F.interpolate(self.rgb, (cfg.input_w, cfg.input_h))
        depth_ds = F.interpolate(self.depth.unsqueeze(1), (cfg.input_w, cfg.input_h), mode='nearest').squeeze(1)
        depth_norm = torch.clip(depth_ds / 1000.0 - (depth_ds / 1000.0).mean(), -1, 1)
        x = torch.cat([depth_norm.unsqueeze(1), rgb_ds], 1)
        # →AnchorNet
        with torch.no_grad():
            pred_2d, per_feat = self.anchor_net(x)
        loc_map, cls_mask, theta_off, h_off, w_off = anchor_output_process(*pred_2d, sigma=cfg.sigma)
        rect_gg = detect_2d_grasp(loc_map, cls_mask, theta_off, h_off, w_off,
                                  ratio=cfg.ratio, anchor_k=cfg.anchor_k, anchor_w=cfg.anchor_w,
                                  anchor_z=cfg.anchor_z, mask_thre=cfg.heatmap_thres,
                                  center_num=cfg.center_num, grid_size=cfg.grid_size,
                                  grasp_nms=cfg.grid_size, reduce='max')
        if rect_gg.size == 0:
            self.get_logger().info('No grasp found'); return []
        # →Point cloud & xyz map
        pc = self.pc_helper.to_point_cloud(self.rgb, self.depth)
        xyzs = self.pc_helper.to_xyz_map(self.depth)
        fusion = feature_fusion(pc[..., :3], per_feat, xyzs)
        pc_group, valid_centers = data_process(fusion, self.depth, [rect_gg],
                                               cfg.center_num, cfg.group_num,
                                               (cfg.input_w, cfg.input_h),
                                               min_points=32, is_training=False)
        grasp_info = torch.tensor(np.vstack([rect_gg.thetas, rect_gg.widths, rect_gg.depths]).T,
                                  dtype=torch.float32, device=self.dev)
        # →LocalNet
        with torch.no_grad():
            _, pred_cls, offset = self.local_net(pc_group, grasp_info)
        _, rect_6d = detect_6d_grasp_multi(rect_gg, pred_cls, offset, valid_centers,
                                           (cfg.input_w, cfg.input_h), self.anchors, k=cfg.local_k)
        # rect_6d 内含 rotation_mats (M,3,3) & translations (M,3)
        Rs: torch.Tensor = rect_6d.rotation_mats  # type: ignore
        Ts: torch.Tensor = rect_6d.translations   # type: ignore
        markers: List[Marker] = []
        for i in range(Rs.shape[0]):
            markers.append(self._pose_to_marker(Rs[i].cpu().numpy(), Ts[i].cpu().numpy(), i))
            self.marker_pub.publish(markers[-1])
        self.get_logger().info(f'Published {len(markers)} grasp markers')
        return markers

    # ───── Pose→Marker ─────
    def _pose_to_marker(self, R: np.ndarray, t: np.ndarray, idx: int) -> Marker:
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = map(float, t)
        R_adj = -R  # 翻转 +Z 方向
        homo = np.eye(4); homo[:3, :3] = R_adj
        qx, qy, qz, qw = quaternion_from_matrix(homo)
        pose.orientation.x = qx; pose.orientation.y = qy; pose.orientation.z = qz; pose.orientation.w = qw
        m = Marker()
        m.header.frame_id = 'd435_color_optical_frame'
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = 'grasp_markers'; m.id = idx; m.type = Marker.CUBE; m.action = Marker.ADD
        m.scale.x = m.scale.y = m.scale.z = 0.05
        m.color.r = m.color.g = m.color.b = m.color.a = 1.0
        m.pose = pose
        return m

# ───── main ─────

def main():
    cfg = parse_args()
    rclpy.init()
    node = AnchorPoseNode(cfg)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node(); rclpy.shutdown()

if __name__ == '__main__':
    main()

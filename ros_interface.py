#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Pose
from cv_bridge import CvBridge
import message_filters
from grasp_msgs.srv import GetGrasp  # 你的自定义服务

import numpy as np
import torch
import torch.nn.functional as F
import random
import cv2

from tf_transformations import quaternion_matrix, quaternion_from_matrix

from dataset.evaluation import (
    anchor_output_process,
    collision_detect,
    detect_2d_grasp,
    detect_6d_grasp_multi
)
from dataset.pc_dataset_tools import data_process, feature_fusion
from models.anchornet import AnchorGraspNet
from models.localgraspnet import PointMultiGraspNet


class PointCloudHelper:
    def __init__(self,
                 all_points_num: int,
                 fx: float, fy: float,
                 cx: float, cy: float,
                 width: int = 1280, height: int = 720,
                 input_w: int = 640, input_h: int = 360,
                 grid_size: int = 8):
        self.all_points_num = all_points_num
        self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy
        self.width, self.height = width, height
        self.input_w, self.input_h = input_w, input_h
        self.grid_size = grid_size

        # full‐resolution x,y maps
        xmap, ymap = np.meshgrid(
            np.arange(self.width),
            np.arange(self.height),
            indexing='ij'
        )
        self.points_x = torch.from_numpy((xmap - cx) / fx).float()
        self.points_y = torch.from_numpy((ymap - cy) / fy).float()

        # downsampled maps for xyz‐map generation
        wd = self.input_w // self.grid_size
        hd = self.input_h // self.grid_size
        self.output_shape = (wd, hd)
        xmap_d, ymap_d = np.meshgrid(
            np.arange(wd),
            np.arange(hd),
            indexing='ij'
        )
        fx_d = fx * (wd / self.width)
        fy_d = fy * (hd / self.height)
        cx_d = cx * (wd / self.width)
        cy_d = cy * (hd / self.height)
        self.points_x_downscale = torch.from_numpy((xmap_d - cx_d) / fx_d).float()
        self.points_y_downscale = torch.from_numpy((ymap_d - cy_d) / fy_d).float()

    def to_scene_points(self,
                        rgbs: torch.Tensor,
                        depths: torch.Tensor,
                        include_rgb: bool = True):
        batch_size = rgbs.shape[0]
        feat_dim = 3 + (3 if include_rgb else 0)
        pts_all = -torch.ones((batch_size, self.all_points_num, feat_dim),
                              dtype=torch.float32, device=rgbs.device)
        masks = (depths > 0)
        zs = depths / 1000.0
        xs = self.points_x.to(rgbs.device) * zs
        ys = self.points_y.to(rgbs.device) * zs

        for b in range(batch_size):
            pts = torch.stack([xs[b], ys[b], zs[b]], dim=-1).reshape(-1, 3)
            m = masks[b].reshape(-1)
            pts = pts[m]
            if include_rgb:
                cols = rgbs[b].reshape(3, -1).T
                cols = cols[m]
            n_pts = pts.shape[0]
            if n_pts >= self.all_points_num:
                idxs = random.sample(range(n_pts), self.all_points_num)
            else:
                idxs = random.choices(range(n_pts), k=self.all_points_num)
            sel = torch.tensor(idxs, device=pts.device)
            pts = pts[sel]
            if include_rgb:
                cols = cols[sel]
                pts_all[b] = torch.cat([pts, cols], dim=1)
            else:
                pts_all[b] = pts
        return pts_all

    def to_xyz_maps(self, depths: torch.Tensor):
        depths = depths.unsqueeze(1)
        down = F.interpolate(depths,
                             size=self.output_shape,
                             mode='nearest').squeeze(1)
        zs = down / 1000.0
        xs = self.points_x_downscale.to(zs.device) * zs
        ys = self.points_y_downscale.to(zs.device) * zs
        xyz = torch.stack([xs, ys, zs], dim=1)
        return xyz


class GraspNode(Node):
    def __init__(self):
        super().__init__('grasp_node')
        # Parameters
        self.declare_parameter('color_topic', '/camera/d435/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/d435/aligned_depth_to_color/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/d435/color/camera_info')
        self.declare_parameter('checkpoint_path', './checkpoints/HGGD_realsense_checkpoint')
        self.declare_parameter('all_points_num', 25600)
        self.declare_parameter('anchor_num', 7)
        self.declare_parameter('ratio', 8)
        self.declare_parameter('anchor_k', 6)
        self.declare_parameter('anchor_w', 50.0)
        self.declare_parameter('anchor_z', 20.0)
        self.declare_parameter('grid_size', 8)
        self.declare_parameter('input_h', 360)
        self.declare_parameter('input_w', 640)
        self.declare_parameter('center_num', 48)
        self.declare_parameter('group_num', 512)
        self.declare_parameter('local_k', 10)
        self.declare_parameter('heatmap_thres', 0.01)
        self.declare_parameter('sigma', 10)

        # Topics
        ct = self.get_parameter('color_topic').value
        dt = self.get_parameter('depth_topic').value
        cit = self.get_parameter('camera_info_topic').value

        self.bridge = CvBridge()
        self.marker_pub = self.create_publisher(MarkerArray, 'grasp_markers', 10)

        self.camera_info = None
        self.create_subscription(CameraInfo, cit, self.camera_info_cb, 10)

        self.color_sub = message_filters.Subscriber(self, Image, ct)
        self.depth_sub = message_filters.Subscriber(self, Image, dt)
        ats = message_filters.ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub], queue_size=10, slop=0.05
        )
        ats.registerCallback(self.image_cb)

        self.pc_helper = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.anchornet = None
        self.localnet = None
        self.anchors = None

        # 缓存最新的 RGB/Depth
        self.latest_rgb   = None
        self.latest_depth = None

        # 在 __init__ 结尾，创建一个 GetGrasp 服务
        self.srv = self.create_service(
            GetGrasp,
            'get_grasps',
            self.get_grasps_callback
        )

    def camera_info_cb(self, msg: CameraInfo):
        if self.camera_info is None:
            self.camera_info = msg
            self.get_logger().info('Got CameraInfo, building models…')
            self.build_models()

    def build_models(self):
        k = self.camera_info.k
        fx, fy = k[0], k[4]
        cx, cy = k[2], k[5]

        self.pc_helper = PointCloudHelper(
            all_points_num=self.get_parameter('all_points_num').value,
            fx=fx, fy=fy, cx=cx, cy=cy,
            width=self.camera_info.width,
            height=self.camera_info.height,
            input_w=self.get_parameter('input_w').value,
            input_h=self.get_parameter('input_h').value,
            grid_size=self.get_parameter('grid_size').value
        )

        self.anchornet = AnchorGraspNet(
            in_dim=4,
            ratio=self.get_parameter('ratio').value,
            anchor_k=self.get_parameter('anchor_k').value
        ).to(self.device)
        self.localnet = PointMultiGraspNet(
            info_size=3,
            k_cls=self.get_parameter('anchor_num').value**2
        ).to(self.device)

        ckpt = torch.load(self.get_parameter('checkpoint_path').value,
                          map_location=self.device)
        self.anchornet.load_state_dict(ckpt['anchor'])
        self.localnet.load_state_dict(ckpt['local'])

        arange = torch.linspace(-1, 1,
                                self.get_parameter('anchor_num').value + 1
                               ).to(self.device)
        self.anchors = {
            'gamma': ckpt['gamma'].to(self.device),
            'beta':  ckpt['beta'].to(self.device)
        }

        self.anchornet.eval()
        self.localnet.eval()
        self.get_logger().info('Models loaded and ready.')

    def image_cb(self, color_msg: Image, depth_msg: Image):
        # 1) 解码并缓存最新的图像／深度
        cv_color = self.bridge.imgmsg_to_cv2(color_msg, 'bgr8')
        cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, 'passthrough')
        self.latest_rgb   = cv2.cvtColor(cv_color, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        self.latest_depth = cv_depth.astype(np.float32)

    def get_grasps_callback(self, request, response):
        # 只对特定命令触发推理
        if request.input != 'start_grasp':
            response.success = False
            response.output.markers = MarkerArray().markers
            return response

        # 确保已有一帧缓存
        if self.latest_rgb is None or self.latest_depth is None:
            self.get_logger().warn("No image+depth buffered yet")
            response.success = False
            response.output.markers = MarkerArray().markers
            return response

        # —— 下面直接复用原来 image_cb 里的推理 pipeline —— #
        # 1) 转 torch.Tensor
        ori_rgb = torch.from_numpy(self.latest_rgb).permute(2,1,0)[None]\
                      .to(self.device, dtype=torch.float32)
        ori_depth = torch.from_numpy(self.latest_depth).T[None]\
                        .to(self.device, dtype=torch.float32)

        view_pts = self.pc_helper.to_scene_points(ori_rgb, ori_depth, include_rgb=True)
        xyzs     = self.pc_helper.to_xyz_maps(ori_depth)

        #    —— 首先对深度做原始范围截断（0–1500 mm），再转换到米，并中心化后 clip 到 [-2,2]
        rgb_in = F.interpolate(
            ori_rgb,
            size=(self.get_parameter('input_w').value,
                self.get_parameter('input_h').value)
        )
        depth_in = F.interpolate(
            ori_depth.unsqueeze(1),
            size=(self.get_parameter('input_w').value,
                self.get_parameter('input_h').value),
            mode='nearest'
        ).squeeze(1)

        # ① 原始 mm 范围截断
        depth_in = torch.clamp(depth_in, 0.0, 2000.0)
        # ② 转成 米
        depth_in = depth_in / 1000.0
        # ③ 中心化（减掉图像平均值）
        depth_in = depth_in - depth_in.mean()
        # ④ 截断到 [-2,2]
        depth_in = torch.clamp(depth_in, -1, 1)

        # 最后拼成 (B,4,H,W) 的网络输入
        x = torch.cat([depth_in.unsqueeze(1), rgb_in], dim=1)

        # 4) 前向推理 + 2D→6D → collision → nms
        with torch.no_grad():
            p2d, feat = self.anchornet(x)
            loc_map, cls_mask, th_off, h_off, w_off = anchor_output_process(
                *p2d, sigma=self.get_parameter('sigma').value
            )
            rect_gg = detect_2d_grasp(
                loc_map, cls_mask, th_off, h_off, w_off,
                ratio=self.get_parameter('ratio').value,
                anchor_k=self.get_parameter('anchor_k').value,
                anchor_w=self.get_parameter('anchor_w').value,
                anchor_z=self.get_parameter('anchor_z').value,
                mask_thre=self.get_parameter('heatmap_thres').value,
                center_num=self.get_parameter('center_num').value,
                grid_size=self.get_parameter('grid_size').value,
                grasp_nms=self.get_parameter('grid_size').value,
                reduce='max'
            )
            if rect_gg.size == 0:
                response.success = False
                response.output.markers = MarkerArray().markers
                return response

            pts_all = feature_fusion(view_pts[..., :3], feat, xyzs)
            pcg, valid_centers = data_process(
                pts_all, ori_depth, [rect_gg],
                self.get_parameter('center_num').value,
                self.get_parameter('group_num').value,
                (self.get_parameter('input_w').value,
                 self.get_parameter('input_h').value),
                min_points=32,
                is_training=False
            )
            # 对齐 gi
            g_t = torch.from_numpy(rect_gg.thetas).to(self.device, dtype=torch.float32)
            g_w = torch.from_numpy(rect_gg.widths).to(self.device, dtype=torch.float32)
            g_d = torch.from_numpy(rect_gg.depths).to(self.device, dtype=torch.float32)
            gi  = torch.stack([g_t, g_w, g_d], dim=1)  # → FloatTensor on self.device

            _, pred_loc, off = self.localnet(pcg, gi)
            _, rect6 = detect_6d_grasp_multi(
                rect_gg, pred_loc, off, valid_centers,
                (self.get_parameter('input_w').value,
                 self.get_parameter('input_h').value),
                self.anchors, k=self.get_parameter('local_k').value
            )
            gg, _ = collision_detect(
                pts_all.squeeze(0),
                rect6.to_6d_grasp_group(depth=0.02),
                mode='graspnet'
            )
            gg = gg.nms()

        # 5) 根据结果构造 MarkerArray 并填入 response
        ma = MarkerArray()
        for i, g in enumerate(gg):
            ma.markers.append(self._grasp_to_marker(g, i))
        self.marker_pub.publish(ma)
        response.output.markers = ma.markers
        response.success = True
        return response


    def _grasp_to_pose(self, g):
        """
        把一个 Grasp 对象 g 转成 geometry_msgs/Pose
        """
        # 1) 取旋转
        rot = g.rotation
        if isinstance(rot, torch.Tensor):
            R = rot.cpu().numpy()
        else:
            R = rot

        # 2) 取平移
        trans = g.translation
        if isinstance(trans, torch.Tensor):
            t = trans.cpu().numpy()
        else:
            t = trans

        # 3) 构造 Pose
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = map(float, t)
        # Z 轴需要翻转
        homo = np.eye(4)
        homo[:3, :3] = -R
        qx, qy, qz, qw = quaternion_from_matrix(homo)
        pose.orientation.x = qx
        pose.orientation.y = qy
        pose.orientation.z = qz
        pose.orientation.w = qw
        return pose
    
    def _grasp_to_marker(self, grasp, idx):
        pose = self._grasp_to_pose(grasp)
        m = Marker()
        m.header.frame_id = 'd435_color_optical_frame'
        m.ns = 'grasp_markers'
        m.id = idx
        m.type   = Marker.ARROW
        m.action = Marker.ADD
        m.pose   = pose
        # 长度和粗细
        m.scale.x = 0.1  # 箭头长度
        m.scale.y = 0.01 # 箭杆直径
        m.scale.z = 0.02 # 箭头直径
        print("grasps",idx,":",pose,"score:",grasp.score)
        # 颜色
        m.color.r = 1.0; m.color.g = 0.0; m.color.b = 0.0; m.color.a = grasp.score
        return m

def main(args=None):
    rclpy.init(args=args)
    node = GraspNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

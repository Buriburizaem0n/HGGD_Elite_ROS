#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import numpy as np
import cv2
import open3d as o3d

def parse_args():
    parser = argparse.ArgumentParser(
        description="将 RGB 图像和深度图像合成为点云，并使用 Open3D 可视化"
    )
    parser.add_argument(
        "--rgb", "-r", required=True,
        help="RGB 图像路径 (e.g. /path/to/frame__color.png)"
    )
    parser.add_argument(
        "--depth", "-d", required=True,
        help="深度图像路径 (单位通常为毫米, e.g. /path/to/frame__depth.png)"
    )
    parser.add_argument(
        "--fx", type=float, default=906.26, help="相机焦距 fx"
    )
    parser.add_argument(
        "--fy", type=float, default=906.77, help="相机焦距 fy"
    )
    parser.add_argument(
        "--cx", type=float, default=651.04, help="相机主点 cx"
    )
    parser.add_argument(
        "--cy", type=float, default=357.40, help="相机主点 cy"
    )
    parser.add_argument(
        "--depth_scale", type=float, default=1000.0,
        help="深度缩放：将原始深度单位转为米 (默认为 1000)"
    )
    parser.add_argument(
        "--depth_trunc", type=float, default=3.0,
        help="深度最大截断值(米)，超过该值的深度点将被置零"
    )
    return parser.parse_args()

def check_file(path):
    if not os.path.isfile(path):
        print(f"[ERROR] 找不到文件: {path}", file=sys.stderr)
        sys.exit(1)

def rgbd_to_pointcloud(rgb_path, depth_path, fx, fy, cx, cy, depth_scale, depth_trunc):
    # 读取 RGB 图
    color_bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    if color_bgr is None:
        print(f"[ERROR] 无法读取 RGB 图像，请检查文件是否损坏或路径是否正确: {rgb_path}", file=sys.stderr)
        sys.exit(1)
    color = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)

    # 读取深度图
    depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth_raw is None:
        print(f"[ERROR] 无法读取深度图像，请检查文件是否损坏或路径是否正确: {depth_path}", file=sys.stderr)
        sys.exit(1)
    depth = depth_raw.astype(np.float32) / depth_scale
    depth[depth > depth_trunc] = 0

    h, w = depth.shape
    u = np.arange(w)
    v = np.arange(h)
    uu, vv = np.meshgrid(u, v)

    z = depth
    x = (uu - cx) * z / fx
    y = (vv - cy) * z / fy

    pts = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    colors = color.reshape(-1, 3) / 255.0

    valid = pts[:, 2] > 0
    pts = pts[valid]
    colors = colors[valid]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

if __name__ == "__main__":
    args = parse_args()
    check_file(args.rgb)
    check_file(args.depth)

    pcd = rgbd_to_pointcloud(
        args.rgb, args.depth,
        fx=args.fx, fy=args.fy, cx=args.cx, cy=args.cy,
        depth_scale=args.depth_scale, depth_trunc=args.depth_trunc
    )

    print(f"点云生成完毕，共 {len(pcd.points)} 个点，开始可视化...")
    o3d.visualization.draw_geometries(
        [pcd],
        window_name="RGB-D Point Cloud",
        width=800, height=600,
        point_show_normal=False
    )

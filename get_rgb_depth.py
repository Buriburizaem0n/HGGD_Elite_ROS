#!/usr/bin/env python3
"""
ROS 2 utility — Save RealSense RGB & Depth frames as PNG files
==============================================================
* Subscribes:
    • /camera/d435/color/image_raw               (sensor_msgs/Image, rgb8)
    • /camera/d435/aligned_depth_to_color/image_raw (sensor_msgs/Image, 16UC1 or 32FC1)
* Saves          : <out_dir>/<prefix>_<stamp>_rgb.png and *_depth.png  (16‑bit depth)

Run:
-----
conda activate hggd   # or system python with rclpy/cv_bridge
python save_realsense_png.py --out-dir captures --prefix sample

Stop with Ctrl‑C.
"""
import os, time, argparse, pathlib
from datetime import datetime

import cv2
import numpy as np
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy

parser = argparse.ArgumentParser("save_realsense_png")
parser.add_argument('--color-topic', default='/camera/d435/color/image_raw')
parser.add_argument('--depth-topic', default='/camera/d435/aligned_depth_to_color/image_raw')
parser.add_argument('--out-dir',     default='captures')
parser.add_argument('--prefix',      default='frame')
args = parser.parse_args()

pathlib.Path(args.out_dir).mkdir(parents=True, exist_ok=True)

class Saver(Node):
    def __init__(self):
        super().__init__('realsense_png_saver')
        self.bridge = CvBridge()
        self.rgb_msg = None
        qos = QoSProfile(depth=10)
        qos.reliability = QoSReliabilityPolicy.BEST_EFFORT
        qos.durability = QoSDurabilityPolicy.VOLATILE
        self.create_subscription(Image, args.color_topic, self.cb_rgb, qos)
        self.create_subscription(Image, args.depth_topic, self.cb_depth, qos)
        self.get_logger().info(f"Saving PNGs to {args.out_dir}/ (Ctrl‑C to stop)")

    def cb_rgb(self, msg):
        self.rgb_msg = msg

    def cb_depth(self, msg):
        if self.rgb_msg is None:
            return                        # wait until RGB arrives
        try:
            rgb = self.bridge.imgmsg_to_cv2(self.rgb_msg, 'rgb8')
            depth = self.bridge.imgmsg_to_cv2(msg, 'passthrough')   # 16UC1 or 32FC1
            if depth.dtype != np.uint16:
                depth = (depth * 1000.0).astype(np.uint16)          # m → mm 16‑bit
            # build filenames
            stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
            rgb_path   = os.path.join(args.out_dir, f"{args.prefix}__rgb.png")
            
            depth_path = os.path.join(args.out_dir, f"{args.prefix}__depth.png")
            cv2.imwrite(rgb_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            cv2.imwrite(depth_path, depth)
            self.get_logger().info(f"Saved {rgb_path} and {depth_path}")
        except Exception as e:
            self.get_logger().error(f"Save error: {e}")


def main():
    rclpy.init()
    node = Saver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node(); rclpy.shutdown()

if __name__ == '__main__':
    main()

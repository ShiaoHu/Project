#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Point
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import torch
import numpy as np
import os
import sys
import yaml
from message_filters import ApproximateTimeSynchronizer, Subscriber
from yolo_color_detection.msg import ObjectInfo

# Add yolov5 module path
yolov5_path = os.path.join(os.path.dirname(__file__), '..', 'yolov5')
sys.path.append(os.path.abspath(yolov5_path))

from models.common import DetectMultiBackend
from utils.general import non_max_suppression
from utils.torch_utils import select_device
from utils.augmentations import letterbox

class YoloColorNode:
    def __init__(self):
        rospy.init_node('yolo_color_node')
        self.bridge = CvBridge()

        # Loading Models
        model_path = os.path.join(os.path.dirname(__file__), '..', 'best.pt')
        self.device = select_device('')
        self.model = DetectMultiBackend(model_path, device=self.device)
        self.model.eval()
        self.img_size = 1280

        # Load Camera External Reference
        with open(os.path.join(os.path.dirname(__file__), '..', 'camera_to_base.yaml')) as f:
            calib = yaml.safe_load(f)
        self.R = np.array(calib['rotation_matrix'])
        self.T = np.array(calib['translation']).reshape((3, 1))

        # publisher
        self.pub_info = rospy.Publisher('/detection_info', String, queue_size=10)
        self.pub_object_info = rospy.Publisher('/object_in_base', ObjectInfo, queue_size=10)

        # Subscribe to Camera Images, Depth, Insights
        self.rgb_sub = Subscriber('/camera/color/image_raw', Image)
        self.depth_sub = Subscriber('/camera/depth/image_rect_raw', Image)
        self.info_sub = Subscriber('/camera/color/camera_info', CameraInfo)

        # simultaneous reception
        self.ts = ApproximateTimeSynchronizer([self.rgb_sub, self.depth_sub, self.info_sub], 10, 0.1)
        self.ts.registerCallback(self.synced_callback)

        rospy.loginfo("YOLO Color Node Ready with TF conversion")
        rospy.spin()

    def synced_callback(self, rgb_msg, depth_msg, info_msg):
        frame = self.bridge.imgmsg_to_cv2(rgb_msg, 'bgr8')
        depth = self.bridge.imgmsg_to_cv2(depth_msg, 'passthrough').astype(np.float32) / 1000.0

        img = letterbox(frame, self.img_size, stride=32, auto=True)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img_tensor = torch.from_numpy(img).to(self.device).float() / 255.0
        if img_tensor.ndimension() == 3:
            img_tensor = img_tensor.unsqueeze(0)

        pred = self.model(img_tensor)
        pred = non_max_suppression(pred, 0.25, 0.45)[0]

        if pred is not None and len(pred):
            pred[:, :4] = self.scale_coords(img_tensor.shape[2:], pred[:, :4], frame.shape).round()
            for *xyxy, conf, cls in pred:
                x1, y1, x2, y2 = map(int, xyxy)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                if cy >= depth.shape[0] or cx >= depth.shape[1]:
                    continue
                z = depth[cy, cx]
                if z == 0 or np.isnan(z): continue

                x_cam, y_cam, z_cam = self.deproject(cx, cy, z, info_msg)
                color = self.detect_color(cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2HSV))
                label = self.model.names[int(cls)]

                # Coordinate conversion
                xyz_cam = np.array([[x_cam], [y_cam], [z_cam]])
                xyz_base = np.dot(self.R, xyz_cam) + self.T
                x_b, y_b, z_b = xyz_base.flatten()

                # Optional: Print debugging information
                msg_str = f'{label} ({color}) , BASE ({x_b:.3f}, {y_b:.3f}, {z_b:.3f})'
                self.pub_info.publish(msg_str)
                rospy.loginfo(msg_str)

                # Publishing Custom ObjectInfo Messages
                obj_msg = ObjectInfo()
                obj_msg.label = label
                obj_msg.color = color
                obj_msg.position = Point(x_b, y_b, z_b)
                self.pub_object_info.publish(obj_msg)

        else:
            rospy.logwarn("YOLOv5 No target detected")

    def deproject(self, u, v, depth, info):
        fx, fy = info.K[0], info.K[4]
        cx, cy = info.K[2], info.K[5]
        x = (u - cx) * depth / fx
        y = (v - cy) * depth / fy
        return x, y, depth

    def scale_coords(self, img1_shape, coords, img0_shape):
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = ((img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2)
        coords[:, [0, 2]] -= pad[0]
        coords[:, [1, 3]] -= pad[1]
        coords[:, :4] /= gain
        coords[:, :4] = coords[:, :4].clamp(min=0)
        return coords

    def detect_color(self, hsv_roi):
        color_ranges = {
            'red': [((0, 100, 100), (10, 255, 255)), ((160, 100, 100), (180, 255, 255))],
            'green': [((35, 100, 100), (85, 255, 255))],
            'blue': [((85, 50, 50), (135, 255, 255))],
            'yellow': [((20, 100, 100), (35, 255, 255))]
        }
        max_count, detected = 0, 'unknown'
        for color, ranges in color_ranges.items():
            mask = None
            for lower, upper in ranges:
                part = cv2.inRange(hsv_roi, np.array(lower), np.array(upper))
                mask = part if mask is None else cv2.bitwise_or(mask, part)
            count = cv2.countNonZero(mask)
            if count > max_count:
                max_count = count
                detected = color
        return detected

if __name__ == '__main__':
    try:
        YoloColorNode()
    except rospy.ROSInterruptException:
        pass
    cv2.destroyAllWindows()


import os
import cv2
import torch
import numpy as np
import scipy.special
from PIL import Image
from model.model import parsingNet
from utils.common import merge_config
import torchvision.transforms as transforms
import time
import sys
import threading
import queue
from data.constant import culane_row_anchor, tusimple_row_anchor
from pathlib import Path
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.torch_utils import select_device
from utils.plots import plot_one_box
import traceback

#!/usr/bin/env python3
import pyrealsense2 as rs
import numpy as np
import cv2
import queue
import threading
import time

from ct import *

class YoloV5Detector:
    def __init__(self, weights='yolov5s.pt', img_size=640, conf_thres=0.45,
                 iou_thres=0.45, device='', classes=[0, 1, 2, 3, 5], stop_event=None):
                 
        self.weights = weights
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.classes = classes
        self.device = select_device(device)
        self.stop_event = stop_event or threading.Event()

        # 模型加载
        self.model = attempt_load(weights, map_location=self.device)
        self.model.eval()
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.stride = int(self.model.stride.max())
        self.img_size = check_img_size(img_size, s=self.stride)

    def stop(self):
        self.stop_event.set()

    def preprocess(self, image):
        img = letterbox(image, self.img_size, stride=self.stride)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
        img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
        return torch.from_numpy(img).unsqueeze(0).to(self.device), image

    def detect(self, img_tensor):
        with torch.no_grad():
            pred = self.model(img_tensor)[0]
            return non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.classes)

    def run(self, q: queue.Queue, q_send: queue.Queue):
        print("[INFO] YOLOv5 Detector started.")
        while not self.stop_event.is_set():
            try:
                frame = q.get(timeout=0.1)
                img_tensor, im0s = self.preprocess(frame)
                preds = self.detect(img_tensor)

                detection = []

                for det in preds:
                    if det is not None and len(det):
                        det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], im0s.shape).round()
                        for *xyxy, conf, cls in det:

                            x1, y1, x2, y2 = map(int, xyxy)
                            class_id = int(cls)
                            # 添加到结果列表中
                            detection.append([class_id, x1, y1, x2, y2])


                            label = f'{self.names[int(cls)]} {conf:.2f}'
                            plot_one_box(xyxy, im0s, label=label, color=(0, 255, 0), line_thickness=2)
                    
                    

                # print(detection)
                # print()
                q_send.put(detection.copy())

                # cv2.imshow("YOLOv5 Detection", im0s)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     self.stop()
                #     break

            except queue.Empty:
                continue
            except Exception as e:
                print(f"[ERROR] {e}")
                self.stop()
                break

        print("[INFO] YOLOv5 Detector stopped.")


def process_lane_and_detection_3d(yolo_send_queue: queue.Queue, depth_queue: queue.Queue, K):
    fx, fy = K[0][0], K[1][1]
    cx, cy = K[0][2], K[1][2]
    detection_3d_box = []
    
    # Get depth image and 2D detections
    depth = depth_queue.get(timeout=0.1)
    detection = yolo_send_queue.get(timeout=0.1)

    if detection:
        # Process each detection box
        for det in detection:
            class_id, x1, y1, x2, y2 = det
            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(depth.shape[1] - 1, int(x2))
            y2 = min(depth.shape[0] - 1, int(y2))

            depth_roi = depth[y1:y2, x1:x2].astype(np.float32)
            mask = depth_roi > 0

            if not np.any(mask):
                continue

            u_coords, v_coords = np.meshgrid(np.arange(x1, x2), np.arange(y1, y2))
            u_coords = u_coords[mask]
            v_coords = v_coords[mask]
            d = depth_roi[mask]

            # Calculate points in camera coordinates
            x_cam = (u_coords - cx) * d / fx
            y_cam = (v_coords - cy) * d / fy
            z_cam = d

            # Transform to vehicle coordinates:
            # x_vehicle = x_camera (right)
            # y_vehicle = z_camera (forward)
            # z_vehicle = -y_camera (up)
            x_veh = x_cam
            y_veh = z_cam
            z_veh = -y_cam

            # Stack points in vehicle coordinates
            points = np.stack([x_veh, y_veh, z_veh], axis=1)

            # Sort by forward distance (Y in vehicle coordinates) instead of depth
            sorted_indices = np.argsort(points[:, 1])  # Y is forward in vehicle frame
            keep_count = int(len(points) * 0.3)
            points = points[sorted_indices[:keep_count]]

            # Calculate bounding box in vehicle coordinates
            x_min, y_min, z_min = points.min(axis=0)
            x_max, y_max, z_max = points.max(axis=0)

            # Create 3D bounding box with dimensions in vehicle coordinates:
            # (class_id, center_x, center_y, center_z, width, length, height)
            detection_3d_box.append((
                class_id,
                (x_max + x_min) / 2,       # Center X (right)
                (y_max + y_min) / 2,       # Center Y (forward)
                (z_max + z_min) / 2,       # Center Z (up)
                x_max - x_min,             # Width along X axis
                y_max - y_min,             # Length along Y axis (forward)
                z_max - z_min              # Height along Z axis (up)
            ))

    return detection_3d_box





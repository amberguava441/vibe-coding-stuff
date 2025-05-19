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

class UFLDDetector:
    def __init__(self, stop_event):
        self.stop_event = stop_event
        self.cfg, self.net, self.transform = self.load_configuration()
        self.row_anchor = culane_row_anchor if self.cfg.dataset == 'CULane' else tusimple_row_anchor
        self.col_sample_w = (800 - 1) / (self.cfg.griding_num - 1)
        self.frame_id, self.time_sum = 0, 0

    def load_configuration(self):
        """加载配置并初始化模型"""
        if len(sys.argv) == 1:  # 没有传递参数时添加默认配置
            # sys.argv += ['configs/tusimple.py', '--test_model', 'experiments/20250511_161010_lr_4e-04_b_32/ep099.pth']
            sys.argv += ['configs/tusimple.py', '--test_model', '/home/chen/Code/hackathon/ep098.pth']
        args, cfg = merge_config()  # 加载配置
        print("Start testing...")
        print("CUDA Available:", torch.cuda.is_available())

        # 加载模型
        net = self.load_model(cfg)
        transform = transforms.Compose([
            transforms.Resize((288, 800)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        return cfg, net, transform

    def load_model(self, cfg):
        """加载并返回训练好的模型"""
        net = parsingNet(
            pretrained=False,
            backbone=cfg.backbone,
            cls_dim=(cfg.griding_num + 1, 56 if cfg.dataset == 'Tusimple' else 18, 4),  # 简化了 cls_num 的判断
            use_aux=False
        ).cuda() 
        
        state_dict = torch.load(cfg.test_model, map_location='cpu')['model']
        net.load_state_dict({k[7:] if 'module.' in k else k: v for k, v in state_dict.items()}, strict=False)
        net.eval()
        return net

    def process_frame(self, frame):
        """处理每帧图像，进行车道线检测"""
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_tensor = self.transform(Image.fromarray(img_rgb)).unsqueeze(0).cuda()
        points = []

        with torch.no_grad():
            out = self.net(img_tensor)

        out_np = out[0].data.cpu().numpy()[:, ::-1, :]
        prob = scipy.special.softmax(out_np[:-1], axis=0)
        idx = np.arange(self.cfg.griding_num).reshape(-1, 1, 1) + 1
        loc = np.sum(prob * idx, axis=0)
        out_idx = np.argmax(out_np, axis=0)
        loc[out_idx == self.cfg.griding_num] = 0

        # 处理检测结果
        for i in range(loc.shape[1]):
            if np.count_nonzero(loc[:, i]) > 2:
                for k in range(loc.shape[0]):
                    if loc[k, i] > 0:
                        x = max(0, min(frame.shape[1] - 1, int(loc[k, i] * self.col_sample_w * frame.shape[1] / 800) - 1))
                        y = max(0, min(frame.shape[0] - 1, int(frame.shape[0] * (self.row_anchor[56 - 1 - k] / 288)) - 1))  # 防止超出范围
                        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                        points.append((x,y))
        return frame, points

    def process_and_display(self, q, q_send):
        """处理图像并显示结果"""
        while not self.stop_event.is_set():
            start_time = time.time()
            try:
                frame = q.get(timeout=0.1)
                frame, points = self.process_frame(frame)
                # print(points)
                # print()
                q_send.put(points.copy())

                cv2.imshow("Lane Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Interrupted by user.")
                    self.stop()
                    break

                self.time_sum += time.time() - start_time
                if (self.frame_id + 1) % 50 == 0:
                    print(f"[Frame {self.frame_id + 1}] Avg FPS: {50 / self.time_sum:.2f}")
                    self.time_sum = 0
                self.frame_id += 1


            except queue.Empty:
                continue
            except Exception as e:
                print(f"[ERROR] {e}")
                self.stop()
                break

    def run(self, q, q_send):
        """启动处理和显示"""
        self.process_and_display(q, q_send)

    def stop(self):
        """停止处理"""
        self.stop_event.set()

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

                cv2.imshow("YOLOv5 Detection", im0s)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.stop()
                    break

            except queue.Empty:
                continue
            except Exception as e:
                print(f"[ERROR] {e}")
                self.stop()
                break

        print("[INFO] YOLOv5 Detector stopped.")

# class get_realsense:
#     def __init__(self):
#         """Initialize the RealSense camera with optimized settings."""
#         # Configure depth and color streams
#         self.pipeline = rs.pipeline()
#         self.config = rs.config()
        
#         # Enable streams with higher framerate
#         self.config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 60)
#         self.config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 60)
        
#         # Start streaming
#         self.profile = self.pipeline.start(self.config)
        
#         # Get device info
#         self.device = self.profile.get_device()
#         self.device_name = self.device.get_info(rs.camera_info.name)
#         print(f"Connected to {self.device_name}")
        
#         # Allow camera to warm up and auto-exposure to stabilize
#         print("Allowing camera to warm up for 2 seconds...")
#         time.sleep(2)
        
#         # Align depth frame to color frame
#         self.align = rs.align(rs.stream.color)
        
#         # Set up post-processing filters
#         self.setup_filters()
        
#         # Image caches for faster repeated access
#         self.last_color_frame_time = 0
#         self.last_depth_frame_time = 0
#         self.cached_color_image = None
#         self.cached_depth_colormap = None
#         self.cache_valid_duration = 0.05  # 50ms cache validity
#         self.cached_raw_depth = None
        
#         print(f"{self.device_name} initialized successfully")
    
#     def setup_filters(self):
#         """Set up post-processing filters with optimized parameters."""
#         # Decimation filter reduces resolution of depth image
#         self.decimation = rs.decimation_filter()
#         self.decimation.set_option(rs.option.filter_magnitude, 2)  # Reduces resolution by 2x
        
#         # Spatial filter smooths depth image by looking at adjacent pixels
#         self.spatial = rs.spatial_filter()
#         self.spatial.set_option(rs.option.filter_magnitude, 3)
#         self.spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
#         self.spatial.set_option(rs.option.filter_smooth_delta, 20)
        
#         # Temporal filter reduces temporal noise
#         self.temporal = rs.temporal_filter()
#         self.temporal.set_option(rs.option.filter_smooth_alpha, 0.4)
#         self.temporal.set_option(rs.option.filter_smooth_delta, 20)
        
#         # Hole filling filter fills small holes in depth image
#         self.hole_filling = rs.hole_filling_filter()
    
#     def get_rs_rgb(self):
#         """
#         Capture RGB image from RealSense camera with caching for performance.
        
#         Returns:
#             numpy.ndarray: RGB image or None if no valid data
#         """
#         current_time = time.time()
        
#         # Check if we have a valid cached image
#         if (self.cached_color_image is not None and 
#             current_time - self.last_color_frame_time < self.cache_valid_duration):
#             return self.cached_color_image.copy()  # Return a copy to avoid modification issues
        
#         try:
#             # Wait for a coherent pair of frames with shorter timeout
#             frames = self.pipeline.wait_for_frames(timeout_ms=1000)
            
#             # Align frames if successful
#             aligned_frames = self.align.process(frames)
#             color_frame = aligned_frames.get_color_frame()
            
#             if not color_frame or not color_frame.get_data():
#                 print("No valid RGB frame received")
#                 return None
            
#             # Convert to numpy array
#             color_image = np.asanyarray(color_frame.get_data())
            
#             # Check if the image is valid
#             if color_image.size == 0 or np.all(color_image == 0):
#                 print("Empty RGB image received")
#                 return None
            
#             # Update cache
#             self.cached_color_image = color_image.copy()
#             self.last_color_frame_time = current_time
            
#             return color_image
            
#         except Exception as e:
#             print(f"Error capturing RGB image: {e}")
#             return None

#     def get_rs_depth(self, colorized=False):
#         """
#         Capture depth image from RealSense camera with caching for performance.
        
#         Args:
#             colorized (bool): If True, return colorized depth map for visualization.
#                             If False, return raw depth values in millimeters.
        
#         Returns:
#             numpy.ndarray: Raw depth image in millimeters or colorized depth image
#         """
#         current_time = time.time()
        
#         # Define cache variables based on return type
#         cache_to_check = self.cached_depth_colormap if colorized else self.cached_raw_depth
#         last_time = self.last_depth_frame_time
        
#         # Check if we have a valid cached depth image
#         if (cache_to_check is not None and 
#             current_time - last_time < self.cache_valid_duration):
#             return cache_to_check.copy()  # Return a copy to avoid modification
        
#         try:
#             # Wait for a coherent pair of frames with shorter timeout
#             frames = self.pipeline.wait_for_frames(timeout_ms=1000)
            
#             # Align frames
#             aligned_frames = self.align.process(frames)
#             depth_frame = aligned_frames.get_depth_frame()
            
#             if not depth_frame or not depth_frame.get_data():
#                 print("No valid depth frame received")
#                 return None
            
#             # Apply filters in sequence for better performance
#             filtered_depth = depth_frame
            
#             # Only apply filters if we have a valid depth frame
#             try:
#                 # Apply filters in sequence
#                 filtered_depth = self.decimation.process(filtered_depth)
#                 filtered_depth = self.spatial.process(filtered_depth)
#                 filtered_depth = self.temporal.process(filtered_depth)
#                 filtered_depth = self.hole_filling.process(filtered_depth)
                
#                 # Check if the filtered depth frame is valid
#                 if not filtered_depth or not filtered_depth.get_data():
#                     print("Filtering produced invalid depth frame")
#                     return None
                
#                 # Convert to numpy array - this contains raw depth values in millimeters
#                 depth_image = np.asanyarray(filtered_depth.get_data())
                
#                 # Check if the depth image is valid
#                 if depth_image.size == 0 or np.all(depth_image == 0):
#                     print("Empty depth image received")
#                     return None
                
#                 # Cache the raw depth values
#                 self.cached_raw_depth = depth_image.copy()
#                 self.last_depth_frame_time = current_time
                
#                 # If colorized version requested, create and cache it
#                 if colorized:
#                     # Colorize depth map for visualization with improved contrast
#                     depth_colormap = cv2.applyColorMap(
#                         cv2.convertScaleAbs(depth_image, alpha=0.03), 
#                         cv2.COLORMAP_JET
#                     )
#                     self.cached_depth_colormap = depth_colormap.copy()
#                     return depth_colormap
#                 else:
#                     # Return the raw depth values in millimeters
#                     return depth_image
                    
#             except Exception as e:
#                 print(f"Error processing depth frame: {e}")
#                 return None
                
#         except Exception as e:
#             print(f"Error capturing depth image: {e}")
#             return None

    
#     def close(self):
#         """
#         Stop the pipeline and release resources
#         """
#         try:
#             self.pipeline.stop()
#             print(f"{self.device_name} stopped")
#             return True
#         except Exception as e:
#             print(f"Error closing RealSense camera: {e}")
#             return False


def process_lane_and_detection_3d(yolo_send_queue: queue.Queue, depth_queue: queue.Queue, K):
        fx, fy = K[0][0], K[1][1]
        cx, cy = K[0][2], K[1][2]
        detection_3d_box = []
        

        # depth_image = depth_queue.get(timeout=0.1)
        depth= depth_queue.get(timeout=0.1)
        #print(depth)

        detection = yolo_send_queue.get(timeout=0.1)
        # print(detection)

        if detection:
            # 处理检测框
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
                d = depth_roi[mask] / 1000

                x = (u_coords - cx) * d / fx
                y = (v_coords - cy) * d / fy
                z = d

                points = np.stack([x, y, z], axis=1)

                sorted_indices = np.argsort(points[:, 2])  # z 值升序
                keep_count = int(len(points) * 0.3)
                points = points[sorted_indices[:keep_count]]



                x_min, y_min, z_min = points.min(axis=0)
                x_max, y_max, z_max = points.max(axis=0)

                # detection_3d_box.append((
                #     class_id,
                #     x_max - x_min,
                #     y_max - y_min,
                #     z_max - z_min,
                #     (x_max + x_min) / 2,
                #     (y_max + y_min) / 2,
                #     (z_max + z_min) / 2
                # ))

                detection_3d_box.append((
                    class_id,
                    (x_max + x_min) / 2,
                    (y_max + y_min) / 2,
                    (z_max + z_min) / 2,
                    x_max - x_min,
                    y_max - y_min,
                    z_max - z_min,
                ))

        return detection_3d_box





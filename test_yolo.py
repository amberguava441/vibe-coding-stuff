#!/usr/bin/env python3
"""
Standalone YOLO Detection Module
Separates YOLO detection from main.py for headless operation
Detects obstacles and calculates 3D coordinates using depth frames
"""

import os
import cv2 as cv
import torch
import numpy as np
import time
import sys
import threading
import traceback
from datetime import datetime
from pathlib import Path

# YOLO imports
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.torch_utils import select_device
from utils.plots import plot_one_box

# Local imports
from cb import createEnv, destroyEnv, getFrame
from dcb import getdeepFrame  # This will need to be provided
from calibration import calibration, getcbpf
from ct import getF
#from inference import LaneDetectionTester
from scipy.io import savemat, loadmat

class StandaloneYoloDetector:
    def __init__(self, weights='yolov5s.pt', img_size=640, conf_thres=0.45,
                 iou_thres=0.45, device='', classes=[0, 1, 2, 3, 5], 
                 output_dir='yolo_detections'):
        """
        Initialize standalone YOLO detector
        
        Args:
            weights (str): Path to YOLO weights file
            img_size (int): Input image size
            conf_thres (float): Confidence threshold
            iou_thres (float): IoU threshold for NMS
            device (str): Device to run on ('', 'cpu', '0', '1', etc.)
            classes (list): List of class IDs to detect
            output_dir (str): Directory to save detection results
        """
        self.weights = weights
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.classes = classes
        self.device = select_device(device)
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"[INFO] Loading YOLO model: {weights}")
        print(f"[INFO] Using device: {self.device}")
        print(f"[INFO] Output directory: {output_dir}")
        
        # Load model
        self.model = attempt_load(weights, map_location=self.device)
        self.model.eval()
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.stride = int(self.model.stride.max())
        self.img_size = check_img_size(img_size, s=self.stride)
        
        print(f"[INFO] Model loaded successfully")
        print(f"[INFO] Detecting classes: {[self.names[i] for i in self.classes]}")

    def preprocess(self, image):
        """Preprocess image for YOLO inference"""
        img = letterbox(image, self.img_size, stride=self.stride)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
        img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
        return torch.from_numpy(img).unsqueeze(0).to(self.device), image

    def detect(self, img_tensor):
        """Run YOLO detection"""
        with torch.no_grad():
            pred = self.model(img_tensor)[0]
            return non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.classes)

    def process_detections(self, preds, im0s, img_tensor):
        """Process detection results and return 2D bounding boxes"""
        detections = []
        
        for det in preds:
            if det is not None and len(det):
                # Scale coordinates back to original image size
                det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], im0s.shape).round()
                
                for *xyxy, conf, cls in det:
                    x1, y1, x2, y2 = map(int, xyxy)
                    class_id = int(cls)
                    confidence = float(conf)
                    
                    # Add to results list
                    detections.append([class_id, x1, y1, x2, y2, confidence])
                    
                    # Draw bounding box on image
                    label = f'{self.names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, im0s, label=label, color=(0, 255, 0), line_thickness=2)
        
        return detections, im0s

    def calculate_3d_coordinates(self, detections, depth_frame, K):
        """
        Calculate 3D coordinates and dimensions of detected objects using the same logic as get_yolo.py
        
        Args:
            detections (list): 2D detection results
            depth_frame (np.ndarray): Depth image
            K (np.ndarray): Camera intrinsic matrix
            
        Returns:
            list: 3D bounding boxes with format:
                  (class_id, center_x, center_y, center_z, width, length, height, confidence)
        """
        if depth_frame is None or len(detections) == 0:
            return []
            
        fx, fy = K[0][0], K[1][1]
        cx, cy = K[0][2], K[1][2]
        detection_3d_boxes = []
        
        for det in detections:
            class_id, x1, y1, x2, y2, confidence = det
            
            # Ensure coordinates are within image bounds
            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(depth_frame.shape[1] - 1, int(x2))
            y2 = min(depth_frame.shape[0] - 1, int(y2))
            
            # Extract depth ROI
            depth_roi = depth_frame[y1:y2, x1:x2].astype(np.float32)
            mask = depth_roi > 0  # Valid depth pixels
            
            if not np.any(mask):
                continue
            
            # Get pixel coordinates within the ROI
            u_coords, v_coords = np.meshgrid(np.arange(x1, x2), np.arange(y1, y2))
            u_coords = u_coords[mask]
            v_coords = v_coords[mask]
            d = depth_roi[mask]  # Already in meters from dcb.py (multiplied by 0.001)
            
            # Calculate 3D points in camera coordinates
            x_cam = (u_coords - cx) * d / fx
            y_cam = (v_coords - cy) * d / fy
            z_cam = d
            
            # Transform to vehicle coordinates with camera mounted at 0.7m height:
            # Camera coordinate system: X=right, Y=down, Z=forward
            # Vehicle coordinate system: X=right, Y=forward, Z=up, origin at ground level
            
            # Camera height above ground (meters)
            camera_height = 2
            
            # Transform from camera coordinates to vehicle coordinates:
            # x_vehicle = x_camera (right - same direction)
            # y_vehicle = z_camera (forward - same direction)  
            # z_vehicle = camera_height - y_camera (up - flip Y and add camera height offset)
            x_veh = x_cam
            y_veh = z_cam
            z_veh = camera_height - y_cam  # Ground level is at z=0, camera Y points down
            
            # Stack points in vehicle coordinates
            points = np.stack([x_veh, y_veh, z_veh], axis=1)
            
            # Sort by forward distance (Y in vehicle coordinates) and use closest 30%
            # This helps reduce noise from background objects
            sorted_indices = np.argsort(points[:, 1])  # Y is forward in vehicle frame
            keep_count = max(1, int(len(points) * 0.3))
            closest_points = points[sorted_indices[:keep_count]]
            
            # For better dimension estimation, also try using points within one standard deviation
            y_mean = np.mean(points[:, 1])
            y_std = np.std(points[:, 1])
            depth_threshold = min(1.0, y_std)  # Limit to 1 meter depth variation
            
            # Keep points within reasonable depth range
            depth_mask = np.abs(points[:, 1] - y_mean) <= depth_threshold
            filtered_points = points[depth_mask] if np.sum(depth_mask) > 10 else closest_points
            
            # Calculate 3D bounding box using filtered points
            x_min, y_min, z_min = filtered_points.min(axis=0)
            x_max, y_max, z_max = filtered_points.max(axis=0)
            
            # Calculate dimensions
            width = x_max - x_min
            length = y_max - y_min
            height = z_max - z_min
            
            # Create 3D bounding box with dimensions in vehicle coordinates
            detection_3d_boxes.append((
                class_id,
                (x_max + x_min) / 2,       # Center X (right)
                (y_max + y_min) / 2,       # Center Y (forward) 
                (z_max + z_min) / 2,       # Center Z (up)
                width,                     # Width along X axis
                length,                    # Length along Y axis (forward) - now limited
                height,                    # Height along Z axis (up)
                confidence                 # Detection confidence
            ))
        
        return detection_3d_boxes

    def save_detection_result(self, image, detections_3d, frame_count):
        """Save detection results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        
        if len(detections_3d) > 0:
            # Save image with detections
            image_filename = f"detection_{frame_count:06d}_{timestamp}.jpg"
            image_path = os.path.join(self.output_dir, image_filename)
            cv.imwrite(image_path, image)
            
            # Save detection data
            data_filename = f"detection_{frame_count:06d}_{timestamp}.txt"
            data_path = os.path.join(self.output_dir, data_filename)
            
            with open(data_path, 'w') as f:
                f.write(f"Timestamp: {timestamp}\n")
                f.write(f"Frame: {frame_count}\n")
                f.write(f"Detections: {len(detections_3d)}\n\n")
                
                for i, det in enumerate(detections_3d):
                    class_id, cx, cy, cz, w, l, h, conf = det
                    class_name = self.names[class_id]
                    f.write(f"Detection {i+1}:\n")
                    f.write(f"  Class: {class_name} (ID: {class_id})\n")
                    f.write(f"  Confidence: {conf:.3f}\n")
                    f.write(f"  Center: ({cx:.3f}, {cy:.3f}, {cz:.3f}) m\n")
                    f.write(f"  Dimensions: {w:.3f} x {l:.3f} x {h:.3f} m (W x L x H)\n\n")
            
            print(f"[INFO] Saved detection result: {image_filename}")
            return image_path, data_path
        
        return None, None

class CameraSystem:
    """Camera system wrapper using the existing cb.py framework"""
    
    def __init__(self, camera_id=100):
        """
        Initialize camera system
        
        Args:
            camera_id (int): Camera ID (100 for RealSense depth camera)
        """
        self.camera_id = camera_id
        self.cnl = [camera_id]
        
        # Camera intrinsic matrix (from c.py)
        self.K = np.array([
            [385.5,   0.0, 326.5],
            [  0.0, 384.6, 242.5], 
            [  0.0,   0.0,   1.0]
        ])
        
        # # Camera extrinsic parameters
        # self.R = np.array([
        #     [1, 0, 0],
        #     [0, 0, -1],
        #     [0, 1, 0]
        # ])
        # self.T = np.array([0, 0, 0])
        
        # Initialize camera environment
        print(f"[INFO] Initializing camera system with ID: {camera_id}")
        self.cl, self.dir = createEnv(self.cnl)
        self.cntok = {c[0]: k for k, c in enumerate(self.cl)}
        
        # Set up calibration
        self._setup_calibration()
        
        # Check if camera is working
        self.work = self._check_camera() == 0
        if not self.work:
            raise Exception(f"Camera {camera_id} initialization failed")
        
        print(f"[INFO] Camera system initialized successfully")
    
    def _setup_calibration(self):
        """Set up camera calibration"""
        for k, C in enumerate(self.cl):
            cn = C[0]
            calib_dir = os.path.join(self.dir, 'calibration', f'{cn}.mat')
            
            if os.path.exists(calib_dir):
                # Load existing calibration
                data = loadmat(calib_dir)
                self.cl[k][3] = data['mtx']
                self.cl[k][4] = data['dist']
                self.cl[k][5] = data['rvecs']
                self.cl[k][6] = data['tvecs']
                print(f"[INFO] Loaded calibration for camera {cn}")
            else:
                # Try to get calibration images
                cbpl = getcbpf(cn)
                if cbpl:
                    data = self._calibrate_camera(cn, cbpl)
                    if data['mtx'] is not None:
                        self.cl[k][3] = data['mtx']
                        self.cl[k][4] = data['dist']
                        self.cl[k][5] = data['rvecs']
                        self.cl[k][6] = data['tvecs']
                        print(f"[INFO] Calibrated camera {cn}")
                    else:
                        print(f"[WARNING] Calibration failed for camera {cn}")
                else:
                    print(f"[WARNING] No calibration images found for camera {cn}")
    
    def _calibrate_camera(self, cn, cbpl):
        """Calibrate camera using checkerboard images"""
        ret, mtx, dist, rvecs, tvecs, size = calibration(cbpl)
        
        data = {
            'ret': ret,
            'mtx': mtx,
            'dist': dist,
            'rvecs': np.array(rvecs).squeeze() if rvecs is not None else None,
            'tvecs': np.array(tvecs).squeeze() if tvecs is not None else None,
            'size': size
        }
        
        if ret is not None:
            savemat(os.path.join(self.dir, 'calibration', f'{cn}.mat'), data)
        
        return data
    
    def _check_camera(self):
        """Check if camera is working properly"""
        flag = 0
        for cn, cflag, c, mtx, _, _, _ in self.cl:
            if cflag == 0:  # Camera initialized successfully
                if cn < 100:
                    fflag, _ = getFrame(c)
                elif cn < 200:
                    fflag, _, _ = getdeepFrame(c)
                
                if fflag == 0 and mtx is not None:
                    print(f"[INFO] Camera {cn}: OK (calibrated)")
                else:
                    print(f"[WARNING] Camera {cn}: Frame={fflag==0}, Calibrated={mtx is not None}")
                    flag += 1
            else:
                print(f"[ERROR] Camera {cn}: Initialization failed")
                flag += 1
        
        return flag
    
    def get_frame(self):
        """
        Get RGB and depth frames from camera
        
        Returns:
            tuple: (flag, calibrated, rgb_frame, depth_frame)
        """
        cn, _, c, mtx, dist, _, _ = self.cl[self.cntok[self.camera_id]]
        
        if cn < 100:
            flag, rgb_frame = getFrame(c)
            depth_frame = None
        elif cn < 200:
            flag, rgb_frame, depth_frame = getdeepFrame(c)
        
        # Apply undistortion if calibration is available
        if mtx is not None and rgb_frame is not None:
            rgb_frame = cv.undistort(rgb_frame, mtx, dist)
        if mtx is not None and depth_frame is not None:
            depth_frame = cv.undistort(depth_frame, mtx, dist)
        
        calibrated = mtx is not None
        return flag, calibrated, rgb_frame, depth_frame
    
    def destroy(self):
        """Cleanup camera resources"""
        destroyEnv(self.cl)

class YoloDetectionSystem:
    def __init__(self, camera_config, yolo_config):
        """
        Initialize the complete YOLO detection system
        
        Args:
            camera_config (dict): Camera configuration
            yolo_config (dict): YOLO configuration
        """
        self.camera_config = camera_config
        self.yolo_config = yolo_config
        
        # Initialize camera system
        print("[INFO] Initializing camera system...")
        self.camera_system = CameraSystem(camera_config['camera_id'])
        
        # Initialize YOLO detector
        print("[INFO] Initializing YOLO detector...")
        self.yolo_detector = StandaloneYoloDetector(**yolo_config)
        
        self.frame_count = 0
        self.running = False
        
    def run_detection_loop(self, duration=None, detection_interval=1):
        """
        Run the detection loop
        
        Args:
            duration (float): Duration to run in seconds (None for infinite)
            detection_interval (float): Time between detections in seconds
        """
        print(f"[INFO] Starting detection loop...")
        print(f"[INFO] Detection interval: {detection_interval:.2f}s")
        
        self.running = True
        start_time = time.time()
        
        try:
            while self.running:
                loop_start = time.time()
                
                # Get camera frames
                try:
                    flag, calibrated, rgb_frame, depth_frame = self.camera_system.get_frame()
                    
                    if flag != 0 or rgb_frame is None:
                        print("[WARNING] Failed to get camera frame")
                        time.sleep(detection_interval)
                        continue
                        
                except Exception as e:
                    print(f"[ERROR] Camera frame acquisition failed: {e}")
                    time.sleep(detection_interval)
                    continue
                
                self.frame_count += 1
                
                # Run YOLO detection
                try:
                    # Preprocess image
                    img_tensor, im0s = self.yolo_detector.preprocess(rgb_frame)
                    
                    # Run detection
                    preds = self.yolo_detector.detect(img_tensor)
                    
                    # Process results
                    detections_2d, annotated_image = self.yolo_detector.process_detections(
                        preds, im0s.copy(), img_tensor
                    )
                    
                    # Calculate 3D coordinates if depth frame is available
                    detections_3d = []
                    if depth_frame is not None and len(detections_2d) > 0:
                        detections_3d = self.yolo_detector.calculate_3d_coordinates(
                            detections_2d, depth_frame, self.camera_system.K
                        )
                    
                    # Print detection results
                    if len(detections_3d) > 0:
                        print(f"[INFO] Frame {self.frame_count}: Detected {len(detections_3d)} obstacles")
                        for i, det in enumerate(detections_3d):
                            class_id, cx, cy, cz, w, l, h, conf = det
                            class_name = self.yolo_detector.names[class_id]
                            print(f"  {class_name}: Center=({cx:.2f}, {cy:.2f}, {cz:.2f})m, "
                                  f"Size=({w:.2f}x{l:.2f}x{h:.2f})m, Conf={conf:.3f}")
                        
                        # Save results
                        self.yolo_detector.save_detection_result(
                            annotated_image, detections_3d, self.frame_count
                        )
                    
                    elif self.frame_count % 50 == 0:  # Print status every 50 frames
                        print(f"[INFO] Frame {self.frame_count}: No obstacles detected")
                        
                except Exception as e:
                    print(f"[ERROR] Detection processing failed: {e}")
                    traceback.print_exc()
                
                # Check duration limit
                if duration is not None and (time.time() - start_time) >= duration:
                    print(f"[INFO] Duration limit reached ({duration}s)")
                    break
                
                # Maintain detection interval
                loop_time = time.time() - loop_start
                sleep_time = detection_interval - loop_time
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
        except KeyboardInterrupt:
            print("\n[INFO] Keyboard interrupt received")
        except Exception as e:
            print(f"[ERROR] Detection loop failed: {e}")
            traceback.print_exc()
        finally:
            self.running = False
            print("[INFO] Detection loop stopped")
    
    def stop(self):
        """Stop the detection system"""
        self.running = False
        if self.camera_system:
            self.camera_system.destroy()

def main():
    """Main entry point for standalone YOLO detection"""
    print("=" * 60)
    print("Standalone YOLO Object Detection System")
    print("=" * 60)
    
    # Configuration
    camera_config = {
        'camera_id': 100  # RealSense depth camera
    }
    
    yolo_config = {
        'weights': 'yolov5s.pt',
        'img_size': 640,
        'conf_thres': 0.45,
        'iou_thres': 0.45,
        'device': '',  # '' for auto-select, 'cpu' for CPU, '0' for GPU 0
        'classes': [0, 1, 2, 3, 5],  # person, bicycle, car, motorcycle, bus
        'output_dir': 'yolo_detections'
    }
    
    try:
        # Initialize detection system
        detection_system = YoloDetectionSystem(camera_config, yolo_config)
        
        # Run detection loop
        # detection_system.run_detection_loop(duration=60, detection_interval=0.1)  # 60 seconds
        detection_system.run_detection_loop(detection_interval=0.1)  # Run indefinitely
        
    except Exception as e:
        print(f"[ERROR] System failed: {e}")
        traceback.print_exc()
    finally:
        try:
            detection_system.stop()
        except:
            pass
        print("[INFO] System terminated")

if __name__ == "__main__":
    main()
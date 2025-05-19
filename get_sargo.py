#!/usr/bin/env python3
# get_sargo.py

import cv2
import numpy as np
import scipy.io
import time
import os
import glob

from c import *

class get_sargo:
    def __init__(self, camera_id='/dev/video4', calibration_path='/home/chen/Code/hackathon/camera/calibration/1.mat'):
        """Initialize the USB camera with optimized settings and load calibration data."""
        
        # Find the correct camera device
        self.camera_id = self._find_camera_device(camera_id)
        
        # Initialize camera
        self.cap = cv2.VideoCapture(self.camera_id)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera at {self.camera_id}")
        
        # Configure camera settings for optimal performance
        self.setup_camera_settings()
        
        # Get camera info
        self.device_name = f"USB Camera {self.camera_id}"
        print(f"Connected to {self.device_name}")
        
        # Allow camera to warm up and auto-exposure to stabilize
        print("Allowing camera to warm up for 2 seconds...")
        time.sleep(2)
        
        # Load calibration data
        self.calibration_path = calibration_path
        self.camera_matrix = None
        self.dist_coeffs = None
        self.load_calibration()
        
        # Image caches for faster repeated access
        self.last_frame_time = 0
        self.cached_image = None
        self.cached_undistorted_image = None
        self.cache_valid_duration = 0.05  # 50ms cache validity
        
        print(f"{self.device_name} initialized successfully")
        
    def _find_camera_device(self, preferred_device):
        """Find available camera device."""
        # If preferred device is specified and exists, use it
        if isinstance(preferred_device, str) and os.path.exists(preferred_device):
            return preferred_device
        elif isinstance(preferred_device, int):
            return preferred_device
        
        # Search for available video devices
        video_devices = glob.glob('/dev/video*')
        if video_devices:
            # Try each device to see which one works
            for device in sorted(video_devices):
                try:
                    test_cap = cv2.VideoCapture(device)
                    if test_cap.isOpened():
                        ret, _ = test_cap.read()
                        test_cap.release()
                        if ret:
                            print(f"Found working camera at {device}")
                            return device
                except:
                    continue
        
        # Fallback to integer device IDs
        for i in range(10):
            try:
                test_cap = cv2.VideoCapture(i)
                if test_cap.isOpened():
                    ret, _ = test_cap.read()
                    test_cap.release()
                    if ret:
                        print(f"Found working camera at device ID {i}")
                        return i
            except:
                continue
        
        raise RuntimeError("No working camera found")
    
    def setup_camera_settings(self):
        """Configure camera settings for optimal performance."""
        # Set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 848)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Set framerate
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        
        # Set buffer size to reduce latency
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Auto-exposure and focus settings
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Auto exposure enabled
        
        # Get actual settings
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        print(f"Camera configured: {width}x{height} @ {fps}fps")
    
    def load_calibration(self):
        """Load camera calibration parameters from .mat file."""
        try:
            if not os.path.exists(self.calibration_path):
                print(f"Warning: Calibration file not found at {self.calibration_path}")
                print("Camera will work without calibration (no undistortion)")
                return
            
            # Load .mat file
            mat_data = scipy.io.loadmat(self.calibration_path)
            
            # Extract calibration parameters (adjust key names based on your .mat file structure)
            # Common naming conventions in MATLAB calibration files:
            possible_matrix_keys = ['K', 'cameraMatrix', 'intrinsic_matrix', 'camera_matrix', 'mtx']
            possible_dist_keys = ['D', 'distCoeffs', 'distortion_coefficients', 'dist_coeffs', 'dist']
            
            # Find camera matrix
            for key in possible_matrix_keys:
                if key in mat_data:
                    self.camera_matrix = mat_data[key].astype(np.float32)
                    break
            
            # Find distortion coefficients
            for key in possible_dist_keys:
                if key in mat_data:
                    self.dist_coeffs = mat_data[key].astype(np.float32)
                    break
            
            if self.camera_matrix is not None and self.dist_coeffs is not None:
                print("Calibration data loaded successfully")
                print(f"Camera matrix shape: {self.camera_matrix.shape}")
                print(f"Distortion coefficients shape: {self.dist_coeffs.shape}")
            else:
                print("Warning: Could not find calibration parameters in .mat file")
                print(f"Available keys: {list(mat_data.keys())}")
                
        except Exception as e:
            print(f"Error loading calibration: {e}")
            print("Camera will work without calibration")
    
    def get_sargo_rgb(self, undistorted=True):
        """
        Capture RGB image from USB camera with caching for performance.
        
        Args:
            undistorted (bool): If True, return undistorted image using calibration data.
                              If False, return raw image from camera.
        
        Returns:
            numpy.ndarray: RGB image or None if no valid data
        """
        current_time = time.time()
        
        # Define cache variables based on return type
        cache_to_check = self.cached_undistorted_image if undistorted else self.cached_image
        
        # Check if we have a valid cached image
        if (cache_to_check is not None and 
            current_time - self.last_frame_time < self.cache_valid_duration):
            return cache_to_check.copy()  # Return a copy to avoid modification issues
        
        try:
            # Capture frame
            ret, frame = self.cap.read()
            
            if not ret or frame is None:
                print("No valid frame received")
                return None
            
            # Check if the image is valid
            if frame.size == 0 or np.all(frame == 0):
                print("Empty image received")
                return None
            
            # Convert BGR to RGB (OpenCV uses BGR by default)
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Update cache for raw image
            self.cached_image = rgb_image.copy()
            self.last_frame_time = current_time
            
            # If undistorted version requested and calibration is available
            if undistorted and self.camera_matrix is not None and self.dist_coeffs is not None:
                # Undistort the image
                undistorted_image = cv2.undistort(
                    rgb_image, 
                    self.camera_matrix, 
                    self.dist_coeffs
                )
                # Cache the undistorted image
                self.cached_undistorted_image = undistorted_image.copy()
                return undistorted_image
            elif undistorted:
                print("Warning: Undistortion requested but calibration not available")
                return rgb_image
            else:
                return rgb_image
                
        except Exception as e:
            print(f"Error capturing image: {e}")
            return None
    
    def get_camera_info(self):
        """Get camera information and settings."""
        info = {
            'device': self.camera_id,
            'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': self.cap.get(cv2.CAP_PROP_FPS),
            'calibrated': self.camera_matrix is not None,
        }
        return info
    
    def close(self):
        """Release camera resources."""
        try:
            if self.cap.isOpened():
                self.cap.release()
            print(f"{self.device_name} stopped")
            return True
        except Exception as e:
            print(f"Error closing USB camera: {e}")
            return False
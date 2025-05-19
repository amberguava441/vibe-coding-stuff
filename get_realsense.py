#!/usr/bin/env python3
# get_realsense.py

import pyrealsense2 as rs
import numpy as np
import cv2
import queue
import threading
import time
import scipy.io
import os

from c import *

class get_realsense:
    def __init__(self, calibration_path='/home/chen/Code/hackathon/camera/calibration/2.mat'):
        """Initialize the RealSense camera with optimized settings."""
        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # Enable streams with higher framerate
        self.config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 60)
        #self.config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 60)
        self.config.enable_stream(rs.stream.color, 640, 360, rs.format.rgb8, 60)

        # Start streaming
        self.profile = self.pipeline.start(self.config)
        
        # Get device info
        self.device = self.profile.get_device()
        self.device_name = self.device.get_info(rs.camera_info.name)
        print(f"Connected to {self.device_name}")
        
        # Allow camera to warm up and auto-exposure to stabilize
        print("Allowing camera to warm up for 2 seconds...")
        time.sleep(2)
        
        # Align depth frame to color frame
        self.align = rs.align(rs.stream.color)
        
        # Set up post-processing filters
        self.setup_filters()
        
        # Load calibration data
        self.calibration_path = calibration_path
        self.camera_matrix = None
        self.dist_coeffs = None
        self.load_calibration()
        
        # Image caches for faster repeated access
        self.last_color_frame_time = 0
        self.last_depth_frame_time = 0
        self.cached_color_image = None
        self.cached_undistorted_image = None
        self.cached_depth_colormap = None
        self.cache_valid_duration = 0.05  # 50ms cache validity
        self.cached_raw_depth = None
        
        print(f"{self.device_name} initialized successfully")
    
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
    
    def setup_filters(self):
        """Set up post-processing filters with optimized parameters."""
        # Decimation filter reduces resolution of depth image
        self.decimation = rs.decimation_filter()
        self.decimation.set_option(rs.option.filter_magnitude, 2)  # Reduces resolution by 2x
        
        # Spatial filter smooths depth image by looking at adjacent pixels
        self.spatial = rs.spatial_filter()
        self.spatial.set_option(rs.option.filter_magnitude, 3)
        self.spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
        self.spatial.set_option(rs.option.filter_smooth_delta, 20)
        
        # Temporal filter reduces temporal noise
        self.temporal = rs.temporal_filter()
        self.temporal.set_option(rs.option.filter_smooth_alpha, 0.4)
        self.temporal.set_option(rs.option.filter_smooth_delta, 20)
        
        # Hole filling filter fills small holes in depth image
        self.hole_filling = rs.hole_filling_filter()
    
    def get_rs_rgb(self, undistorted=True):
        """
        Capture RGB image from RealSense camera with caching for performance.
        
        Args:
            undistorted (bool): If True, return undistorted image using calibration data.
                              If False, return raw image from camera.
        
        Returns:
            numpy.ndarray: RGB image or None if no valid data
        """
        current_time = time.time()
        
        # Define cache variables based on return type
        cache_to_check = self.cached_undistorted_image if undistorted else self.cached_color_image
        
        # Check if we have a valid cached image
        if (cache_to_check is not None and 
            current_time - self.last_color_frame_time < self.cache_valid_duration):
            return cache_to_check.copy()  # Return a copy to avoid modification issues
        
        try:
            # Wait for a coherent pair of frames with shorter timeout
            frames = self.pipeline.wait_for_frames(timeout_ms=1000)
            
            # Align frames if successful
            aligned_frames = self.align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            
            if not color_frame or not color_frame.get_data():
                print("No valid RGB frame received")
                return None
            
            # Convert to numpy array (RealSense provides BGR by default)
            color_image = np.asanyarray(color_frame.get_data())
            
            # Check if the image is valid
            if color_image.size == 0 or np.all(color_image == 0):
                print("Empty RGB image received")
                return None
            
            # Convert BGR to RGB (RealSense provides BGR, but we want RGB)
            rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            
            # Update cache for raw image
            self.cached_color_image = rgb_image.copy()
            self.last_color_frame_time = current_time
            
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
            print(f"Error capturing RGB image: {e}")
            return None

    def get_rs_depth(self, colorized=False):
        """
        Capture depth image from RealSense camera with caching for performance.
        
        Args:
            colorized (bool): If True, return colorized depth map for visualization.
                            If False, return raw depth values in millimeters.
        
        Returns:
            numpy.ndarray: Raw depth image in millimeters or colorized depth image
        """
        current_time = time.time()
        
        # Define cache variables based on return type
        cache_to_check = self.cached_depth_colormap if colorized else self.cached_raw_depth
        last_time = self.last_depth_frame_time
        
        # Check if we have a valid cached depth image
        if (cache_to_check is not None and 
            current_time - last_time < self.cache_valid_duration):
            return cache_to_check.copy()  # Return a copy to avoid modification
        
        try:
            # Wait for a coherent pair of frames with shorter timeout
            frames = self.pipeline.wait_for_frames(timeout_ms=1000)
            
            # Align frames
            aligned_frames = self.align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            
            if not depth_frame or not depth_frame.get_data():
                print("No valid depth frame received")
                return None
            
            # Apply filters in sequence for better performance
            filtered_depth = depth_frame
            
            # Only apply filters if we have a valid depth frame
            try:
                # Apply filters in sequence
                filtered_depth = self.decimation.process(filtered_depth)
                filtered_depth = self.spatial.process(filtered_depth)
                filtered_depth = self.temporal.process(filtered_depth)
                filtered_depth = self.hole_filling.process(filtered_depth)
                
                # Check if the filtered depth frame is valid
                if not filtered_depth or not filtered_depth.get_data():
                    print("Filtering produced invalid depth frame")
                    return None
                
                # Convert to numpy array - this contains raw depth values in millimeters
                depth_image = np.asanyarray(filtered_depth.get_data())
                
                # Check if the depth image is valid
                if depth_image.size == 0 or np.all(depth_image == 0):
                    print("Empty depth image received")
                    return None
                
                # Cache the raw depth values
                self.cached_raw_depth = depth_image.copy()
                self.last_depth_frame_time = current_time
                
                # If colorized version requested, create and cache it
                if colorized:
                    # Colorize depth map for visualization with improved contrast
                    depth_colormap = cv2.applyColorMap(
                        cv2.convertScaleAbs(depth_image, alpha=0.03), 
                        cv2.COLORMAP_JET
                    )
                    self.cached_depth_colormap = depth_colormap.copy()
                    return depth_colormap
                else:
                    # Return the raw depth values in millimeters
                    return depth_image
                    
            except Exception as e:
                print(f"Error processing depth frame: {e}")
                return None
                
        except Exception as e:
            print(f"Error capturing depth image: {e}")
            return None

    def get_camera_info(self):
        """Get camera information and settings."""
        info = {
            'device': self.device_name,
            'calibrated': self.camera_matrix is not None,
        }
        return info
    
    def close(self):
        """
        Stop the pipeline and release resources
        """
        try:
            self.pipeline.stop()
            print(f"{self.device_name} stopped")
            return True
        except Exception as e:
            print(f"Error closing RealSense camera: {e}")
            return False
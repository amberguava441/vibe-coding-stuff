#!/usr/bin/env python3
# get_realsense.py

import pyrealsense2 as rs
import numpy as np
import cv2
import queue
import threading
import time

class get_realsense:
    def __init__(self):
        """Initialize the RealSense camera with optimized settings."""
        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # Enable streams with higher framerate
        self.config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 60)
        self.config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 60)
        
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
        
        # Image caches for faster repeated access
        self.last_color_frame_time = 0
        self.last_depth_frame_time = 0
        self.cached_color_image = None
        self.cached_depth_colormap = None
        self.cache_valid_duration = 0.05  # 50ms cache validity
        
        print(f"{self.device_name} initialized successfully")
    
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
    
    def get_rs_rgb(self):
        """
        Capture RGB image from RealSense camera with caching for performance.
        
        Returns:
            numpy.ndarray: RGB image or None if no valid data
        """
        current_time = time.time()
        
        # Check if we have a valid cached image
        if (self.cached_color_image is not None and 
            current_time - self.last_color_frame_time < self.cache_valid_duration):
            return self.cached_color_image.copy()  # Return a copy to avoid modification issues
        
        try:
            # Wait for a coherent pair of frames with shorter timeout
            frames = self.pipeline.wait_for_frames(timeout_ms=1000)
            
            # Align frames if successful
            aligned_frames = self.align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            
            if not color_frame or not color_frame.get_data():
                print("No valid RGB frame received")
                return None
            
            # Convert to numpy array
            color_image = np.asanyarray(color_frame.get_data())
            
            # Check if the image is valid
            if color_image.size == 0 or np.all(color_image == 0):
                print("Empty RGB image received")
                return None
            
            # Update cache
            self.cached_color_image = color_image.copy()
            self.last_color_frame_time = current_time
            
            return color_image
            
        except Exception as e:
            print(f"Error capturing RGB image: {e}")
            return None

    def get_rs_depth(self):
        """
        Capture depth image from RealSense camera with caching for performance.
        
        Returns:
            numpy.ndarray: Colorized depth image or None if no valid data
        """
        current_time = time.time()
        
        # Check if we have a valid cached depth image
        if (self.cached_depth_colormap is not None and 
            current_time - self.last_depth_frame_time < self.cache_valid_duration):
            return self.cached_depth_colormap.copy()  # Return a copy to avoid modification
        
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
                
                # Convert to numpy array
                depth_image = np.asanyarray(filtered_depth.get_data())
                
                # Check if the depth image is valid
                if depth_image.size == 0 or np.all(depth_image == 0):
                    print("Empty depth image received")
                    return None
                
                # Colorize depth map for visualization with improved contrast
                depth_colormap = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_image, alpha=0.03), 
                    cv2.COLORMAP_JET
                )
                
                # Update cache
                self.cached_depth_colormap = depth_colormap.copy()
                self.last_depth_frame_time = current_time
                
                return depth_colormap
                
            except Exception as e:
                print(f"Error processing depth frame: {e}")
                return None
                
        except Exception as e:
            print(f"Error capturing depth image: {e}")
            return None
    
    def get_frames_synchronized(self):
        """
        Get synchronized RGB and depth frames in a single call.
        
        Returns:
            tuple: (rgb_image, depth_colormap) or (None, None) if no valid data
        """
        try:
            # Wait for a coherent pair of frames
            frames = self.pipeline.wait_for_frames(timeout_ms=1000)
            
            # Align frames
            aligned_frames = self.align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                return None, None
            
            # Process RGB image
            color_image = np.asanyarray(color_frame.get_data())
            
            # Process depth image with filters
            filtered_depth = self.decimation.process(depth_frame)
            filtered_depth = self.spatial.process(filtered_depth)
            filtered_depth = self.temporal.process(filtered_depth)
            filtered_depth = self.hole_filling.process(filtered_depth)
            
            depth_image = np.asanyarray(filtered_depth.get_data())
            
            # Colorize depth map for visualization
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03), 
                cv2.COLORMAP_JET
            )
            
            # Update caches
            current_time = time.time()
            self.cached_color_image = color_image.copy()
            self.cached_depth_colormap = depth_colormap.copy()
            self.last_color_frame_time = current_time
            self.last_depth_frame_time = current_time
            
            return color_image, depth_colormap
            
        except Exception as e:
            print(f"Error capturing synchronized frames: {e}")
            return None, None
    
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
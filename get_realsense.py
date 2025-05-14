#!/usr/bin/env python3
import pyrealsense2 as rs
import numpy as np
import cv2
import queue
import threading
import time


class GetRealSense:
    def __init__(self):
        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # Enable streams
        self.config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
        
        # Start streaming
        self.pipeline.start(self.config)
        
        # Allow camera to warm up
        time.sleep(2)
        
        # Align depth frame to color frame
        self.align = rs.align(rs.stream.color)
        
        # Post-processing filters
        self.decimation = rs.decimation_filter()
        self.spatial = rs.spatial_filter()
        self.temporal = rs.temporal_filter()
        self.hole_filling = rs.hole_filling_filter()
        
        print("RealSense D455 camera initialized successfully")
    
    def get_rs_rgb(self):
        """
        Capture RGB image from RealSense camera
        
        Returns:
            numpy.ndarray: RGB image
        """
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        
        if not color_frame:
            return None
        
        # Convert to numpy array
        color_image = np.asanyarray(color_frame.get_data())
        
        return color_image
    
    def get_rs_depth(self):
        """
        Capture depth image from RealSense camera
        
        Returns:
            numpy.ndarray: Depth image
        """
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        
        if not depth_frame:
            return None
        
        # Apply filters to enhance depth data
        filtered_depth = self.decimation.process(depth_frame)
        filtered_depth = self.spatial.process(filtered_depth)
        filtered_depth = self.temporal.process(filtered_depth)
        filtered_depth = self.hole_filling.process(filtered_depth)
        
        # Convert to numpy array
        depth_image = np.asanyarray(filtered_depth.get_data())
        
        # Colorize depth map for visualization
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03), 
            cv2.COLORMAP_JET
        )
        
        return depth_colormap
    
    def close(self):
        """
        Stop the pipeline and release resources
        """
        self.pipeline.stop()
        print("RealSense D455 camera stopped")


def main():
    # Create image queues
    rgb_queue = queue.Queue(maxsize=30)  # Store up to 30 frames
    depth_queue = queue.Queue(maxsize=30)
    
    # Initialize camera
    rs_camera = GetRealSense()
    
    # Capture frames in a loop for 10 seconds
    start_time = time.time()
    frame_count = 0
    
    try:
        while time.time() - start_time < 10:  # Run for 10 seconds
            # Get RGB image
            rgb_img = rs_camera.get_rs_rgb()
            if rgb_img is not None:
                # Add to queue, remove oldest if full
                if rgb_queue.full():
                    rgb_queue.get()
                rgb_queue.put(rgb_img)
                
                # Display RGB image
                cv2.imshow("RGB Image", rgb_img)
            
            # Get depth image
            depth_img = rs_camera.get_rs_depth()
            if depth_img is not None:
                # Add to queue, remove oldest if full
                if depth_queue.full():
                    depth_queue.get()
                depth_queue.put(depth_img)
                
                # Display depth image
                cv2.imshow("Depth Image", depth_img)
            
            frame_count += 1
            
            # Wait for 1ms for key press to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        # Close camera and clean up
        rs_camera.close()
        cv2.destroyAllWindows()
        
        # Print statistics
        elapsed_time = time.time() - start_time
        print(f"Captured {frame_count} frames in {elapsed_time:.2f} seconds")
        print(f"Average FPS: {frame_count / elapsed_time:.2f}")
        print(f"RGB queue size: {rgb_queue.qsize()}")
        print(f"Depth queue size: {depth_queue.qsize()}")
        
        # Save the last frame as example
        if not rgb_queue.empty() and not depth_queue.empty():
            last_rgb = rgb_queue.get()
            last_depth = depth_queue.get()
            
            cv2.imwrite("last_rgb_frame.png", last_rgb)
            cv2.imwrite("last_depth_frame.png", last_depth)
            print("Last frames saved to disk")


if __name__ == "__main__":
    main()
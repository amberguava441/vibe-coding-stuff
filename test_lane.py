#!/usr/bin/env python3
"""
Lane Detection Test Script
Tests lane detection pipeline: Camera -> Lane Detection -> 3D Coordinate Transformation -> Visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import os

# Import custom modules
from c import camera, LaneDetectionTester
from ct import getF

class LaneDetectionTest:
    def __init__(self, camera_id=100):
        """
        Initialize the lane detection test system
        
        Args:
            camera_id (int): Camera ID (100 for RealSense depth camera)
        """
        print("[INFO] Initializing Lane Detection Test...")
        
        self.camera_id = camera_id
        
        # Camera intrinsic matrix (from c.py)
        self.K = np.array([
            [385.5,   0.0, 326.5],
            [  0.0, 384.6, 242.5],
            [  0.0,   0.0,   1.0]
        ])
        
        # Camera extrinsic parameters for coordinate transformation
        self.R = np.array([
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0]
        ])
        self.T = np.array([0, 0, 0])
        
        # Initialize camera system
        print("[INFO] Initializing camera system...")
        self.camera_system = camera([self.camera_id])
        if not self.camera_system.work:
            raise Exception("Camera system initialization failed")
        print("[INFO] Camera system initialized successfully")
        
        # Initialize lane detection
        print("[INFO] Initializing lane detection...")
        self.lane_detector = LaneDetectionTester()
        print("[INFO] Lane detection initialized successfully")
        
        print("[INFO] Lane detection test system ready!")
    
    def capture_and_detect(self):
        """
        Capture frames and perform lane detection
        
        Returns:
            tuple: (rgb_frame, depth_frame, detected_lanes) where detected_lanes is list of (x,y) points
        """
        print("[INFO] Capturing camera frames...")
        
        # Get camera frames
        flag, calibrated, rgb_frame, depth_frame = self.camera_system.getFrame(self.camera_id)
        
        if flag != 0 or rgb_frame is None:
            raise Exception("Failed to get camera frame")
        
        print(f"[INFO] RGB frame shape: {rgb_frame.shape}")
        if depth_frame is not None:
            print(f"[INFO] Depth frame shape: {depth_frame.shape}")
        else:
            raise Exception("No depth frame available")
        
        # Perform lane detection
        print("[INFO] Performing lane detection...")
        detected_lanes = self.lane_detector.detect_lanes(rgb_frame)
        W = 640
        H = 480
        detected_lanes = [(x, y) for x, y in detected_lanes if 0 < x < W - 5 and 0 < y < H - 5]
        
        if detected_lanes:
            print(f"[INFO] Detected {len(detected_lanes)} lane points")
        else:
            print("[WARNING] No lanes detected")
        
        return rgb_frame, depth_frame, detected_lanes
    
    def convert_to_3d(self, detected_lanes, depth_frame):
        """
        Convert detected lane points to 3D world coordinates
        
        Args:
            detected_lanes: List of (x,y) pixel coordinates
            depth_frame: Depth image array
            
        Returns:
            list: List of (rx, ry, rz) world coordinates
        """
        if not detected_lanes:
            print("[WARNING] No lane points to convert")
            return []
        
        print("[INFO] Converting lane points to 3D coordinates...")
        
        # Convert lane points to 3D pixel coordinates with depth
        lane_points_pixel = []
        for x, y in detected_lanes:
            # Get depth value (with bounds checking)
            if y < depth_frame.shape[0] and x < depth_frame.shape[1]:
                depth_value = depth_frame[y, x]  # Note: depth_frame[row, col] = depth_frame[y, x]
            else:
                depth_value = 0
            
            lane_points_pixel.append((x, y, depth_value))
            
        W = 640
        H = 480
        lane_points_pixel = [(x, y, depth_value) for x, y, depth_value in lane_points_pixel if 0 < x < W - 5 and 0 < y < H - 5]
        
        print(f"[INFO] Created {len(lane_points_pixel)} 3D pixel points")
        
        # Convert from pixel coordinates to world coordinates using getF()
        world_coordinates = getF(lane_points_pixel, self.K, self.R, self.T)

        
        
        if world_coordinates:
            print(f"[INFO] Converted to {len(world_coordinates)} world coordinate points")
            # Print first few points for debugging
            for i, (rx, ry, rz) in enumerate(world_coordinates[:5]):
                print(f"[DEBUG] Point {i}: ({rx:.3f}, {ry:.3f}, {rz:.3f})")
        else:
            print("[WARNING] No valid world coordinates generated")
        
        return world_coordinates
    
    def visualize_lane_on_rgb(self, rgb_frame, detected_lanes, save_path="lane_detection_rgb.png", show_plot=False):
        """
        Visualize detected lane points overlaid on RGB image
        
        Args:
            rgb_frame: RGB camera frame
            detected_lanes: List of (x,y) pixel coordinates
            save_path: Path to save the visualization
            show_plot: Whether to display the plot
        """
        if not detected_lanes:
            print("[WARNING] No lane points to visualize on RGB")
            return
        
        print(f"[INFO] Creating RGB visualization with {len(detected_lanes)} lane points...")
        
        # Create a copy of the RGB frame for visualization
        rgb_viz = rgb_frame.copy()
        
        # Draw lane points on the image
        for i, (x, y) in enumerate(detected_lanes):
            # Draw small circles for each lane point
            # Convert to int in case coordinates are float
            x_int, y_int = int(x), int(y)
            
            # Ensure coordinates are within image bounds
            if 0 <= x_int < rgb_viz.shape[1] and 0 <= y_int < rgb_viz.shape[0]:
                # Draw a small circle (radius=2) in red color
                rgb_viz[max(0, y_int-1):min(rgb_viz.shape[0], y_int+2), 
                        max(0, x_int-1):min(rgb_viz.shape[1], x_int+2)] = [255, 0, 0]  # Red color
        
        # Create matplotlib figure
        plt.figure(figsize=(12, 8))
        
        # Display the image with lane points
        plt.imshow(rgb_viz)
        plt.title(f'Lane Detection Results on RGB Image\n{len(detected_lanes)} lane points detected', fontsize=14)
        plt.axis('off')  # Hide axes for cleaner look
        
        # Add some information text
        plt.figtext(0.02, 0.02, f'Lane points: {len(detected_lanes)}', 
                   fontsize=10, color='white', 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
        print(f"[INFO] RGB lane visualization saved to {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()  # Close to free memory
    
    def visualize_3d(self, world_coordinates, save_path="lane_detection_3d.png", show_plot=False):
        """
        Create 3D visualization of lane points
        
        Args:
            world_coordinates: List of (rx, ry, rz) world coordinates
            save_path: Path to save the visualization
            show_plot: Whether to display the plot
        """
        if not world_coordinates:
            print("[WARNING] No coordinates to visualize")
            return
        
        print(f"[INFO] Creating 3D visualization with {len(world_coordinates)} points...")
        
        # Extract coordinates
        x_coords = [point[0] for point in world_coordinates]
        y_coords = [point[1] for point in world_coordinates]
        # n = len(world_coordinates)
        # z_coords = [0] * n
        z_coords = [point[2] for point in world_coordinates]
        
        # Create 3D plot
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot points with smaller size
        scatter = ax.scatter(x_coords, y_coords, z_coords, 
                           c=y_coords, cmap='viridis', 
                           s=20, alpha=0.8)  # Reduced size from 50 to 20
        
        # Customize plot
        ax.set_xlabel('X (Right, m)', fontsize=12)
        ax.set_ylabel('Y (Forward, m)', fontsize=12)
        ax.set_zlabel('Z (Up, m)', fontsize=12)
        ax.set_title('Lane Detection Results in 3D World Coordinates', fontsize=14)
        
        # Add colorbar
        plt.colorbar(scatter, ax=ax, shrink=0.8, label='Forward Distance (m)')
        
        # Set equal aspect ratio if possible
        max_range = max(max(x_coords) - min(x_coords),
                       max(y_coords) - min(y_coords),
                       max(z_coords) - min(z_coords)) / 2.0
        
        mid_x = (max(x_coords) + min(x_coords)) * 0.5
        mid_y = (max(y_coords) + min(y_coords)) * 0.5
        mid_z = (max(z_coords) + min(z_coords)) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Save plot
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] 3D visualization saved to {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()  # Close to free memory
        
        # Also create a top-down view (Bird's Eye View)
        self.create_bev_plot(world_coordinates, save_path.replace('.png', '_bev.png'), show_plot)
    
    def create_bev_plot(self, world_coordinates, save_path, show_plot=False):
        """
        Create Bird's Eye View (top-down) plot
        
        Args:
            world_coordinates: List of (rx, ry, rz) world coordinates  
            save_path: Path to save the BEV plot
            show_plot: Whether to display the plot
        """
        print("[INFO] Creating Bird's Eye View plot...")
        
        # Extract X and Y coordinates (ignoring Z for top-down view)
        x_coords = [point[0] for point in world_coordinates]
        y_coords = [point[1] for point in world_coordinates]
        
        # Create 2D plot
        plt.figure(figsize=(10, 12))
        
        # Plot points with smaller size
        plt.scatter(x_coords, y_coords, c='blue', s=15, alpha=0.7, label='Lane Points')  # Reduced size from 30 to 15
        
        # Customize plot
        plt.xlabel('X (Right, m)', fontsize=12)
        plt.ylabel('Y (Forward, m)', fontsize=12)
        plt.title('Lane Detection Results - Bird\'s Eye View', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        # Invert Y axis so forward is up
        plt.gca().invert_yaxis()
        
        # Add vehicle position indicator (origin)
        plt.scatter([0], [0], c='red', s=100, marker='s', label='Vehicle Position')
        
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Bird's Eye View saved to {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()  # Close to free memory
    
    def run_test(self, save_images=True, show_plots=True):
        """
        Run the complete lane detection test
        
        Args:
            save_images (bool): Whether to save visualization images
            show_plots (bool): Whether to display plots after completion
        """
        try:
            print("\n" + "="*60)
            print("STARTING LANE DETECTION TEST")
            print("="*60)
            
            # Step 1: Capture frames and detect lanes
            rgb_frame, depth_frame, detected_lanes = self.capture_and_detect()
            
            # Step 2: Convert to 3D world coordinates
            world_coordinates = self.convert_to_3d(detected_lanes, depth_frame)
            
            # Step 3: Save visualizations first (without showing)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            
            if detected_lanes and save_images:
                rgb_save_path = f"lane_test_rgb_{timestamp}.png"
                self.visualize_lane_on_rgb(rgb_frame, detected_lanes, rgb_save_path, show_plot=False)
            
            if world_coordinates and save_images:
                save_path_3d = f"lane_test_3d_{timestamp}.png"
                self.visualize_3d(world_coordinates, save_path_3d, show_plot=False)
            
            # Step 4: Print summary
            print("\n" + "="*60)
            print("LANE DETECTION TEST SUMMARY")
            print("="*60)
            print(f"RGB Frame Shape: {rgb_frame.shape}")
            print(f"Depth Frame Shape: {depth_frame.shape}")
            print(f"Detected Lane Points (2D): {len(detected_lanes)}")
            print(f"Converted World Points (3D): {len(world_coordinates)}")
            
            # Print sample 2D points (pixel coordinates)
            if detected_lanes:
                print(f"\nSample 2D Lane Points (pixel coordinates):")
                for i, (x, y) in enumerate(detected_lanes[:5]):
                    print(f"  Point {i+1}: ({x}, {y}) pixels")
            
            # Print sample 3D points (world coordinates)
            if world_coordinates:
                print(f"\nSample 3D World Points (meters):")
                for i, (rx, ry, rz) in enumerate(world_coordinates[:5]):
                    print(f"  Point {i+1}: ({rx:.3f}, {ry:.3f}, {rz:.3f}) meters")
            
            print("="*60)
            
            # Step 5: Show plots after everything is complete
            if show_plots:
                print("\n[INFO] Displaying visualizations...")
                print("[INFO] Close each plot window to proceed to the next one")
                
                if detected_lanes:
                    print("[INFO] Showing RGB visualization...")
                    self.visualize_lane_on_rgb(rgb_frame, detected_lanes, 
                                             f"lane_test_rgb_{timestamp}.png", show_plot=True)
                
                if world_coordinates:
                    print("[INFO] Showing 3D visualization...")
                    self.visualize_3d(world_coordinates, 
                                    f"lane_test_3d_{timestamp}.png", show_plot=True)
            
            return world_coordinates
            
        except Exception as e:
            print(f"[ERROR] Lane detection test failed: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def cleanup(self):
        """Cleanup resources"""
        if hasattr(self, 'camera_system'):
            self.camera_system.destroy()
        print("[INFO] Cleanup completed")

def main():
    """Main entry point for lane detection test"""
    print("Lane Detection 3D Test")
    print("This will capture camera frames, detect lanes, and create 3D visualizations")
    
    # Configuration
    CAMERA_ID = 100  # RealSense depth camera
    
    lane_test = None
    try:
        # Initialize test system
        lane_test = LaneDetectionTest(camera_id=CAMERA_ID)
        
        # Run the test with plots showing after completion
        world_coordinates = lane_test.run_test(save_images=True, show_plots=True)
        
        if world_coordinates:
            print(f"\n[SUCCESS] Test completed successfully with {len(world_coordinates)} 3D points")
        else:
            print("\n[WARNING] Test completed but no valid 3D points were generated")
        
    except KeyboardInterrupt:
        print("\n[INFO] Test interrupted by user")
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if lane_test:
            lane_test.cleanup()
        print("\n[INFO] Lane detection test finished")

if __name__ == "__main__":
    main()
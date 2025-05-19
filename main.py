#!/usr/bin/env python3
# main_pipeline.py

import time
import signal
import sys
import os
import threading
import queue
import numpy as np
from datetime import datetime

# Import sensor modules
from get_ins import get_ins
from get_realsense import get_realsense
from get_sensors import get_sensors
from get_yolo import YoloV5Detector, process_lane_and_detection_3d

# Import path planner
from p import planner

class MainPipeline:
    def __init__(self):
        """Initialize the main autonomous driving pipeline."""
        
        # Fixed destination coordinates
        self.destination_lat = 45.72713182
        self.destination_lon = 126.62565161
        
        # Fixed lane detection data
        self.fixed_lane_detection = [
            [(-100, 0), (-100, 1), (-100, 2), (-100, 3), (-100, 4), (-100, 5), 
             (-100, 6), (-100, 7), (-100, 8), (-100, 9), (-100, 10), (-100, 11), 
             (-100, 12), (-100, 13), (-100, 14), (-100, 15), (-100, 16), (-100, 17), 
             (-100, 18), (-100, 19)],
            [(100, 0), (100, 1), (100, 2), (100, 3), (100, 4), (100, 5), 
             (100, 6), (100, 7), (100, 8), (100, 9), (100, 10), (100, 11), 
             (100, 12), (100, 13), (100, 14), (100, 15), (100, 16), (100, 17), 
             (100, 18), (100, 19)]
        ]

        #self.fixed_lane_detection = [[(-2.5, 0), (-2.5, 1), (-2.5, 2), (-2.5, 3), (-2.5, 4), (-2.5, 5), (-2.5, 6), (-2.5, 7), (-2.5, 8), (-2.5, 9), (-2.5, 10), (-2.5, 11), (-2.5, 12), (-2.5, 13), (-2.5, 14), (-2.5, 15), (-2.5, 16), (-2.5, 17), (-2.5, 18), (-2.5, 19)], [(2.5, 0), (2.5, 1), (2.5, 2), (2.5, 3), (2.5, 4), (2.5, 5), (2.5, 6), (2.5, 7), (2.5, 8), (2.5, 9), (2.5, 10), (2.5, 11), (2.5, 12), (2.5, 13), (2.5, 14), (2.5, 15), (2.5, 16), (2.5, 17), (2.5, 18), (2.5, 19)]]
        
        # Camera intrinsic parameters for 3D processing
        self.K = np.array([
            [385.5,   0.0, 326.5],
            [  0.0, 384.6, 242.5],
            [  0.0,   0.0,   1.0]
        ])
        
        # Update interval
        self.update_interval = 0.1
        
        # Initialize device handles
        self.ins_device = None
        self.rs_camera = None
        self.yolo_detector = None
        self.path_planner = None
        
        # Results folder
        self.results_folder = "p_results"
        self.setup_results_folder()
        
        # Control flags
        self.running = True
        self.stop_event = threading.Event()
        
        # Statistics
        self.iteration_count = 0
        self.successful_paths = 0
        self.total_obstacles_detected = 0
        
    def setup_results_folder(self):
        """Create results folder if it doesn't exist."""
        if not os.path.exists(self.results_folder):
            os.makedirs(self.results_folder)
            print(f"Created results folder: {self.results_folder}")
        
    def initialize_devices(self):
        """Initialize all sensor devices and path planner."""
        try:
            print("Initializing devices...")
            
            # Initialize INS device
            print("1. Initializing INS device...")
            try:
                # Try common serial ports
                serial_ports = ['/dev/ttyUSB0', '/dev/ttyUSB1', '/dev/ttyACM0', '/dev/ttyACM1']
                self.ins_device = None
                
                for port in serial_ports:
                    try:
                        self.ins_device = get_ins(port=port, baudrate=115200)
                        if self.ins_device.connect():
                            print(f"   INS device connected on {port}")
                            break
                    except Exception as e:
                        continue
                
                if self.ins_device and self.ins_device.is_connected:
                    # Start GPS and IMU modes
                    self.ins_device.start_gps_mode()
                    self.ins_device.start_imu_mode()
                    time.sleep(1.0)  # Allow time for sensor data to start flowing
                    print("   INS device initialized successfully")
                else:
                    print("   Warning: Could not connect to INS device")
                    self.ins_device = None
                    
            except Exception as e:
                print(f"   Error initializing INS device: {e}")
                self.ins_device = None
            
            # Initialize RealSense camera
            print("2. Initializing RealSense camera...")
            try:
                self.rs_camera = get_realsense()
                time.sleep(0.5)  # Wait for camera to initialize
                print("   RealSense camera initialized successfully")
            except Exception as e:
                print(f"   Error initializing RealSense camera: {e}")
                return False
            
            # Initialize YOLO detector
            print("3. Initializing YOLO detector...")
            try:
                self.yolo_detector = YoloV5Detector(stop_event=self.stop_event)
                print("   YOLO detector initialized successfully")
            except Exception as e:
                print(f"   Error initializing YOLO detector: {e}")
                return False
            
            # Initialize path planner
            print("4. Initializing path planner...")
            try:
                self.path_planner = planner()
                print("   Path planner initialized successfully")
            except Exception as e:
                print(f"   Error initializing path planner: {e}")
                return False
            
            # Get initial position from sensors before setting origin and destination
            print("5. Getting initial sensor data...")
            sensor_data = get_sensors(
                ins_device=self.ins_device,
                rs_camera=self.rs_camera,
                sargo_camera=None,
                gps_mode=True,
                imu_mode=True,
                max_wait_time=2.0  # Longer timeout for initial reading
            )
            
            # Check if we have valid GPS data
            if sensor_data.get('gps_data') is None:
                print("   Error: Failed to get initial GPS data")
                return False
            
            # Extract initial position
            initial_lon = sensor_data['gps_data'][0]
            initial_lat = sensor_data['gps_data'][1]
            
            print(f"   Initial GPS position: ({initial_lon}, {initial_lat})")
            
            # Set origin and destination for the planner (proper initialization sequence)
            self.path_planner.setOrigin(initial_lon, initial_lat)
            self.path_planner.setDestination(self.destination_lon, self.destination_lat)
            print(f"   Origin set to: ({initial_lon}, {initial_lat})")
            print(f"   Destination set to: ({self.destination_lon}, {self.destination_lat})")
            
            print("All devices initialized successfully!\n")
            return True
            
        except Exception as e:
            print(f"Critical error during device initialization: {e}")
            return False
    
    # def get_obstacle_detection(self):
    #     """Get obstacle detection using YOLO and 3D processing."""
    #     try:
    #         # Get RGB and depth images
    #         rgb_image = self.rs_camera.get_rs_rgb(undistorted=True)
    #         depth_image = self.rs_camera.get_rs_depth(colorized=False)  # Raw depth values
            
    #         if rgb_image is None or depth_image is None:
    #             return []
            
    #         # Run YOLO detection
    #         img_tensor, im0s = self.yolo_detector.preprocess(rgb_image)
    #         preds = self.yolo_detector.detect(img_tensor)
            
    #         # Process detections
    #         detections_2d = []
    #         for det in preds:
    #             if det is not None and len(det):
    #                 # Scale coordinates back to original image size
    #                 try:
    #                     from utils.general import scale_coords
    #                     det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], im0s.shape).round()
    #                 except ImportError:
    #                     pass  # Skip scaling if not available
                    
    #                 for *xyxy, conf, cls in det:
    #                     x1, y1, x2, y2 = map(int, xyxy)
    #                     class_id = int(cls)
    #                     detections_2d.append([class_id, x1, y1, x2, y2])
            
    #         # Convert to 3D using depth information
    #         if detections_2d:
    #             # Create temporary queues for the 3D processing function
    #             yolo_queue = queue.Queue()
    #             depth_queue = queue.Queue()
                
    #             yolo_queue.put(detections_2d)
    #             depth_queue.put(depth_image)
                
    #             # Get 3D obstacle positions
    #             obstacles_3d = process_lane_and_detection_3d(yolo_queue, depth_queue, self.K)
    #             return obstacles_3d if obstacles_3d else []
            
    #         return []
            
    #     except Exception as e:
    #         print(f"Error in obstacle detection: {e}")
    #         return []

    def get_obstacle_detection(self):
        """Get obstacle detection using YOLO and 3D processing."""
        try:
            # Get RGB and depth images
            rgb_image = self.rs_camera.get_rs_rgb(undistorted=True)
            depth_image = self.rs_camera.get_rs_depth(colorized=False)
            
            if rgb_image is None or depth_image is None:
                return []
            
            # Run YOLO detection
            img_tensor, im0s = self.yolo_detector.preprocess(rgb_image)
            preds = self.yolo_detector.detect(img_tensor)
            
            # Process detections
            detections_2d = []
            for det in preds:
                if det is not None and len(det):
                    # Scale coordinates back to original image size
                    try:
                        from utils.general import scale_coords
                        det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], im0s.shape).round()
                    except ImportError:
                        pass  # Skip scaling if not available
                    
                    for *xyxy, conf, cls in det:
                        x1, y1, x2, y2 = map(int, xyxy)
                        class_id = int(cls)
                        detections_2d.append([class_id, x1, y1, x2, y2])
            
            # Convert to 3D using depth information
            # Convert to 3D using depth information
            if detections_2d:
                yolo_queue = queue.Queue()
                depth_queue = queue.Queue()
                yolo_queue.put(detections_2d)
                depth_queue.put(depth_image)
                
                # Get 3D obstacles in camera coordinates
                obstacles_3d_camera = process_lane_and_detection_3d(yolo_queue, depth_queue, self.K)
                
                # Transform to vehicle coordinates for path planner
                obstacles_3d_vehicle = []
                for obs in obstacles_3d_camera:
                    if len(obs) >= 7:
                        class_id, cam_x, cam_y, cam_z, width, height, depth = obs
                        
                        # KEY TRANSFORMATION: Camera → Vehicle coordinates
                        vehicle_x = cam_x    # right → lateral (same direction)
                        vehicle_y = cam_z    # forward → forward (CRITICAL FIX!)
                        vehicle_z = cam_y    # keep as-is (planner ignores this anyway)
                        
                        # Dimensions (planner only uses width and length)
                        vehicle_width = width   # lateral dimension
                        vehicle_length = depth  # forward dimension  
                        vehicle_height = height # vertical (ignored by planner)
                        
                        obstacles_3d_vehicle.append((
                            class_id, 
                            vehicle_x, vehicle_y, vehicle_z,
                            vehicle_width, vehicle_length, vehicle_height
                        ))
                
                return obstacles_3d_vehicle if obstacles_3d_vehicle else []
            
            return []
            
        except Exception as e:
            print(f"Error in obstacle detection: {e}")
            return []
    
    def run_planning_cycle(self):
        """Run one planning cycle."""
        try:
            # Get sensor data
            sensor_data = get_sensors(
                ins_device=self.ins_device,
                rs_camera=self.rs_camera,
                sargo_camera=None,  # Disabled as requested
                gps_mode=True,
                imu_mode=True,
                max_wait_time=0.05  # Short timeout for real-time operation
            )
            
            # Extract GPS data
            gps_data = sensor_data.get('gps_data')
            if gps_data is None:
                print(f"[{self.iteration_count:04d}] No GPS data available - skipping planning cycle")
                return False
            
            # Extract required data from GPS
            lon, lat, x, y, dx, dy, cog, dir_vec_x, dir_vec_y = gps_data
            
            # Get obstacle detection
            obstacle_detection = self.get_obstacle_detection()
            num_obstacles = len(obstacle_detection)
            self.total_obstacles_detected += num_obstacles
            
            # Prepare planner input
            # planner.getpath(xy, dxy, jw, o, lane, ol)
            xy = (x, y)
            dxy = (dir_vec_x, dir_vec_y)
            jw = (lon, lat)
            o = cog
            lane = self.fixed_lane_detection
            ol = obstacle_detection
            #ol = [(0, 2, 12, 2, 2, 2, 2)]
            #print(ol)
            
            # Run path planning
            try:
                path = self.path_planner.getpath(xy, dxy, jw, o, lane, ol)
                path_available = path is not None and len(path) > 0
                
                # Output status
                status = "YES" if path_available else "NO"
                print(f"[{self.iteration_count:04d}] Path available: {status}, Obstacles detected: {num_obstacles}")
                
                # Save path if available
                if path_available:
                    self.successful_paths += 1
                    self.save_path_result(path, obstacle_detection)
                
                return True
                
            except Exception as e:
                print(f"[{self.iteration_count:04d}] Path planning error: {e}")
                return False
            
        except Exception as e:
            print(f"[{self.iteration_count:04d}] Planning cycle error: {e}")
            return False
    
    def save_path_result(self, path, obstacles):
        """Save path result to file."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = f"{self.results_folder}/path_{self.iteration_count:04d}_{timestamp}.txt"
            
            with open(filename, 'w') as f:
                f.write(f"# Path result for iteration {self.iteration_count}\n")
                f.write(f"# Timestamp: {timestamp}\n")
                f.write(f"# Number of path points: {len(path)}\n")
                f.write(f"# Number of obstacles: {len(obstacles)}\n\n")
                
                f.write("# Path points (x, y):\n")
                for i, (x, y) in enumerate(path):
                    f.write(f"{i}: ({x:.6f}, {y:.6f})\n")
                
                f.write("\n# Obstacles (class_id, width, height, depth, center_x, center_y, center_z):\n")
                for i, obstacle in enumerate(obstacles):
                    f.write(f"{i}: {obstacle}\n")
            
            # Try to create a simple plot as well
            try:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(10, 8))
                
                # Plot path
                if path:
                    path_x, path_y = zip(*path)
                    plt.plot(path_x, path_y, 'b-', linewidth=2, label='Planned Path')
                    plt.scatter(path_x[0], path_y[0], color='green', s=100, label='Start', zorder=5)
                    plt.scatter(path_x[-1], path_y[-1], color='red', s=100, label='Goal', zorder=5)
                
                # Plot obstacles
                if obstacles:
                    for obs in obstacles:
                        if len(obs) >= 7:
                            center_x, center_y = obs[1], obs[2]
                            plt.scatter(center_x, center_y, color='orange', s=50, marker='x')
                
                plt.xlabel('X coordinate')
                plt.ylabel('Y coordinate')
                plt.title(f'Path Planning Result - Iteration {self.iteration_count}')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.axis('equal')
                
                plot_filename = f"{self.results_folder}/path_{self.iteration_count:04d}_{timestamp}.png"
                plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
                plt.close()
                
            except Exception as e:
                print(f"   Warning: Could not create plot: {e}")
            
        except Exception as e:
            print(f"   Error saving path result: {e}")
    
    def run(self):
        """Main execution loop."""
        if not self.initialize_devices():
            print("Failed to initialize devices. Exiting.")
            return
        
        print("Starting main pipeline...")
        print("Press Ctrl+C to stop gracefully\n")
        
        try:
            start_time = time.time()
            
            while self.running and not self.stop_event.is_set():
                cycle_start = time.time()
                
                # Run planning cycle
                success = self.run_planning_cycle()
                self.iteration_count += 1
                
                # Calculate timing
                cycle_time = time.time() - cycle_start
                remaining_time = max(0, self.update_interval - cycle_time)
                
                # Sleep for remaining time
                if remaining_time > 0:
                    time.sleep(remaining_time)
                
                # Print statistics periodically
                if self.iteration_count % 50 == 0:
                    elapsed_time = time.time() - start_time
                    avg_frequency = self.iteration_count / elapsed_time
                    success_rate = (self.successful_paths / self.iteration_count) * 100
                    avg_obstacles = self.total_obstacles_detected / self.iteration_count
                    
                    print(f"\n--- Statistics after {self.iteration_count} iterations ---")
                    print(f"Average frequency: {avg_frequency:.2f} Hz")
                    print(f"Path planning success rate: {success_rate:.1f}%")
                    print(f"Average obstacles per cycle: {avg_obstacles:.2f}")
                    print("---\n")
        
        except KeyboardInterrupt:
            print("\nReceived interrupt signal. Stopping gracefully...")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up all devices and resources."""
        print("\nCleaning up devices...")
        
        self.running = False
        self.stop_event.set()
        
        # Stop INS device
        if self.ins_device:
            try:
                self.ins_device.disconnect()
                print("INS device disconnected")
            except Exception as e:
                print(f"Error disconnecting INS device: {e}")
        
        # Stop RealSense camera
        if self.rs_camera:
            try:
                self.rs_camera.close()
                print("RealSense camera closed")
            except Exception as e:
                print(f"Error closing RealSense camera: {e}")
        
        # Stop YOLO detector
        if self.yolo_detector:
            try:
                self.yolo_detector.stop()
                print("YOLO detector stopped")
            except Exception as e:
                print(f"Error stopping YOLO detector: {e}")
        
        # Print final statistics
        if self.iteration_count > 0:
            success_rate = (self.successful_paths / self.iteration_count) * 100
            avg_obstacles = self.total_obstacles_detected / self.iteration_count
            
            print(f"\n--- Final Statistics ---")
            print(f"Total iterations: {self.iteration_count}")
            print(f"Successful paths: {self.successful_paths}")
            print(f"Success rate: {success_rate:.1f}%")
            print(f"Total obstacles detected: {self.total_obstacles_detected}")
            print(f"Average obstacles per cycle: {avg_obstacles:.2f}")
            print(f"Results saved in: {self.results_folder}/")
        
        print("Cleanup completed successfully")

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    print("\nSignal received, initiating shutdown...")
    sys.exit(0)

def main():
    """Main function."""
    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    # Create and run the pipeline
    pipeline = MainPipeline()
    pipeline.run()

if __name__ == "__main__":
    main()
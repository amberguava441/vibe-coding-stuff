#!/usr/bin/env python3

import time
import queue
import threading
import numpy as np
import cv2 as cv
import sys
import traceback
from datetime import datetime

# Import custom modules
from get_ins import get_ins
from c import camera, LaneDetectionTester
from get_yolo import YoloV5Detector, process_lane_and_detection_3d
from ct import getF
from p import planner

class SensoryStack:
    def __init__(self, gps_port='/dev/ttyUSB0', camera_id=100, yolo_weights='yolov5s.pt'):
        """
        Initialize the sensory stack with all components
        
        Args:
            gps_port (str): Serial port for GPS/IMU device
            camera_id (int): Camera ID (100 for RealSense depth camera)
            yolo_weights (str): Path to YOLO weights file
        """
        print("[INFO] Initializing Sensory Stack...")
        
        # Configuration parameters
        self.gps_port = gps_port
        self.camera_id = camera_id
        self.yolo_weights = yolo_weights
        self.update_rate = 10  # Hz (10 times per second)
        self.loop_interval = 1.0 / self.update_rate  # 0.1 seconds
        
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
        
        # Initialize components
        self.gps_device = None
        self.camera_system = None
        self.lane_detector = None
        self.yolo_detector = None
        self.path_planner = None
        
        # Destination coordinates (fixed)
        self.destination_lon = 126.62565295
        self.destination_lat = 45.72713751
        self.planner_initialized = False
        
        # Threading components for YOLO
        self.yolo_input_queue = queue.Queue(maxsize=2)
        self.yolo_output_queue = queue.Queue(maxsize=2)
        self.depth_queue = queue.Queue(maxsize=2)
        self.stop_event = threading.Event()
        
        # Initialize all systems
        self.initialize_systems()
        
    def initialize_systems(self):
        """Initialize all sensory systems"""
        try:
            # 1. Initialize GPS/IMU device
            print("[INFO] Initializing GPS device...")
            self.gps_device = get_ins(self.gps_port)
            if not self.gps_device.connect():
                raise Exception("Failed to connect to GPS device")
            if not self.gps_device.start_gps_mode():
                raise Exception("Failed to start GPS mode")
            print("[INFO] GPS device initialized successfully")
            
            # 2. Initialize camera system
            print("[INFO] Initializing camera system...")
            self.camera_system = camera([self.camera_id])
            if not self.camera_system.work:
                raise Exception("Camera system initialization failed")
            print("[INFO] Camera system initialized successfully")
            
            # 3. Initialize lane detection
            print("[INFO] Initializing lane detection...")
            self.lane_detector = LaneDetectionTester()
            print("[INFO] Lane detection initialized successfully")
            
            # 4. Initialize YOLO detector
            print("[INFO] Initializing YOLO detector...")
            self.yolo_detector = YoloV5Detector(
                weights=self.yolo_weights,
                img_size=640,
                conf_thres=0.45,
                iou_thres=0.45,
                classes=[0, 1, 2, 3, 5],  # person, bicycle, car, motorcycle, bus
                stop_event=self.stop_event
            )
            print("[INFO] YOLO detector initialized successfully")
            
            # 5. Initialize path planner
            print("[INFO] Initializing path planner...")
            self.path_planner = planner()
            print("[INFO] Path planner initialized successfully")
            
            print("[INFO] All systems initialized successfully!")
            
        except Exception as e:
            print(f"[ERROR] System initialization failed: {e}")
            traceback.print_exc()
            self.cleanup()
            sys.exit(1)
    
    def initialize_planner_origin(self):
        """Initialize planner origin using GPS data"""
        print("[INFO] Getting initial GPS position for planner...")
        max_attempts = 10
        attempt = 0
        
        while attempt < max_attempts:
            gps_data = self.gps_device.get_gps(max_wait_time=1.0)
            if gps_data is not None:
                lon, lat = gps_data[0], gps_data[1]  # Extract lon, lat
                print(f"[INFO] Initial position: lat={lat:.8f}, lon={lon:.8f}")
                
                # Set origin and destination for planner
                self.path_planner.setOrigin(lon, lat)
                self.path_planner.setDestination(self.destination_lon, self.destination_lat)
                self.planner_initialized = True
                
                print(f"[INFO] Planner initialized with destination: lat={self.destination_lat:.8f}, lon={self.destination_lon:.8f}")
                return True
            
            attempt += 1
            print(f"[WARNING] GPS data not available, attempt {attempt}/{max_attempts}")
            time.sleep(0.5)
        
        print("[ERROR] Failed to get initial GPS position for planner")
        return False
    
    def start_yolo_thread(self):
        """Start YOLO detection in a separate thread"""
        print("[INFO] Starting YOLO detection thread...")
        self.yolo_thread = threading.Thread(
            target=self.yolo_detector.run,
            args=(self.yolo_input_queue, self.yolo_output_queue),
            daemon=True
        )
        self.yolo_thread.start()
        print("[INFO] YOLO detection thread started")
    
    def get_sensory_data_and_plan(self):
        """
        Get complete sensory data and generate path plan
        
        Returns:
            tuple: (sensory_data, planned_path) where:
                   sensory_data = [x, y, dir_vec_x, dir_vec_y, lon, lat, cog, lane_points, obstacles_3d]
                   planned_path = list of waypoints from planner
        """
        try:
            # 1. Get GPS data
            gps_data = self.gps_device.get_gps(max_wait_time=0.3)
            if gps_data is None:
                print("[WARNING] No GPS data available")
                return None, []
            
            # Extract GPS components: [lon, lat, x, y, dx, dy, cog, dir_vec_x, dir_vec_y]
            lon, lat, x, y, dx, dy, cog, dir_vec_x, dir_vec_y = gps_data
            
            # 2. Get camera frames
            flag, calibrated, rgb_frame, depth_frame = self.camera_system.getFrame(self.camera_id)
            if flag != 0 or rgb_frame is None:
                print("[WARNING] Failed to get camera frame")
                return None, []
            
            print(f"[DEBUG] RGB frame shape: {rgb_frame.shape}")
            if depth_frame is not None:
                print(f"[DEBUG] Depth frame shape: {depth_frame.shape}")
            
            # 3. Lane detection and coordinate transformation
            lane_points = []
            try:
                detected_lanes = self.lane_detector.detect_lanes(rgb_frame)
                if detected_lanes:
                    # Convert lane points to 3D coordinates if depth is available
                    if depth_frame is not None:
                        lane_points_pixel = [(x, y, depth_frame[y, x] if y < depth_frame.shape[0] and x < depth_frame.shape[1] else 0) 
                                           for x, y in detected_lanes]
                    else:
                        lane_points_pixel = [(x, y, 0) for x, y in detected_lanes]
                    
                    # Convert from pixel coordinates to world coordinates using getF()
                    lane_points = getF(lane_points_pixel, self.K, self.R, self.T)
                    
                print(f"[DEBUG] Detected {len(lane_points)} lane points")
            except Exception as e:
                print(f"[WARNING] Lane detection failed: {e}")
                lane_points = []
            
            # 4. YOLO object detection
            obstacles_3d = []
            try:
                # Add frame to YOLO input queue (non-blocking)
                if not self.yolo_input_queue.full():
                    self.yolo_input_queue.put(rgb_frame, block=False)
                
                # Add depth frame to depth queue for 3D processing
                if depth_frame is not None and not self.depth_queue.full():
                    self.depth_queue.put(depth_frame, block=False)
                
                # Try to get YOLO results (non-blocking)
                if not self.yolo_output_queue.empty():
                    detection_2d = self.yolo_output_queue.get(block=False)
                    
                    # Process 3D coordinates if we have depth data
                    if depth_frame is not None and not self.depth_queue.empty():
                        try:
                            depth_for_3d = self.depth_queue.get(block=False)
                            # Put data into temporary queues for the function
                            temp_yolo_queue = queue.Queue()
                            temp_depth_queue = queue.Queue()
                            temp_yolo_queue.put(detection_2d)
                            temp_depth_queue.put(depth_for_3d)
                            
                            obstacles_3d = process_lane_and_detection_3d(
                                temp_yolo_queue, temp_depth_queue, self.K
                            )
                        except Exception as e:
                            print(f"[WARNING] 3D processing failed: {e}")
                            obstacles_3d = []
                    
                print(f"[DEBUG] Detected {len(obstacles_3d)} 3D obstacles")
            except Exception as e:
                print(f"[WARNING] YOLO detection failed: {e}")
                obstacles_3d = []
            
            # 5. Combine all sensory data
            sensory_data = [
                x,            # x position (m)
                y,            # y position (m)  
                dir_vec_x,    # direction vector x component
                dir_vec_y,    # direction vector y component
                lon,          # longitude
                lat,          # latitude
                cog,          # course over ground (degrees)
                lane_points,  # list of lane points [(x,y,z), ...]
                obstacles_3d  # list of obstacles [class_id, center_x, center_y, center_z, width, length, height]
            ]
            
            # 6. Generate path plan if planner is initialized
            planned_path = []
            fixed_lane = [[(-100, 0), (-100, 1), (-100, 2), (-100, 3), (-100, 4), (-100, 5), 
                                (-100, 6), (-100, 7), (-100, 8), (-100, 9), (-100, 10), (-100, 11), 
                                    (-100, 12), (-100, 13), (-100, 14), (-100, 15), (-100, 16), (-100, 17), 
                                        (-100, 18), (-100, 19)], [(100, 0), (100, 1), (100, 2), (100, 3), (100, 4), (100, 5), 
                                            (100, 6), (100, 7), (100, 8), (100, 9), (100, 10), (100, 11), 
                                                (100, 12), (100, 13), (100, 14), (100, 15), (100, 16), (100, 17), 
                                                    (100, 18), (100, 19)]]

            if self.planner_initialized:
                try:
                    # Prepare planner inputs according to getpath() signature:
                    # getpath(self, xy, dxy, jw, o, lane, ol)
                    xy = (x, y)              # current position
                    dxy = (dx, dy)           # current velocity
                    jw = (lon, lat)          # current GPS coordinates
                    o = cog                  # orientation (course over ground)
                    #lane = lane_points       # lane points
                    lane = fixed_lane
                    ol = obstacles_3d        # obstacle list
                    
                    # Get path from planner
                    planned_path = self.path_planner.getpath(xy, dxy, jw, o, lane, ol)
                    
                    #print(f"[DEBUG] Generated path with {len(planned_path)} waypoints")
                    
                except Exception as e:
                    print(f"[WARNING] Path planning failed: {e}")
                    planned_path = []
            else:
                print("[WARNING] Planner not initialized, skipping path planning")
            
            return sensory_data, planned_path
            
        except Exception as e:
            print(f"[ERROR] Sensory data acquisition failed: {e}")
            traceback.print_exc()
            return None, []
    
    def run(self, duration=None, save_to_file=False):
        """
        Run the complete autonomous system (sensory + planning)
        
        Args:
            duration (float): Duration to run in seconds (None for infinite)
            save_to_file (bool): Whether to save data to file
        """
        print(f"[INFO] Starting autonomous system at {self.update_rate} Hz...")
        
        # Start YOLO detection thread
        self.start_yolo_thread()
        
        # Initialize planner with GPS origin
        if not self.initialize_planner_origin():
            print("[ERROR] Failed to initialize planner, stopping system")
            return
        
        # Setup data logging if requested
        if save_to_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = f"autonomous_data_{timestamp}.txt"
            print(f"[INFO] Logging data to {log_file}")
        
        start_time = time.time()
        frame_count = 0
        
        try:
            while not self.stop_event.is_set():
                loop_start_time = time.time()
                
                # Get sensory data and generate path plan
                sensory_data, planned_path = self.get_sensory_data_and_plan()
                
                if sensory_data is not None:
                    frame_count += 1
                    
                    # Extract key information for status display
                    x, y, dir_vec_x, dir_vec_y, lon, lat, cog, lane_points, obstacles_3d = sensory_data
                    
                    # Print status every 10 frames
                    if frame_count % 10 == 0:
                        print(f"[INFO] Frame {frame_count}: Pos=({x:.2f}, {y:.2f}), "
                              f"GPS=({lat:.6f}, {lon:.6f}), COG={cog:.1f}°, "
                              f"Lanes={len(lane_points)}, Obstacles={len(obstacles_3d)}, "
                              f"Path_waypoints={len(planned_path)}")
                    
                    # Save to file if requested
                    if save_to_file:
                        with open(log_file, 'a') as f:
                            timestamp = time.time()
                            # Save both sensory data and planned path
                            #f.write(f"{timestamp}: SENSORY={sensory_data}\n")
                            f.write(f"{timestamp}: PLANNED_PATH={planned_path}\n")
                
                # Check duration limit
                if duration is not None and (time.time() - start_time) >= duration:
                    print(f"[INFO] Duration limit reached ({duration}s)")
                    break
                
                # Maintain update rate at 10Hz
                loop_time = time.time() - loop_start_time
                sleep_time = self.loop_interval - loop_time
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    # If processing took longer than expected, warn but continue
                    if frame_count % 50 == 0:  # Only warn every 50 frames to avoid spam
                        print(f"[WARNING] Processing time ({loop_time:.3f}s) exceeded target interval ({self.loop_interval:.3f}s)")
                
        except KeyboardInterrupt:
            print("\n[INFO] Keyboard interrupt received")
        except Exception as e:
            print(f"[ERROR] Runtime error: {e}")
            traceback.print_exc()
        finally:
            print("[INFO] Stopping autonomous system...")
            self.stop()
    
    def stop(self):
        """Stop all systems and cleanup"""
        print("[INFO] Stopping all systems...")
        
        # Signal stop to all threads
        self.stop_event.set()
        
        # Stop YOLO detector
        if self.yolo_detector:
            self.yolo_detector.stop()
        
        # Wait for YOLO thread to finish
        if hasattr(self, 'yolo_thread') and self.yolo_thread.is_alive():
            self.yolo_thread.join(timeout=2.0)
        
        # Stop GPS device
        if self.gps_device:
            self.gps_device.disconnect()
        
        # Stop camera system
        if self.camera_system:
            self.camera_system.destroy()
        
        print("[INFO] All systems stopped")
    
    def cleanup(self):
        """Cleanup resources"""
        self.stop()

def main():
    """Main entry point"""
    print("=" * 60)
    print("Autonomous Vehicle System (Sensory + Planning)")
    print("=" * 60)
    
    # Configuration
    GPS_PORT = '/dev/ttyUSB0'
    CAMERA_ID = 100
    YOLO_WEIGHTS = 'yolov5s.pt'
    
    try:
        # Initialize complete autonomous system
        autonomous_system = SensoryStack(
            gps_port=GPS_PORT,
            camera_id=CAMERA_ID, 
            yolo_weights=YOLO_WEIGHTS
        )
        
        # Run the complete system (sensory + planning)
        # autonomous_system.run(duration=60, save_to_file=True)  # Run for 60 seconds with logging
        autonomous_system.run(save_to_file=True)  # Run indefinitely with logging
        
    except Exception as e:
        print(f"[ERROR] Main execution failed: {e}")
        traceback.print_exc()
    finally:
        print("[INFO] Program terminated")

if __name__ == "__main__":
    main()
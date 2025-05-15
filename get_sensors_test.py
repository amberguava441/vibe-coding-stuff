#!/usr/bin/env python3
# get_sensors_test.py

import cv2
import numpy as np
import time
import argparse
import os
import sys
from datetime import datetime

# Import the required classes and functions
from get_ins import get_ins
from get_realsense import get_realsense
from get_sensors import get_sensors

def main():
    """
    Test the get_sensors function with minimal output.
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Test sensor data collection')
    parser.add_argument('--port', type=str, default='/dev/ttyUSB0',
                        help='Serial port for the GPS/INS device')
    parser.add_argument('--timeout', type=float, default=0.5,
                        help='Maximum time to wait for sensor data (seconds)')
    parser.add_argument('--save_dir', type=str, default='sensor_data',
                        help='Directory to save sensor data')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of sensor samples to collect')
    parser.add_argument('--interval', type=float, default=0.1,
                        help='Interval between samples (seconds)')
    args = parser.parse_args()
    
    # Create save directory if it doesn't exist
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    # Initialize devices silently
    try:
        # Initialize INS device
        ins_device = get_ins(port=args.port)
        ins_device.connect()
        ins_device.start_gps_mode()
        ins_device.start_imu_mode()
        
        # Allow time for sensor data to start flowing
        time.sleep(1.0)
        
        # Initialize RealSense camera
        rs_camera = get_realsense()
        
        # Wait for camera to initialize
        time.sleep(0.5)
        
    except Exception as e:
        print(f"Initialization error: {e}")
        sys.exit(1)
    
    # Collect samples
    for sample_idx in range(args.num_samples):
        try:
            # Get sensor data
            sensor_data = get_sensors(
                ins_device=ins_device,
                rs_camera=rs_camera,
                max_wait_time=args.timeout
            )
            
            # Create consolidated array with all sensor data
            output = []
            
            # Add GPS data (7 values)
            if sensor_data['gps_data'] is not None:
                output.extend(sensor_data['gps_data'])  # [lon, lat, x, y, dx, dy, cog]
            else:
                output.extend([None] * 7)
                
            # Add IMU data (4 values)
            if sensor_data['imu_data'] is not None:
                output.extend(sensor_data['imu_data'])  # [accel_x, accel_y, accel_e, accel_n]
            else:
                output.extend([None] * 4)
                
            # Add RGB and depth images (2 items)
            output.append(sensor_data['rgb_image'])  # rgb_img
            output.append(sensor_data['depth_image'])  # depth_img
            
            # Print only the compact output array
            print(f"Output {sample_idx + 1}: {output}")
            
            # Save to disk silently
            if output[0] is not None or output[7] is not None or output[11] is not None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                sample_dir = os.path.join(args.save_dir, f"sample_{timestamp}")
                
                if not os.path.exists(sample_dir):
                    os.makedirs(sample_dir)
                
                # Save RGB image if available
                if output[11] is not None:
                    cv2.imwrite(os.path.join(sample_dir, "rgb_image.png"), output[11])
                
                # Save depth image if available
                if output[12] is not None:
                    cv2.imwrite(os.path.join(sample_dir, "depth_image.png"), output[12])
                
                # Save data file
                with open(os.path.join(sample_dir, "data.txt"), 'w') as f:
                    f.write(f"Timestamp: {timestamp}\n")
                    f.write(f"GPS: {output[0:7]}\n")
                    f.write(f"IMU: {output[7:11]}\n")
                    f.write(f"Images: {'RGB Available' if output[11] is not None else 'RGB Not Available'}, " +
                            f"{'Depth Available' if output[12] is not None else 'Depth Not Available'}\n")
            
            # Wait for specified interval
            if sample_idx < args.num_samples - 1:
                time.sleep(args.interval)
                
        except Exception as e:
            print(f"Error in sample {sample_idx + 1}: {e}")
    
    # Clean up resources silently
    try:
        ins_device.disconnect()
        rs_camera.close()
    except Exception:
        pass

if __name__ == "__main__":
    main()
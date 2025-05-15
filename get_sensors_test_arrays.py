#!/usr/bin/env python3
# get_sensors_test.py

import time
import argparse
import os
import sys

# Import the required functions
from get_ins import get_ins
from get_realsense import get_realsense
from get_sensors import get_sensors

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Sensor data collection')
    parser.add_argument('--port', type=str, default='/dev/ttyUSB0',
                        help='Serial port for GPS/INS device')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of samples to collect')
    parser.add_argument('--interval', type=float, default=0.01,
                        help='Interval between samples (seconds)')
    parser.add_argument('--output', type=str, default='sensor_arrays.txt',
                        help='Output file for sensor arrays')
    args = parser.parse_args()
    
    # Initialize devices
    try:
        ins_device = get_ins(port=args.port)
        ins_device.connect()
        ins_device.start_gps_mode()
        ins_device.start_imu_mode()
        time.sleep(1.0)  # Buffer filling time
        rs_camera = get_realsense()
    except Exception as e:
        print(f"Initialization error: {e}")
        sys.exit(1)
    
    # Open output file
    with open(args.output, 'w') as f:
        # Collect samples
        for _ in range(args.num_samples):
            try:
                # Get sensor data
                sensor_data = get_sensors(ins_device=ins_device, rs_camera=rs_camera)
                
                # Process data into a simple flat list
                output = []
                
                # Add GPS data
                if sensor_data['gps_data'] is not None:
                    output.extend(sensor_data['gps_data'])
                else:
                    output.extend([None] * 7)
                    
                # Add IMU data
                if sensor_data['imu_data'] is not None:
                    output.extend(sensor_data['imu_data'])
                else:
                    output.extend([None] * 4)
                
                # Add image status (not the images themselves - just True/False)
                output.append(sensor_data['rgb_image'] is not None)
                output.append(sensor_data['depth_image'] is not None)
                
                # Print the raw output array to console
                print(output)
                
                # Write array to file
                f.write(f"{output}\n")
                
                # Wait for next sample
                time.sleep(args.interval)
                
            except Exception as e:
                print(f"Error: {e}")
    
    # Clean up
    ins_device.disconnect()
    rs_camera.close()

if __name__ == "__main__":
    main()
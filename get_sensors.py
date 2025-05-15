#!/usr/bin/env python3
# get_sensors.py

import threading
import time
from get_ins import get_ins
from get_realsense import get_realsense

def get_sensors(ins_device, rs_camera, gps_mode=True, imu_mode=True, max_wait_time=0.5):
    """
    Capture data from GPS/IMU and RealSense camera simultaneously.
    
    Args:
        ins_device: Initialized get_ins instance
        rs_camera: Initialized GetRealSense instance (or None if not available)
        gps_mode: Whether to collect GPS data
        imu_mode: Whether to collect IMU data
        max_wait_time (float): Maximum time to wait for all data in seconds
    
    Returns:
        dict: Dictionary containing all sensor data with keys:
            'gps_data': [lon, lat, x, y, dx, dy, cog] or None if indoors/no signal
            'imu_data': [accel_x, accel_y, accel_e, accel_n] or None if unavailable
            'rgb_image': numpy array with RGB image or None if no camera/image
            'depth_image': numpy array with depth colormap or None if no depth data
    """
    # Initialize result containers
    results = {
        'gps_data': None,
        'imu_data': None,
        'rgb_image': None,
        'depth_image': None
    }
    
    # Capture results flags - to check if a thread completed successfully
    capture_success = {
        'gps': threading.Event(),
        'imu': threading.Event(),
        'rgb': threading.Event(),
        'depth': threading.Event()
    }
    
    # Define thread functions to capture each data type
    def capture_gps():
        if gps_mode:
            try:
                # Use a timeout appropriate for 5Hz data
                gps_data = ins_device.get_gps(max_wait_time=0.3)
                if gps_data:
                    results['gps_data'] = gps_data
                    capture_success['gps'].set()
            except Exception as e:
                print(f"Error capturing GPS data: {e}")
    
    def capture_imu():
        if imu_mode:
            try:
                # Use a shorter timeout for 100Hz data
                imu_data = ins_device.get_imu(max_wait_time=0.1)
                if imu_data:
                    results['imu_data'] = imu_data
                    capture_success['imu'].set()
            except Exception as e:
                print(f"Error capturing IMU data: {e}")
    
    def capture_rgb():
        if rs_camera:
            try:
                rgb_image = rs_camera.get_rs_rgb()
                if rgb_image is not None:
                    results['rgb_image'] = rgb_image
                    capture_success['rgb'].set()
            except Exception as e:
                print(f"Error capturing RGB image: {e}")
    
    def capture_depth():
        if rs_camera:
            try:
                depth_image = rs_camera.get_rs_depth()
                if depth_image is not None:
                    results['depth_image'] = depth_image
                    capture_success['depth'].set()
            except Exception as e:
                print(f"Error capturing depth image: {e}")
    
    # Create threads for each sensor
    threads = []
    
    # Add GPS and IMU threads if appropriate
    if gps_mode:
        threads.append(threading.Thread(target=capture_gps, name="GPS Thread"))
    if imu_mode:
        threads.append(threading.Thread(target=capture_imu, name="IMU Thread"))
    
    # Add camera threads if camera is available
    if rs_camera:
        threads.append(threading.Thread(target=capture_rgb, name="RGB Thread"))
        threads.append(threading.Thread(target=capture_depth, name="Depth Thread"))
    
    try:
        # Start all threads
        print("Starting sensor data collection...")
        for thread in threads:
            thread.start()
        
        # Wait for all threads to complete or timeout
        start_time = time.time()
        
        # Define maximum wait times for specific data types
        remaining_time = max_wait_time
        
        # Wait for threads to complete
        for thread in threads:
            # Don't wait longer than the remaining time
            thread_timeout = min(remaining_time, max_wait_time)
            thread.join(timeout=thread_timeout)
            
            # Update remaining time
            elapsed = time.time() - start_time
            remaining_time = max(0, max_wait_time - elapsed)
        
        # Check if any threads are still running
        for thread in threads:
            if thread.is_alive():
                print(f"Warning: {thread.name} timed out after {max_wait_time} seconds")
        
    except Exception as e:
        print(f"Error during sensor data collection: {e}")
    
    # Print status of collected data
    for key, value in results.items():
        if (key == 'gps_data' and not gps_mode) or (key == 'imu_data' and not imu_mode):
            continue  # Skip reporting if we're not collecting this data type
        status = "Collected" if value is not None else "Not available"
        print(f"{key}: {status}")
    
    return results
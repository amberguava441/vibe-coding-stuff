#!/usr/bin/env python3

import serial
import time
from helpers import geo_to_cartesian
from parsers import parse_gpgga, parse_gprmc, parse_gtimu, imu_to_world

class get_gps:
    def __init__(self, port, baudrate=115200, timeout=1):
        """Initialize the GPS/IMU converter with serial port settings."""
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.serial_connection = None
        self.reference_lat = None
        self.reference_lon = None
        
    def connect(self):
        """Establish connection to the GPS/IMU device."""
        try:
            self.serial_connection = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout
            )
            return True
        except serial.SerialException:
            return False
    
    def send_command(self, command):
        """Send a command to the GPS/IMU device."""
        try:
            if self.serial_connection and self.serial_connection.is_open:
                self.serial_connection.write((command + '\r\n').encode())
                return True
            else:
                return False
        except Exception:
            return False
    
    def gps_fetcher(self, max_wait_time=2.0):
        """
        Connect, request data, get readings from GPGGA and GPRMC sentence types, then disconnect.
        
        Args:
            max_wait_time (float): Maximum time to wait for all sentence types in seconds
            
        Returns:
            list: [lon, lat, x, y, dx, dy, cog] - None if no valid data available
                 lon: Longitude
                 lat: Latitude
                 x: East-West position in meters
                 y: North-South position in meters
                 dx: East component of velocity in m/s
                 dy: North component of velocity in m/s
                 cog: Course over ground in degrees
        """
        # Store the latest data for each sentence type
        latest_gpgga = None
        latest_gprmc = None
        
        # Connect to the GPS device
        if not self.connect():
            return None
        
        try:
            # Send commands to get GPGGA and GPRMC sentences
            self.send_command("$CMD,OUTPUT,COM0,GPGGA,1*FF")
            self.send_command("$CMD,THROUGH,COM0,GPRMC,1*FF")
            
            # Small delay to ensure commands are processed
            time.sleep(0.1)
            
            # Record start time
            start_time = time.time()
            
            # Continue reading until we have data from both sources or timeout
            while time.time() - start_time < max_wait_time:
                if self.serial_connection.in_waiting:
                    line = self.serial_connection.readline().decode('ascii', errors='replace').strip()
                    
                    # Skip empty lines
                    if not line:
                        continue
                    
                    # Process GPGGA sentences
                    if line.startswith('$GPGGA'):
                        data = parse_gpgga(line, self.reference_lat, self.reference_lon)
                        if data:
                            latest_gpgga = data
                            # Update reference coordinates if needed
                            if self.reference_lat is None or self.reference_lon is None:
                                self.reference_lat = data[1]  # lat at index 1
                                self.reference_lon = data[0]  # lon at index 0
                    
                    # Process GPRMC sentences
                    elif line.startswith('$GPRMC'):
                        data = parse_gprmc(line)
                        if data:
                            latest_gprmc = data
                    
                    # If we have data from both sources, we can stop
                    if latest_gpgga and latest_gprmc:
                        break
                
                # Small sleep to avoid hogging CPU
                time.sleep(0.01)
            
            # If we have both GPGGA and GPRMC data, combine them
            if latest_gpgga and latest_gprmc:
                # GPGGA data: [lon, lat, x, y]
                # GPRMC data: [dx, dy, cog]
                combined_data = [
                    latest_gpgga[0],  # lon
                    latest_gpgga[1],  # lat
                    latest_gpgga[2],  # x
                    latest_gpgga[3],  # y
                    latest_gprmc[0],  # dx
                    latest_gprmc[1],  # dy
                    latest_gprmc[2]   # cog
                ]
                return combined_data
            else:
                return None
            
        except Exception as e:
            print(f"Error in gps_fetcher: {e}")
            return None
        finally:
            # Always disconnect and send NULL commands
            self.disconnect()
    
    def imu_fetcher(self, max_wait_time=2.0):
        """
        Connect, request data, get readings from GTIMU sentence type,
        transform accelerations from IMU to World coordinates, then disconnect.
        
        Args:
            max_wait_time (float): Maximum time to wait for IMU data in seconds
            
        Returns:
            list: [accel_x, accel_y, accel_e, accel_n] - None if no valid data available
                 accel_x: X-axis acceleration in IMU coordinates (m/s²)
                 accel_y: Y-axis acceleration in IMU coordinates (m/s²)
                 accel_e: East component of acceleration in world coordinates (m/s²)
                 accel_n: North component of acceleration in world coordinates (m/s²)
        """
        # Store the latest data
        latest_gtimu = None
        latest_gprmc = None  # We need GPRMC for course over ground
        
        # Connect to the device
        if not self.connect():
            return None
        
        try:
            # Send commands to get GTIMU and GPRMC (needed for coordinate transformation)
            self.send_command("$CMD,OUTPUT,COM0,GTIMU,1*FF")
            self.send_command("$CMD,THROUGH,COM0,GPRMC,1*FF")
            
            # Small delay to ensure commands are processed
            time.sleep(0.1)
            
            # Record start time
            start_time = time.time()
            
            # Continue reading until we have both data types or timeout
            while time.time() - start_time < max_wait_time:
                if self.serial_connection.in_waiting:
                    line = self.serial_connection.readline().decode('ascii', errors='replace').strip()
                    
                    # Skip empty lines
                    if not line:
                        continue
                    
                    # Process GTIMU sentences
                    if line.startswith('$GTIMU'):
                        data = parse_gtimu(line)
                        if data:
                            latest_gtimu = data
                    
                    # Process GPRMC sentences (need for course over ground)
                    elif line.startswith('$GPRMC'):
                        data = parse_gprmc(line)
                        if data:
                            latest_gprmc = data
                    
                    # If we have both data types, we can stop
                    if latest_gtimu and latest_gprmc:
                        break
                
                # Small sleep to avoid hogging CPU
                time.sleep(0.01)
            
            # If we have IMU data and GPRMC data for orientation, transform coordinates
            if latest_gtimu and latest_gprmc:
                # Get course over ground from GPRMC
                cog_deg = latest_gprmc[2]  # COG is at index 2 in GPRMC data
                
                # Transform IMU accelerations to world coordinates
                accel_e, accel_n = imu_to_world(latest_gtimu, cog_deg)
                
                # Return combined data
                return [
                    latest_gtimu[0],  # accel_x (IMU)
                    latest_gtimu[1],  # accel_y (IMU)
                    accel_e,          # accel_e (East)
                    accel_n           # accel_n (North)
                ]
            else:
                return None
            
        except Exception as e:
            print(f"Error in imu_fetcher: {e}")
            return None
        finally:
            # Always disconnect and send NULL commands
            self.disconnect()
    
    def disconnect(self):
        """Close the serial connection after sending NULL commands."""
        if self.serial_connection and self.serial_connection.is_open:
            try:
                # Send NULL commands to stop data output
                self.send_command("$CMD,OUTPUT,COM0,NULL*FF")
                time.sleep(0.1)  # Brief pause to ensure command is processed
                self.send_command("$CMD,THROUGH,COM0,NULL*FF")
                time.sleep(0.1)  # Brief pause to ensure command is processed
                
                # Now close the serial connection
                self.serial_connection.close()
            except Exception:
                pass
                

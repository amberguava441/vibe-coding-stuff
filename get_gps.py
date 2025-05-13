#!/usr/bin/env python3

import serial
import time
from helpers import haversine_distance, geo_to_cartesian
from sentence_parser import parse_gpgga, parse_gprmc, parse_gtimu, imu_to_world_transform

class get_gps:
    def __init__(self, port, baudrate=115200, timeout=1):
        """Initialize the GPS converter with serial port settings."""
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.serial_connection = None
        self.reference_lat = None
        self.reference_lon = None
        
    def connect(self):
        """Establish connection to the GPS device."""
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
        """Send a command to the GPS device."""
        try:
            if self.serial_connection and self.serial_connection.is_open:
                self.serial_connection.write((command + '\r\n').encode())
                return True
            else:
                return False
        except Exception:
            return False
    
    def gps_fetcher_a(self, max_wait_time=2.0, imu_to_world=False):
        """
        Connect, request data, get the latest readings from all three sentence types,
        then disconnect.
        
        Args:
            max_wait_time (float): Maximum time to wait for all sentence types in seconds
            imu_to_world (bool): Whether to transform IMU accelerations to world coordinates
            
        Returns:
            tuple: If imu_to_world is False:
                  (gpgga_data, gprmc_data, gtimu_data) - 'NULL' if no valid data available
                  
                  If imu_to_world is True:
                  (gpgga_data, gprmc_data, latest_gtimu_world) - 'NULL' if no valid data available
        """
        # Store the latest data for each sentence type
        latest_gpgga = None
        latest_gprmc = None
        latest_gtimu = None
        
        # Connect to the GPS device
        if not self.connect():
            return "NULL", "NULL", "NULL"
        
        try:
            # Send commands to get all three sentence types
            self.send_command("$CMD,OUTPUT,COM0,GPGGA,1*FF")
            self.send_command("$CMD,THROUGH,COM0,GPRMC,1*FF")
            self.send_command("$CMD,OUTPUT,COM0,GTIMU,1*FF")
            
            # Small delay to ensure commands are processed
            time.sleep(0.1)
            
            # Record start time
            start_time = time.time()
            
            # Continue reading until we have data from all three sources or timeout
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
                    
                    # Process GTIMU sentences
                    elif line.startswith('$GTIMU'):
                        data = parse_gtimu(line)
                        if data:
                            latest_gtimu = data
                    
                    # If we have data from all three sources, we can stop
                    if latest_gpgga and latest_gprmc and latest_gtimu:
                        break
                
                # Small sleep to avoid hogging CPU
                time.sleep(0.01)
            
            # Transform IMU data to world coordinates if requested
            if imu_to_world and latest_gprmc != None and latest_gtimu != None:
                cog_deg = latest_gprmc[2]  # COG is at index 2 in GPRMC data
                latest_gtimu_world = imu_to_world_transform(latest_gtimu, cog_deg)
                
                # Return with transformed acceleration data
                return (
                    latest_gpgga if latest_gpgga else "NULL",
                    latest_gprmc if latest_gprmc else "NULL",
                    latest_gtimu_world
                )
            else:
                # Return the data without transformation
                return (
                    latest_gpgga if latest_gpgga else "NULL",
                    latest_gprmc if latest_gprmc else "NULL",
                    latest_gtimu if latest_gtimu else "NULL"
                )
            
        except Exception as e:
            print(f"Error in gps_fetcher_a: {e}")
            return "NULL", "NULL", "NULL"
        finally:
            # Always disconnect and send NULL commands
            self.disconnect()
    
    def gps_fetcher_b(self, max_wait_time=2.0):
        """
        Convenience method to get GPS data with IMU accelerations transformed to world coordinates.
        
        Args:
            max_wait_time (float): Maximum time to wait for all sentence types in seconds
            
        Returns:
            tuple: (gpgga_data, gprmc_data, latest_gtimu_world) - 'NULL' if no valid data available
        """
        return self.gps_fetcher_a(max_wait_time, imu_to_world=True)
    
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

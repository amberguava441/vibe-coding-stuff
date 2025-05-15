#!/usr/bin/env python3
# get_ins.py

import serial
import time
import queue
import threading
from helpers import geo_to_cartesian
from parsers import parse_gpgga, parse_gprmc, parse_gtimu, imu_to_world

class get_ins:
    def __init__(self, port, baudrate=115200, timeout=1):
        """Initialize the GPS/IMU converter with serial port settings."""
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.serial_connection = None
        self.reference_lat = None
        self.reference_lon = None
        self.is_connected = False
        self.gps_mode_active = False
        self.imu_mode_active = False
        
        # Sentence queues - store the latest valid sentences
        self.gpgga_queue = queue.Queue(maxsize=5)
        self.gprmc_queue = queue.Queue(maxsize=5)
        self.gtimu_queue = queue.Queue(maxsize=20)
        
        # Reader thread
        self.reader_thread = None
        self.keep_reading = False
        
    def connect(self):
        """Establish connection to the GPS/IMU device."""
        try:
            if not self.is_connected:
                self.serial_connection = serial.Serial(
                    port=self.port,
                    baudrate=self.baudrate,
                    timeout=self.timeout
                )
                self.is_connected = True
                print(f"Connected to INS device on port {self.port}")
            return True
        except serial.SerialException as e:
            print(f"Failed to connect to INS device: {e}")
            return False
    
    def send_command(self, command):
        """Send a command to the GPS/IMU device."""
        try:
            if self.serial_connection and self.serial_connection.is_open:
                self.serial_connection.write((command + '\r\n').encode())
                # Wait briefly to allow the device to process the command
                time.sleep(0.1)
                
                # Clear any "OK" responses to avoid parser confusion
                self._clear_buffer()
                return True
            else:
                print("Cannot send command: device not connected")
                return False
        except Exception as e:
            print(f"Error sending command: {e}")
            return False
    
    def _clear_buffer(self):
        """Clear any pending data from the input buffer."""
        try:
            if self.serial_connection and self.serial_connection.is_open:
                if self.serial_connection.in_waiting:
                    # Read and discard all available data
                    self.serial_connection.read(self.serial_connection.in_waiting)
        except Exception as e:
            print(f"Error clearing buffer: {e}")
    
    def _validate_nmea_sentence(self, sentence):
        """
        Basic validation of NMEA sentence format and checksum.
        
        Args:
            sentence (str): NMEA sentence to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        if not sentence or len(sentence) < 5:
            return False
            
        # Check if it starts with $
        if not sentence.startswith('$'):
            return False
            
        # Check if it has a checksum
        if '*' not in sentence:
            return False
            
        try:
            # Split the sentence and checksum
            data, checksum = sentence.split('*')
            
            # Remove the $ at the beginning
            data = data[1:]
            
            # Calculate the checksum (XOR of all chars between $ and *)
            calculated_checksum = 0
            for char in data:
                calculated_checksum ^= ord(char)
                
            # Convert the hex checksum string to integer
            provided_checksum = int(checksum[:2], 16)
            
            # Compare calculated and provided checksums
            return calculated_checksum == provided_checksum
        except Exception:
            return False
    
    def _reader_thread_func(self):
        """Thread function to continuously read and sort NMEA sentences."""
        while self.keep_reading:
            try:
                if self.serial_connection and self.serial_connection.is_open and self.serial_connection.in_waiting:
                    # Read a line from the serial port
                    line = self.serial_connection.readline().decode('ascii', errors='replace').strip()
                    
                    # Skip empty lines or OK responses
                    if not line or line == "OK":
                        continue
                    
                    # Process only sentences that start properly
                    if line.startswith('$'):
                        # Basic validation
                        if not self._validate_nmea_sentence(line):
                            continue
                            
                        # Sort into appropriate queues
                        if line.startswith('$GPGGA'):
                            data = parse_gpgga(line, self.reference_lat, self.reference_lon)
                            if data:
                                # Update reference coordinates if needed
                                if self.reference_lat is None or self.reference_lon is None:
                                    self.reference_lat = data[1]  # lat at index 1
                                    self.reference_lon = data[0]  # lon at index 0
                                    
                                # Add to queue, remove oldest if full
                                if self.gpgga_queue.full():
                                    self.gpgga_queue.get()
                                self.gpgga_queue.put(data)
                                
                        elif line.startswith('$GPRMC'):
                            data = parse_gprmc(line)
                            if data:
                                if self.gprmc_queue.full():
                                    self.gprmc_queue.get()
                                self.gprmc_queue.put(data)
                                
                        elif line.startswith('$GTIMU'):
                            data = parse_gtimu(line)
                            if data:
                                if self.gtimu_queue.full():
                                    self.gtimu_queue.get()
                                self.gtimu_queue.put(data)
                else:
                    # Small sleep to avoid hogging CPU when there's no data
                    time.sleep(0.001)
                    
            except Exception as e:
                print(f"Error in reader thread: {e}")
                time.sleep(0.1)  # Pause briefly on error to avoid rapid error loops
    
    def start_sentence_reader(self):
        """Start the background thread that reads and sorts NMEA sentences."""
        if not self.is_connected:
            print("Cannot start reader: device not connected")
            return False
            
        if self.reader_thread and self.reader_thread.is_alive():
            print("Reader thread already running")
            return True
            
        # Clear queues
        while not self.gpgga_queue.empty():
            self.gpgga_queue.get()
        while not self.gprmc_queue.empty():
            self.gprmc_queue.get()
        while not self.gtimu_queue.empty():
            self.gtimu_queue.get()
            
        # Start reader thread
        self.keep_reading = True
        self.reader_thread = threading.Thread(
            target=self._reader_thread_func,
            name="NMEA Reader Thread",
            daemon=True
        )
        self.reader_thread.start()
        print("NMEA sentence reader started")
        return True
    
    def stop_sentence_reader(self):
        """Stop the background NMEA sentence reader thread."""
        if self.reader_thread and self.reader_thread.is_alive():
            self.keep_reading = False
            self.reader_thread.join(timeout=1.0)
            print("NMEA sentence reader stopped")
            return True
        return False
    
    def start_gps_mode(self):
        """
        Configure the device to output GPS data (GPGGA and GPRMC sentences).
        Returns True if successful, False otherwise.
        """
        if not self.is_connected and not self.connect():
            return False
            
        try:
            # Send commands to get GPGGA and GPRMC sentences
            success1 = self.send_command("$CMD,OUTPUT,COM0,GPGGA,0.2*FF")
            success2 = self.send_command("$CMD,THROUGH,COM0,GPRMC,0.2*FF")
            
            if success1 and success2:
                self.gps_mode_active = True
                # Start the sentence reader if not already running
                if not self.reader_thread or not self.reader_thread.is_alive():
                    self.start_sentence_reader()
                print("GPS mode activated successfully")
                return True
            else:
                print("Failed to activate GPS mode")
                return False
        except Exception as e:
            print(f"Error starting GPS mode: {e}")
            return False
    
    def start_imu_mode(self):
        """
        Configure the device to output IMU data (GTIMU and GPRMC sentences).
        Returns True if successful, False otherwise.
        """
        if not self.is_connected and not self.connect():
            return False
            
        try:
            # Send commands to get GTIMU and GPRMC (needed for coordinate transformation)
            success1 = self.send_command("$CMD,OUTPUT,COM0,GTIMU,0.01*FF")
            # Note: GPRMC is already set up in GPS mode, but we'll set it again in case GPS mode wasn't activated
            success2 = self.send_command("$CMD,THROUGH,COM0,GPRMC,0.2*FF")
            
            if success1 and success2:
                self.imu_mode_active = True
                # Start the sentence reader if not already running
                if not self.reader_thread or not self.reader_thread.is_alive():
                    self.start_sentence_reader()
                print("IMU mode activated successfully")
                return True
            else:
                print("Failed to activate IMU mode")
                return False
        except Exception as e:
            print(f"Error starting IMU mode: {e}")
            return False
    
    def get_gps(self, max_wait_time=0.3):
        """
        Get readings from GPGGA and GPRMC sentence types.
        
        Args:
            max_wait_time (float): Maximum time to wait for all sentence types in seconds
            
        Returns:
            list: [lon, lat, x, y, dx, dy, cog] - None if no valid data available
        """
        # Check if we're in GPS mode and device is connected
        if not self.is_connected:
            print("Device not connected, cannot get GPS data")
            return None
            
        if not self.gps_mode_active:
            print("GPS mode not active, call start_gps_mode() first")
            return None
            
        if not self.reader_thread or not self.reader_thread.is_alive():
            print("NMEA reader not active")
            return None
        
        try:
            # Record start time
            start_time = time.time()
            
            # Keep checking for both sentence types until we have them or timeout
            while time.time() - start_time < max_wait_time:
                # Check if we have both sentence types
                if not self.gpgga_queue.empty() and not self.gprmc_queue.empty():
                    # Get the latest data from queues without removing
                    latest_gpgga = self.gpgga_queue.queue[-1]
                    latest_gprmc = self.gprmc_queue.queue[-1]
                    
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
                
                # Small sleep to avoid hogging CPU
                time.sleep(0.01)
            
            # If we get here, we've timed out
            print("Timed out waiting for GPS data")
            return None
            
        except Exception as e:
            print(f"Error in get_gps: {e}")
            return None
    
    def get_imu(self, max_wait_time=0.1):
        """
        Get readings from GTIMU sentence type and transform accelerations
        from IMU to World coordinates.
        
        Args:
            max_wait_time (float): Maximum time to wait for IMU data in seconds
            
        Returns:
            list: [accel_x, accel_y, accel_e, accel_n] - None if no valid data available
        """
        # Check if we're in IMU mode and device is connected
        if not self.is_connected:
            print("Device not connected, cannot get IMU data")
            return None
            
        if not self.imu_mode_active:
            print("IMU mode not active, call start_imu_mode() first")
            return None
            
        if not self.reader_thread or not self.reader_thread.is_alive():
            print("NMEA reader not active")
            return None
        
        try:
            # Record start time
            start_time = time.time()
            
            # Keep checking for both sentence types until we have them or timeout
            while time.time() - start_time < max_wait_time:
                # Check if we have both sentence types
                if not self.gtimu_queue.empty() and not self.gprmc_queue.empty():
                    # Get the latest data from queues without removing
                    latest_gtimu = self.gtimu_queue.queue[-1]
                    latest_gprmc = self.gprmc_queue.queue[-1]
                    
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
                
                # Small sleep to avoid hogging CPU
                time.sleep(0.01)
            
            # If we get here, we've timed out
            print("Timed out waiting for IMU data")
            return None
            
        except Exception as e:
            print(f"Error in get_imu: {e}")
            return None
    
    def stop_data_output(self):
        """Stop all data output from the device."""
        if self.is_connected:
            # First stop the sentence reader
            self.stop_sentence_reader()
            
            # Then stop data output from the device
            success1 = self.send_command("$CMD,OUTPUT,COM0,NULL*FF")
            success2 = self.send_command("$CMD,THROUGH,COM0,NULL*FF")
            
            if success1 and success2:
                self.gps_mode_active = False
                self.imu_mode_active = False
                print("All data output stopped")
                return True
            else:
                print("Failed to stop data output")
                return False
        return False
    
    def disconnect(self):
        """Close the serial connection after stopping data output."""
        try:
            if self.is_connected:
                # First stop data output and reader
                self.stop_data_output()
                
                # Now close the serial connection
                if self.serial_connection and self.serial_connection.is_open:
                    self.serial_connection.close()
                    
                self.is_connected = False
                print("Disconnected from INS device")
                return True
            return False
        except Exception as e:
            print(f"Error disconnecting: {e}")
            return False
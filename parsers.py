#!/usr/bin/env python3

import pynmea2
import math
from helpers import geo_to_cartesian

def parse_gpgga(sentence, ref_lat=None, ref_lon=None):
    """
    Parse GPGGA sentence and extract relevant information.
    
    Args:
        sentence: NMEA sentence string
        ref_lat: Reference latitude for Cartesian conversion
        ref_lon: Reference longitude for Cartesian conversion
        
    Returns:
        list: [lon, lat, x, y] or None if invalid
    """
    try:
        # Check if it's a GPGGA message first
        if not sentence.startswith('$GPGGA'):
            return None
            
        msg = pynmea2.parse(sentence)
            
        # Check if we have a valid fix
        if msg.gps_qual == 0:  # 0 means no fix
            return None
            
        # Extract latitude and longitude
        lat = float(msg.latitude)
        lon = float(msg.longitude)
        
        # Convert to Cartesian coordinates
        x, y = geo_to_cartesian(lat, lon, ref_lat, ref_lon)
        
        # Return as an array [lon, lat, x, y]
        return [lon, lat, x, y]
    except Exception as e:
        print(f"Error parsing GPGGA: {e}")
        return None

def parse_gprmc(sentence):
    """
    Parse GPRMC sentence and extract relevant information.
    
    Args:
        sentence: NMEA sentence string
        
    Returns:
        list: [dx, dy, cog_deg] or None if invalid
    """
    try:
        # Check if it's a GPRMC message first
        if not sentence.startswith('$GPRMC'):
            return None
            
        msg = pynmea2.parse(sentence)
            
        # Check if we have a valid fix
        if not msg.status == 'A':  # 'A' means valid position
            return None
            
        # Extract speed over ground (SOG) in knots and convert to m/s
        sog_knots = float(msg.spd_over_grnd) if msg.spd_over_grnd else 0.0
        sog_ms = sog_knots * 0.5144  # Convert knots to m/s
        
        # Extract course over ground (COG) in degrees
        cog_deg = float(msg.true_course) if msg.true_course else 0.0
        
        # Calculate dx and dy components (North, East velocity)
        # Note: COG is measured clockwise from True North, so:
        # - North component (dy) = SOG * cos(COG)
        # - East component (dx) = SOG * sin(COG)
        cog_rad = math.radians(cog_deg)
        dy = sog_ms * math.cos(cog_rad)  # North component
        dx = sog_ms * math.sin(cog_rad)  # East component
        
        # Return as an array [dx, dy, cog_deg]
        return [dx, dy, cog_deg]
    except Exception as e:
        print(f"Error parsing GPRMC: {e}")
        return None

def parse_gpfpd(sentence):
    """
    Parse GTIMU sentence and extract accelerations.
    
    Args:
        sentence: NMEA sentence string
        
    Returns:
        list: [accel_x, accel_y] or None if invalid
    """
    try:
        # Validate sentence starts with $GTIMU
        if not sentence.startswith('$GPFPD'):
            return None
        
        # Split the sentence by comma
        parts = sentence.split(',')
        
        # Validate the sentence has enough parts
        if len(parts) < 16:
            return None
        
        # Parse acceleration values (fields 6, 7, 8 - indexes 5, 6, 7)
        # Convert from g to m/s^2 by multiplying by 9.8
        dx = float(parts[9])
        dy = float(parts[10])
        cog_deg = float(parts[3])
        
        # Return as an array [accel_x, accel_y]
        return [dx, dy, cog_deg]
    except (ValueError, IndexError) as e:
        print(f"Error parsing GPFPD: {e}")
        return None

def parse_gtimu(sentence):
    """
    Parse GTIMU sentence and extract accelerations.
    
    Args:
        sentence: NMEA sentence string
        
    Returns:
        list: [accel_x, accel_y] or None if invalid
    """
    try:
        # Validate sentence starts with $GTIMU
        if not sentence.startswith('$GTIMU'):
            return None
        
        # Split the sentence by comma
        parts = sentence.split(',')
        
        # Validate the sentence has enough parts
        if len(parts) < 10:
            return None
        
        # Parse acceleration values (fields 6, 7, 8 - indexes 5, 6, 7)
        # Convert from g to m/s^2 by multiplying by 9.8
        accel_x = float(parts[6]) * 9.8
        accel_y = float(parts[7]) * 9.8
        
        # Return as an array [accel_x, accel_y]
        return [accel_x, accel_y]
    except (ValueError, IndexError) as e:
        print(f"Error parsing GTIMU: {e}")
        return None

def imu_to_world(imu_data, cog_deg):
    """
    Transform IMU accelerations from device coordinates to world coordinates.
    
    Args:
        imu_data (list): [accel_x, accel_y] in device coordinates
        cog_deg (float): Course Over Ground in degrees
        
    Returns:
        list: [accel_e, accel_n] in world coordinates (East, North)
    """
    accel_x, accel_y = imu_data
    cog_rad = math.radians(cog_deg)
    
    # Transform accelerations to world coordinates (East-North)
    accel_n = accel_y * math.cos(cog_rad) - accel_x * math.sin(cog_rad)
    accel_e = accel_y * math.sin(cog_rad) + accel_x * math.cos(cog_rad)
    
    return [accel_e, accel_n]
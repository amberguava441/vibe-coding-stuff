#!/usr/bin/env python3

import math

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees).
    Returns distance in meters.
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a)) 
    r = 6371000  # Radius of earth in meters
    return c * r

def geo_to_cartesian(lat, lon, ref_lat, ref_lon):
    """
    Convert latitude and longitude to 2D Cartesian coordinates (in meters)
    relative to the reference position.
    
    Args:
        lat: Latitude of the point
        lon: Longitude of the point
        ref_lat: Reference latitude
        ref_lon: Reference longitude
        
    Returns:
        tuple: (x, y) coordinates in meters
    """
    if ref_lat is None or ref_lon is None:
        return 0, 0
    
    # Calculate east-west distance (x)
    x = haversine_distance(ref_lat, ref_lon, ref_lat, lon)
    if lon < ref_lon:
        x = -x  # Negative direction
        
    # Calculate north-south distance (y)
    y = haversine_distance(ref_lat, ref_lon, lat, ref_lon)
    if lat < ref_lat:
        y = -y  # Negative direction
        
    return x, y
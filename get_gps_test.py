from get_gps import get_gps

# Create an instance with your serial port
gps = get_gps("/dev/ttyUSB0")  # or COM1 on Windows, etc.

# Get GPS data (position and velocity)
gps_data = gps.gps_fetcher()
if gps_data:
    lon, lat, x, y, dx, dy, cog = gps_data
    print(f"Position: {lon}, {lat}")
    print(f"Cartesian: {x}m, {y}m")
    print(f"Velocity: {dx}m/s, {dy}m/s")
    print(f"Course: {cog}°")

# Get IMU data (accelerations)
imu_data = gps.imu_fetcher()
if imu_data:
    accel_x, accel_y, accel_e, accel_n = imu_data
    print(f"IMU accelerations: {accel_x}m/s², {accel_y}m/s²")
    print(f"World accelerations: {accel_e}m/s², {accel_n}m/s²")
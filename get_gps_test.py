from get_gps import get_gps

gps = get_gps('/dev/ttyUSB0')

while True:
    gpgga, gprmc, gtimu = gps.gps_fetcher_b()
    print(gpgga, gprmc, gtimu)
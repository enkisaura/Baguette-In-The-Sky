"""
Parse NMEA files
"""

import pandas as pd
import numpy as np

from bits.src.reference_frame_object import GnssTimestamp
from bits.src.convert.space_conversion import wgs_to_ecef

def gga(filepath:str) -> pd.DataFrame:
    """
    Parse GGA from nmea text file. Requires RMC inside the NMEA file to get the date.

    source: https://docs.novatel.com/OEM7/Content/Logs/GPGGA.htm

    :param filepath: Path of the NMEA file
    :return: Dataframe with parsed GGA data
    """
    records = []

    current_date = None

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()

            # RMC : Getting date
            if line.startswith("$") and line[3:6] == "RMC":
                fields = line.split(",")

                try:
                    # fields[9] = DDMMYY
                    if len(fields) > 9 and fields[9]:
                        current_date = fields[9]

                except (ValueError, IndexError):
                    pass

                continue

            # GGA : parsing position
            if not (line.startswith("$") and line[3:6] == "GGA"):
                continue

            fields = line.split(",")

            if len(fields) < 10:
                continue

            if current_date is None:
                continue

            try:
                time_str    = fields[1]   # HHMMSS.ss
                lat_raw     = float(fields[2])
                lat_hem     = fields[3]
                lon_raw     = float(fields[4])
                lon_hem     = fields[5]
                fix_quality = int(fields[6])

                # Ignore invalid fixes
                if fix_quality == 0:
                    continue

                num_sats = int(fields[7]) if fields[7] else None
                hdop     = float(fields[8]) if fields[8] else None
                altitude = float(fields[9]) if fields[9] else None

                # Convert NMEA -> decimal degrees
                lat_deg = int(lat_raw / 100) + (lat_raw % 100) / 60
                lon_deg = int(lon_raw / 100) + (lon_raw % 100) / 60

                if lat_hem == "S":
                    lat_deg *= -1

                if lon_hem == "W":
                    lon_deg *= -1

                # Convert lla to ecef
                x_ecef, y_ecef, z_ecef = wgs_to_ecef(lat_deg, lon_deg, altitude)

                # Timestamp UTC
                dt_str = f"{current_date} {time_str}"

                dt = pd.to_datetime(
                    dt_str,
                    format="%d%m%y %H%M%S.%f",
                    utc=True
                )

                unix_time = dt.timestamp()

                gnss_timestamp = GnssTimestamp(unix_time, unit='s')

                records.append({
                    "timestamp": gnss_timestamp,
                    "lat":         lat_deg,
                    "lon":         lon_deg,
                    "altitude_m":  altitude,
                    "x_rx_m": x_ecef,
                    "y_rx_m": y_ecef,
                    "z_rx_m": z_ecef,
                    "fix_quality": fix_quality,
                    "num_sats":    num_sats,
                    "hdop":        hdop,
                })

            except (ValueError, IndexError):
                continue

    return pd.DataFrame(records)


def rmc(filepath: str) -> pd.DataFrame:
    """
    Parse RMC from nmea text file

    source: https://docs.novatel.com/OEM7/Content/Logs/GPRMC.htm

    :param filepath: Path of the NMEA file
    :return: Dataframe with parsed RMC data
    """
    records = []

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not (line.startswith("$") and line[3:6] == "RMC"):
                continue

            fields = line.split(",")

            if len(fields) < 10 or fields[2] != "A":
                continue

            try:
                time_str    = fields[1]   # HHMMSS.ss
                date_str    = fields[9]   # DDMMYY
                lat_raw     = float(fields[3])
                lat_hem     = fields[4]
                lon_raw     = float(fields[5])
                lon_hem     = fields[6]
                speed_knots = float(fields[7])
                cog_deg     = float(fields[8]) if fields[8] else None

                # Convert NMEA -> decimal degrees
                lat_deg = int(lat_raw / 100) + (lat_raw % 100) / 60
                lon_deg = int(lon_raw / 100) + (lon_raw % 100) / 60
                if lat_hem == "S": lat_deg *= -1
                if lon_hem == "W": lon_deg *= -1

                # Convert lla to ecef. Altitude is not available in RMC message; setting to 0
                x_ecef, y_ecef, z_ecef = wgs_to_ecef(lat_deg, lon_deg, 0)

                dt_str = f"{date_str} {time_str}"
                dt = pd.to_datetime(dt_str, format="%d%m%y %H%M%S.%f", utc=True)
                unix_time = dt.timestamp()
                gnss_timestamp = GnssTimestamp(unix_time, unit='s')

                cog_rad = -np.deg2rad(cog_deg) if cog_deg is not None else None

                records.append({
                    "timestamp": gnss_timestamp,
                    "lat":         lat_deg,
                    "lon":         lon_deg,
                    "x_rx_m":      x_ecef,
                    "y_rx_m":      y_ecef,
                    "z_rx_m":      z_ecef,
                    "speed_mps":   speed_knots/1.944,
                    "cog_rad":     cog_rad,
                })

            except (ValueError, IndexError):
                continue

    return pd.DataFrame(records)
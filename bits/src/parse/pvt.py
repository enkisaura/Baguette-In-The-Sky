"""
Parse NMEA files
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

from bits.src import convert

def gga(filepath:str|Path) -> pd.DataFrame:
    """
    Parse GGA from nmea text file. Requires RMC inside the NMEA file to get the date.

    Speed is unavailable with GGA.

    source: https://docs.novatel.com/OEM7/Content/Logs/GPGGA.htm

    :param filepath: Path of the NMEA file
    :return: BITS PVT dataframe
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
                x_ecef, y_ecef, z_ecef = convert.space.wgs_to_ecef(lat_deg, lon_deg, altitude)

                # Timestamp UTC
                time_str = f"{current_date} {time_str}"
                time = np.datetime64(datetime.strptime(time_str, "%d%m%y %H%M%S.%f"), "ns")

                records.append({
                    "time": time,
                    "corr_time": time,
                    "lat": lat_deg,
                    "lon": lon_deg,
                    "alt": altitude,
                    "x_rx_m": float(x_ecef),
                    "y_rx_m": float(y_ecef),
                    "z_rx_m": float(z_ecef),
                    "b_rx_m": 0,
                    "vx_rx_mps": None,
                    "vy_rx_mps": None,
                    "vz_rx_mps": None,
                    "vb_rx_mps": None,
                    "fix_quality": fix_quality,
                    "num_sats": num_sats,
                    "hdop": hdop,
                })

            except (ValueError, IndexError):
                continue

    return pd.DataFrame(records)


def rmc(filepath: str|Path) -> pd.DataFrame:
    """
    Parse RMC from nmea text file.

    Altitude is unavailable with RMC.

    source: https://docs.novatel.com/OEM7/Content/Logs/GPRMC.htm

    :param filepath: Path of the NMEA file
    :return: BITS PVT dataframe
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
                x_ecef, y_ecef, z_ecef = convert.space.wgs_to_ecef(lat_deg, lon_deg, 0)

                # Timestamp UTC
                time_str = f"{date_str} {time_str}"
                time = np.datetime64(datetime.strptime(time_str, "%d%m%y %H%M%S.%f"), "ns")

                cog_rad = -np.deg2rad(cog_deg) if cog_deg is not None else None
                speed_mps = speed_knots/1.944

                # Convert speed to ENU
                v_east = speed_mps * np.sin(cog_rad)
                v_north = speed_mps * np.cos(cog_rad)

                # Convert speed to ECEF
                vx, vy, vz = convert.space.enu_to_ecef(v_east, v_north, 0, x_ecef, y_ecef, z_ecef, with_translation=False)

                records.append({
                    "time": time,
                    "corr_time": time,
                    "lat": float(lat_deg),
                    "lon": float(lon_deg),
                    "alt": None,
                    "x_rx_m": float(x_ecef),
                    "y_rx_m": float(y_ecef),
                    "z_rx_m": float(z_ecef),
                    "b_rx_m": 0,
                    "vx_rx_mps": vx,
                    "vy_rx_mps": vy,
                    "vz_rx_mps": vz,
                    "vb_rx_mps": 0,
                    "speed_mps": float(speed_mps),
                    "cog_rad": float(cog_rad),
                })

            except (ValueError, IndexError):
                continue

    return pd.DataFrame(records)


def rtklib(filepath: str|Path) -> pd.DataFrame:
    """
    Parse RTKlib .pos file.

    :param filepath: Path of .pos file
    :return: BITS PVT dataframe
    """
    found_header = False
    ref_pos = None

    lost_in_translation = {
        "ecef": {
            "x-ecef(m)": "x_rx_m",
            "y-ecef(m)": "y_rx_m",
            "z-ecef(m)": "z_rx_m",
            "sdx(m)": "cov_xx_rx_m",
            "sdy(m)": "cov_yy_rx_m",
            "sdz(m)": "cov_zz_rx_m",
            "sdxy(m)": "cov_yx_rx_m",
            "sdyz(m)": "cov_zy_rx_m",
            "sdzx(m)": "cov_zx_rx_m",
        },
        "ecef_with_speed": {
            "x-ecef(m)": "x_rx_m",
            "y-ecef(m)": "y_rx_m",
            "z-ecef(m)": "z_rx_m",
            "sdx(m)": "cov_xx_rx_m",
            "sdy(m)": "cov_yy_rx_m",
            "sdz(m)": "cov_zz_rx_m",
            "sdxy(m)": "cov_yx_rx_m",
            "sdyz(m)": "cov_zy_rx_m",
            "sdzx(m)": "cov_zx_rx_m",
            "vx(m/s)": "vx_rx_mps",
            "vy(m/s)": "vy_rx_mps",
            "vz(m/s)": "vz_rx_mps",
            "sdvx": "cov_vxvx_rx_mps",
            "sdvy": "cov_vyvy_rx_mps",
            "sdvz": "cov_vzvz_rx_mps",
            "sdvxy": "cov_vyvx_rx_mps",
            "sdvyz": "cov_vzvy_rx_mps",
            "sdvzx": "cov_vzvx_rx_mps",
        },
        "lla":{
            "latitude(deg)": "lat",
            "longitude(deg)": "lon",
            "height(m)": "alt",
            "sde(m)": "cov_ee_rx_m",
            "sdn(m)": "cov_nn_rx_m",
            "sdu(m)": "cov_uu_rx_m",
            "sdne(m)": "cov_ne_rx_m",
            "sdun(m)": "cov_un_rx_m",
            "sdeu(m)": "cov_ue_rx_m",
        },
        "lla_with_speed": {
            "latitude(deg)": "lat",
            "longitude(deg)": "lon",
            "height(m)": "alt",
            "sde(m)": "cov_ee_rx_m",
            "sdn(m)": "cov_nn_rx_m",
            "sdu(m)": "cov_uu_rx_m",
            "sdne(m)": "cov_ne_rx_m",
            "sdun(m)": "cov_un_rx_m",
            "sdeu(m)": "cov_ue_rx_m",
            "ve(m/s)": "ve_rx_mps",
            "vn(m/s)": "vn_rx_mps",
            "vu(m/s)": "vu_rx_mps",
            "sdve": "cov_veve_rx_mps",
            "sdvn": "cov_vnvn_rx_mps",
            "sdvu": "cov_vuvu_rx_mps",
            "sdvne": "cov_vnve_rx_mps",
            "sdvun": "cov_vuvn_rx_mps",
            "sdveu": "cov_vuve_rx_mps",
        },
        "baseline": {
            "e-baseline(m)": "be_rx_m",
            "n-baseline(m)": "bn_rx_m",
            "u-baseline(m)": "bu_rx_m",
            "sde(m)": "cov_be_rx_m",
            "sdn(m)": "cov_bn_rx_m",
            "sdu(m)": "cov_bu_rx_m",
            "sdne(m)": "cov_bnbe_rx_m",
            "sdun(m)": "cov_bubn_rx_m",
            "sdeu(m)": "cov_bebu_rx_m",
        },
    }

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            fields = line.split()

            # Parse header
            if fields[0] == "%":
                if len(fields) < 3:
                    continue
                if fields[1] == "ref" and fields[2] == "pos":
                    ref_pos = (float(fields[4]), float(fields[5]), float(fields[6]))
                if fields[1] == "GPST" or fields[1] == "JST":
                    raise NotImplementedError('GPST and JST are not implemented yet. Please select "UTC" time format.')
                if fields[1] == "UTC":
                    found_header = True

                    # Check datatype
                    filetype = None
                    for datatype in lost_in_translation.keys():
                        number_of_match=0
                        for col in lost_in_translation[datatype].keys():
                            if col in fields:
                                number_of_match+=1
                            if number_of_match == len(lost_in_translation[datatype]):
                                filetype = datatype
                    if filetype is None:
                        raise KeyError("RTKlib file cannot be parsed.")
                    elif filetype == "baseline":
                        raise NotImplementedError("Cannot parse E/N/U-Baseline format yet.")

                    data = {}
                    fields = fields[1:]
                    for field in fields:
                        data[field] = []

            elif not found_header:
                raise KeyError("Data description not found in RTKlib file header.")
            else:
                # Parse data
                if len(fields) != len(data) + 1:
                    txt = f"Header fields/data mismatch, cannot parse line: {line}"
                    raise KeyError(txt)
                # Parse UTC
                date = fields[0].replace("/", "-")
                fields = [date + "T" + fields[1]] + fields[2:]

                for index, datatype in enumerate(data.keys()):
                    data[datatype].append(fields[index])

    # Parse time
    df = pd.DataFrame(data)
    df["UTC"] = df["UTC"].astype("datetime64[ns]")
    df.rename({"UTC": "time"}, axis="columns", inplace=True)

    # Convert str to float
    for col in lost_in_translation[filetype].keys():
        df[col] = df[col].astype("float64")

    # Parse ECEF type
    if filetype == "ecef" or filetype == "ecef_with_speed":
        # Variance instead of std
        diag_cols = ["sdx(m)", "sdy(m)", "sdz(m)"]
        vdiag_cols = ["sdvx", "sdvy", "sdvz"]
        offdiag_cols = ["sdxy(m)", "sdyz(m)", "sdzx(m)"]
        voffdiag_cols = ["sdvxy", "sdvyz", "sdvzx"]

        df[diag_cols] = df[diag_cols] ** 2
        df[offdiag_cols] = df[offdiag_cols] * df[offdiag_cols].abs()

        if filetype == "ecef_with_speed":
            df[vdiag_cols] = df[vdiag_cols] ** 2
            df[voffdiag_cols] = df[voffdiag_cols] * df[voffdiag_cols].abs()

        # Add lla
        lat, lon, alt = convert.space.ecef_to_wgs(df["x-ecef(m)"], df["y-ecef(m)"], df["z-ecef(m)"])
        df["lon"] = lon
        df["lat"] = lat
        df["alt"] = alt

    # Parse LLA type
    if filetype == "lla" or filetype == "lla_with_speed":
        # Variance instead of std
        diag_cols = ["sde(m)", "sdn(m)", "sdu(m)"]
        vdiag_cols = ["sdve", "sdvn", "sdvu"]
        offdiag_cols = ["sdne(m)", "sdeu(m)", "sdun(m)"]
        voffdiag_cols = ["sdvne", "sdveu", "sdvun"]

        df[diag_cols] = df[diag_cols] ** 2
        df[offdiag_cols] = df[offdiag_cols] * df[offdiag_cols].abs()

        if filetype == "lla_with_speed":
            df[vdiag_cols] = df[vdiag_cols] ** 2
            df[voffdiag_cols] = df[voffdiag_cols] * df[voffdiag_cols].abs()

        # Add ecef
        x_ecef, y_ecef, z_ecef = convert.space.wgs_to_ecef(df["latitude(deg)"], df["longitude(deg)"], df["height(m)"])
        df["x_rx_m"] = x_ecef
        df["y_rx_m"] = y_ecef
        df["z_rx_m"] = z_ecef

        # Covariance matrix
        ref_ecef = convert.space.wgs_to_ecef(*ref_pos)

        r = convert.space.ecef_to_enu_rotation_matrix(*ref_ecef, wgs=True)
        r = np.linalg.inv(r)

        df[[
            "cov_xx_rx_m", "cov_yy_rx_m", "cov_zz_rx_m",
            "cov_yx_rx_m", "cov_zy_rx_m", "cov_zx_rx_m"
        ]] = df.apply(lambda row: pd.Series(convert.space.covariance_enu_to_ecef(*row[diag_cols + offdiag_cols], r=r)),
                      axis=1)

        if filetype == "lla_with_speed":
            vx, vy, vz = (
                convert.space.enu_to_ecef(df["ve(m/s)"], df["vn(m/s)"], df["vu(m/s)"], *ref_ecef, with_translation=False))
            df["vx_rx_mps"] = vx
            df["vy_rx_mps"] = vy
            df["vz_rx_mps"] = vz

            df[[
                "cov_vxvx_rx_mps", "cov_vyvy_rx_mps", "cov_vzvz_rx_mps",
                "cov_vyvx_rx_mps", "cov_vzvy_rx_mps", "cov_vzvx_rx_mps"
            ]] = df.apply(lambda row: pd.Series(
                convert.space.covariance_enu_to_ecef(*row[vdiag_cols + voffdiag_cols], r=r)), axis=1)

    df.rename(lost_in_translation[filetype], axis="columns", inplace=True)

    return df

import pandas as pd
import numpy as np
from typing import Tuple

UNIX_EPOCH = np.datetime64("1970-01-01T00:00:00", "ns") # Starting epoch of UNIX time (UTC)
GPST_EPOCH = np.datetime64("1980-01-06T00:00:00", "ns") # Starting epoch of GPS Time (GPST) (UTC)
BDT_EPOCH = np.datetime64("2006-01-01 00:00:00", "ns") # Starting epoch of Beidou Time (BDT) (UTC)

NS_PER_SECOND = 1_000_000_000
SECONDS_PER_DAY = 86400
SECONDS_PER_WEEK = 604800

LEAP_SECONDS = np.array([
    "1981-06-30T23:59:59", "1982-06-30T23:59:59", "1983-06-30T23:59:59",
    "1985-06-30T23:59:59", "1987-12-31T23:59:59", "1989-12-31T23:59:59",
    "1990-12-31T23:59:59", "1992-06-30T23:59:59", "1993-06-30T23:59:59",
    "1994-06-30T23:59:59", "1995-12-31T23:59:59", "1997-06-30T23:59:59",
    "1998-12-31T23:59:59", "2005-12-31T23:59:59", "2008-12-31T23:59:59",
    "2012-06-30T23:59:59", "2015-06-30T23:59:59", "2016-12-31T23:59:59",
], dtype="datetime64[ns]")


def process_time(time: np.ndarray|pd.Series|np.datetime64|pd.Timestamp) -> Tuple[np.ndarray, bool]:
    """
    Convert np.ndarray, pd.Series, np.datetime64, pd.Timestamp to a proper np.ndarray

    :param time:
    :return: converted format, is_scalar set to True if input was a scalar
    """
    is_scalar = isinstance(time, (np.datetime64, pd.Timestamp))

    # Use DatetimeIndex to get tz
    idx = pd.DatetimeIndex([time]) if is_scalar else pd.DatetimeIndex(time)

    # Convert to UTC
    if idx.tz is not None:
        idx = idx.tz_convert("UTC")

    # Convert to ndarray datetime64
    arr = idx.to_numpy(dtype="datetime64[ns]")

    if is_scalar:
        arr = arr[0]

    return arr, is_scalar


def count_leap_seconds(time: np.ndarray, gnss_id: str | np.ndarray) -> np.ndarray:
    """
    Counts the number of leap seconds that occurred between epoch (exclusif) and dt (inclusif).

    :param time: Time of the measurements (UTC, datetime64)
    :param gnss_id: constellation id (str: "gps", "gal", "bei", "glo")
    :return: leap seconds count
    """
    time, _ = process_time(time)
    gnss_id = np.asarray(gnss_id)

    reference_epoch = np.where(gnss_id == "bei", BDT_EPOCH, GPST_EPOCH)

    count_time = np.searchsorted(LEAP_SECONDS, time, side="left")
    count_epoch = np.searchsorted(LEAP_SECONDS, reference_epoch, side="left")

    leap_sec = count_time - count_epoch
    glo_leap_sec = np.zeros_like(leap_sec)

    return np.where(gnss_id == "glo", glo_leap_sec, leap_sec)


# Constellation-specific system time (datetime)
def constellation_time(time: np.ndarray, gnss_id: str | np.ndarray, to_utc: bool) -> np.ndarray:
    """
    Converts a constellation-specific system time (GPST, GST, BDT or GLONASST) from/to UTC.

    Each constellation maintains its own time reference. GPS Time (GPST), Galileo System Time (GST), and Beidou Time
    (BDT) are continuous time scales synchronized with the Coordinated Universal Time (UTC). At the time of writing,
    GPST and GST are offset from UTC by 18 leap seconds, while BDT is offset from UTC by 4 leap seconds. GLONASS Time
    (GLONASST) differs from this approach, as it is a time scale directly steered to UTC and therefore has no offset
    with UTC.

    :param time: Constellation system time (datetime64) or UTC time (UTC, datetime64)
    :param gnss_id: constellation id (str: "gps", "gal", "bei", "glo")
    :param to_utc: set to True to convert constellation-specific system time to UTC, False otherwise
    :return: UTC time (UTC, datetime64) or Constellation system time (datetime64)
    """
    time, _ = process_time(time)
    gnss_id = np.asarray(gnss_id)

    if to_utc:
        sign = -1
    else:
        sign = 1

    # Continuous time scales
    leap_seconds = count_leap_seconds(time, gnss_id)

    return time + sign * leap_seconds.astype("timedelta64[s]")


def constellation_time_to_utc(time: np.ndarray, gnss_id: str | np.ndarray) -> np.ndarray:
    """
    Converts a constellation-specific system time (GPST, GST, BDT or GLONASST) to UTC.

    :param time: Constellation system time (datetime64)
    :param gnss_id: str or np.ndarray of str ("gps", "gal", "bei", "glo")
    :return: UTC time (UTC, datetime64)
    """
    return constellation_time(time, gnss_id, to_utc=True)

def utc_to_constellation_time(time: np.ndarray, gnss_id: str | np.ndarray) -> np.ndarray:
    """
    Converts a UTC time to constellation-specific system time (GPST, GST, BDT or GLONASST).

    :param time: UTC time (UTC, datetime64)
    :param gnss_id: str or np.ndarray of str ("gps", "gal", "bei", "glo")
    :return: Constellation system time (datetime64)
    """
    return constellation_time(time, gnss_id, to_utc=False)


def galileo_time_to_utc(time: np.ndarray) -> np.ndarray:
    """
    Converts GST to UTC.
    :param time: GST time (datetime64)
    :return: UTC time (datetime64)
    """
    return constellation_time(time, gnss_id="gal", to_utc=True)

def utc_to_galileo_time(time: np.ndarray) -> np.ndarray:
    """
    Converts UTC to GST.
    :param time: UTC time (datetime64)
    :return: GST time (datetime64)
    """
    return constellation_time(time, gnss_id="gal", to_utc=False)


def gps_time_to_utc(time: np.ndarray) -> np.ndarray:
    """
    Converts GPST to UTC.
    :param time: GPST time (datetime64)
    :return: UTC time (datetime64)
    """
    return constellation_time(time, gnss_id="gps", to_utc=True)

def utc_to_gps_time(time: np.ndarray) -> np.ndarray:
    """
    Converts UTC to GPST.
    :param time: UTC time (datetime64)
    :return: GPST time (datetime64)
    """
    return constellation_time(time, gnss_id="gps", to_utc=False)


def beidou_time_to_utc(time: np.ndarray) -> np.ndarray:
    """
    Converts BDT to UTC.
    :param time: BDT time (datetime64)
    :return: UTC time (datetime64)
    """
    return constellation_time(time, gnss_id="bei", to_utc=True)

def utc_to_beidou_time(time: np.ndarray) -> np.ndarray:
    """
    Converts UTC to BDT.
    :param time: UTC time (datetime64)
    :return: BDT time (datetime64)
    """
    return constellation_time(time, gnss_id="bei", to_utc=False)


def glonass_time_to_utc(time: np.ndarray) -> np.ndarray:
    """
    Converts GLONASST to UTC. This script won't do anything, GLONASST = UTC.
    :param time: GLONASST time (datetime64)
    :return: UTC time (datetime64)
    """
    return constellation_time(time, gnss_id="glo", to_utc=True)

def utc_to_glonass_time(time: np.ndarray) -> np.ndarray:
    """
    Converts UTC to GLONASST. This script won't do anything, GLONASST = UTC.
    :param time: UTC time (datetime64)
    :return: GLONASST time (datetime64)
    """
    return constellation_time(time, gnss_id="glo", to_utc=False)

# Constellation-specific system time (seconds)

# Time of week
def utc_to_tow(time: np.ndarray, leap_seconds: int | np.ndarray) -> np.ndarray:
    """
    convert UTC to time of week (seconds).

    :param time: UTC time (datetime64)
    :param leap_seconds: number of leap seconds to add (seconds)
    :return: time of week (seconds)
    """
    time, _ = process_time(time)
    leap_seconds = np.asarray(leap_seconds)

    # Midnight (00:00:00) of the current UTC calendar day
    days = time.astype("datetime64[D]")
    midnight = days.astype("datetime64[ns]")

    # Day of week: Unix epoch (1970-01-01) was a Thursday -> index 4 if Sunday=0
    day_of_week = (days.astype(np.int64) + 4) % 7  # 0=Sunday, ..., 6=Saturday

    # Nanoseconds elapsed since midnight (exact int64, no precision loss)
    ns_since_midnight = (time - midnight).astype("timedelta64[ns]").astype(np.int64)
    sec_int = ns_since_midnight // NS_PER_SECOND
    ns_frac = ns_since_midnight % NS_PER_SECOND
    seconds_since_midnight = sec_int.astype(np.float64) + ns_frac.astype(np.float64) / NS_PER_SECOND

    tow_utc = day_of_week * SECONDS_PER_DAY + seconds_since_midnight

    # Add leap seconds, then wrap around the week boundary
    tow = (tow_utc + leap_seconds) % SECONDS_PER_WEEK

    return tow


# Sidereal time
def utc_to_sidereal(time: np.ndarray|pd.Series|np.datetime64|pd.Timestamp) -> np.ndarray|float:
    """
    Converts a UTC timestamp to Greenwich Mean Sidereal Time (GMST) in radians.

    :param time: pd.Timestamp, pd.Series, np.ndarray or np.datetime64 (UTC)
    :return: GMST in radians (0 - 2π)
    """
    unix_epoch_jd = 2440587.5  # Julian Date of the Unix Epoch (1970-01-01 00:00:00 UTC)
    j200_jd = 2451545.0  # Julian Date of epoch J2000.0

    time, is_scalar = process_time(time)

    ns_since_epoch = (time - UNIX_EPOCH).astype("timedelta64[ns]").astype(np.int64)
    seconds_int = ns_since_epoch // NS_PER_SECOND
    ns_frac = ns_since_epoch % NS_PER_SECOND
    seconds_since_epoch = seconds_int.astype(np.float64) + ns_frac.astype(np.float64) / NS_PER_SECOND

    # Julian Date
    jd = unix_epoch_jd + seconds_since_epoch / SECONDS_PER_DAY

    # Julian century since J2000.0
    T = (jd - j200_jd) / 36525

    # GMST seconds (IAU 1982)
    GMST_sec = (
            67310.54841
            + (876600 * 3600 + 8640184.812866) * T
            + 0.093104 * (T ** 2)
            - 6.2e-6 * (T ** 3)
    )
    GMST_sec = GMST_sec % SECONDS_PER_DAY

    # Convert to radians
    GMST_rad = (GMST_sec / SECONDS_PER_DAY) * (2 * np.pi)
    GMST_rad = GMST_rad % (2 * np.pi)

    if is_scalar:
        return float(GMST_rad)
    return GMST_rad
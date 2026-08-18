"""
Time reference system conversions algorithms.

Conversions are based on the Coordinated Universal Time (UTC) reference system and on the numpy datetime64 and
timedelta64 formats.
"""

import pandas as pd
import numpy as np

from bits.src.reference_frame_object import GnssTimestamp

UNIX_EPOCH = np.datetime64("1970-01-01T00:00:00", "ns") # Starting epoch of UNIX time (UTC)
J2000_EPOCH = np.datetime64("2000-01-01T12:00:00", "ns")  # Julian Date 2451545.0 (UTC)
GST_EPOCH = np.datetime64("1999-08-21T23:59:47", "ns") # Starting epoch of Galileo System Time (GST) (UTC)
GPST_EPOCH = np.datetime64("1980-01-06T00:00:00", "ns") # Starting epoch of GPS Time (GPST) (UTC)
BDT_EPOCH = np.datetime64("2006-01-01 00:00:00", "ns") # Starting epoch of Beidou Time (BDT) (UTC)

GST_EPOCH_SEC_TO_MIDNIGHT = np.timedelta64(13, "s")

LEAP_SECONDS = np.array([
    "1981-06-30T23:59:59", "1982-06-30T23:59:59", "1983-06-30T23:59:59",
    "1985-06-30T23:59:59", "1987-12-31T23:59:59", "1989-12-31T23:59:59",
    "1990-12-31T23:59:59", "1992-06-30T23:59:59", "1993-06-30T23:59:59",
    "1994-06-30T23:59:59", "1995-12-31T23:59:59", "1997-06-30T23:59:59",
    "1998-12-31T23:59:59", "2005-12-31T23:59:59", "2008-12-31T23:59:59",
    "2012-06-30T23:59:59", "2015-06-30T23:59:59", "2016-12-31T23:59:59",
], dtype="datetime64[ns]")


transparent = False

def gnss_timestamp_to_datetime(series: pd.Series) -> np.ndarray:
    """
    TODO script provisoire
    """
    if transparent:
        return series

    def safe_pd_timestamp(g):
        if False:
            return g.pd_timestamp()
        try:
            return g.pd_timestamp()
        except Exception:
            return None

    sv_time_series = series.apply(safe_pd_timestamp)
    sv_time = process_time(sv_time_series)

    return sv_time

def datetime_to_gnss_timestamp(array: np.ndarray) -> np.ndarray:
    """
    TODO script provisoire
    """
    if transparent:
        return array

    sv_time_array = np.array([GnssTimestamp.from_pd_timestamp(pd.Timestamp(t)) for t in array], dtype=object)

    return sv_time_array

def process_time(time: np.ndarray|pd.Series|np.datetime64|pd.Timestamp) -> np.ndarray:
    """
    Convert np.ndarray, pd.Series, np.datetime64, pd.Timestamp to a proper np.ndarray

    :param time: time to process (UTC)
    :return: converted format (UTC)
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

    return arr

def process_timedelta(timedelta: np.ndarray|pd.Series|np.timedelta64|pd.Timedelta|float) -> np.ndarray:
    """
    Convert np.ndarray, pd.Series, np.timedelta64, pd.Timedelta or float/int (interpreted as
    seconds) to a proper np.ndarray of timedelta64[ns], preserving nanosecond precision.

    :param timedelta: timedelta to convert (s)
    :return: converted timedelta
    """
    is_scalar = isinstance(timedelta, (np.timedelta64, pd.Timedelta, int, float))

    arr = np.asarray([timedelta]) if is_scalar else np.asarray(timedelta)

    if np.issubdtype(arr.dtype, np.timedelta64):
        arr = arr.astype("timedelta64[ns]")
    else:
        ns = np.round(arr.astype(np.float64) * 1e9).astype(np.int64)
        arr = ns.astype("timedelta64[ns]")

    if is_scalar:
        arr = arr[0]

    return arr


def get_reference_epoch(gnss_id: str | np.ndarray) -> np.ndarray:
    """
    Finds the appropriate reference epoch for each constellation

    :param gnss_id: constellation id (str: "gps", "gal", "bei")
    :return: reference epoch (UTC)
    """
    gnss_id = np.asarray(gnss_id)
    constellation_mapping = [gnss_id == "gal", gnss_id == "gps", gnss_id == "bei"]
    epoch_mapping = [
        np.broadcast_to(np.asarray(GST_EPOCH, dtype="datetime64[ns]"), gnss_id.shape),
        np.broadcast_to(np.asarray(GPST_EPOCH, dtype="datetime64[ns]"), gnss_id.shape),
        np.broadcast_to(np.asarray(BDT_EPOCH, dtype="datetime64[ns]"), gnss_id.shape)
    ]
    return np.select(constellation_mapping, epoch_mapping, default=np.datetime64("NaT", "ns"))


def count_leap_seconds(time: np.ndarray, gnss_id: str | np.ndarray) -> np.ndarray:
    """
    Computes the time difference between UTC and the constellation time at "time".

    :param time: Time of the measurements (UTC, datetime64)
    :param gnss_id: constellation id (str: "gps", "gal", "bei", "glo")
    :return: time difference to UTC (timedelta)
    """
    time = process_time(time)
    gnss_id = np.asarray(gnss_id)

    reference_epoch = get_reference_epoch(gnss_id)

    count_time = np.searchsorted(LEAP_SECONDS, time, side="left")
    count_epoch = np.searchsorted(LEAP_SECONDS, reference_epoch, side="left")

    leap_sec = (count_time - count_epoch).astype("timedelta64[s]")

    # Leap seconds are mostly used as system time difference to UTC, therefore, this script actually return this time
    # difference and not the actual number of leap seconds.
    # GST time difference to UTC = leap seconds + system time difference to UTC at reference epoch (13s)
    leap_sec = np.where(gnss_id == "gal", leap_sec + GST_EPOCH_SEC_TO_MIDNIGHT, leap_sec)

    # GLONASST does not have leap seconds to UTC
    leap_sec = np.where(gnss_id == "glo", np.timedelta64(0, "s"), leap_sec)

    return leap_sec


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
    time = process_time(time)
    gnss_id = np.asarray(gnss_id)

    if to_utc:
        sign = -1
    else:
        sign = 1

    # Continuous time scales
    leap_seconds = count_leap_seconds(time, gnss_id)

    return time + sign * leap_seconds


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


# Constellation-specific system time of week (seconds) and week (week)
# Time of week
def utc_to_tow(time: np.ndarray, leap_seconds: int | np.ndarray | None = None, gnss_id: str | np.ndarray | None = None)\
        -> np.ndarray:
    """
    Convert UTC to time of week (seconds).

    :param time: UTC time (datetime64)
    :param leap_seconds: number of leap seconds to add (seconds)
    :param gnss_id: str or np.ndarray of str ("gps", "gal", "bei", "glo")
    :return: time of week (timedelta)
    """
    time = process_time(time)

    if leap_seconds is None:
        if gnss_id is None:
            raise ValueError("At least one of leap_seconds or gnss_id must be specified")
        leap_seconds = count_leap_seconds(time, gnss_id)
    leap_seconds = process_timedelta(leap_seconds)

    # time - epoch(1970-01-01, Thursday) gives a timedelta64; +4 days realigns it on Sunday.
    elapsed = time - UNIX_EPOCH + np.timedelta64(4, "D")

    # Add leap seconds, then wrap around the week boundary
    tow = (elapsed + leap_seconds) % np.timedelta64(1, 'W')

    return tow

def utc_to_week(time: np.ndarray, gnss_id: str | np.ndarray)\
        -> np.ndarray:
    """
    Convert UTC to week number of the constellation system time.

    :param time: UTC time (datetime64)
    :param gnss_id: str or np.ndarray of str ("gps", "gal", "bei", "glo")
    :return: week (timedelta)
    """
    time = process_time(time)

    time = utc_to_constellation_time(time, gnss_id)
    reference_epoch = get_reference_epoch(gnss_id)

    elapsed_time = time - reference_epoch
    return (elapsed_time // np.timedelta64(1, "W")).astype("timedelta64[W]")


def tow_to_utc(week: np.ndarray, tow: np.ndarray, gnss_id: str | np.ndarray) -> np.ndarray:
    """
    Converts constellation-specific time (week, seconds of week) to UTC

    :param week: constellation-specific week number
    :param tow: seconds elapsed since the beginning of the week
    :param gnss_id: str or np.ndarray of str ("gps", "gal", "bei")
    :return: corresponding UTC time (datetime)
    """
    week = week.astype("timedelta64[W]")
    tow = process_timedelta(tow)

    # Compute constellation-specific time (ignoring leap seconds).
    const_time = get_reference_epoch(gnss_id) + week + tow

    # GST starts 13 secs before midnight, constellation_time_to_utc is leap second based and does not account for that.
    # Correcting leap sec for galileo
    const_time = np.where(gnss_id == "gal", const_time + GST_EPOCH_SEC_TO_MIDNIGHT, const_time)

    return constellation_time_to_utc(const_time, gnss_id)


# Ellapsed time
def utc_to_secondes(time: np.ndarray, gnss_id: str | np.ndarray | None=None) -> np.ndarray:
    """
    Converts a UTC time to ellapsed time from reference epoch in seconds. Default to UNIX time.

    :param time: time to convert (UTC, datetime)
    :param gnss_id: str or np.ndarray of str ("gps", "gal", "bei", "glo")
    :return: ellapsed time since reference epoch (timedelta)
    """
    time = process_time(time)
    gnss_id = np.asarray(gnss_id)

    if gnss_id is None:
        reference_epoch = UNIX_EPOCH
    else:
        reference_epoch = get_reference_epoch(gnss_id)

    reference_epoch = np.where(gnss_id == "glo", UNIX_EPOCH, reference_epoch)

    return time - reference_epoch

def secondes_to_utc(timedelta: np.ndarray, gnss_id: str | np.ndarray | None = None) -> np.ndarray:
    """
    Converts ellapsed time from reference epoch in seconds to UTC time. Default to UNIX time.

    :param timedelta: ellapsed time since reference epoch (timedelta)
    :param gnss_id: str or np.ndarray of str ("gps", "gal", "bei", "glo")
    :return: time in utc (datetime)
    """
    timedelta = process_timedelta(timedelta)
    gnss_id = np.asarray(gnss_id)

    if gnss_id is None:
        reference_epoch = UNIX_EPOCH
    else:
        reference_epoch = get_reference_epoch(gnss_id)

    reference_epoch = np.where(gnss_id == "glo", UNIX_EPOCH, reference_epoch)

    return reference_epoch + timedelta


# Greenwich Mean Sidereal Time
def utc_to_sidereal(time: np.ndarray|pd.Series|np.datetime64|pd.Timestamp) -> np.ndarray|float:
    """
    Converts a UTC timestamp to Greenwich Mean Sidereal Time (GMST) in radians.

    :param time: pd.Timestamp, pd.Series, np.ndarray or np.datetime64 (UTC)
    :return: GMST in radians (0 - 2pi)
    """
    time = process_time(time)

    # Elapsed time since J2000.0, kept as a native timedelta64 as long as possible
    elapsed_since_j2000 = time - J2000_EPOCH

    # Julian centuries since J2000.0: the ONLY point where a plain float is required,
    # since T feeds a nonlinear polynomial (not a duration-preserving operation)
    T = elapsed_since_j2000 / np.timedelta64(36525, "D")

    # GMST seconds (IAU 1982), as a scalar polynomial evaluation
    GMST_sec = (
        67310.54841
        + (876600 * 3600 + 8640184.812866) * T
        + 0.093104 * (T ** 2)
        - 6.2e-6 * (T ** 3)
    )
    GMST_sec = GMST_sec % 86400

    # Convert to radians
    GMST_rad = (GMST_sec / 86400) * (2 * np.pi)

    return GMST_rad % (2 * np.pi)

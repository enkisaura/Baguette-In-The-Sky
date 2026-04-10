"""
Functions for time conversions
"""

__authors__ = ("Enki SAURA")
__contact__ = ("esaura@ikosconsulting.com")
__copyright__ = "IKOS"
__date__ = "12/02/2025"
__version__ = "0.0.1"

from pandas import Timestamp, Timedelta
import math
from decimal import Decimal, ROUND_HALF_EVEN
from typing import Tuple, Union

# Define GPS epoch (January 6, 1980)
GPS_EPOCH = Timestamp('1980-01-06T00:00:00', tz='UTC')

# List of UTC leap seconds (dates when they were introduced)
LEAP_SECONDS = [
    Timestamp("1981-06-30T23:59:59", tz='UTC'),
    Timestamp("1982-06-30T23:59:59", tz='UTC'),
    Timestamp("1983-06-30T23:59:59", tz='UTC'),
    Timestamp("1985-06-30T23:59:59", tz='UTC'),
    Timestamp("1987-12-31T23:59:59", tz='UTC'),
    Timestamp("1989-12-31T23:59:59", tz='UTC'),
    Timestamp("1990-12-31T23:59:59", tz='UTC'),
    Timestamp("1992-06-30T23:59:59", tz='UTC'),
    Timestamp("1993-06-30T23:59:59", tz='UTC'),
    Timestamp("1994-06-30T23:59:59", tz='UTC'),
    Timestamp("1995-12-31T23:59:59", tz='UTC'),
    Timestamp("1997-06-30T23:59:59", tz='UTC'),
    Timestamp("1998-12-31T23:59:59", tz='UTC'),
    Timestamp("2005-12-31T23:59:59", tz='UTC'),
    Timestamp("2008-12-31T23:59:59", tz='UTC'),
    Timestamp("2012-06-30T23:59:59", tz='UTC'),
    Timestamp("2015-06-30T23:59:59", tz='UTC'),
    Timestamp("2016-12-31T23:59:59", tz='UTC')
]


def decimal_seconds_to_sec_ns(value: Decimal) -> Tuple[int, int]:
    """
    Convert Decimal seconds into (whole_seconds, nanoseconds).
    Nanoseconds are rounded to the nearest integer.
    """
    whole_seconds = int(value // Decimal("1"))
    frac = value - Decimal(whole_seconds)
    nanoseconds = int((frac * Decimal("1000000000")).to_integral_value(rounding=ROUND_HALF_EVEN))

    # Handle carry if rounding gives 1_000_000_000 ns
    if nanoseconds >= 1_000_000_000:
        whole_seconds += 1
        nanoseconds -= 1_000_000_000

    return whole_seconds, nanoseconds

def _split_seconds_to_sec_ns(value: Union[int, float , Decimal]) -> Tuple[int, int]:
    """
    Convert a numeric seconds value into (whole_seconds, nanoseconds).
    Supports int, float, and Decimal.

    Float inputs are still limited by float precision.
    Decimal inputs preserve high precision.
    """
    if isinstance(value, Decimal):
        whole_seconds = int(value // Decimal("1"))
        frac = value - Decimal(whole_seconds)
        nanoseconds = int(
            (frac * Decimal("1000000000")).to_integral_value(rounding=ROUND_HALF_EVEN)
        )
    elif isinstance(value, int):
        whole_seconds = value
        nanoseconds = 0
    else:
        # float path: precision already limited by float
        whole_seconds = int(value)
        nanoseconds = int(round((value - whole_seconds) * 1e9))

    if nanoseconds >= 1_000_000_000:
        whole_seconds += 1
        nanoseconds -= 1_000_000_000
    elif nanoseconds < 0:
        whole_seconds -= 1
        nanoseconds += 1_000_000_000

    return whole_seconds, nanoseconds

def count_leap_seconds(dt: Timestamp) -> int:
    """
    Counts the number of leap seconds that have occurred up to a given UTC datetime.
    :param dt: Time of the measurements
    :return: Number of leap seconds
    """
    return sum(1 for leap in LEAP_SECONDS if dt > leap)


# To timestamp
def gps_time_ts_to_utc_ts(gps_time_ts: Timestamp) -> Timestamp:
    """
    Converts GPS time to UTC by adding leap seconds.
    :param gps_time_ts: GPS time
    :return: UTC time
    """
    # Get leap seconds
    leap_seconds = count_leap_seconds(gps_time_ts)

    # Convert to UTC by subtracting leap seconds
    utc_time_ts = gps_time_ts - Timedelta(leap_seconds, 's')

    return utc_time_ts

def gps_time_to_timestamp(gps_time: Union[int , float , Decimal]) -> Timestamp:
    """
    Convert GPS seconds since GPS epoch to UTC pandas Timestamp.
    Supports int, float, and Decimal.
    """
    sec, ns = _split_seconds_to_sec_ns(gps_time)
    gps_time_ts = GPS_EPOCH + Timedelta(sec, "s") + Timedelta(ns, "ns")
    utc_time_ts = gps_time_ts_to_utc_ts(gps_time_ts)
    return utc_time_ts


def gps_week_to_timestamp_v1(gps_week: int, tow: float) -> Timestamp:
    """
    Converts GPS time (week, seconds of week) to pandas.Timestamp.
    Precision to the nanosecond (ns).
    :param gps_week: GPS week number (since January 6, 1980).
    :param tow: Seconds elapsed since the beginning of the week.
    :return: The corresponding UTC timestamp.
    """
    # Compute GPS time (ignoring leap seconds). Time of week is converted to nanoseconds to preserve the best accuracy
    # possible.
    gps_time_ts = GPS_EPOCH + Timedelta(gps_week * 7 * 86400, 's') + Timedelta(int(tow * 1e9), 'ns')

    utc_time_ts = gps_time_ts_to_utc_ts(gps_time_ts)

    return utc_time_ts

def gps_week_to_timestamp(gps_week: int, tow: Union[int , float , Decimal]) -> Timestamp:
    sec, ns = _split_seconds_to_sec_ns(tow)
    gps_time_ts = (
        GPS_EPOCH
        + Timedelta(gps_week * 7 * 86400, "s")
        + Timedelta(sec, "s")
        + Timedelta(ns, "ns")
    )
    utc_time_ts = gps_time_ts_to_utc_ts(gps_time_ts)
    return utc_time_ts


def timestamp_to_gps_time(ts: Timestamp) -> float:
    """
    Convert a UTC timestamp to GPS time (seconds since GPS epoch), accounting for leap seconds.
    :param ts: The UTC timestamp to convert.
    :return: GPS time in seconds.
    """
    # Ensure the timestamp is timezone-aware
    ts = ts.tz_convert('UTC') if ts.tzinfo else ts.tz_localize('UTC')

    # Compute total seconds since GPS epoch
    gps_seconds = (ts - GPS_EPOCH).total_seconds()

    # Add leap seconds to get GPS time
    leap_seconds = count_leap_seconds(ts)
    gps_time = gps_seconds + leap_seconds

    return gps_time

def timestamp_to_gps_time_decimal(ts: Timestamp) -> Decimal:
    ts = ts.tz_convert("UTC") if ts.tzinfo else ts.tz_localize("UTC")

    leap_seconds = count_leap_seconds(ts)
    delta = ts - GPS_EPOCH

    total_ns = (
        delta.days * 86400 * 1_000_000_000
        + delta.seconds * 1_000_000_000
        + delta.microseconds * 1_000
        + delta.nanoseconds
    )

    gps_ns = total_ns + leap_seconds * 1_000_000_000
    return Decimal(gps_ns) / Decimal("1000000000")


def timestamp_to_gps_tow(ts: Timestamp) -> (int, float):
    """
    Converts a UTC datetime to GPS Time of Week (TOW), considering leap seconds.
    Precision to the nanosecond (ns).
    :param ts: UTC datetime
    :return: (gps week, time of week)
    """
    # Ensure the timestamp is timezone-aware
    ts = ts.tz_convert('UTC') if ts.tzinfo else ts.tz_localize('UTC')

    # Subtract leap seconds to convert UTC to GPS time
    leap_seconds = count_leap_seconds(ts)
    gps_time = ts + Timedelta(seconds=leap_seconds)

    # Compute time difference from GPS epoch
    delta = gps_time - GPS_EPOCH

    # Compute GPS Week
    gps_week = delta.days // 7

    # Compute Time of Week (TOW)
    tow = (delta.days % 7) * 86400 + delta.seconds + delta.microseconds / 1e6 + delta.nanoseconds / 1e9

    return gps_week, tow

def timestamp_to_gps_tow_decimal(ts: Timestamp) -> Tuple[int, Decimal]:
    ts = ts.tz_convert("UTC") if ts.tzinfo else ts.tz_localize("UTC")

    leap_seconds = count_leap_seconds(ts)
    gps_time = ts + Timedelta(seconds=leap_seconds)

    delta = gps_time - GPS_EPOCH

    gps_week = delta.days // 7

    tow_ns = (
        (delta.days % 7) * 86400 * 1_000_000_000
        + delta.seconds * 1_000_000_000
        + delta.microseconds * 1_000
        + delta.nanoseconds
    )

    tow = Decimal(tow_ns) / Decimal("1000000000")
    return gps_week, tow

def timestamp_to_unix_decimal(ts: Timestamp) -> Decimal:
    ts = ts.tz_convert("UTC") if ts.tzinfo else ts.tz_localize("UTC")
    unix_epoch = Timestamp("1970-01-01T00:00:00Z")

    delta = ts - unix_epoch

    total_ns = (
        delta.days * 86400 * 1_000_000_000
        + delta.seconds * 1_000_000_000
        + delta.microseconds * 1_000
        + delta.nanoseconds
    )

    return Decimal(total_ns) / Decimal("1000000000")


def utc_to_gmst_radians(timestamp: Timestamp) -> float:
    """
    Converts a UTC timestamp to Greenwich Mean Sidereal Time (GMST) in radians.

    :param timestamp: pd.Timestamp (must be in UTC)
    :return: GMST in radians (0 - 2π)
    """
    # S'assurer que le timestamp est en UTC
    timestamp = timestamp.tz_convert('UTC') if timestamp.tzinfo else timestamp.tz_localize('UTC')

    # Calcul du Julian Date (JD)
    unix_epoch_JD = 2440587.5  # JD de l'époque Unix (1970-01-01 00:00:00 UTC)
    jd = unix_epoch_JD + timestamp.timestamp() / 86400  # 86400 sec/jour

    # Calcul du temps en siècles juliens depuis J2000.0
    T = (jd - 2451545.0) / 36525

    # Formule GMST en secondes
    GMST_sec = 67310.54841 + (876600 * 3600 + 8640184.812866) * T + 0.093104 * (T ** 2) - 6.2e-6 * (T ** 3)
    GMST_sec = GMST_sec % 86400  # Reste du jour sidéral

    # Conversion en radians (86400s = 2π rad)
    GMST_rad = (GMST_sec / 86400) * (2 * math.pi)
    return GMST_rad % (2 * math.pi)  # Retourne une valeur entre 0 et 2π

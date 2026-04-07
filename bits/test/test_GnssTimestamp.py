"""
Tests for the GnssTimestamp constructor

Usage: Used from pytest
======
    python -m pytest -v
"""

__authors__ = ("Enki SAURA")
__contact__ = ("esaura@ikosconsulting.com")
__copyright__ = "IKOS"
__date__ = "17/02/2025"
__version__ = "0.0.1"


import pandas as pd
import math
from ..src.reference_frame_object import GnssTimestamp

required_precision = 1e-9  # Precisions to the nanosecond
required_precision_timedelta = pd.Timedelta(required_precision, unit="seconds")
required_precision_sidereal = 1e-5

year = 2025
month = 2
day = 13
hour = 12
minute = 4
second = 54
microsecond = 123456
nanosecond = 789
leap_seconds = 18
gps_week = 2353
tow = 389112.123456789123456789
gps_time = 1423483512.123456789123456789
GMST_sec = 21 * 60 * 60 + 40 * 60 + 0.86
GMST_rad = (GMST_sec / 86400) * (2 * math.pi)
reference_ts = GnssTimestamp(year=year, month=month, day=day, hour=hour, minute=minute, second=second,
                             microsecond=microsecond, nanosecond=nanosecond, unit="ns", tz="UTC")


def test_from_gps_tow():
    tow_ts = GnssTimestamp.from_gps_tow(gps_week, tow)
    diff = reference_ts.timestamp_pd - tow_ts.timestamp_pd
    assert diff < required_precision_timedelta, f"Precision requirement is not met. Current precision is " \
                                                f"{diff.value}ns, target precision is {required_precision}s"


def test_from_gps_time():
    gps_time_ts = GnssTimestamp.from_gps_time(gps_time)
    diff = reference_ts.timestamp_pd - gps_time_ts.timestamp_pd
    assert diff < required_precision_timedelta, f"Precision requirement is not met. Current precision is " \
                                                f"{diff.value}ns, target precision is {required_precision}s"


def test_to_gps_time():
    gps_time_computed = reference_ts.gps_time()
    diff = gps_time - gps_time_computed
    assert diff < required_precision, f"Precision requirement is not met. Current precision is {diff}s, " \
                                      f"target precision is {required_precision}s"


def test_to_tow():
    timestamp_tow = reference_ts.tow()
    diff = tow - timestamp_tow
    assert diff < required_precision, f"Precision requirement is not met. Current precision is {diff}s, " \
                                      f"target precision is {required_precision}s"


def test_to_gps_week():
    timestamp_gps_week = reference_ts.gps_week()
    diff = gps_week - timestamp_gps_week
    assert diff < required_precision, f"Precision requirement is not met. Current precision is {diff}s, " \
                                      f"target precision is {required_precision}s"

def test_to_sidereal():
    gmst = reference_ts.sidereal()
    diff = GMST_rad - gmst
    assert diff < required_precision, f"Precision requirement is not met. Current precision is {diff}s, " \
                                      f"target precision is {required_precision_sidereal}rad"

if __name__ == "__main__":
    test_from_gps_tow()
    test_from_gps_time()
    test_to_gps_time()
    test_to_tow()
    test_to_gps_week()
    test_to_sidereal()

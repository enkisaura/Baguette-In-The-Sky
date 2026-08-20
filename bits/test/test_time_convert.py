"""
Tests for time conversion functions

Reference time from https://gnsscalc.com/, reference sidereal time from https://calcbe.com/en/calculators/sidereal-time/

Usage: Used from pytest
======
    python -m pytest -v
"""
import numpy as np

from bits.src import convert

required_precision = np.timedelta64("1", "ns")

utc_time = np.datetime64("2027-02-04T08:42:42.123456789")
gps_time = np.datetime64("2027-02-04T08:43:00.123456789")
gal_time = np.datetime64("2027-02-04T08:43:00.123456789")
bei_time = np.datetime64("2027-02-04T08:42:46.123456789")
glo_time = np.datetime64("2027-02-04T08:42:42.123456789")

gps_week = np.timedelta64(2456, "W")
gps_time_of_week = np.timedelta64(376980123456789, "ns")

unix_sec = np.timedelta64(1801730562123456789, "ns")
gps_sec = np.timedelta64(1485765780123456789, "ns")
gal_sec = np.timedelta64(866450580123456789, "ns")
bei_sec = np.timedelta64(665656966123456789, "ns")

day_of_year = np.timedelta64(35, "D")

GMST_rad = 	np.radians(280.737294)

def test_constellation_time():
    from_gps = convert.time.constellation_time_to_utc(gps_time, "gps")
    from_gal = convert.time.constellation_time_to_utc(gal_time, "gal")
    from_bei = convert.time.constellation_time_to_utc(bei_time, "bei")
    from_glo = convert.time.constellation_time_to_utc(glo_time, "glo")

    to_gps = convert.time.utc_to_constellation_time(utc_time, "gps")
    to_gal = convert.time.utc_to_constellation_time(utc_time, "gal")
    to_bei = convert.time.utc_to_constellation_time(utc_time, "bei")
    to_glo = convert.time.utc_to_constellation_time(utc_time, "glo")

    assert np.abs(from_gps - utc_time) < required_precision, f"Precision not met from GPST to UTC ({from_gps - utc_time})"
    assert np.abs(from_gal - utc_time) < required_precision, f"Precision not met from GST to UTC ({from_gal - utc_time})"
    assert np.abs(from_bei - utc_time) < required_precision, f"Precision not met from BDT to UTC ({from_bei - utc_time})"
    assert np.abs(from_glo - utc_time) < required_precision, f"Precision not met from GLONASST to UTC ({from_glo - utc_time})"

    assert np.abs(to_gps - gps_time) < required_precision, f"Precision not met from UTC to GPST ({to_gps - gps_time})"
    assert np.abs(to_gal - gal_time) < required_precision, f"Precision not met from UTC to GST ({to_gal - gal_time})"
    assert np.abs(to_bei - bei_time) < required_precision, f"Precision not met from UTC to BDT ({to_bei - bei_time})"
    assert np.abs(to_glo - utc_time) < required_precision, f"Precision not met from UTC to GLONASST ({to_glo - glo_time})"

def test_time_of_week():
    tow_from_utc_gnss_id = convert.time.utc_to_tow(utc_time, gnss_id="gps")
    tow_from_utc_leapsec = convert.time.utc_to_tow(utc_time, 18)
    week_from_utc = convert.time.utc_to_week(utc_time, "gps")
    utc_from_tow = convert.time.tow_to_utc(gps_week, gps_time_of_week, "gps")

    assert np.abs(tow_from_utc_gnss_id - gps_time_of_week) < required_precision, \
        f"Precision not met from UTC to TOW ({tow_from_utc_gnss_id - gps_time_of_week})"
    assert np.abs(tow_from_utc_leapsec - gps_time_of_week) < required_precision, \
        f"Precision not met from UTC to TOW ({tow_from_utc_leapsec - gps_time_of_week})"
    assert np.abs(week_from_utc - gps_week) < required_precision, \
        f"Precision not met from UTC to week ({week_from_utc - gps_week})"
    assert np.abs(utc_from_tow - utc_time) < required_precision, \
        f"Precision not met from TOW to UTC ({utc_from_tow - utc_time})"

def test_timestamp_seconds():
    to_unix = convert.time.utc_to_seconds(utc_time)
    to_gps = convert.time.utc_to_seconds(utc_time, "gps")
    to_gal = convert.time.utc_to_seconds(utc_time, "gal")
    to_bei = convert.time.utc_to_seconds(utc_time, "bei")

    from_unix = convert.time.seconds_to_utc(unix_sec)
    from_gps = convert.time.seconds_to_utc(gps_sec, "gps")
    from_gal = convert.time.seconds_to_utc(gal_sec, "gal")
    from_bei = convert.time.seconds_to_utc(bei_sec, "bei")

    assert np.abs(to_unix - unix_sec) < required_precision, \
        f"Precision not met from UTC to UNIX seconds ({to_unix - unix_sec})"
    assert np.abs(to_gps - gps_sec) < required_precision, \
        f"Precision not met from UTC to GPST seconds ({to_gps - gps_sec})"
    assert np.abs(to_gal - gal_sec) < required_precision, \
        f"Precision not met from UTC to GST seconds ({to_gal - gal_sec})"
    assert np.abs(to_bei - bei_sec) < required_precision, \
        f"Precision not met from UTC to BDT seconds ({to_bei - bei_sec})"

    assert np.abs(from_unix - utc_time) < required_precision, \
        f"Precision not met from UNIX seconds to UTC ({from_unix - utc_time})"
    assert np.abs(from_gps - utc_time) < required_precision, \
        f"Precision not met from GPST seconds to UTC ({from_gps - utc_time})"
    assert np.abs(from_gal - utc_time) < required_precision, \
        f"Precision not met from GST seconds to UTC ({from_gal - utc_time})"
    assert np.abs(from_bei - utc_time) < required_precision, \
        f"Precision not met from BDT seconds to UTC ({from_bei - utc_time})"

def test_gmst():
    #TODO
    assert True

def test_day_of_year():
    from_utc = convert.time.day_of_year(utc_time)
    assert np.abs(from_utc - day_of_year) < required_precision, \
        f"Precision not met from UTC to day of year ({from_utc - day_of_year})"

if __name__ == "__main__":
    test_constellation_time()
    test_time_of_week()
    test_timestamp_seconds()
    test_gmst()
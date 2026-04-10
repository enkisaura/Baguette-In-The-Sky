"""
Tests for the GnssTimestamp constructor

Usage: Used from pytest
======
    python -m pytest -v
"""

__authors__ = ("Corentin M")
__contact__ = ("esaura@ikosconsulting.com")
__copyright__ = "Snap&Co"
__date__ = ""
__version__ = "0.0.1"

import math
from decimal import Decimal
import pandas as pd
import pytest
from pandas import Timestamp, Timedelta

from  src import time_frame_object as mod
from src.time_frame_object import GnssTime, TimeSystem

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
tow = Decimal("389112.123456789123456789")
gps_time = Decimal("1423483512.123456789123456789")
GMST_sec = 21 * 60 * 60 + 40 * 60 + 0.86
GMST_rad = (GMST_sec / 86400) * (2 * math.pi)
# reference_ts = GnssTimestamp(year=year, month=month, day=day, hour=hour, minute=minute, second=second,
#                              microsecond=microsecond, nanosecond=nanosecond, unit="ns", tz="UTC")



def test_formats():
    json_data = {"gps_week": gps_week, "tow": tow}
    t1 = GnssTime.from_value(json_data)
    t2 = GnssTime.from_value(gps_time, TimeSystem.GPS)
    json_data = {"year":year, 
                 "month":month, 
                 "day":day, 
                 "hour":hour, 
                 "minute":minute, 
                 "second":second,
                 "microsecond":microsecond, 
                 "nanosecond":nanosecond,
                "tz":"UTC",}
    t3 = GnssTime.from_value(json_data, TimeSystem.UTC)

    assert t1==t2
    return




# ----------------------------
# Basic construction / post_init
# ----------------------------

def test_post_init_localizes_naive_timestamp_to_utc():
    t = GnssTime(pd.Timestamp("2024-01-01 12:00:00"))
    assert t.utc == pd.Timestamp("2024-01-01 12:00:00", tz="UTC")


def test_post_init_converts_aware_timestamp_to_utc():
    paris = pd.Timestamp("2024-01-01 13:00:00", tz="Europe/Paris")
    t = GnssTime(paris)
    assert t.utc == pd.Timestamp("2024-01-01 12:00:00", tz="UTC")


def test_post_init_accepts_non_timestamp_input():
    t = GnssTime("2024-01-01T12:00:00Z")
    assert t.utc == pd.Timestamp("2024-01-01T12:00:00Z")


# ----------------------------
# from_utc
# ----------------------------

def test_from_utc_with_string():
    t = GnssTime.from_utc("2024-01-01T12:00:00Z")
    assert t.utc == pd.Timestamp("2024-01-01T12:00:00Z")
    assert t._source_system == TimeSystem.UTC


def test_from_utc_with_dict():
    value = {
        "year": 2024,
        "month": 1,
        "day": 1,
        "hour": 12,
        "minute": 0,
        "second": 0,
        "tz": "UTC",
    }
    t = GnssTime.from_utc(value)
    assert t.utc == pd.Timestamp(**value)
    assert t._source_system == TimeSystem.UTC


# ----------------------------
# from_unix
# ----------------------------

def test_from_unix_uses_split_seconds_helper(monkeypatch):
    calls = {}

    def fake_split(seconds):
        calls["value"] = seconds
        return 10, 123

    monkeypatch.setattr(mod.tc, "_split_seconds_to_sec_ns", fake_split)

    t = GnssTime.from_unix(Decimal("10.000000123"))

    assert calls["value"] == Decimal("10.000000123")
    assert t.utc == pd.Timestamp("1970-01-01T00:00:10Z") + Timedelta(123, "ns")
    assert t._source_system == TimeSystem.UNIX


def test_from_unix_accepts_decimal(monkeypatch):
    def fake_split(seconds):
        assert isinstance(seconds, Decimal)
        return 42, 0

    monkeypatch.setattr(mod.tc, "_split_seconds_to_sec_ns", fake_split)

    t = GnssTime.from_unix(Decimal("42"))
    assert t.utc == pd.Timestamp("1970-01-01T00:00:42Z")


# ----------------------------
# from_gps_seconds / from_gps_week_tow
# ----------------------------

def test_from_gps_seconds(monkeypatch):
    fake_ts = pd.Timestamp("2024-01-01T00:00:00Z")

    def fake_convert(value):
        assert value == Decimal("123.5")
        return fake_ts

    monkeypatch.setattr(mod, "gps_time_to_timestamp", fake_convert)

    t = GnssTime.from_gps_seconds(Decimal("123.5"))
    assert t.utc == fake_ts
    assert t._source_system == TimeSystem.GPS


# def test_from_gps_week_tow(monkeypatch):
#     fake_ts = pd.Timestamp("2024-01-01T00:00:00Z")

#     def fake_convert(week, tow):
#         assert week == 2000
#         assert tow == Decimal("345600.25")
#         return fake_ts

#     monkeypatch.setattr(mod, "gps_week_to_timestamp", fake_convert)

#     t = GnssTime.from_gps_week_tow(2000, Decimal("345600.25"))
#     assert t.utc == fake_ts
#     assert t._source_system == TimeSystem.GPS_WEEK_TOW


# ----------------------------
# _from_value_explicit
# ----------------------------

def test_from_value_explicit_returns_same_instance():
    t = GnssTime.from_utc("2024-01-01T00:00:00Z")
    out = GnssTime._from_value_explicit(t, TimeSystem.UTC)
    assert out is t


def test_from_value_explicit_none_system_returns_none():
    out = GnssTime._from_value_explicit("2024-01-01T00:00:00Z", None)
    assert out is None


def test_from_value_explicit_utc():
    t = GnssTime.from_value("2024-01-01T00:00:00Z", system=TimeSystem.UTC)
    assert t.utc == pd.Timestamp("2024-01-01T00:00:00Z")


def test_from_value_explicit_unix(monkeypatch):
    expected = GnssTime.from_utc("2024-01-01T00:00:00Z")

    def fake_from_unix(value):
        assert value == Decimal("123")
        return expected

    monkeypatch.setattr(GnssTime, "from_unix", staticmethod(fake_from_unix))

    out = GnssTime._from_value_explicit(123, TimeSystem.UNIX)
    assert out is expected


def test_from_value_explicit_gps_week_tow_requires_pair():
    with pytest.raises(ValueError, match="GPS_WEEK_TOW input must be a tuple/list"):
        GnssTime._from_value_explicit(123, TimeSystem.GPS_WEEK_TOW)


def test_from_value_explicit_invalid_system():
    with pytest.raises(ValueError, match="Unsupported system"):
        GnssTime._from_value_explicit("x", "bad-system")  # intentionally invalid


# ----------------------------
# _from_value_autodetect_safe
# ----------------------------

def test_autodetect_safe_returns_same_instance():
    t = GnssTime.from_utc("2024-01-01T00:00:00Z")
    assert GnssTime._from_value_autodetect_safe(t) is t


def test_autodetect_safe_timestamp():
    ts = pd.Timestamp("2024-01-01T00:00:00Z")
    t = GnssTime._from_value_autodetect_safe(ts)
    assert t.utc == ts


def test_autodetect_safe_string():
    t = GnssTime._from_value_autodetect_safe("2024-01-01T00:00:00Z")
    assert t.utc == pd.Timestamp("2024-01-01T00:00:00Z")


def test_autodetect_safe_week_tow_pair(monkeypatch):
    expected = GnssTime.from_utc("2024-01-01T00:00:00Z")

    def fake_from_gps_week_tow(week, tow):
        assert week == 2000
        assert tow == 100.5
        return expected

    monkeypatch.setattr(GnssTime, "from_gps_week_tow", staticmethod(fake_from_gps_week_tow))

    out = GnssTime._from_value_autodetect_safe((2000, 100.5))
    assert out is expected


def test_autodetect_safe_dict_utc():
    t = GnssTime._from_value_autodetect_safe({"utc": "2024-01-01T00:00:00Z"})
    assert t.utc == pd.Timestamp("2024-01-01T00:00:00Z")


def test_autodetect_safe_dict_unix(monkeypatch):
    expected = GnssTime.from_utc("2024-01-01T00:00:00Z")

    def fake_from_unix(value):
        assert value == 123.0
        return expected

    monkeypatch.setattr(GnssTime, "from_unix", staticmethod(fake_from_unix))

    out = GnssTime._from_value_autodetect_safe({"unix": 123})
    assert out is expected


def test_autodetect_safe_dict_gps(monkeypatch):
    expected = GnssTime.from_utc("2024-01-01T00:00:00Z")

    def fake_from_gps_seconds(value):
        assert value == 456.0
        return expected

    monkeypatch.setattr(GnssTime, "from_gps_seconds", staticmethod(fake_from_gps_seconds))

    out = GnssTime._from_value_autodetect_safe({"gps": 456})
    assert out is expected


def test_autodetect_safe_dict_gps_seconds(monkeypatch):
    expected = GnssTime.from_utc("2024-01-01T00:00:00Z")

    def fake_from_gps_seconds(value):
        assert value == 789.0
        return expected

    monkeypatch.setattr(GnssTime, "from_gps_seconds", staticmethod(fake_from_gps_seconds))

    out = GnssTime._from_value_autodetect_safe({"gps_seconds": 789})
    assert out is expected


def test_autodetect_safe_dict_gps_week_tow(monkeypatch):
    expected = GnssTime.from_utc("2024-01-01T00:00:00Z")

    def fake_from_gps_week_tow(week, tow):
        assert week == 2000
        assert tow == Decimal("100.25")
        return expected

    monkeypatch.setattr(GnssTime, "from_gps_week_tow", staticmethod(fake_from_gps_week_tow))

    out = GnssTime._from_value_autodetect_safe({"gps_week": 2000, "tow": "100.25"})
    assert out is expected


def test_autodetect_safe_dict_invalid_format():
    with pytest.raises(ValueError, match="Unsupported dict format"):
        GnssTime._from_value_autodetect_safe({"foo": "bar"})


def test_autodetect_safe_returns_none_for_unknown_input():
    assert GnssTime._from_value_autodetect_safe(object()) is None


# ----------------------------
# _from_value_autodetect_numeric
# ----------------------------

def test_autodetect_numeric_disabled_returns_none():
    assert GnssTime._from_value_autodetect_numeric(123, infer_numeric=False) is None


def test_autodetect_numeric_non_numeric_returns_none():
    assert GnssTime._from_value_autodetect_numeric("123", infer_numeric=True) is None


def test_autodetect_numeric_invalid_nan():
    with pytest.raises(ValueError, match="Invalid numeric time value"):
        GnssTime._from_value_autodetect_numeric(float("nan"), infer_numeric=True)


def test_autodetect_numeric_prefers_unambiguous_unix(monkeypatch):
    unix_candidate = GnssTime.from_utc("2020-01-01T00:00:00Z")
    gps_candidate = GnssTime.from_utc("2205-01-01T00:00:00Z")  # implausible

    monkeypatch.setattr(GnssTime, "from_unix", staticmethod(lambda v: unix_candidate))
    monkeypatch.setattr(GnssTime, "from_gps_seconds", staticmethod(lambda v: gps_candidate))

    out = GnssTime._from_value_autodetect_numeric(123, infer_numeric=True)
    assert out is unix_candidate


def test_autodetect_numeric_prefers_unambiguous_gps(monkeypatch):
    unix_candidate = GnssTime.from_utc("1975-01-01T00:00:00Z")  # implausible
    gps_candidate = GnssTime.from_utc("2020-01-01T00:00:00Z")

    monkeypatch.setattr(GnssTime, "from_unix", staticmethod(lambda v: unix_candidate))
    monkeypatch.setattr(GnssTime, "from_gps_seconds", staticmethod(lambda v: gps_candidate))

    out = GnssTime._from_value_autodetect_numeric(123, infer_numeric=True)
    assert out is gps_candidate


def test_autodetect_numeric_ambiguous():
    unix_candidate = GnssTime.from_utc("2020-01-01T00:00:00Z")
    gps_candidate = GnssTime.from_utc("2021-01-01T00:00:00Z")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(GnssTime, "from_unix", staticmethod(lambda v: unix_candidate))
        mp.setattr(GnssTime, "from_gps_seconds", staticmethod(lambda v: gps_candidate))

        with pytest.raises(ValueError, match="Ambiguous numeric time value"):
            GnssTime._from_value_autodetect_numeric(123, infer_numeric=True)


def test_autodetect_numeric_no_plausible_interpretation():
    unix_candidate = GnssTime.from_utc("1900-01-01T00:00:00Z")
    gps_candidate = GnssTime.from_utc("2200-01-01T00:00:00Z")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(GnssTime, "from_unix", staticmethod(lambda v: unix_candidate))
        mp.setattr(GnssTime, "from_gps_seconds", staticmethod(lambda v: gps_candidate))

        with pytest.raises(ValueError, match="Could not infer time system from numeric value"):
            GnssTime._from_value_autodetect_numeric(123, infer_numeric=True)


# ----------------------------
# from_value
# ----------------------------

def test_from_value_accepts_string_system_case_insensitive(monkeypatch):
    expected = GnssTime.from_utc("2024-01-01T00:00:00Z")

    def fake_explicit(value, system):
        assert system == TimeSystem.UNIX
        return expected

    monkeypatch.setattr(GnssTime, "_from_value_explicit", classmethod(lambda cls, value, system: fake_explicit(value, system)))
    monkeypatch.setattr(GnssTime, "_from_value_autodetect_safe", classmethod(lambda cls, value: None))
    monkeypatch.setattr(GnssTime, "_from_value_autodetect_numeric", classmethod(lambda cls, value, infer_numeric: None))

    out = GnssTime.from_value(123, system="UNIX")
    assert out is expected


def test_from_value_invalid_system_string():
    with pytest.raises(ValueError, match="Unsupported system"):
        GnssTime.from_value(123, system="BAD")


def test_from_value_uses_explicit_first(monkeypatch):
    expected = GnssTime.from_utc("2024-01-01T00:00:00Z")

    monkeypatch.setattr(GnssTime, "_from_value_explicit", classmethod(lambda cls, value, system: expected))
    monkeypatch.setattr(GnssTime, "_from_value_autodetect_safe", classmethod(lambda cls, value: pytest.fail("should not be called")))
    monkeypatch.setattr(GnssTime, "_from_value_autodetect_numeric", classmethod(lambda cls, value, infer_numeric: pytest.fail("should not be called")))

    out = GnssTime.from_value(123, system=TimeSystem.UNIX)
    assert out is expected


def test_from_value_uses_safe_after_explicit(monkeypatch):
    expected = GnssTime.from_utc("2024-01-01T00:00:00Z")

    monkeypatch.setattr(GnssTime, "_from_value_explicit", classmethod(lambda cls, value, system: None))
    monkeypatch.setattr(GnssTime, "_from_value_autodetect_safe", classmethod(lambda cls, value: expected))
    monkeypatch.setattr(GnssTime, "_from_value_autodetect_numeric", classmethod(lambda cls, value, infer_numeric: pytest.fail("should not be called")))

    out = GnssTime.from_value("2024-01-01T00:00:00Z")
    assert out is expected


def test_from_value_uses_numeric_last(monkeypatch):
    expected = GnssTime.from_utc("2024-01-01T00:00:00Z")

    monkeypatch.setattr(GnssTime, "_from_value_explicit", classmethod(lambda cls, value, system: None))
    monkeypatch.setattr(GnssTime, "_from_value_autodetect_safe", classmethod(lambda cls, value: None))
    monkeypatch.setattr(GnssTime, "_from_value_autodetect_numeric", classmethod(lambda cls, value, infer_numeric: expected))

    out = GnssTime.from_value(123, infer_numeric=True)
    assert out is expected


def test_from_value_raises_when_all_strategies_fail(monkeypatch):
    monkeypatch.setattr(GnssTime, "_from_value_explicit", classmethod(lambda cls, value, system: None))
    monkeypatch.setattr(GnssTime, "_from_value_autodetect_safe", classmethod(lambda cls, value: None))
    monkeypatch.setattr(GnssTime, "_from_value_autodetect_numeric", classmethod(lambda cls, value, infer_numeric: None))

    with pytest.raises(ValueError, match="Ambiguous time input"):
        GnssTime.from_value(object())


# ----------------------------
# Properties
# ----------------------------

def test_unix_property():
    t = GnssTime.from_utc("1970-01-01T00:00:01Z")
    assert t.unix == 1.0


# def test_gps_seconds_property(monkeypatch):
#     t = GnssTime.from_utc("2024-01-01T00:00:00Z")
#     monkeypatch.setattr(mod, "timestamp_to_gps_time", lambda ts: 123.5)
#     assert t.gps_seconds == 123.5


# def test_gps_seconds_decimal_property(monkeypatch):
#     t = GnssTime.from_utc("2024-01-01T00:00:00Z")
#     monkeypatch.setattr(mod.tc, "timestamp_to_gps_time_decimal", lambda ts: Decimal("123.456"))
#     assert t.gps_seconds_decimal == Decimal("123.456")


# def test_gps_week_tow_property(monkeypatch):
#     t = GnssTime.from_utc("2024-01-01T00:00:00Z")
#     monkeypatch.setattr(mod, "timestamp_to_gps_tow", lambda ts: (2000, 345600.5))
#     assert t.gps_week_tow == (2000, 345600.5)


# def test_gps_week_and_tow_properties(monkeypatch):
#     t = GnssTime.from_utc("2024-01-01T00:00:00Z")
#     monkeypatch.setattr(mod, "timestamp_to_gps_tow", lambda ts: (2000, 345600.5))
#     assert t.gps_week == 2000
#     assert t.tow == 345600.5


# def test_unix_decimal_property(monkeypatch):
#     t = GnssTime.from_utc("2024-01-01T00:00:00Z")
#     monkeypatch.setattr(mod.tc, "timestamp_to_unix_decimal", lambda ts: Decimal("1704067200.0"))
#     assert t.unix_decimal == Decimal("1704067200.0")


# def test_gps_week_tow_decimal_property(monkeypatch):
#     t = GnssTime.from_utc("2024-01-01T00:00:00Z")
#     monkeypatch.setattr(mod.tc, "timestamp_to_gps_tow_decimal", lambda ts: (2000, Decimal("123.456")))
#     assert t.gps_week_tow_decimal == (2000, Decimal("123.456"))


# def test_tow_decimal_property(monkeypatch):
#     t = GnssTime.from_utc("2024-01-01T00:00:00Z")
#     monkeypatch.setattr(mod.tc, "timestamp_to_gps_tow_decimal", lambda ts: (2000, Decimal("123.456")))
#     assert t.tow_decimal == Decimal("123.456")


# ----------------------------
# Utility methods
# ----------------------------

def test_to_timezone():
    t = GnssTime.from_utc("2024-01-01T12:00:00Z")
    paris = t.to_timezone("Europe/Paris")
    assert str(paris.tz) == "Europe/Paris"
    assert paris.hour == 13


# def test_sidereal(monkeypatch):
#     t = GnssTime.from_utc("2024-01-01T00:00:00Z")
#     monkeypatch.setattr(mod, "utc_to_gmst_radians", lambda ts: 1.234)
#     assert t.sidereal() == 1.234


def test_str():
    t = GnssTime.from_utc("2024-01-01T00:00:00.123456789Z")
    assert str(t) == "2024-01-01T00:00:00.123456789+00:00"


def test_repr():
    t = GnssTime.from_utc("2024-01-01T00:00:00.123456789Z")
    assert repr(t) == "GnssTime(2024-01-01T00:00:00.123456789+00:00)"


# ----------------------------
# Arithmetic
# ----------------------------

def test_add_timedelta():
    t = GnssTime.from_utc("2024-01-01T00:00:00Z")
    out = t + Timedelta(seconds=10)
    assert isinstance(out, GnssTime)
    assert out.utc == pd.Timestamp("2024-01-01T00:00:10Z")
    assert out._source_system == t._source_system


def test_subtract_timedelta():
    t = GnssTime.from_utc("2024-01-01T00:00:10Z")
    out = t - Timedelta(seconds=10)
    assert isinstance(out, GnssTime)
    assert out.utc == pd.Timestamp("2024-01-01T00:00:00Z")
    assert out._source_system == t._source_system


def test_subtract_gnss_time():
    t1 = GnssTime.from_utc("2024-01-01T00:00:10Z")
    t2 = GnssTime.from_utc("2024-01-01T00:00:00Z")
    delta = t1 - t2
    assert delta == Timedelta(seconds=10)


def test_add_invalid_type_returns_notimplemented():
    t = GnssTime.from_utc("2024-01-01T00:00:00Z")
    result = t.__add__(123)
    assert result is NotImplemented


def test_sub_invalid_type_returns_notimplemented():
    t = GnssTime.from_utc("2024-01-01T00:00:00Z")
    result = t.__sub__(123)
    assert result is NotImplemented


# ----------------------------
# Dataclass behavior
# ----------------------------

def test_frozen_dataclass():
    t = GnssTime.from_utc("2024-01-01T00:00:00Z")
    with pytest.raises(Exception):
        t._utc = pd.Timestamp("2025-01-01T00:00:00Z")


def test_ordering_and_equality():
    t1 = GnssTime.from_utc("2024-01-01T00:00:00Z")
    t2 = GnssTime.from_utc("2024-01-02T00:00:00Z")
    t3 = GnssTime.from_utc("2024-01-01T00:00:00Z")

    assert t1 < t2
    assert t1 == t3


def test_source_system_not_used_for_equality():
    t1 = GnssTime(pd.Timestamp("2024-01-01T00:00:00Z"), TimeSystem.UTC)
    t2 = GnssTime(pd.Timestamp("2024-01-01T00:00:00Z"), TimeSystem.GPS)
    assert t1 == t2
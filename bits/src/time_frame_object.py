from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any
import pandas as pd
from pandas import Timestamp, Timedelta
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Union


from .convert.time_conversion import (
    gps_time_to_timestamp,
    gps_week_to_timestamp,
    timestamp_to_gps_time,
    timestamp_to_gps_tow,
    utc_to_gmst_radians,
)
from .convert import time_conversion  as tc

class TimeSystem(str, Enum):
    UTC = "utc"
    UNIX = "unix"
    GPS = "gps"
    GPS_WEEK_TOW = "gps_week_tow"


@dataclass(frozen=True, order=True)
class GnssTime:
    _utc: Timestamp
    _source_system: TimeSystem | None = field(default=None, compare=False)

    def __post_init__(self):
        utc = self._utc
        if not isinstance(utc, pd.Timestamp):
            utc = pd.Timestamp(utc)

        if utc.tz is None:
            utc = utc.tz_localize("UTC")
        else:
            utc = utc.tz_convert("UTC")

        object.__setattr__(self, "_utc", utc)

    @classmethod
    def from_utc(cls, value: Any) -> "GnssTime":
        if isinstance(value, dict):
            return cls(pd.Timestamp(**value), TimeSystem.UTC)
        return cls(pd.Timestamp(value), TimeSystem.UTC)
    
    @classmethod
    def from_unix(cls, seconds: Union[int , float , Decimal]) -> "GnssTime":
        sec, ns = tc._split_seconds_to_sec_ns(seconds)

        ts = (
            pd.Timestamp("1970-01-01T00:00:00Z")
            + Timedelta(sec, "s")
            + Timedelta(ns, "ns")
        )

        return cls(ts, TimeSystem.UNIX)
    
    @classmethod
    def from_gps_seconds(cls, gps_seconds: int | float | Decimal) -> "GnssTime":
        ts = gps_time_to_timestamp(gps_seconds)
        return cls(ts, TimeSystem.GPS)

    @classmethod
    def from_gps_week_tow(cls, week: int, tow: int | float | Decimal) -> "GnssTime":
        ts = gps_week_to_timestamp(week, tow)
        return cls(ts, TimeSystem.GPS_WEEK_TOW)

    @classmethod
    def _from_value_explicit(
        cls,
        value: Any,
        system: TimeSystem | None = None,
    ) -> "GnssTime | None":
        if isinstance(value, cls):
            return value

        if system is None:
            return None

        if system == TimeSystem.UTC:
            return cls.from_utc(value)

        if system == TimeSystem.UNIX:
            return cls.from_unix(Decimal(value))

        if system == TimeSystem.GPS:
            return cls.from_gps_seconds(Decimal(value))

        if system == TimeSystem.GPS_WEEK_TOW:
            if not isinstance(value, (tuple, list)) or len(value) != 2:
                raise ValueError(
                    "GPS_WEEK_TOW input must be a tuple/list like (gps_week, tow)."
                )
            week, tow = value
            return cls.from_gps_week_tow(int(week), float(tow))

        raise ValueError(f"Unsupported system: {system}")

    @classmethod
    def _from_value_autodetect_safe(cls, value: Any) -> "GnssTime | None":
        if isinstance(value, cls):
            return value

        if isinstance(value, pd.Timestamp):
            return cls.from_utc(value)

        if isinstance(value, str):
            return cls.from_utc(value)

        if isinstance(value, (tuple, list)) and len(value) == 2:
            week, tow = value
            if isinstance(week, int) and isinstance(tow, (int, float)):
                return cls.from_gps_week_tow(week, float(tow))

        if isinstance(value, dict):
            if "utc" in value:
                return cls.from_utc(value["utc"])

            if "unix" in value:
                return cls.from_unix(float(value["unix"]))

            if "gps" in value:
                return cls.from_gps_seconds(float(value["gps"]))

            if "gps_seconds" in value:
                return cls.from_gps_seconds(float(value["gps_seconds"]))

            if "gps_week" in value and "tow" in value:
                return cls.from_gps_week_tow(int(value["gps_week"]), Decimal(value["tow"]))

            raise ValueError(
                "Unsupported dict format. Expected one of: "
                "{'utc': ...}, {'unix': ...}, {'gps': ...}, "
                "{'gps_seconds': ...}, {'gps_week': ..., 'tow': ...}"
            )

        return None

    @classmethod
    def _from_value_autodetect_numeric(
        cls,
        value: Any,
        infer_numeric: bool = False,
    ) -> "GnssTime | None":
        if not infer_numeric or not isinstance(value, (int, float)):
            return None

        numeric_value = float(value)

        if not pd.notna(numeric_value):
            raise ValueError(f"Invalid numeric time value: {value}")

        unix_candidate = None
        gps_candidate = None

        try:
            unix_candidate = cls.from_unix(numeric_value)
        except Exception:
            pass

        try:
            gps_candidate = cls.from_gps_seconds(numeric_value)
        except Exception:
            pass

        def plausible(ts_obj: "GnssTime | None") -> bool:
            if ts_obj is None:
                return False
            year = ts_obj.utc.year
            return 1980 <= year <= 2100

        unix_ok = plausible(unix_candidate)
        gps_ok = plausible(gps_candidate)

        if unix_ok and not gps_ok:
            return unix_candidate

        if gps_ok and not unix_ok:
            return gps_candidate

        if unix_ok and gps_ok:
            raise ValueError(
                f"Ambiguous numeric time value: {value}. "
                "It could be interpreted as either UNIX seconds or GPS seconds. "
                "Please specify system=TimeSystem.UNIX or system=TimeSystem.GPS."
            )

        raise ValueError(
            f"Could not infer time system from numeric value: {value}. "
            "Please specify system=TimeSystem.UNIX or system=TimeSystem.GPS."
        )

    @classmethod
    def from_value(
        cls,
        value: Any,
        system: TimeSystem | str | None = None,
        infer_numeric: bool = False,
    ) -> "GnssTime":
        if isinstance(system, str):
            try:
                system = TimeSystem(system.lower())
            except ValueError as exc:
                raise ValueError(f"Unsupported system: {system}") from exc

        instance = cls._from_value_explicit(value, system)
        if instance is not None:
            return instance

        instance = cls._from_value_autodetect_safe(value)
        if instance is not None:
            return instance

        instance = cls._from_value_autodetect_numeric(value, infer_numeric)
        if instance is not None:
            return instance

        raise ValueError(
            "Ambiguous time input. Please specify "
            "system=TimeSystem.UTC/UNIX/GPS/GPS_WEEK_TOW, "
            "or use infer_numeric=True for raw numeric values."
        )

    @property
    def utc(self) -> Timestamp:
        return self._utc

    @property
    def unix(self) -> float:
        return self._utc.timestamp()

    @property
    def gps_seconds(self) -> float:
        return timestamp_to_gps_time(self._utc)
    
    @property
    def gps_seconds_decimal(self) -> Decimal:
        gps_time = tc.timestamp_to_gps_time_decimal(self._utc)
        return gps_time

    @property
    def gps_week_tow(self) -> tuple[int, float]:
        return timestamp_to_gps_tow(self._utc)

    @property
    def gps_week(self) -> int:
        return self.gps_week_tow[0]

    @property
    def tow(self) -> float:
        return self.gps_week_tow[1]
    
    @property
    def unix_decimal(self) -> Decimal:  
        return tc.timestamp_to_unix_decimal(self._utc)

    @property
    def gps_week_tow_decimal(self) -> tuple[int, Decimal]:
        return tc.timestamp_to_gps_tow_decimal(self._utc)

    @property
    def tow_decimal(self) -> Decimal:
        return self.gps_week_tow_decimal[1]

    def to_timezone(self, tz: str) -> Timestamp:
        return self._utc.tz_convert(tz)

    def sidereal(self) -> float:
        return utc_to_gmst_radians(self._utc)

    def __str__(self) -> str:
        return self._utc.isoformat(timespec="nanoseconds")

    def __repr__(self) -> str:
        return f"GnssTime({self._utc.isoformat(timespec='nanoseconds')})"

    def __add__(self, other: Timedelta) -> "GnssTime":
        if not isinstance(other, Timedelta):
            return NotImplemented
        return type(self)(self._utc + other, self._source_system)

    def __sub__(self, other):
        if isinstance(other, Timedelta):
            return type(self)(self._utc - other, self._source_system)
        if isinstance(other, GnssTime):
            return self._utc - other._utc
        return NotImplemented
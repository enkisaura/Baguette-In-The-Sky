"""
Object constructors to be used in BITS to handle metrics in a reference frame. Solves reference frame ambiguity and ease
conversions.
"""

__authors__ = ("Enki SAURA")
__contact__ = ("esaura@ikosconsulting.com")
__copyright__ = "IKOS"
__date__ = "12/02/2025"
__version__ = "0.0.1"

import pandas as pd
from pandas import Timestamp, Timedelta
from tzlocal import get_localzone
from bits.src.convert import time_conversion


class GnssTimestamp:
    """
    BITS' time reference frame object. This constructor must be used to handle time metrics.
    Based on Panda's Timestamp, enables basics operations and GPS specific conversions. Use GnssTimestamp.timestamp_pd
    for panda's Timestamp functions.

    ### Panda's Timestamp ###
        It looks like datetime.datetime, it tastes like datetime.datetime, but it is not datetime.datetime, it's
        pd.Timestamp !
        Accuracy down to the nanoseconds, unlike the microsecond accuracy of datetime. Easier to use than np.datetime64
        and takes timezone into account.
        https://pandas.pydata.org/docs/reference/api/pandas.Timestamp.html
    """

    def __init__(self, *args, **kwargs):
        """Initialize the GnssTimestamp with a pandas Timestamp."""
        self.timestamp_pd = pd.Timestamp(*args, **kwargs)
        self.check_utc()

    def __repr__(self):
        """Return a string representation of the GnssTimestamp."""
        return f"GnssTimestamp({self.timestamp_pd.isoformat(timespec='nanoseconds')})"

    def __str__(self):
        """Return the ISO 8601 string representation of the timestamp."""
        return self.timestamp_pd.isoformat(timespec='nanoseconds')

    def __float__(self):
        return self.timestamp_pd.timestamp()

    def __add__(self, other: Timedelta):
        """
        Allow addition with a Timedelta.
        :param other: Timedelta to add
        :return: GnssTimestamp
        """
        if isinstance(other, Timedelta):
            return GnssTimestamp(self.timestamp_pd + other)
        raise TypeError(f"Unsupported type for addition: {type(other)}")

    def __sub__(self, other):
        """
        Allow subtraction:
        - If subtracting another GnssTimestamp, return a Timedelta.
        - If subtracting a Timedelta, return a new GnssTimestamp.
        :param other: object to substract (GnssTimestamp or Timedelta)
        :return: rest of the substraction (GnssTimestamp or Timedelta)
        """
        if isinstance(other, GnssTimestamp):
            return self.timestamp_pd - other.timestamp_pd  # Returns a Timedelta
        elif isinstance(other, Timedelta):
            return GnssTimestamp(self.timestamp_pd - other)  # Returns a new GnssTimestamp
        raise TypeError(f"Unsupported type for subtraction: {type(other)}")

    def __hash__(self):
        return hash(self.timestamp_pd)

    def __eq__(self, other) -> bool:
        """Check if this timestamp is equal to another GnssTimestamp."""
        if isinstance(other, GnssTimestamp):
            return self.timestamp_pd == other.timestamp_pd
        raise TypeError(f"Unsupported type for comparison: {type(other)}")

    def __gt__(self, other) -> bool:
        """Check if this timestamp is greater than another GnssTimestamp."""
        if isinstance(other, GnssTimestamp):
            return self.timestamp_pd > other.timestamp_pd
        raise TypeError(f"Unsupported type for comparison: {type(other)}")

    def __ge__(self, other) -> bool:
        """Check if this timestamp is greater than or equal to another GnssTimestamp."""
        if isinstance(other, GnssTimestamp):
            return self.timestamp_pd >= other.timestamp_pd
        raise TypeError(f"Unsupported type for comparison: {type(other)}")

    def __lt__(self, other) -> bool:
        """Check if this timestamp is less than another GnssTimestamp."""
        if isinstance(other, GnssTimestamp):
            return self.timestamp_pd < other.timestamp_pd
        raise TypeError(f"Unsupported type for comparison: {type(other)}")

    def __le__(self, other) -> bool:
        """Check if this timestamp is less than or equal to another GnssTimestamp."""
        if isinstance(other, GnssTimestamp):
            return self.timestamp_pd <= other.timestamp_pd
        raise TypeError(f"Unsupported type for comparison: {type(other)}")

    @classmethod
    def from_pd_timestamp(cls, pd_ts: Timestamp):
        timestamp_str = pd_ts.isoformat(timespec="nanoseconds")
        return cls(timestamp_str)

    @classmethod
    def from_gps_time(cls, gps_time: float):
        pd_ts = time_conversion.gps_time_to_timestamp(gps_time)
        timestamp_str = pd_ts.isoformat(timespec="nanoseconds")
        return cls(timestamp_str)

    @classmethod
    def from_gps_tow(cls, gps_week: int, tow: float):
        pd_ts = time_conversion.gps_week_to_timestamp(gps_week, tow)
        timestamp_str = pd_ts.isoformat()
        return cls(timestamp_str)

    def check_utc(self):
        """Ensures that a pandas Timestamp is in UTC."""
        if self.timestamp_pd.tz is None:
            # If no timezone is set, assume it was naive and localize it to UTC
            self.timestamp_pd = self.timestamp_pd.tz_localize("UTC")
        else:
            # Convert any timezone-aware timestamp to UTC
            self.timestamp_pd = self.timestamp_pd.tz_convert("UTC")

    def pd_timestamp(self) -> Timestamp:
        return self.timestamp_pd

    def gps_time(self) -> float:
        gps_time = time_conversion.timestamp_to_gps_time(self.timestamp_pd)
        return gps_time

    def gps_week(self) -> int:
        gps_week, _ = time_conversion.timestamp_to_gps_tow(self.timestamp_pd)
        return gps_week

    def tow(self) -> float:
        _, tow = time_conversion.timestamp_to_gps_tow(self.timestamp_pd)
        return tow

    def local_time(self) -> str:
        # Get the local timezone of the computer
        local_timezone = get_localzone()

        # Convert the Timestamp to the local timezone
        local_ts = self.timestamp_pd.astimezone(local_timezone)

        # Convert the Timestamp to string
        local_ts_str = local_ts.isoformat(timespec="nanoseconds")
        return local_ts_str

    def sidereal(self) -> float:
        gmst = time_conversion.utc_to_gmst_radians(self.timestamp_pd)

        return gmst

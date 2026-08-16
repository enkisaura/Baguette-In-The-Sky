import pandas as pd
import numpy as np
from typing import Tuple

UNIX_EPOCH = np.datetime64("1970-01-01T00:00:00", "ns")

SECONDS_PER_DAY = 86400

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


def utc_to_sidereal(time: np.ndarray|pd.Series|np.datetime64|pd.Timestamp) -> np.ndarray|float:
    """
    Converts a UTC timestamp to Greenwich Mean Sidereal Time (GMST) in radians.

    :param time: pd.Timestamp, pd.Series, np.ndarray or np.datetime64 (UTC)
    :return: GMST in radians (0 - 2π)
    """
    unix_epoch_jd = 2440587.5  # Julian Date de l'epoch Unix (1970-01-01 00:00:00 UTC)
    j200_jd = 2451545.0  # Julian Date de l'epoch J2000.0
    ns_per_sec = 1_000_000_000

    time, is_scalar = process_time(time)

    ns_since_epoch = (time - UNIX_EPOCH).astype("timedelta64[ns]").astype(np.int64)
    seconds_int = ns_since_epoch // ns_per_sec
    ns_frac = ns_since_epoch % ns_per_sec
    seconds_since_epoch = seconds_int.astype(np.float64) + ns_frac.astype(np.float64) / ns_per_sec

    # Julian Date
    jd = unix_epoch_jd + seconds_since_epoch / SECONDS_PER_DAY

    # Siècles juliens depuis J2000.0
    T = (jd - j200_jd) / 36525

    # GMST secondes (IAU 1982)
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
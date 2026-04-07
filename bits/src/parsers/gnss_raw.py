"""
GNSS raw data parser to be used with baguette in the sky
"""

__authors__ = ("Enki SAURA")
__contact__ = ("esaura@ikosconsulting.com")
__copyright__ = "IKOS"
__date__ = "17/02/2025"
__version__ = "0.0.1"

import pandas
import pandas as pd
import numpy as np
import os
from bits.src.reference_frame_object import GnssTimestamp
from bits.src.naming import normalize_gnss_constellation
from bits.src.convert.other import doppler_to_pr_rate


def skydel_raw(filepath: str) -> pandas.DataFrame:
    """
    Parse skydel raw data to pandas Dataframe.
    :param filepath: Path of the file
    :return: BITS raw dataframe
    """
    translation_dict = {
        "ECEF X (m)": 'x_sv_m',
        "ECEF Y (m)": 'y_sv_m',
        "ECEF Z (m)": 'z_sv_m',
        "PSR (m)": 'pr_m',
        "Range (m)": 'corr_pr_m', # Exact range
        "PSR Change Rate (m/s)": 'pr_rate_mps',  # TODO renormer le nom
    }

    pd_data = pd.read_csv(filepath)

    filename = os.path.basename(filepath)
    sv_id = filename.split(" ")[-1].split(".")[0]  # TODO normer nom sv -> sv id ou prn number ??
    gnss_id = filename[0]
    pd_data["sv_id"] = int(sv_id)
    pd_data["gnss_id"] = gnss_id
    pd_data["gnss_id"] = pd_data["gnss_id"].apply(normalize_gnss_constellation)

    pd_data["time"] = \
        pd_data.apply(lambda row: GnssTimestamp.from_gps_tow(row["GPS Week Number"], row["GPS TOW"]), axis=1)

    pd_data.rename(columns=translation_dict, inplace=True)

    return pd_data


def micdrop_raw(filepath: str) -> pandas.DataFrame:
    """
    Parse micdrop raw data to pandas Dataframe.
    :param filepath: Path of the file
    :return: BITS raw dataframe
    """
    translation_dict = {
        "timestamp": "time",
        "pseudorange": 'pr_m', # Exact range
        "doppler": 'doppler_hz',
        "sv_id": "sv_id",
        "sv_const": "gnss_id"
    }

    pd_data = pd.read_csv(filepath)
    pd_data.rename(columns=translation_dict, inplace=True)

    # Convert gps time milliseconds to GnssTimestamp
    pd_data["time"] = \
        pd_data["time"].apply(lambda timestamp: GnssTimestamp.from_gps_time(timestamp/1000))

    # Convert Doppler shift to pr_rate -> Works only with L1 !!!!
    pd_data['pr_rate_mps'] = np.nan
    pd_data["pr_rate_mps"] = \
        pd_data["doppler_hz"].apply(lambda doppler: doppler_to_pr_rate(doppler)) # TODO works only with L1...

    # Convert sv_id to int
    pd_data["sv_id"] = pd_data["sv_id"].apply(int)

    # Normalize GNSS constellation name
    pd_data["gnss_id"] = pd_data["gnss_id"].apply(normalize_gnss_constellation)

    return pd_data
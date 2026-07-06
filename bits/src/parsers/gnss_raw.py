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
import georinex
import os
from bits.src.reference_frame_object import GnssTimestamp
from bits.src.naming import normalize_gnss_constellation
from bits.src.convert.other import doppler_to_pr_rate

# Get read of FutureWarning from georinex
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="georinex")


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

def rinex_obs(filepath: str) -> pandas.DataFrame:
    # Parsing rinex file to dataframe
    obs = georinex.load(filepath, verbose=True)
    obs_df = obs.to_dataframe()
    obs_df = obs_df.reset_index()

    # Getting timestamp and sv IDs
    obs_df = obs_df.rename(columns={
        obs_df.columns[0]: "time",
        obs_df.columns[1]: "sv"
    })

    # Convert Timestamp to GnssTimestamp
    obs_df["time"] = \
        obs_df["time"].apply(lambda timestamp: GnssTimestamp.from_pd_timestamp_gps_time(timestamp))

    # Get constellation id
    obs_df["gnss_id"] = obs_df["sv"].str[0]

    # Normalize GNSS constellation name
    obs_df["gnss_id"] = obs_df["gnss_id"].apply(normalize_gnss_constellation)

    # Get PRN #
    obs_df["sv_id"] = obs_df["sv"].str[1:]
    obs_df["sv_id"] = obs_df["sv_id"].astype(int)

    # Get pseudorange
    obs_df["pr_m"] = obs_df["C1C"].combine_first(obs_df["C2I"])

    # Get doppler
    obs_df["doppler_hz"] = obs_df["D1C"].combine_first(obs_df["D2I"])
    obs_df["pr_rate_mps"] = \
        obs_df["doppler_hz"].apply(lambda doppler: doppler_to_pr_rate(doppler)) # TODO works only with L1...

    # Get CN0
    obs_df["CN0"] = obs_df["S1C"].combine_first(obs_df["S2I"])

    # Clean up
    obs_df = obs_df[["time", "gnss_id", "sv_id", "pr_m", "doppler_hz", "pr_rate_mps", "CN0"]]
    obs_df = obs_df.dropna()
    obs_df = obs_df.reset_index(drop=True)

    return obs_df
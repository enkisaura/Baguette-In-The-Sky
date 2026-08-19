"""
GNSS raw data parser to be used with baguette in the sky
"""

__authors__ = ("Enki SAURA")
__contact__ = ("esaura@ikosconsulting.com")
__copyright__ = "IKOS"
__date__ = "17/02/2025"
__version__ = "0.0.1"

import pandas as pd
import numpy as np
import georinex
import os
from pathlib import Path

from bits.src import convert, const
from bits.src.parse.utils import normalize_gnss_constellation
from bits.src.convert.other import doppler_to_pr_rate

# Get read of FutureWarning from georinex
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="georinex")


def rinex(filepath: str|Path) -> pd.DataFrame:
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
    obs_df["time"] = convert.time.constellation_time_to_utc(obs_df["time"], gnss_id="gps") # TODO issue #15

    # Get constellation id
    obs_df["gnss_id"] = obs_df["sv"].str[0]

    # Normalize GNSS constellation name
    obs_df["gnss_id"] = obs_df["gnss_id"].apply(normalize_gnss_constellation)

    # Get PRN #
    obs_df["prn_id"] = obs_df["sv"].str[1:]
    obs_df["prn_id"] = obs_df["prn_id"].astype(int)

    # Add sv_id
    obs_df["sv_id"] = obs_df["gnss_id"] + obs_df["prn_id"].astype(str)

    # Get pseudorange
    obs_df["pr_m"] = obs_df["C1C"].combine_first(obs_df["C2I"])

    # Get doppler
    obs_df["doppler_hz"] = obs_df["D1C"].combine_first(obs_df["D2I"])
    obs_df["pr_rate_mps"] = \
        obs_df["doppler_hz"].apply(lambda doppler: doppler_to_pr_rate(doppler)) # TODO works only with L1...

    # Get CN0
    obs_df["CN0"] = obs_df["S1C"].combine_first(obs_df["S2I"])

    # Clean up
    obs_df = obs_df[["time", "gnss_id", "sv_id", "prn_id", "pr_m", "doppler_hz", "pr_rate_mps", "CN0"]]
    obs_df = obs_df.dropna()
    obs_df = obs_df.reset_index(drop=True)

    return obs_df


def skydel_folder(folderpath: str|Path) -> pd.DataFrame:
    df_list = []
    for filename in os.listdir(folderpath):
        raw_filepath = os.path.join(folderpath, filename)
        df_list.append(skydel_file(raw_filepath))

    df = pd.concat(df_list)
    df = df.reset_index(drop=True)
    df = df.sort_values("time")

    return df

def skydel_file(filepath: str|Path) -> pd.DataFrame:
    """
    Parse skydel raw data to pandas Dataframe.
    :param filepath: Path of the file
    :return: BITS raw dataframe
    """
    translation_dict = {
        "PSR (m)": 'pr_m',
        "Range (m)": 'corr_pr_m',
        "PSR Change Rate (m/s)": 'pr_rate_mps',
        "Doppler Frequency (Hz)": "doppler_hz",
        "ECEF X (m)": 'x_sv_m',
        "ECEF Y (m)": 'y_sv_m',
        "ECEF Z (m)": 'z_sv_m',
        "Body Elevation(rad)": "elevation_rad",
        "Body Azimuth(rad)": "azimuth_rad",
        "Clock Correction (s)": "poly_clock_corr_m",
        "Iono Correction (m)": "iono_corr_m",
        "Tropo Correction (m)": "tropo_corr_m",
    }

    skydel_const_dict = {
        "B": "bei",
        "E": "gal",
        "G": "glo",
        "L": "gps"
    }

    pd_data = pd.read_csv(filepath)

    filename = os.path.basename(filepath)
    prn_id = filename.split(" ")[-1].split(".")[0]
    pd_data["prn_id"] = int(prn_id)
    gnss_id = filename[0]
    if gnss_id in skydel_const_dict.keys():
        gnss_id = skydel_const_dict[gnss_id]
    else:
        gnss_id = "UKNOWN"
    pd_data["gnss_id"] = gnss_id
    pd_data["sv_id"] = pd_data["gnss_id"] + pd_data["prn_id"].astype(str)

    pd_data["time"] = convert.time.tow_to_utc(pd_data["GPS Week Number"], pd_data["GPS TOW"], gnss_id="gps")

    pd_data["Clock Correction (s)"] = pd_data["Clock Correction (s)"] * const.C

    pd_data.rename(columns=translation_dict, inplace=True)

    return pd_data


def micdrop(filepath: str|Path) -> pd.DataFrame:
    """
    Parse micdrop raw data to pandas Dataframe.
    :param filepath: Path of the file
    :return: BITS raw dataframe
    """
    translation_dict = {
        "timestamp": "time",
        "pseudorange": 'pr_m', # Exact range
        "doppler": 'doppler_hz',
        "sv_id": "prn_id",
        "sv_const": "gnss_id"
    }

    pd_data = pd.read_csv(filepath)
    pd_data.rename(columns=translation_dict, inplace=True)

    # Convert gps time milliseconds to UTC
    pd_data["time"] = convert.time.secondes_to_utc(pd_data["time"]/1000, gnss_id="gps")

    # Convert Doppler shift to pr_rate -> Works only with L1 !!!!
    pd_data['pr_rate_mps'] = np.nan
    pd_data["pr_rate_mps"] = \
        pd_data["doppler_hz"].apply(lambda doppler: doppler_to_pr_rate(doppler)) # TODO works only with L1...

    # Convert sv_id to int
    pd_data["prn_id"] = pd_data["prn_id"].apply(int)

    # Normalize GNSS constellation name
    pd_data["gnss_id"] = pd_data["gnss_id"].apply(normalize_gnss_constellation)

    # Add sv_id
    pd_data["sv_id"] = pd_data["gnss_id"] + pd_data["prn_id"].astype(str)

    return pd_data
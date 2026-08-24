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
import re
import os
from pathlib import Path

from bits.src import convert, const, utils

# Get rid of FutureWarning from georinex
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="georinex")


def rinex(filepath: str|Path) -> pd.DataFrame:
    # Parsing rinex file to dataframe
    obs = georinex.load(filepath, verbose=True)

    if obs.rinextype != "obs":
        txt = f"Rinex {obs.rinextype} cannot be parsed with the rinex observation parser."
        if obs.rinextype == "nav":
            txt += f" Please use bits.parse.ephemeris.rinex('{filepath}') instead."
        raise ValueError(txt)

    header = georinex.rinexheader(filepath)
    obs_df = obs.to_dataframe()
    obs_df = obs_df.reset_index()

    # Getting timestamp and sv IDs
    obs_df = obs_df.rename(columns={
        obs_df.columns[0]: "time",
        obs_df.columns[1]: "sv"
    })

    # Get sv id info
    # Get constellation id
    obs_df["gnss_id"] = obs_df["sv"].str[0]
    obs_df["gnss_id"] = obs_df["gnss_id"].apply(utils.normalize_gnss_constellation)
    # Get PRN #
    obs_df["prn_id"] = obs_df["sv"].str[1:]
    obs_df["prn_id"] = obs_df["prn_id"].astype(int)
    # Add sv_id
    obs_df["sv_id"] = obs_df["gnss_id"] + obs_df["prn_id"].astype(str)

    # Get SV frequencies
    obs_df["frequency_hz"] = np.nan
    # Galileo and GPS
    if "L1C" in obs_df.columns:
        obs_df.loc[((obs_df["L1C"].notna()) & (obs_df["gnss_id"]!="glo")), "frequency_hz"] = const.band_frequency["l1"]
    # Beidou
    if "L2I" in obs_df.columns:
        obs_df.loc[obs_df["L2I"].notna(), "frequency_hz"] = const.band_frequency["b1i"]
    # Glonass
    glo_slot = header.get('GLONASS SLOT / FRQ #')
    if glo_slot is not None:
        # Parse Glonass slots from header
        if isinstance(glo_slot, (list, tuple)):
            glo_slot = ' '.join(glo_slot)
        else:
            glo_slot = str(glo_slot)
        pairs = re.findall(r'R(\d{1,2})\s+(-?\d+)', glo_slot)
        # Get frequency from slot
        for prn, slot in pairs:
            obs_df.loc[((obs_df["prn_id"] == int(prn)) &
                        (obs_df["gnss_id"] == "glo") &
                        (obs_df["L1C"].notna())), "frequency_hz"] = const.get_glo_freq("g1", int(slot))


    # Convert Timestamp to UTC
    time_system = utils.normalize_gnss_constellation(obs.time_system)
    obs_df["time"] = convert.time.constellation_time_to_utc(obs_df["time"], gnss_id=time_system)

    # Get pseudorange
    obs_df["pr_m"] = obs_df["C1C"].combine_first(obs_df["C2I"])

    # Get doppler and pseudorange rate
    obs_df["doppler_hz"] = obs_df["D1C"].combine_first(obs_df["D2I"])
    obs_df["pr_rate_mps"] = convert.other.doppler_to_pr_rate(obs_df["doppler_hz"], obs_df["frequency_hz"])

    # Get CN0
    obs_df["CN0"] = obs_df["S1C"].combine_first(obs_df["S2I"])

    # Clean up
    obs_df = obs_df[["time", "gnss_id", "sv_id", "prn_id", "pr_m", "doppler_hz", "pr_rate_mps", "CN0", "frequency_hz"]]
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
        "Body Elevation (rad)": "elevation_rad",
        "Body Azimuth (rad)": "azimuth_rad",
        "Clock Correction (s)": "clock_corr_m",
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

    pd_data["Clock Correction (s)"] = pd_data["Clock Correction (s)"].astype("float64") * const.C

    pd_data.rename(columns=translation_dict, inplace=True)

    # Ensure types
    pd_data['pr_m'] = pd_data['pr_m'].astype("float64")
    pd_data['corr_pr_m'] = pd_data['corr_pr_m'].astype("float64")
    pd_data['pr_rate_mps'] = pd_data['pr_rate_mps'].astype("float64")
    pd_data['doppler_hz'] = pd_data["doppler_hz"].astype("float64")
    pd_data['x_sv_m'] = pd_data['x_sv_m'].astype("float64")
    pd_data['y_sv_m'] = pd_data['y_sv_m'].astype("float64")
    pd_data['z_sv_m'] = pd_data['z_sv_m'].astype("float64")
    pd_data['elevation_rad'] = pd_data["elevation_rad"].astype("float64")
    pd_data['azimuth_rad'] = pd_data["azimuth_rad"].astype("float64")
    pd_data['clock_corr_m'] = pd_data["clock_corr_m"].astype("float64")
    pd_data['iono_corr_m'] = pd_data["iono_corr_m"].astype("float64")
    pd_data['tropo_corr_m'] = pd_data["tropo_corr_m"].astype("float64")

    # Compute frequency
    wavelength = -pd_data['pr_rate_mps'] / pd_data['doppler_hz']
    pd_data["frequency_hz"] = const.C / wavelength

    return pd_data

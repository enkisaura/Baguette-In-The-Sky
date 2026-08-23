"""
Ephemeris parser to be used with baguette in the sky
"""

__authors__ = ("Enki SAURA")
__contact__ = ("esaura@ikosconsulting.com")
__copyright__ = "IKOS"
__date__ = "12/02/2025"
__version__ = "0.0.1"

import georinex
import warnings
import pandas as pd
from pathlib import Path

from bits.src import convert, utils


def rinex(filepath: str|Path) -> pd.DataFrame:
    """
    Parse rinex nav into pandas dataframe using georinex.
    :param filepath: Path of the rinex nav file
    :return: BITS ephemeris dataframe
    """
    lost_in_translation = {
        "sqrtA": "sqrta",  # Square root of the semi-major axis (sqrt(m))
        "Eccentricity": "e",  # Eccentricity (dimensionless)
        "Io": "i0",  # Inclination angle at reference time (semicircles)
        "IDOT": "idot",  # Rate of change of inclination (semicircles/s)
        "Omega0": "omega0",  # Longitude of ascending node at reference time (semicircles)
        "omega": "omega",  # Argument of perigee (semicircles)
        "M0": "m0",  # Mean anomaly at reference time (semicircles)
        "OmegaDot": "omegadot",  # Rate of change of right ascension (semicircles/s)
        "DeltaN": "deltan",  # Mean motion difference from computed value (semicircles/s)
        "Cuc": "cuc",  # Amplitude of the cosine harmonic correction term to the argument of latitude (rad)
        "Cus": "cus",  # Amplitude of the sine harmonic correction term to the argument of latitude (rad)
        "Crc": "crc",  # Amplitude of the cosine harmonic correction term to the orbit radius (m)
        "Crs": "crs",  # Amplitude of the sine harmonic correction term to the orbit radius (m)
        "Cic": "cic",  # Amplitude of the cosine harmonic correction term to the angle of inclination (rad)
        "Cis": "cis",  # Amplitude of the sine harmonic correction term to the angle of inclination (rad)
        "SVclockBias": "clock_bias",  # Clock bias (s)
        "SVclockDrift": "clock_drift",  # Clock drift (s/s)
        "SVclockDriftRate": "clock_drift_rate",  # Clock drift rate (s/s2)
        "TGD": "tgd", # Time Group Delay (s)
    }

    ephemeris = georinex.load(filepath)

    if ephemeris.rinextype != "nav":
        txt = f"Rinex {ephemeris.rinextype} cannot be parsed with the rinex navigation parser."
        if ephemeris.rinextype == "obs":
            txt += f" Please use bits.parse.raw.rinex('{filepath}') instead."
        raise ValueError(txt)

    pd_ephemeris = ephemeris.to_dataframe().dropna(how='all')

    # Get rid of unhealthy satellites
    if "SatH1" in pd_ephemeris.columns:
        pd_ephemeris = pd_ephemeris[pd_ephemeris["SatH1"]!=1]
    if "health" in pd_ephemeris.columns:
        pd_ephemeris = pd_ephemeris[pd_ephemeris["health"]!=1]

    # Rename and rearrange
    indexes = pd_ephemeris.index
    pd_ephemeris["gnss_id"] = indexes.get_level_values(1)
    pd_ephemeris["prn_id"] = pd_ephemeris["gnss_id"].apply(lambda sv: int(sv[1:]))
    pd_ephemeris["gnss_id"] = pd_ephemeris["gnss_id"].apply(lambda sv: sv[0])
    pd_ephemeris["gnss_id"] = pd_ephemeris["gnss_id"].apply(utils.normalize_gnss_constellation)
    pd_ephemeris["sv_id"] = pd_ephemeris["gnss_id"].astype(str) + pd_ephemeris["prn_id"].astype(str)  # Add sv_id
    pd_ephemeris["time_rinex"] = indexes.get_level_values(0)


    # Convert time
    pd_ephemeris["time"] = convert.time.constellation_time_to_utc(pd_ephemeris["time_rinex"], pd_ephemeris["gnss_id"])
    # Convert time_of_ephemeris
    pd_ephemeris["time_of_ephemeris"] = pd.Series(pd.NaT, index=pd_ephemeris.index, dtype="datetime64[ns]")
    mask_unsteered = pd_ephemeris["gnss_id"] != "glo"
    gnss_id_unsteered = pd_ephemeris.loc[mask_unsteered, "gnss_id"]

    # Unsteered constellations
    if "Toe" in pd_ephemeris.columns:
        pd_ephemeris.loc[mask_unsteered, "time_of_ephemeris"] = (
            convert.time.tow_to_utc(week=convert.time.utc_to_week(pd_ephemeris.loc[mask_unsteered, "time"], gnss_id_unsteered),
                                    tow=pd_ephemeris.loc[mask_unsteered, "Toe"], gnss_id=gnss_id_unsteered))

    # Constellation steered to UTC
    pd_ephemeris.loc[~mask_unsteered, "time_of_ephemeris"] = (
        convert.time.process_time(pd_ephemeris.loc[~mask_unsteered, "time"]))

    # Get gps clock corrections
    if "TGD" not in pd_ephemeris.columns:
        pd_ephemeris["TGD"] = 0.0

    # Get glo clock corrections
    mask = pd_ephemeris["gnss_id"] == "glo"
    if mask.any():
        pd_ephemeris.loc[mask, "SVclockDrift"] = pd_ephemeris.loc[mask, "SVrelFreqBias"]
        if "SVclockDriftRate" in pd_ephemeris.columns:
            pd_ephemeris["SVclockDriftRate"] = pd_ephemeris["SVclockDriftRate"].fillna(0)
        else:
            pd_ephemeris["SVclockDriftRate"] = 0

    # Get Galileo time group delay
    mask = pd_ephemeris["gnss_id"] == "gal"
    if mask.any():
        pd_ephemeris.loc[mask, "TGD"] = pd_ephemeris.loc[mask, "BGDe5a"]

    # Get Beidou time group delay
    mask = pd_ephemeris["gnss_id"] == "bei"
    if mask.any():
        pd_ephemeris.loc[mask, "TGD"] = pd_ephemeris.loc[mask, "TGD1"]

    # Clean up
    pd_ephemeris = pd_ephemeris.rename(columns=lost_in_translation)
    pd_ephemeris = pd_ephemeris.dropna(axis=1, how='all')
    pd_ephemeris = pd_ephemeris.reset_index(drop=True)
    if "tgd" not in pd_ephemeris.columns:
        pd_ephemeris["tgd"] = pd_ephemeris["tgd"].astype("float64")

    try:
        pd_ephemeris["klo_a0"] = [ephemeris.ionospheric_corr_GPS[0]] * len(pd_ephemeris)
        pd_ephemeris["klo_a1"] = [ephemeris.ionospheric_corr_GPS[1]] * len(pd_ephemeris)
        pd_ephemeris["klo_a2"] = [ephemeris.ionospheric_corr_GPS[2]] * len(pd_ephemeris)
        pd_ephemeris["klo_a3"] = [ephemeris.ionospheric_corr_GPS[3]] * len(pd_ephemeris)
        pd_ephemeris["klo_b0"] = [ephemeris.ionospheric_corr_GPS[4]] * len(pd_ephemeris)
        pd_ephemeris["klo_b1"] = [ephemeris.ionospheric_corr_GPS[5]] * len(pd_ephemeris)
        pd_ephemeris["klo_b2"] = [ephemeris.ionospheric_corr_GPS[6]] * len(pd_ephemeris)
        pd_ephemeris["klo_b3"] = [ephemeris.ionospheric_corr_GPS[7]] * len(pd_ephemeris)
    except:
        txt = f"No ionospheric parameters found in rinex file {filepath}"
        warnings.warn(txt)

    return pd_ephemeris


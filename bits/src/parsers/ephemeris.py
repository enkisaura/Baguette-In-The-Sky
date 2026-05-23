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
from pandas import Timedelta
from copy import copy

from bits.src.reference_frame_object import GnssTimestamp
from bits.src.naming import normalize_gnss_constellation


def rinex_nav(filepath):
    """
    Parse rinex nav into pandas dataframe using georinex.
    :param filepath: Path of the rinex nav file
    :return: BITS ephemeris dataframe
    """
    lost_in_translation = {
        #"Toe": "toe",  # Reference time, ephemeris parameters (s)
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

    pd_ephemeris = ephemeris.to_dataframe().dropna(how='all')

    # Rename and rearrange
    indexes = pd_ephemeris.index
    pd_ephemeris["gnss_id"] = indexes.get_level_values(1)
    pd_ephemeris["sv_id"] = pd_ephemeris["gnss_id"].apply(lambda sv: int(sv[1:]))
    pd_ephemeris["gnss_id"] = pd_ephemeris["gnss_id"].apply(lambda sv: sv[0])
    pd_ephemeris["gnss_id"] = pd_ephemeris["gnss_id"].apply(normalize_gnss_constellation)
    pd_ephemeris["time_rinex"] = indexes.get_level_values(0)


    # Convert time
    pd_ephemeris["time"] = None
    pd_ephemeris["time_of_ephemeris"] = None
    # GPS time system has an 18s bias with respect to UTC
    mask = pd_ephemeris["gnss_id"] == "gps"
    pd_ephemeris.loc[mask, "time"] = (
        pd_ephemeris.loc[mask, "time_rinex"].apply(lambda ts: GnssTimestamp.from_pd_timestamp_gps_time(ts)))
    pd_ephemeris.loc[mask, "time_of_ephemeris"] = pd_ephemeris.loc[mask].apply(lambda row: get_gps_toe(row), axis=1)


    # Galileo time system has an 18s bias with respect to UTC
    mask = pd_ephemeris["gnss_id"] == "gal"
    pd_ephemeris.loc[mask, "time"] = (
        pd_ephemeris.loc[mask, "time_rinex"].apply(lambda ts: GnssTimestamp.from_pd_timestamp_gps_time(ts)))
    pd_ephemeris.loc[mask, "time_of_ephemeris"] = pd_ephemeris.loc[mask].apply(lambda row: get_gps_toe(row), axis=1)

    # Glonass is equivalent to UTC (but used to be in UTC+3)
    mask = pd_ephemeris["gnss_id"] == "glo"
    pd_ephemeris.loc[mask, "time"] = (
        pd_ephemeris.loc[mask, "time_rinex"].apply(lambda ts: GnssTimestamp.from_pd_timestamp(ts)))
    pd_ephemeris.loc[mask, "time_of_ephemeris"] = (
        pd_ephemeris.loc[mask, "time_rinex"].apply(lambda ts: GnssTimestamp.from_pd_timestamp(ts)))

    # Beidou time system has a 4s bias with respect to UTC
    mask = pd_ephemeris["gnss_id"] == "bei"
    pd_ephemeris.loc[mask, "time"] = (
        pd_ephemeris.loc[mask, "time_rinex"].apply(lambda ts: GnssTimestamp.from_pd_timestamp_beidou_time(ts)))
    pd_ephemeris.loc[mask, "time_of_ephemeris"] = pd_ephemeris.loc[mask].apply(lambda row: get_bei_toe(row), axis=1)


    # Get glo clock corrections
    mask = pd_ephemeris["gnss_id"] == "glo"
    pd_ephemeris.loc[mask, "SVclockDrift"] = pd_ephemeris.loc[mask, "SVrelFreqBias"]
    pd_ephemeris["SVclockDriftRate"] = pd_ephemeris["SVclockDriftRate"].fillna(0)

    # Get Galileo time group delay
    mask = pd_ephemeris["gnss_id"] == "gal"
    pd_ephemeris.loc[mask, "TGD"] = pd_ephemeris.loc[mask, "BGDe5a"]

    # Get Beidou time group delay
    mask = pd_ephemeris["gnss_id"] == "bei"
    pd_ephemeris.loc[mask, "TGD"] = pd_ephemeris.loc[mask, "TGD1"]

    # Clean up
    pd_ephemeris = pd_ephemeris.rename(columns=lost_in_translation)
    pd_ephemeris = pd_ephemeris.dropna(axis=1, how='all')
    pd_ephemeris = pd_ephemeris.reset_index(drop=True)

    try:
        pd_ephemeris["ionospheric_param"] = [ephemeris.ionospheric_corr_GPS] * len(pd_ephemeris)
    except:
        txt = f"No ionospheric parameters found in rinex file {filepath}"
        warnings.warn(txt)

    return pd_ephemeris


def get_gps_toe(row):
    gps_week = row["time"].gps_week()
    return GnssTimestamp.from_gps_tow(gps_week, row["Toe"])

def get_bei_toe(row):
    bei_week = row["time"].bei_week()
    return GnssTimestamp.from_bei_tow(bei_week, row["Toe"])

def get_toe_beidou(toe_bds: float) -> float:
    """
    Convertit un Toe BeiDou (sec of BDT week) en TOW GPS (sec of GPS week).

    BDT = GPS - 14s  →  toe_gps = toe_bds - 14
    Les deux semaines démarrent le dimanche 00h00 → même rollover.

    Args:
        toe_bds : Toe en secondes dans la semaine BDT [0, 604800]
    Returns:
        toe_gps : Toe en secondes dans la semaine GPS [0, 604800]
    """
    toe_gps = toe_bds + 14

    # Rollover si toe_bds < 14s (rarissime, début de semaine)
    if toe_gps < 0:
        toe_gps += 604_800

    return toe_gps
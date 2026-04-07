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
from bits.src.reference_frame_object import GnssTimestamp
from bits.src.naming import normalize_gnss_constellation


def rinex_nav(filepath):
    """
    Parse rinex nav into pandas dataframe using georinex.
    :param filepath: Path of the rinex nav file
    :return: BITS ephemeris dataframe
    """
    lost_in_translation = {
        "Toe": "toe",  # Reference time, ephemeris parameters (s)
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

    pd_ephemeris = ephemeris.to_dataframe()

    # Rename and rearrange
    indexes = pd_ephemeris.index
    pd_ephemeris["gnss_id"] = indexes.get_level_values(1)
    pd_ephemeris["sv_id"] = pd_ephemeris["gnss_id"].apply(lambda sv: int(sv[1:]))
    pd_ephemeris["gnss_id"] = pd_ephemeris["gnss_id"].apply(lambda sv: sv[0])
    pd_ephemeris["gnss_id"] = pd_ephemeris["gnss_id"].apply(normalize_gnss_constellation)
    pd_ephemeris["time"] = indexes.get_level_values(0)
    pd_ephemeris["time"] = pd_ephemeris["time"].apply(lambda ts: GnssTimestamp(ts))

    # Clean up
    pd_ephemeris = pd_ephemeris.rename(columns=lost_in_translation)
    if 'X' in pd_ephemeris.columns:
        pd_ephemeris = pd_ephemeris.dropna(subset=['X'], how='all')
    if 'crs' in pd_ephemeris.columns:
        pd_ephemeris = pd_ephemeris.dropna(subset=['crs'], how='all')
    pd_ephemeris = pd_ephemeris.dropna(axis=1, how='all')
    pd_ephemeris = pd_ephemeris.reset_index(drop=True)

    try:
        pd_ephemeris["ionospheric_param"] = [ephemeris.ionospheric_corr_GPS] * len(pd_ephemeris)
    except:
        txt = f"No ionospheric parameters found in rinex file {filepath}"
        warnings.warn(txt)

    return pd_ephemeris


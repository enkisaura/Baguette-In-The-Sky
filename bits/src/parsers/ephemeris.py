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
    # GPS
    mask = pd_ephemeris["gnss_id"] == "gps"
    pd_ephemeris.loc[mask, "time"] = (
        pd_ephemeris.loc[mask, "time_rinex"].apply(lambda ts: GnssTimestamp.from_pd_timestamp_gps_time(ts)))
    # Galileo
    mask = pd_ephemeris["gnss_id"] == "gal"
    pd_ephemeris.loc[mask, "time"] = (
        pd_ephemeris.loc[mask, "time_rinex"].apply(lambda ts: GnssTimestamp.from_pd_timestamp_gps_time(ts)))
    # Glonass
    mask = pd_ephemeris["gnss_id"] == "glo"
    pd_ephemeris.loc[mask, "time"] = (
        pd_ephemeris.loc[mask, "time_rinex"].apply(lambda ts: GnssTimestamp.from_pd_timestamp(ts)))
    # Beidou TODO
    mask = pd_ephemeris["gnss_id"] == "bei"
    pd_ephemeris.loc[mask, "time"] = (
        pd_ephemeris.loc[mask, "time_rinex"].apply(lambda ts: GnssTimestamp.from_pd_timestamp(ts)))

    # Convert toe
    mask = pd_ephemeris["gnss_id"] == "glo"
    pd_ephemeris["glo_toe"] = None
    pd_ephemeris.loc[mask, "glo_toe"] = pd_ephemeris.loc[mask].apply(get_toe_glonass, axis=1) #TODO

    # Get glo clock corrections
    pd_ephemeris.loc[mask, "SVclockDrift"] = pd_ephemeris.loc[mask, "SVrelFreqBias"] # TODO
    pd_ephemeris["SVclockDriftRate"] = pd_ephemeris["SVclockDriftRate"].fillna(0)

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


#GLONASS_UTC_OFFSET = 3 * 3600  # GLONASS time = UTC + 3h
GLONASS_UTC_OFFSET = 3 * 3600  # GLONASS time = UTC + 3h


def _old_get_toe_glonass(row) -> float:
    """
    Reconstruit t_e depuis MessageFrameTime (secondes dans la journée GLONASS).

    MessageFrameTime = secondes dans la journée courante en GLONASS time (UTC+3h)
    → convertir en UTC Unix
    """
    # Date du message en UTC (depuis nav.time)
    t_msg_unix = row["time"].pd_timestamp().timestamp()

    # Début de la journée GLONASS (UTC+3h) correspondante
    import datetime
    t_msg_dt = datetime.datetime.utcfromtimestamp(t_msg_unix)

    # Jour courant en GLONASS time = UTC + 3h
    t_glonass = t_msg_unix + GLONASS_UTC_OFFSET
    t_glo_dt = datetime.datetime.utcfromtimestamp(t_glonass)

    # Début du jour GLONASS (minuit Moscou)
    day_start_glo = datetime.datetime(
        t_glo_dt.year, t_glo_dt.month, t_glo_dt.day
    ).timestamp() - GLONASS_UTC_OFFSET  # → en UTC Unix

    # t_e = début du jour GLONASS + MessageFrameTime
    mft = row["MessageFrameTime"]  # secondes dans la journée [s]
    t_e = day_start_glo + mft  # UTC Unix [s]

    # Correction si rollover minuit (MessageFrameTime proche de 86400)
    if t_e - t_msg_unix > 43200: t_e -= 86400
    if t_e - t_msg_unix < -43200: t_e += 86400

    return GnssTimestamp(t_e, unit='s')

def get_toe_glonass(row) -> float:
    """
    Reconstruit t_e depuis MessageFrameTime (secondes dans la journée GLONASS).

    MessageFrameTime = secondes dans la journée courante en GLONASS time (UTC+3h)
    → convertir en UTC Unix
    """
    message_reception_time = copy(row["time"]).pd_timestamp()
    #message_reception_time = message_reception_time.tz_convert("Europe/Moscow") # Convert to Glonass time
    days_since_sunday = (message_reception_time.weekday() + 1) % 7
    week_start = (message_reception_time - Timedelta(days=days_since_sunday)).normalize()

    #message_start_of_day = message_reception_time.normalize() # Get start of day

    toe = week_start + Timedelta(seconds=row["MessageFrameTime"])

    dt = toe - message_reception_time
    if dt.total_seconds() >  3.5 * 86400: toe -= Timedelta(weeks=1)
    if dt.total_seconds() < -3.5 * 86400: toe += Timedelta(weeks=1)

    #toe = toe.tz_convert("UTC")

    # Date du message en UTC (depuis nav.time)
    #t_msg_unix = row["time"].timestamp()

    # Début de la journée GLONASS (UTC+3h) correspondante
    #t_msg_dt = datetime.datetime.utcfromtimestamp(t_msg_unix)

    # Jour courant en GLONASS time = UTC + 3h
    #t_glonass = t_msg_unix + GLONASS_UTC_OFFSET
    #t_glo_dt = datetime.datetime.utcfromtimestamp(t_glonass)

    # Début du jour GLONASS (minuit Moscou)
    #day_start_glo = datetime.datetime(
    #    t_glo_dt.year, t_glo_dt.month, t_glo_dt.day
    #).timestamp() - GLONASS_UTC_OFFSET  # → en UTC Unix

    # t_e = début du jour GLONASS + MessageFrameTime
    #mft = row["MessageFrameTime"]  # secondes dans la journée [s]
    #t_e = day_start_glo + mft  # UTC Unix [s]

    # Correction si rollover minuit (MessageFrameTime proche de 86400)
    #if t_e - t_msg_unix > 43200: t_e -= 86400
    #if t_e - t_msg_unix < -43200: t_e += 86400

    return GnssTimestamp.from_pd_timestamp(toe)
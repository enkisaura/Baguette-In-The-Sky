"""
Used to find sv states
"""

__authors__ = ("Enki SAURA")
__contact__ = ("esaura@ikosconsulting.com")
__copyright__ = "IKOS"
__date__ = "12/02/2025"
__version__ = "0.0.1"

import pandas as pd
import math
import numpy as np
import warnings
from typing import Tuple

from .reference_frame_object import GnssTimestamp
from .convert import space_conversion
from . import const
from .parsers.ephemeris import rinex_nav
from .utils import check_dataframe


def compute_eccentric_anomaly(pd_ephemeris_row: pd.Series, time: GnssTimestamp, ek_iterations=5):
    """
    Compute eccentric anomaly for a specific satellite vehicle at a specific time.
    :param pd_ephemeris_row: Satellite ephemeris. Use a pd.Series parsed with the BITS ephemeris parser.
    :param time: Time at which the satellite's position should be computed
    :param ek_iterations: Number of iterations to compute the eccentric anomaly
    :return: Eccentric anomaly
    """
    a = pd_ephemeris_row["sqrta"] ** 2  # Semi-major axis
    n0 = math.sqrt(const.NU / a ** 3)  # Computed mean motion (rad/sec)w
    # This line is from stanford
    gpsweek_diff = (np.mod(time.gps_week(), 1024)
                    - np.mod(pd_ephemeris_row["time_navdata"].gps_week(), 1024)) * 604800.
    tk = time.tow() - pd_ephemeris_row["toe"] + gpsweek_diff  # Time from ephemeris reference epoch
    n = n0 + pd_ephemeris_row["deltan"]  # Corrected mean motion
    mk = pd_ephemeris_row["m0"] + n * tk  # Mean anomaly

    # Kepler’s equation(𝑀𝑘=𝐸𝑘 − 𝑒 sin 𝐸𝑘 ) may be solved for Eccentric anomaly(𝐸𝑘) by iteration:
    ek = mk  # Initial Value (radians)
    for i in range(ek_iterations):  # Refined Value, minimum of three iterations
        ek = ek + (mk - ek + pd_ephemeris_row["e"] * math.sin(ek)) / (1 - pd_ephemeris_row["e"] * math.cos(ek))

    return ek, n

def _get_sv_state_row(pd_ephemeris_row: pd.Series, time: GnssTimestamp, ek_iterations=5) \
        -> Tuple[float, float, float, float, float, float, float, float, float, float]:
    """
    Compute GPS, Galileo or Beidou SV states.
    Computes one satellite position at a specific time using its ephemeris parameters.
    Based on: https://www.gps.gov/technical/icwg/IS-GPS-200M.pdf (Table 20-IV. Broadcast Navigation User Equations)
    :param pd_ephemeris_row: Satellite ephemeris. Use a pd.Series parsed with the BITS ephemeris parser.
    :param time: Time at which the satellite's position should be computed
    :param ek_iterations: Number of iterations to compute the eccentric anomaly
    :return: (x_ecef, y_ecef, z_ecef, vx_ecef, vy_ecef, vz_ecef, ax_ecef, ay_ecef, az_ecef, ek) -> Satellite position,
    speed and acceleration in ECEF and eccentric anomaly
    """
    a = pd_ephemeris_row["sqrta"] ** 2  # Semi-major axis

    # This line is from stanford
    gpsweek_diff = (np.mod(time.gps_week(), 1024)
                    - np.mod(pd_ephemeris_row["time_navdata"].gps_week(), 1024)) * 604800.
    tk = time.tow() - pd_ephemeris_row["toe"] + gpsweek_diff  # Time from ephemeris reference epoch

    ek, n = compute_eccentric_anomaly(pd_ephemeris_row, time, ek_iterations=ek_iterations)

    # True Anomaly (unambiguous quadrant)
    vk = 2 * math.atan(math.sqrt((1 + pd_ephemeris_row["e"]) / (1 - pd_ephemeris_row["e"])) * math.tan(ek / 2))

    phik = vk + pd_ephemeris_row["omega"]  # Argument of latitude

    # Second harmonic perturbations
    # Argument of latitude correction
    delta_uk = pd_ephemeris_row["cuc"] * math.cos(2 * phik) + pd_ephemeris_row["cus"] * math.sin(2 * phik)
    # Radius correction
    delta_rk = pd_ephemeris_row["crc"] * math.cos(2 * phik) + pd_ephemeris_row["crs"] * math.sin(2 * phik)
    # Inclination correction
    delta_ik = pd_ephemeris_row["cic"] * math.cos(2 * phik) + pd_ephemeris_row["cis"] * math.sin(2 * phik)

    # Corrected argument of latitude
    uk = phik + delta_uk

    # Corrected radius
    rk = a * (1 - pd_ephemeris_row["e"] * math.cos(ek)) + delta_rk

    # Corrected inclination
    ik = pd_ephemeris_row["i0"] + delta_ik + pd_ephemeris_row["idot"] * tk

    # Position in the orbital plane
    xprimek = rk * math.cos(uk)
    yprimek = rk * math.sin(uk)

    # Corrected longitude of ascending node
    omegak = pd_ephemeris_row["omega0"] + (pd_ephemeris_row["omegadot"] - const.OMEGA_E) * tk \
             - const.OMEGA_E * pd_ephemeris_row["toe"]

    # Earth-fixed geocentric satellite coordinate
    xk = xprimek * math.cos(omegak) - yprimek * math.cos(ik) * math.sin(omegak)
    yk = xprimek * math.sin(omegak) + yprimek * math.cos(ik) * math.cos(omegak)
    zk = yprimek * math.sin(ik)


    # SV velocity
    # Eccentric Anomaly Rate
    ek_dot = n/(1 - pd_ephemeris_row["e"] * math.cos(ek))

    # True Anomaly Rate
    vk_dot = ek_dot * math.sqrt(1 - pd_ephemeris_row["e"]**2) / (1 - pd_ephemeris_row["e"] * math.cos(ek))

    # Corrected Inclination Angle Rate
    dik_dt = pd_ephemeris_row["idot"] + 2 * vk_dot * (pd_ephemeris_row["cis"] * math.cos(2 * phik)
                                                      - pd_ephemeris_row["cic"] * math.sin(2 * phik))
    # Corrected Argument of Latitude Rate
    uk_dot = vk_dot + 2 * vk_dot * (pd_ephemeris_row["cus"] * math.cos(2 * phik)
                                    - pd_ephemeris_row["cuc"] * math.sin(2 * phik))
    # Corrected Radius Rate
    rk_dot = (pd_ephemeris_row["e"] * a * ek_dot * math.sin(ek) + 2 * vk_dot *
              (pd_ephemeris_row["crs"] * math.cos(2 * phik) - pd_ephemeris_row["crc"] * math.sin(2 * phik)))

    # Longitude of Ascending Node Rate
    omegak_dot = pd_ephemeris_row["omegadot"] - const.OMEGA_E

    # In-plane velocity
    xprimek_dot = rk_dot * math.cos(uk) - rk * uk_dot * math.sin(uk)
    yprimek_dot = rk_dot * math.sin(uk) + rk * uk_dot * math.cos(uk)

    # Earth_fixed velocity (m/s)
    xk_dot = (-xprimek * omegak_dot * math.sin(omegak) + xprimek_dot * math.cos(omegak)
              - yprimek_dot * math.sin(omegak) * math.cos(ik)
              - yprimek * (omegak_dot * math.cos(omegak) * math.cos(ik) - dik_dt * math.sin(omegak) * math.sin(ik)))
    yk_dot = (xprimek * omegak_dot * math.cos(omegak) + xprimek_dot * math.sin(omegak)
              + yprimek_dot * math.cos(omegak) * math.cos(ik)
              - yprimek * (omegak_dot * math.sin(omegak) * math.cos(ik) + dik_dt * math.cos(omegak) * math.sin(ik)))

    zk_dot = yprimek_dot * math.sin(ik) + yprimek * dik_dt * math.cos(ik)

    # SV acceleration
    # Oblate Earth acceleration Factor
    F = -(3/2) * const.J2 * (const.NU/(rk**2)) * (const.RE/rk)**2

    # Earth-Fixed acceleration (m/s2)
    xk_dotdot = (-const.NU * (xk/(rk**3)) + F * ((1 - 5 * (zk/rk)**2) * (xk/rk)) + 2 * yk_dot * const.OMEGA_E
                 + xk * const.OMEGA_E**2)
    yk_dotdot = (-const.NU * (yk / (rk ** 3)) + F * ((1 - 5 * (zk / rk) ** 2) * (yk / rk)) + 2 * xk_dot * const.OMEGA_E
                 + yk * const.OMEGA_E ** 2)
    zk_dotdot = -const.NU * (zk / (rk ** 3)) + F * ((3 - 5 * (zk / rk) ** 2) * (zk / rk))

    return xk, yk, zk, xk_dot, yk_dot, zk_dot, xk_dotdot, yk_dotdot, zk_dotdot, ek


def _get_glo_sv_state_row(pd_ephemeris_row: pd.Series, time: GnssTimestamp) -> Tuple[float, float, float]:
    """
    Compute Glonass SV states.
    Computes one satellite position at a specific time using its ephemeris parameters.
    Based on https://gssc.esa.int/navipedia/index.php?title=GLONASS_Satellite_Coordinates_Computation
    :param pd_ephemeris_row: Satellite ephemeris. Use a pd.Series parsed with the BITS ephemeris parser.
    :param time: Time at which the satellite's position should be computed
    :return: (x_ecef, y_ecef, z_ecef) -> Satellite position in ECEF
    """
    delta_t = time.tow() - pd_ephemeris_row["time_navdata"].tow()

    # Coordinates transformation to an inertial reference frame
    xa, ya, za = space_conversion.ecef_to_eci_position(
        pd_ephemeris_row["X"], pd_ephemeris_row["Y"], pd_ephemeris_row["Z"], pd_ephemeris_row["time_navdata"])
    dxa, dya, dza = space_conversion.ecef_to_eci_velocity(
        pd_ephemeris_row["X"], pd_ephemeris_row["Y"], pd_ephemeris_row["Z"],
        pd_ephemeris_row["dX"], pd_ephemeris_row["dY"], pd_ephemeris_row["dZ"], pd_ephemeris_row["time_navdata"])
    ddxa, ddya, ddza = space_conversion.ecef_to_eci_position(
        pd_ephemeris_row["dX2"], pd_ephemeris_row["dY2"], pd_ephemeris_row["dZ2"], pd_ephemeris_row["time_navdata"])

    # To be improved (Not so much catholique but that go)
    x_eci = xa + dxa * delta_t + 0.5 * ddxa * delta_t ** 2
    y_eci = ya + dya * delta_t + 0.5 * ddya * delta_t ** 2
    z_eci = za + dza * delta_t + 0.5 * ddza * delta_t ** 2

    # Coordinates transformation back to the PZ-90 reference system
    x_pz90, y_pz90, z_pz90 = space_conversion.eci_to_ecef_position(x_eci, y_eci, z_eci, time)

    x, y, z = space_conversion.pz_90_to_ecef(x_pz90, y_pz90, z_pz90)

    return x, y, z


def get_sv_states(pd_gnss_raw: pd.DataFrame, pd_ephemeris: pd.DataFrame = None, ephem_filepath: str= None) -> pd.DataFrame:
    """
    Compute SV states (positions only) using a pd.Dataframe ephemeris from the BITS ephemeris parser for GPS, Galileo,
    Glonass and Beidou.
    Based on https://gssc.esa.int/navipedia/index.php?title=Satellite_Coordinates_Computation
    :param pd_gnss_raw: BITS raw dataframe
    :param pd_ephemeris: BITS ephemeris dataframe
    :param ephem_filepath: Path of a rinex nav file
    :return: BITS raw dataframe with corresponding sv positions
    """
    raw_required_columns = ["time", "pr_m", "gnss_id", "sv_id"]
    ephemeris_required_columns = ["time", "toe", "sqrta", "e", "i0", "idot", "omega0", "omega", "m0", "omegadot",
                                  "deltan", "cuc", "cus", "crc", "crs", "cic", "cis"]

    if not check_dataframe(pd_gnss_raw, raw_required_columns):
        warnings.warn("Missing columns in pd_gnss_raw, cannot add SV states.")
        return pd_gnss_raw

    if {"corr_time"}.issubset(pd_gnss_raw.columns):
        timestamp_column_name = "corr_time"
    else:
        timestamp_column_name = "time"
    if {"corr_pr_m"}.issubset(pd_gnss_raw.columns):
        pr_column_name = "corr_pr_m"
    else:
        pr_column_name = "pr_m"

    # Get ephemeris
    pd_gnss = retrieve_ephemeris(pd_gnss_raw, pd_ephemeris, ephem_filepath=ephem_filepath)
    if not check_dataframe(pd_gnss, ephemeris_required_columns):
        warnings.warn("Missing ephemeris data, cannot add SV states.")
        return pd_gnss

    # 1. Calculate satellite coordinates at the emission time in the associated ECEF reference frame (i.e., tied to the
    # emission time).
    # Find emission time
    pd_gnss["delta_time"] = \
        pd_gnss.apply(lambda row: pd.Timedelta(row[pr_column_name] / const.C, unit="seconds"), axis=1)
    pd_gnss["emission_time"] = \
        pd_gnss.apply(lambda row: row[timestamp_column_name] - row["delta_time"], axis=1)

    # Compute sv states at emission time
    pd_gnss_glo = pd_gnss[pd_gnss["gnss_id"] == "glo"]
    if not pd_gnss_glo.empty:
        pd_gnss_glo[["x_sv_m", "y_sv_m", "z_sv_m"]] = \
            pd_gnss_glo.apply(lambda row: pd.Series(_get_glo_sv_state_row(row, row["emission_time"])), axis=1)
    else:
        pd_gnss_glo = pd.DataFrame()

    pd_gnss = pd_gnss[pd_gnss["gnss_id"] != "glo"]
    if not pd_gnss.empty:
        pd_gnss[["x_sv_m", "y_sv_m", "z_sv_m", "vx_sv_mps", "vy_sv_mps", "vz_sv_mps", "ax_sv_mpss", "ay_sv_mpss",
                 "az_sv_mpss", "eccentric_anomaly"]] = \
            pd_gnss.apply(lambda row: pd.Series(_get_sv_state_row(row, row["emission_time"])), axis=1)
    else:
        pd_gnss = pd.DataFrame()

    pd_gnss = pd.concat([pd_gnss, pd_gnss_glo], axis=0)

    # 2. Transform satellite coordinates from the system tied to the earth at "emission time" to the system tied to the
    # earth at "reception time" (which is common for all measurements). In order to do so, one must consider the earth
    # rotation during the time interval that the signal takes to propagate from the satellite to the receiver:
    pd_gnss[["x_sv_m", "y_sv_m", "z_sv_m"]] = \
        pd_gnss.apply(
            lambda row: pd.Series(space_conversion.rotate_ecef(row["x_sv_m"], row["y_sv_m"], row["z_sv_m"],
                                                               row["delta_time"])), axis=1)
    return pd_gnss


def retrieve_ephemeris(pd_gnss_raw: pd.DataFrame, pd_ephemeris: pd.DataFrame = None, ephem_filepath: str = None)\
        -> pd.DataFrame:
    """
    Finds the closest ephemeris parameters for each satellite vehicle
    :param pd_gnss_raw: GNSS raw dataframe from BITS parser
    :param pd_ephemeris: ephemeris dataframe from BITS parser
    :param ephem_filepath: Path of a rinex nav file
    :return: GNSS raw dataframe with ephemeris
    """
    ephemeris_required_columns = ["time", "toe", "sqrta", "e", "i0", "idot", "omega0", "omega", "m0", "omegadot",
                                  "deltan", "cuc", "cus", "crc", "crs", "cic", "cis"]

    # Check if ephemeris is already retrieved
    if not check_dataframe(pd_gnss_raw, ephemeris_required_columns, with_warning=False):
        if pd_ephemeris is None:
            if ephem_filepath is None:
                pd_ephemeris = ephemeris_loader(pd_gnss_raw["time"].iloc[0]) # Get ephemeris from the internet
            else:
                pd_ephemeris = rinex_nav(ephem_filepath)
        # Find corresponding ephemeris for each SV from gnss_raw
        merged = pd_gnss_raw.merge(pd_ephemeris, on=['gnss_id', 'sv_id'], suffixes=('', '_navdata'))
        # Find difference between ephemeris and gnss_raw timestamp
        merged['time_diff'] = (
            abs(merged["time"] - merged[f'time_navdata']).astype('timedelta64[ns]'))
        # Find the ephemeris that has the closest timestamp to gnss_raw
        closest_matches = merged.loc[merged.groupby(["time", 'gnss_id', 'sv_id'])['time_diff'].idxmin()]
    else:
        closest_matches = pd_gnss_raw

    return closest_matches


def ephemeris_loader(timestamp: GnssTimestamp):
    """
    Loads ephemeris from https://cddis.nasa.gov/.
    To be implemented.
    :param timestamp: ephemeris time required
    :return: ephemeris dataframe (same format as the ephemeris parser)
    """
    raise NotImplementedError("Getting navdata from the internet is not yet implemented. "
                              "Please use a downloaded rinex nav file.")


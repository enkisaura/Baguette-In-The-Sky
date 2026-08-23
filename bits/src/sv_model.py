"""
Used to find sv states
"""

__authors__ = ("Enki SAURA")
__contact__ = ("esaura@ikosconsulting.com")
__copyright__ = "IKOS"
__date__ = "12/02/2025"
__version__ = "0.0.1"

import pandas as pd
import numpy as np
import warnings
from typing import NamedTuple

from bits.src import const, convert, parse, utils


class SVState(NamedTuple):
    """
    State of a satellite vehicle.
    Every array must be vectors of same size.
    """
    # Position
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray

    # Speed
    vx: np.ndarray
    vy: np.ndarray
    vz: np.ndarray

    # Acceleration
    ax: np.ndarray
    ay: np.ndarray
    az: np.ndarray


class KeplerianParameters(NamedTuple):
    """
    Keplerian parameters of a satellite vehicle with additional orbit and harmonic corrections.
    Every array must be vectors of same size.
    """
    # Elliptical orbit shape
    sqrta: np.ndarray  # Square root of semi-major axis (sqrt(m))
    e: np.ndarray  # Eccentricity ()

    # Orbit orientation
    omega: np.ndarray  # Argument of perigee (semi-circles)
    omega0: np.ndarray  # Longitude of Ascending Node of Orbit Plane at Weekly Epoch (semi-circles)
    i0: np.ndarray  # Inclination angle at reference time (semi-circles)

    # Orbit corrections
    # -Secular corrections
    deltan: np.ndarray  # Mean Motion difference from computed value at reference time (semi-circles/s)
    omegadot: np.ndarray  # Rate of right ascension difference (semi-circles/s)
    idot: np.ndarray  # Rate of inclination angle  (semi-circles/s)

    # Harmonic corrections
    cis: np.ndarray  # Amplitude of the sine harmonic correction term to the angle of inclination (radians)
    cic: np.ndarray  # Amplitude of the cosine harmonic correction term to the angle of inclination (radians)
    crs: np.ndarray  # Amplitude of the sine correction term to the orbit radius (meters)
    crc: np.ndarray  # Amplitude of the cosine correction term to the orbit radius (meters)
    cus: np.ndarray  # Amplitude of the sine harmonic correction term to the argument of latitude (radians)
    cuc: np.ndarray  # Amplitude of the cosine harmonic correction term to the argument of latitude (radians)

    # Satellite position on orbit
    m0: np.ndarray  # Mean anomaly at reference time (semi-circles)


def compute_corrected_mean_motion(a: np.ndarray, deltan: np.ndarray) -> np.ndarray:
    """
    Computes corrected mean motion

    :param a: semi-major axis (m)
    :param deltan: mean Motion difference from computed value at reference time (semi-circles/s)
    :return: corrected mean motion (rad/sec)
    """
    n0 = np.sqrt(const.NU / a ** 3)  # Computed mean motion (rad/sec)
    return n0 + deltan

def compute_eccentric_anomaly(tk: np.ndarray[float], e: np.ndarray, m0: np.ndarray, n: np.ndarray,
                              ek_iterations: int = 5) -> np.ndarray:
    """
    Computes eccentric anomaly at tk

    :param tk: elapsed time since ephemeris (s)
    :param e: eccentricity ()
    :param m0: mean anomaly at reference time (semi-circles)
    :param n: corrected mean motion (rad/sec)
    :param ek_iterations: number of iterations to compute the eccentric anomaly (int)
    :return: eccentric anomaly (rad)
    """
    mk = m0 + n * tk  # Mean anomaly

    # Kepler’s equation(𝑀𝑘=𝐸𝑘 − 𝑒 sin 𝐸𝑘 ) may be solved for Eccentric anomaly(𝐸𝑘) by iteration:
    ek = mk  # Initial Value (radians)
    for i in range(ek_iterations):  # Refined Value, minimum of three iterations
        ek = ek + (mk - ek + e * np.sin(ek)) / (1 - e * np.cos(ek))

    return ek

def kepler_based_sv_model(orbit_param: KeplerianParameters, toe: np.ndarray, time: np.ndarray, leap_sec: np.ndarray,
                          ek_iterations: int = 5, omega_e: float|np.ndarray = const.OMEGA_E) -> SVState:
    """
    Compute GPS, Galileo or Beidou SV states.

    Based on: https://www.navcen.uscg.gov/sites/default/files/pdf/gps/IS_GPS_200M.pdf
    (Table Broadcast Navigation User Equations)

    :param orbit_param: Keplerian set of parameters that describe the orbit
    :param toe: Ephemeris data reference time in UTC (datetime64)
    :param time: Time at which the satellite's position should be computed in UTC (datetime64)
    :param leap_sec: number of leap seconds to add to UTC (seconds)
    :param ek_iterations: Number of iterations to compute the eccentric anomaly
    :return: SVState(x, y, z, vx, vy, vz, ax, ay, az) in ECEF (m)
    """
    # Elapsed time since ephemeris
    tk = (time - toe) / np.timedelta64(1, "s")

    toe_tow = convert.time.utc_to_tow(toe, leap_sec)  / np.timedelta64(1, "s")

    a = orbit_param.sqrta ** 2  # Semi-major axis

    # Corrected mean motion
    n = compute_corrected_mean_motion(a, orbit_param.deltan)

    # Eccentric anomaly
    ek = compute_eccentric_anomaly(tk, orbit_param.e, orbit_param.m0, n, ek_iterations)

    # True Anomaly (unambiguous quadrant)
    vk = 2 * np.arctan(np.sqrt((1 + orbit_param.e) / (1 - orbit_param.e)) * np.tan(ek / 2))

    phik = vk + orbit_param.omega  # Argument of latitude

    # Second harmonic perturbations
    # Argument of latitude correction
    delta_uk = orbit_param.cuc * np.cos(2 * phik) + orbit_param.cus * np.sin(2 * phik)
    # Radius correction
    delta_rk = orbit_param.crc * np.cos(2 * phik) + orbit_param.crs * np.sin(2 * phik)
    # Inclination correction
    delta_ik = orbit_param.cic * np.cos(2 * phik) + orbit_param.cis * np.sin(2 * phik)

    # Corrected argument of latitude
    uk = phik + delta_uk

    # Corrected radius
    rk = a * (1 - orbit_param.e * np.cos(ek)) + delta_rk

    # Corrected inclination
    ik = orbit_param.i0 + delta_ik + orbit_param.idot * tk

    # Position in the orbital plane
    xprimek = rk * np.cos(uk)
    yprimek = rk * np.sin(uk)

    # Corrected longitude of ascending node
    omegak = orbit_param.omega0 + (orbit_param.omegadot - omega_e) * tk - omega_e * toe_tow

    # Earth-fixed geocentric satellite coordinate
    xk = xprimek * np.cos(omegak) - yprimek * np.cos(ik) * np.sin(omegak)
    yk = xprimek * np.sin(omegak) + yprimek * np.cos(ik) * np.cos(omegak)
    zk = yprimek * np.sin(ik)

    # SV velocity
    # Eccentric Anomaly Rate
    ek_dot = n / (1 - orbit_param.e * np.cos(ek))

    # True Anomaly Rate
    vk_dot = ek_dot * np.sqrt(1 - orbit_param.e ** 2) / (1 - orbit_param.e * np.cos(ek))

    # Corrected Inclination Angle Rate
    dik_dt = orbit_param.idot + 2 * vk_dot * (orbit_param.cis * np.cos(2 * phik) - orbit_param.cic * np.sin(2 * phik))
    # Corrected Argument of Latitude Rate
    uk_dot = vk_dot + 2 * vk_dot * (orbit_param.cus * np.cos(2 * phik) - orbit_param.cuc * np.sin(2 * phik))
    # Corrected Radius Rate
    rk_dot = (orbit_param.e * a * ek_dot * np.sin(ek) + 2 * vk_dot * (orbit_param.crs * np.cos(2 * phik)
                                                                      - orbit_param.crc * np.sin(2 * phik)))

    # Longitude of Ascending Node Rate
    omegak_dot = orbit_param.omegadot - omega_e

    # In-plane velocity
    xprimek_dot = rk_dot * np.cos(uk) - rk * uk_dot * np.sin(uk)
    yprimek_dot = rk_dot * np.sin(uk) + rk * uk_dot * np.cos(uk)

    # Earth_fixed velocity (m/s)
    xk_dot = (-xprimek * omegak_dot * np.sin(omegak) + xprimek_dot * np.cos(omegak)
              - yprimek_dot * np.sin(omegak) * np.cos(ik)
              - yprimek * (omegak_dot * np.cos(omegak) * np.cos(ik) - dik_dt * np.sin(omegak) * np.sin(ik)))
    yk_dot = (xprimek * omegak_dot * np.cos(omegak) + xprimek_dot * np.sin(omegak)
              + yprimek_dot * np.cos(omegak) * np.cos(ik)
              - yprimek * (omegak_dot * np.sin(omegak) * np.cos(ik) + dik_dt * np.cos(omegak) * np.sin(ik)))

    zk_dot = yprimek_dot * np.sin(ik) + yprimek * dik_dt * np.cos(ik)

    # SV acceleration
    # Oblate Earth acceleration Factor
    F = -(3 / 2) * const.J2 * (const.NU / (rk ** 2)) * (const.RE / rk) ** 2

    # Earth-Fixed acceleration (m/s2)
    xk_dotdot = (-const.NU * (xk / (rk ** 3)) + F * ((1 - 5 * (zk / rk) ** 2) * (xk / rk)) + 2 * yk_dot * omega_e
                 + xk * omega_e ** 2)
    yk_dotdot = (-const.NU * (yk / (rk ** 3)) + F * ((1 - 5 * (zk / rk) ** 2) * (yk / rk)) - 2 * xk_dot * omega_e
                 + yk * omega_e ** 2)
    zk_dotdot = -const.NU * (zk / (rk ** 3)) + F * ((3 - 5 * (zk / rk) ** 2) * (zk / rk))

    return SVState(xk, yk, zk, xk_dot, yk_dot, zk_dot, xk_dotdot, yk_dotdot, zk_dotdot)


def state_derivatives(State:SVState) -> SVState:
    """
    dx/dt  = vx
    dy/dt  = vy
    dz/dt  = vz
    dvx/dt = - mu_bar x_bar - (3/2) J2 mu_bar x_bar rho² (1-5 z_bar²) + ax
    dvy/dt = - mu_bar y_bar - (3/2) J2 mu_bar y_bar rho² (1-5 z_bar²) + ay
    dvz/dt = - mu_bar z_bar - (3/2) J2 mu_bar z_bar rho² (3-5 z_bar²) + az

    Where:
    mu_bar = NU/r²
    x_bar = x/r, y_bar = y/r, z_bar = z/r
    rho = RE/r
    r = sqrt(x² + y² + z²)

    NU= 398 600.44*10^9 m3/s2 -Gravitational constant
    RE= 6 378 136 m -Equatorial radius of Earth
    J2 = 0.0010826262 -Oblate Earth Gravity Coefficient

    Source: GLONASS Interface Control Document (ICD), Edition 5.1, Appendix 3

    :param State: SVState to be derived in ECI (m)
    :return: SVState(0, 0, 0, vx, vy, vz, ax, ay, az) in ECI (m)
    """
    r = np.sqrt(State.x ** 2 + State.y ** 2 + State.z ** 2)
    mu_bar = const.NU / r ** 2
    x_bar, y_bar, z_bar = State.x / r, State.y / r, State.z / r
    rho = const.RE / r

    ax = - mu_bar * x_bar - (3 / 2) * const.J2 * mu_bar * x_bar * rho ** 2 * (1 - 5 * z_bar ** 2) + State.ax
    ay = - mu_bar * y_bar - (3 / 2) * const.J2 * mu_bar * y_bar * rho ** 2 * (1 - 5 * z_bar ** 2) + State.ay
    az = - mu_bar * z_bar - (3 / 2) * const.J2 * mu_bar * z_bar * rho ** 2 * (3 - 5 * z_bar ** 2) + State.az

    return SVState(np.zeros_like(State.x), np.zeros_like(State.y), np.zeros_like(State.z),
                   State.vx, State.vy, State.vz,
                   ax, ay, az)


def runge_kutta(State: SVState, dt: np.ndarray) -> SVState:
    """
    Propagate the State of a satellite over the integration step dt.

    :param State: SVState to propagate in ECI (m)
    :param dt: Integration step (seconds)
    :return: propagated SVState in ECI (m)
    """
    def propagate_k(state_place:int, K:SVState, factor:float) -> np.ndarray:
        """
        Propagate one K on State.
        :param state_place: 0, 1, 2 -> x, y, z; 3, 4, 5 -> vx, vy, vz; state_place + 3 -> derivative
        :param K: K state vector to propagate
        :param factor: multiplying factor
        :return: propagated state vector
        """
        return State[state_place]  + factor * dt * K[state_place+3]

    K1 = state_derivatives(State)

    State_k2 = SVState(propagate_k(0, K1, 0.5), propagate_k(1, K1, 0.5), propagate_k(2, K1, 0.5),
                       propagate_k(3, K1, 0.5), propagate_k(4, K1, 0.5), propagate_k(5, K1, 0.5),
                       State.ax, State.ay, State.az)
    K2 = state_derivatives(State_k2)

    State_k3 = SVState(propagate_k(0, K2, 0.5), propagate_k(1, K2, 0.5), propagate_k(2, K2, 0.5),
                       propagate_k(3, K2, 0.5), propagate_k(4, K2, 0.5), propagate_k(5, K2, 0.5),
                       State.ax, State.ay, State.az)
    K3 = state_derivatives(State_k3)

    State_k3 = SVState(propagate_k(0, K3, 1), propagate_k(1, K3, 1), propagate_k(2, K3, 1),
                       propagate_k(3, K3, 1), propagate_k(4, K3, 1), propagate_k(5, K3, 1),
                       State.ax, State.ay, State.az)
    K4 = state_derivatives(State_k3)

    def propagate(state_place:int) -> np.ndarray:
        """
        Propagate all K on State.
        :param state_place: 0, 1, 2 -> x, y, z; 3, 4, 5 -> vx, vy, vz; state_place + 3 -> derivative
        :return: propagated state vector
        """
        return (State[state_place]
                + (dt / 6) * (K1[state_place + 3] + 2 * K2[state_place + 3]
                              + 2 * K3[state_place + 3] + K4[state_place + 3]))

    return SVState(propagate(0), propagate(1), propagate(2),
                   propagate(3), propagate(4), propagate(5),
                   State.ax, State.ay, State.az)


def state_propagation_based_sv_model(EphemState: SVState, toe: np.ndarray, time: np.ndarray,
                                     rk_step_s: float = 60) -> SVState:
    """
    Compute Glonass SV states.
    Based on https://gssc.esa.int/navipedia/index.php?title=GLONASS_Satellite_Coordinates_Computation
    :param EphemState: Broadcast SVState in ECEF
    :param toe: Ephemeris data reference time in UTC (datetime64)
    :param time: Time at which the satellite's position should be computed in UTC (datetime64)
    :param rk_step_s: Runge-Kutta step size (seconds)
    :return: SVState(x, y, z, vx, vy, vz, ax, ay, az) in ECEF (m)
    """
    # Elapsed time since ephemeris
    tk = (time - toe) / np.timedelta64(1, "s")

    # 1. Coordinates transformation to an inertial reference frame:
    pv_state = convert.space.ecef_to_eci(toe, *EphemState[:6])
    # Luni-solar acceleration must not be corrected from coriolis/centrifugal force
    luni_solar_acceleration = convert.space.ecef_to_eci(toe, *EphemState[-3:])
    EphemState = SVState(*pv_state, *luni_solar_acceleration)

    # 2. Numerical integration of differential equations that describe the motion of the satellites.
    State = EphemState # Initial state
    remaining = tk
    sign = np.where(tk >= 0, 1.0, -1.0)

    # Runge-Kutta integration algorithm
    while np.abs(remaining).max() > 1e-9:
        dt = sign * np.minimum(rk_step_s, np.abs(remaining))
        dt = np.where(np.abs(remaining) > 1e-9, dt, 0.0)
        State = runge_kutta(State, dt)
        remaining -= dt

    # Compute acceleration
    state_acceleration = state_derivatives(State)[-3:]

    # 3. Coordinates transformation back to ECEF reference system:
    State = SVState(*convert.space.eci_to_ecef(time, *State[:6], *state_acceleration))

    return State


def get_sv_states(pd_gnss_raw: pd.DataFrame, pd_ephemeris: pd.DataFrame = None,
                  ephem_filepath: str = None) -> pd.DataFrame:
    """
    Compute SV states using a pd.Dataframe ephemeris from the BITS ephemeris parser for GPS, Galileo, Glonass and
    Beidou.

    Based on https://gssc.esa.int/navipedia/index.php?title=Satellite_Coordinates_Computation
    :param pd_gnss_raw: BITS raw dataframe
    :param pd_ephemeris: BITS ephemeris dataframe
    :param ephem_filepath: Path of a rinex nav file
    :return: BITS raw dataframe with corresponding sv positions
    """

    pd_gnss_raw = pd_gnss_raw.copy()

    raw_required_columns = ["time", "pr_m", "gnss_id", "sv_id"]
    kepler_ephemeris_required_columns = ["time", "time_of_ephemeris", "sqrta", "deltan", "m0", "e",
                                      "omega", "omega0", "omegadot", "i0", "idot",
                                      "cis", "cic", "crs", "crc", "cus", "cuc"]
    state_ephemeris_required_columns = ["time", "time_of_ephemeris", "X", "Y", "Z", "dX", "dY", "dZ", "dX2", "dY2", "dZ2"]

    if not utils.check_dataframe(pd_gnss_raw, raw_required_columns):
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
    pd_ephemeris = utils.get_ephemeris(pd_gnss_raw, pd_ephemeris, ephem_filepath, ignore_warnings=True)

    kepler_ok = utils.check_dataframe(pd_ephemeris, kepler_ephemeris_required_columns)
    state_ok = utils.check_dataframe(pd_ephemeris, state_ephemeris_required_columns)

    if not kepler_ok or not state_ok:
        warnings.warn("Missing ephemeris data, cannot add SV states.")
        return pd_gnss_raw


    # 1. Compute satellite coordinates at the emission time in the associated ECEF reference frame (i.e., tied to the
    # emission time).
    # Find emission time
    tof = convert.time.process_timedelta(pd_gnss_raw[pr_column_name] / const.C)
    pd_gnss_raw["emission_time"] = pd_gnss_raw[timestamp_column_name] - tof

    # Compute sv states at emission time
    cols = [
        "x_sv_m", "y_sv_m", "z_sv_m",
        "vx_sv_mps", "vy_sv_mps", "vz_sv_mps",
        "ax_sv_mpss", "ay_sv_mpss", "az_sv_mpss",
    ]

    # Compute state propagation-based sv state (Glonass)
    propagation_based_mask = pd_gnss_raw["gnss_id"] == "glo"
    pd_glo = pd_gnss_raw[propagation_based_mask]
    if not pd_glo.empty and state_ok:
        # Get initial SV state from ephemeris
        col_to_field = {
            "X": "x", "Y": "y", "Z": "z",
            "dX": "vx", "dY": "vy", "dZ": "vz",
            "dX2": "ax", "dY2": "ay", "dZ2": "az",
        }

        # Get ephemeris data
        pd_glo = utils.get_data_from_ephemeris(pd_glo, pd_ephemeris, state_ephemeris_required_columns[1:])

        Ephem_state = SVState(**{field: pd_glo[col].to_numpy() for col, field in col_to_field.items()})

        # Compute SV state
        sv_state = state_propagation_based_sv_model(Ephem_state, toe=pd_glo['time_of_ephemeris'],
                                                    time=pd_glo["emission_time"])
        pd_glo.loc[:, cols] = np.column_stack(sv_state)
        pd_glo.drop(kepler_ephemeris_required_columns[1:], inplace=True, axis="columns", errors="ignore")
    else:
        pd_glo = pd.DataFrame()

    # Compute orbit-based sv state (Galileo, GPS, Beidou)
    orbit_based_mask = pd_gnss_raw["gnss_id"].isin(["gal", "gps", "bei"])
    pd_gps = pd_gnss_raw[orbit_based_mask]

    if not pd_gps.empty and kepler_ok:
        # Get ephemeris data
        pd_gps = utils.get_data_from_ephemeris(pd_gps, pd_ephemeris, kepler_ephemeris_required_columns[1:])

        # Get leap seconds
        leap_sec = convert.time.count_leap_seconds(pd_gps["emission_time"], pd_gps["gnss_id"])

        # Get Keplerian set of parameters alongside correction parameters
        orbit_param = KeplerianParameters(**{field: pd_gps[field].to_numpy() for field in KeplerianParameters._fields})

        # Get omega_e (earth rotation rate)
        omega_e = const.get_omega_e(pd_gps["gnss_id"])

        # Compute SV state
        sv_state_np_tuple = kepler_based_sv_model(orbit_param, toe=pd_gps['time_of_ephemeris'],
                                                  time=pd_gps["emission_time"], leap_sec=leap_sec,
                                                  omega_e=omega_e)
        pd_gps.loc[:, cols] = np.column_stack(sv_state_np_tuple)
        pd_gps.drop(kepler_ephemeris_required_columns[1:], axis="columns", inplace=True, errors="ignore")
    else:
        pd_gps = pd.DataFrame()

    pd_gnss = pd.concat([pd_gps, pd_glo], axis=0)
    pd_gnss.drop("emission_time", axis="columns", inplace=True, errors="ignore")

    # 2. Transform satellite coordinates from the system tied to the earth at "emission time" to the system tied to the
    # earth at "reception time" (which is common for all measurements). In order to do so, one must consider the earth
    # rotation during the time interval that the signal takes to propagate from the satellite to the receiver:
    # Find emission time
    tof = convert.time.process_timedelta(pd_gnss[pr_column_name] / const.C)
    pd_gnss[["x_sv_m", "y_sv_m", "z_sv_m"]] = pd.Series(convert.space.rotate_ecef(pd_gnss["x_sv_m"], pd_gnss["y_sv_m"],
                                                                                  pd_gnss["z_sv_m"], tof))
    pd_gnss[["vx_sv_mps", "vy_sv_mps", "vz_sv_mps"]] = pd.Series(convert.space.rotate_ecef(pd_gnss["vx_sv_mps"],
                                                                                           pd_gnss["vy_sv_mps"],
                                                                                           pd_gnss["vz_sv_mps"], tof))
    pd_gnss[["ax_sv_mpss", "ay_sv_mpss", "az_sv_mpss"]] = pd.Series(convert.space.rotate_ecef(pd_gnss["ax_sv_mpss"],
                                                                                              pd_gnss["ay_sv_mpss"],
                                                                                              pd_gnss["az_sv_mpss"],
                                                                                              tof))

    return pd_gnss


def retrieve_ephemeris(pd_gnss_raw: pd.DataFrame, pd_ephemeris: pd.DataFrame = None, ephem_filepath: str = None) \
        -> pd.DataFrame:
    """
    Finds the closest ephemeris parameters for each satellite vehicle
    :param pd_gnss_raw: GNSS raw dataframe from BITS parser
    :param pd_ephemeris: ephemeris dataframe from BITS parser
    :param ephem_filepath: Path of a rinex nav file
    :return: GNSS raw dataframe with ephemeris
    """
    gps_ephemeris_required_columns = ["time", "time_of_ephemeris", "sqrta", "e", "i0", "idot", "omega0", "omega", "m0",
                                      "omegadot", "deltan", "cuc", "cus", "crc", "crs", "cic", "cis"]
    glo_ephemeris_required_columns = ["time", "time_of_ephemeris", "X", "Y", "Z", "dX", "dY", "dZ", "dX2", "dY2", "dZ2"]
    ephemeris_keep_columns = ["clock_bias", "clock_drift", "clock_drift_rate", "tgd"]

    # Check if ephemeris is already retrieved
    gps_present = utils.check_dataframe(pd_gnss_raw, gps_ephemeris_required_columns, with_warning=False)
    glo_present = utils.check_dataframe(pd_gnss_raw, glo_ephemeris_required_columns, with_warning=False)
    if gps_present or glo_present: # Ephemeris are already there
        return pd_gnss_raw

    if pd_ephemeris is None:
        if ephem_filepath is None:
            pd_ephemeris = ephemeris_loader(pd_gnss_raw["time"].iloc[0])  # Get ephemeris from the internet
        else:
            pd_ephemeris = parse.ephemeris.rinex(ephem_filepath)

    cols_to_get = gps_ephemeris_required_columns[1:] + glo_ephemeris_required_columns[2:] + ephemeris_keep_columns
    pd_gnss_raw = utils.get_data_from_ephemeris(pd_gnss_raw, pd_ephemeris, cols_to_get)
    return pd_gnss_raw


def ephemeris_loader(time: np.ndarray):
    """
    Loads ephemeris from https://cddis.nasa.gov/.
    To be implemented.
    :param time: ephemeris time required (UTC)
    :return: ephemeris dataframe (same format as the ephemeris parser)
    """
    raise NotImplementedError("Getting navdata from the internet is not yet implemented. "
                              "Please use a downloaded rinex nav file.")

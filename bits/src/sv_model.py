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
from bits.src.reference_frame_object import GnssTimestamp
from bits.src.convert import space_conversion
from bits.src import const
from bits.src.parsers.ephemeris import rinex_nav
from bits.src.utils import check_dataframe


class SVState(NamedTuple):
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
    # Elliptical orbit shape
    sqrta: np.ndarray  # Square root of semi-major axis
    e: np.ndarray  # Eccentricity

    # Orbit orientation
    omega: np.ndarray  # Argument of perigee
    omega0: np.ndarray  # Longitude of Ascending Node of Orbit Plane at Weekly Epoch
    i0: np.ndarray  # Inclination angle at reference time

    # Orbit corrections
    # Secular corrections
    deltan: np.ndarray  # Mean Motion difference from computed value at reference time
    omegadot: np.ndarray  # Rate of right ascension difference
    idot: np.ndarray  # Rate of inclination angle

    # Harmonic corrections
    cis: np.ndarray  # Amplitude of the sine harmonic correction term to the angle of inclination
    cic: np.ndarray  # Amplitude of the cosine harmonic correction term to the angle of inclination
    crs: np.ndarray  # Amplitude of the sine correction term to the orbit radius
    crc: np.ndarray  # Amplitude of the cosine correction term to the orbit radius
    cus: np.ndarray  # Amplitude of the sine harmonic correction term to the argument of latitude
    cuc: np.ndarray  # Amplitude of the cosine harmonic correction term to the argument of latitude

    # Satellite position on orbit
    m0: np.ndarray  # Mean anomaly at reference time


def kepler_based_sv_model(orbit_param: KeplerianParameters, toe: np.ndarray, tow: None | np.ndarray = None,
                          tk: None | np.ndarray = None, ek_iterations: int = 5) -> SVState:
    """
    Compute GPS, Galileo or Beidou SV states.

    Based on: https://www.navcen.uscg.gov/sites/default/files/pdf/gps/IS_GPS_200M.pdf
    (Table Broadcast Navigation User Equations)

    :param orbit_param: Keplerian set of parameters that describe the orbit
    :param toe: Ephemeris data reference time of week (secondes)
    :param tow: Time of week at which the satellite's position should be computed (secondes)
    :param tk: Elapsed time since ephemeris data reference (secondes)
    :param ek_iterations: Number of iterations to compute the eccentric anomaly
    :return: SVState(x_ecef, y_ecef, z_ecef, vx_ecef, vy_ecef, vz_ecef, ax_ecef, ay_ecef, az_ecef)
    speed and acceleration in ECEF and eccentric anomaly
    """
    # Elapsed time since ephemeris
    if tk is None:
        if tow is None:
            raise ValueError("Either tk or tow must be specified")

        tk = tow - toe
        # Week crossover
        tk = np.where(tk > 302400, tk - 604800, tk)
        tk = np.where(tk < -302400, tk + 604800, tk)

    a = orbit_param.sqrta ** 2  # Semi-major axis
    n0 = np.sqrt(const.NU / a ** 3)  # Computed mean motion (rad/sec)
    n = n0 + orbit_param.deltan  # Corrected mean motion
    mk = orbit_param.m0 + n * tk  # Mean anomaly

    # Kepler’s equation(𝑀𝑘=𝐸𝑘 − 𝑒 sin 𝐸𝑘 ) may be solved for Eccentric anomaly(𝐸𝑘) by iteration:
    ek = mk  # Initial Value (radians)
    for i in range(ek_iterations):  # Refined Value, minimum of three iterations
        ek = ek + (mk - ek + orbit_param.e * np.sin(ek)) / (1 - orbit_param.e * np.cos(ek))

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
    omegak = orbit_param.omega0 + (orbit_param.omegadot - const.OMEGA_E) * tk - const.OMEGA_E * toe

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
    omegak_dot = orbit_param.omegadot - const.OMEGA_E

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
    xk_dotdot = (-const.NU * (xk / (rk ** 3)) + F * ((1 - 5 * (zk / rk) ** 2) * (xk / rk)) + 2 * yk_dot * const.OMEGA_E
                 + xk * const.OMEGA_E ** 2)
    yk_dotdot = (-const.NU * (yk / (rk ** 3)) + F * ((1 - 5 * (zk / rk) ** 2) * (yk / rk)) - 2 * xk_dot * const.OMEGA_E
                 + yk * const.OMEGA_E ** 2)
    zk_dotdot = -const.NU * (zk / (rk ** 3)) + F * ((3 - 5 * (zk / rk) ** 2) * (zk / rk))

    return SVState(xk, yk, zk, xk_dot, yk_dot, zk_dot, xk_dotdot, yk_dotdot, zk_dotdot)


def _glo_equations_of_motion(state: np.ndarray,
                             ddx: float, ddy: float, ddz: float) -> np.ndarray:
    """
    Function F(t, Y): derivatives of the state vector in ECI.

    dx/dt  = vx
    dy/dt  = vy
    dz/dt  = vz
    dvx/dt = -μ̄·x̄ + (3/2)·C20·μ̄·x̄·ρ²·(1 − 5z̄²) + ddx
    dvy/dt = -μ̄·ȳ + (3/2)·C20·μ̄·ȳ·ρ²·(1 − 5z̄²) + ddy
    dvz/dt = -μ̄·z̄ + (3/2)·C20·μ̄·z̄·ρ²·(3 − 5z̄²) + ddz

    Args:
        state : [x, y, z, vx, vy, vz] ECI [m, m/s]
        ddx,ddy,ddz : Lunisolar accelerations ECI [m/s²]

    Returns:
        dY/dt : [vx, vy, vz, ax, ay, az]
    """
    x, y, z, vx, vy, vz = state

    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    if r < 1e3:
        raise ValueError(f"Invalid state vector: {state}")

    # Normalised variables
    mu_bar = const.NU / r ** 2  # μ/r²   [m/s²/m normalised]
    x_bar = x / r  # x/r    [-]
    y_bar = y / r
    z_bar = z / r
    rho = const.RE / r  # ae/r   [-]
    rho2 = rho ** 2
    z_bar2 = z_bar ** 2

    # Oblate Earth acceleration Factor
    j2_coeff = 1.5 * -const.J2 * mu_bar * rho2

    # Accelerations [m/s²]
    ax = -mu_bar * x_bar + j2_coeff * x_bar * (1.0 - 5.0 * z_bar2) + ddx
    ay = -mu_bar * y_bar + j2_coeff * y_bar * (1.0 - 5.0 * z_bar2) + ddy
    az = -mu_bar * z_bar + j2_coeff * z_bar * (3.0 - 5.0 * z_bar2) + ddz

    return np.array([vx, vy, vz, ax, ay, az])


def _rk4_step(state: np.ndarray, ddx: float, ddy: float, ddz: float, h: float) -> np.ndarray:
    """
    RK4 integration step (Runge-Kutta)

    K1 = F(tn,      Yn)
    K2 = F(tn+h/2,  Yn + h·K1/2)
    K3 = F(tn+h/2,  Yn + h·K2/2)
    K4 = F(tn+h,    Yn + h·K3)
    Y_{n+1} = Yn + h/6·(K1 + 2K2 + 2K3 + K4)

    Args:
        state : state vector [m, m/s]
        ddx,ddy,ddz : Lunisolar accelerations [m/s²]
        h     : time step [s]

    Returns:
        state vector after step h
    """
    K1 = _glo_equations_of_motion(state, ddx, ddy, ddz)
    K2 = _glo_equations_of_motion(state + h * K1 / 2, ddx, ddy, ddz)
    K3 = _glo_equations_of_motion(state + h * K2 / 2, ddx, ddy, ddz)
    K4 = _glo_equations_of_motion(state + h * K3, ddx, ddy, ddz)

    return state + (h / 6.0) * (K1 + 2 * K2 + 2 * K3 + K4)


def state_propagation_based_sv_model(pd_ephemeris_row: pd.Series, time: GnssTimestamp, step_s: float = 30) -> SVState:
    """
    Compute Glonass SV states.
    Computes one satellite position at a specific time using its ephemeris parameters.
    Based on https://gssc.esa.int/navipedia/index.php?title=GLONASS_Satellite_Coordinates_Computation
    :param pd_ephemeris_row: Satellite ephemeris. Use a pd.Series parsed with the BITS ephemeris parser.
    :param time: Time at which the satellite's position should be computed
    :return: SVState(x_ecef, y_ecef, z_ecef, vx_ecef, vy_ecef, vz_ecef, ax_ecef, ay_ecef, az_ecef)
    """
    required_columns = ["time_of_ephemeris", "X", "Y", "Z", "dX", "dY", "dZ", "dX2", "dY2", "dZ2", ]

    if pd.isna(time) or pd_ephemeris_row[required_columns].isna().any():
        return [np.nan] * 3

    delta_t = (time - pd_ephemeris_row["time_of_ephemeris"]).total_seconds()

    # 1. Coordinates transformation to an inertial reference frame:
    xa, ya, za = space_conversion.ecef_to_eci_position(
        pd_ephemeris_row["X"], pd_ephemeris_row["Y"], pd_ephemeris_row["Z"], pd_ephemeris_row["time_of_ephemeris"])
    dxa, dya, dza = space_conversion.ecef_to_eci_velocity(
        pd_ephemeris_row["X"], pd_ephemeris_row["Y"], pd_ephemeris_row["Z"],
        pd_ephemeris_row["dX"], pd_ephemeris_row["dY"], pd_ephemeris_row["dZ"], pd_ephemeris_row["time_of_ephemeris"])
    ddxa, ddya, ddza = space_conversion.ecef_to_eci_position(
        pd_ephemeris_row["dX2"], pd_ephemeris_row["dY2"], pd_ephemeris_row["dZ2"],
        pd_ephemeris_row["time_of_ephemeris"])

    # 2. Numerical integration of differential equations that describe the motion of the satellites.
    Y = np.array([xa, ya, za, dxa, dya, dza])  # Initial state
    remaining = delta_t
    sign = 1.0 if delta_t >= 0 else -1.0

    # Runge-Kutta integration algorithm
    while abs(remaining) > 1e-9:
        h = sign * min(step_s, abs(remaining))
        Y = _rk4_step(Y, ddxa, ddya, ddza, h)
        remaining -= h

    x_eci, y_eci, z_eci, vx_eci, vy_eci, vz_eci = tuple(Y)

    # Compute acceleration
    r = np.sqrt(x_eci ** 2 + y_eci ** 2 + z_eci ** 2)
    mu_bar = const.NU / r ** 2
    x_bar, y_bar, z_bar = x_eci / r, y_eci / r, z_eci / r
    rho2 = (const.RE / r) ** 2
    z_bar2 = z_bar ** 2
    j2 = 1.5 * -const.J2 * mu_bar * rho2

    ax_eci = -mu_bar * x_bar + j2 * x_bar * (1.0 - 5.0 * z_bar2) + dxa
    ay_eci = -mu_bar * y_bar + j2 * y_bar * (1.0 - 5.0 * z_bar2) + dya
    az_eci = -mu_bar * z_bar + j2 * z_bar * (3.0 - 5.0 * z_bar2) + dza

    # 3. Coordinates transformation back to ECEF reference system:
    # Position
    x_ecef, y_ecef, z_ecef = space_conversion.eci_to_ecef_position(x_eci, y_eci, z_eci, time)

    # Speed
    theta_ge = time.sidereal()
    s = np.sin(theta_ge)
    c = np.cos(theta_ge)

    omega_cross_r = np.array([-const.OMEGA_E * y_eci, const.OMEGA_E * x_eci, 0.0])
    v_ecef_eci = np.array([vx_eci, vy_eci, vz_eci]) - omega_cross_r

    vx_ecef = v_ecef_eci[0] * c + v_ecef_eci[1] * s
    vy_ecef = -v_ecef_eci[0] * s + v_ecef_eci[1] * c
    vz_ecef = v_ecef_eci[2]

    # Acceleration
    a_eci = np.array([ax_eci, ay_eci, az_eci])

    #   Coriolis  : 2·ω × v_ECEF
    omega_cross_v = np.array([-const.OMEGA_E * vy_ecef, const.OMEGA_E * vx_ecef, 0.0])  # ω×v_ECEF (approx en ECI)
    coriolis = 2.0 * omega_cross_v

    #   Centrifuge: ω×(ω×r_ECI) = [−ω²·xi, −ω²·yi, 0]
    centrifuge = np.array([-const.OMEGA_E ** 2 * x_eci, -const.OMEGA_E ** 2 * y_eci, 0.0])

    a_ecef_eci = a_eci - coriolis - centrifuge

    ax_ecef = a_ecef_eci[0] * c + a_ecef_eci[1] * s
    ay_ecef = -a_ecef_eci[0] * s + a_ecef_eci[1] * c
    az_ecef = a_ecef_eci[2]

    return SVState(x_ecef, y_ecef, z_ecef, vx_ecef, vy_ecef, vz_ecef, ax_ecef, ay_ecef, az_ecef)


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
    raw_required_columns = ["time", "pr_m", "gnss_id", "sv_id"]
    gps_ephemeris_required_columns = ["time", "time_of_ephemeris", "sqrta", "deltan", "m0", "e",
                                      "omega", "omega0", "omegadot", "i0", "idot",
                                      "cis", "cic", "crs", "crc", "cus", "cuc"]
    glo_ephemeris_required_columns = ["time", "X", "Y", "Z", "dX", "dY", "dZ", "dX2", "dY2", "dZ2", ]

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
    if (not check_dataframe(pd_gnss, gps_ephemeris_required_columns)
            and not check_dataframe(pd_gnss, glo_ephemeris_required_columns)):
        warnings.warn("Missing ephemeris data, cannot add SV states.")
        return pd_gnss

    # 1. Compute satellite coordinates at the emission time in the associated ECEF reference frame (i.e., tied to the
    # emission time).
    # Find emission time
    pd_gnss["delta_time"] = \
        pd_gnss.apply(lambda row: pd.Timedelta(row[pr_column_name] / const.C, unit="seconds"), axis=1)

    valid_mask = pd_gnss["delta_time"].notna()

    pd_gnss.loc[valid_mask, ["emission_time"]] = pd_gnss.loc[valid_mask].apply(
        lambda row: row[timestamp_column_name] - row["delta_time"], axis=1)

    # Compute sv states at emission time
    cols = [
        "x_sv_m", "y_sv_m", "z_sv_m",
        "vx_sv_mps", "vy_sv_mps", "vz_sv_mps",
        "ax_sv_mpss", "ay_sv_mpss", "az_sv_mpss",
    ]

    # Compute state propagation-based sv state (Glonass)
    pd_glo = pd_gnss[pd_gnss["gnss_id"] == "glo"]
    if not pd_glo.empty and check_dataframe(pd_gnss, glo_ephemeris_required_columns):
        pd_glo.loc[:, cols] = pd_glo.apply(
            lambda row: state_propagation_based_sv_model(row, row["emission_time"]), axis=1,
            result_type="expand").to_numpy()
    else:
        pd_glo = pd.DataFrame()

    # Compute orbit-based sv state (Galileo, GPS, Beidou)
    pd_gps = pd_gnss[pd_gnss["gnss_id"].isin(["gal", "gps", "bei"])]

    if not pd_gps.empty and check_dataframe(pd_gnss, gps_ephemeris_required_columns):
        # Convert time to time of week
        # Conversion algorithms will be numpy compatible in a future version
        # tk is passed instead of tow to handle ephemeris data that has more than one week difference with toe
        tk = pd_gps.apply(lambda row: (row["emission_time"] - row["time_of_ephemeris"]).total_seconds(), axis=1)

        toe = np.array([t.bei_tow() if const == "bei"
                        else t.tow()
                        for const, t in zip(pd_gps["gnss_id"], pd_gps["time_of_ephemeris"])])

        # Get Keplerian set of parameters alongside correction parameters
        orbit_param = KeplerianParameters(**{field: pd_gps[field].to_numpy() for field in KeplerianParameters._fields})

        # Compute SV state
        sv_state_np_tuple = kepler_based_sv_model(orbit_param, toe=toe, tk=tk)
        pd_gps.loc[:, cols] = np.column_stack(sv_state_np_tuple)
    else:
        pd_gps = pd.DataFrame()

    pd_gnss = pd.concat([pd_gps, pd_glo], axis=0)

    # 2. Transform satellite coordinates from the system tied to the earth at "emission time" to the system tied to the
    # earth at "reception time" (which is common for all measurements). In order to do so, one must consider the earth
    # rotation during the time interval that the signal takes to propagate from the satellite to the receiver:
    pd_gnss[["x_sv_m", "y_sv_m", "z_sv_m"]] = \
        pd_gnss.apply(
            lambda row: pd.Series(space_conversion.rotate_ecef(row["x_sv_m"], row["y_sv_m"], row["z_sv_m"],
                                                               row["delta_time"])), axis=1)
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
                                      "omegadot",
                                      "deltan", "cuc", "cus", "crc", "crs", "cic", "cis"]
    glo_ephemeris_required_columns = ["time", "X", "Y", "Z", "dX", "dY", "dZ", "dX2", "dY2", "dZ2", ]

    # Check if ephemeris is already retrieved
    gps_present = check_dataframe(pd_gnss_raw, gps_ephemeris_required_columns, with_warning=False)
    glo_present = check_dataframe(pd_gnss_raw, glo_ephemeris_required_columns, with_warning=False)
    if not gps_present and not glo_present:
        if pd_ephemeris is None:
            if ephem_filepath is None:
                pd_ephemeris = ephemeris_loader(pd_gnss_raw["time"].iloc[0])  # Get ephemeris from the internet
            else:
                pd_ephemeris = rinex_nav(ephem_filepath)
        # Find corresponding ephemeris for each SV from gnss_raw
        merged = pd_gnss_raw.merge(pd_ephemeris, on=['sv_id'], suffixes=('', '_navdata'))
        # Find difference between ephemeris and gnss_raw timestamp
        merged['time_diff'] = (
            abs(merged["time"] - merged[f'time_of_ephemeris']).astype('timedelta64[ns]'))
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

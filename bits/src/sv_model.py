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
from bits.src.reference_frame_object import GnssTimestamp
from bits.src.convert import space_conversion
from bits.src import const
from bits.src.parsers.ephemeris import rinex_nav
from bits.src.utils import check_dataframe


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
    #gpsweek_diff = (np.mod(time.gps_week(), 1024)
    #                - np.mod(pd_ephemeris_row["time_navdata"].gps_week(), 1024)) * 604800.
    #tk = time.tow() - pd_ephemeris_row["toe"] + gpsweek_diff  # Time from ephemeris reference epoch
    tk = (time - pd_ephemeris_row["time_of_ephemeris"]).total_seconds()
    n = n0 + pd_ephemeris_row["deltan"]  # Corrected mean motion
    mk = pd_ephemeris_row["m0"] + n * tk  # Mean anomaly

    # Kepler’s equation(𝑀𝑘=𝐸𝑘 − 𝑒 sin 𝐸𝑘 ) may be solved for Eccentric anomaly(𝐸𝑘) by iteration:
    ek = mk  # Initial Value (radians)
    for i in range(ek_iterations):  # Refined Value, minimum of three iterations
        ek = ek + (mk - ek + pd_ephemeris_row["e"] * math.sin(ek)) / (1 - pd_ephemeris_row["e"] * math.cos(ek))

    return ek, n

def _get_sv_state_row(pd_ephemeris_row: pd.Series, time: GnssTimestamp, ek_iterations=5) \
        -> tuple[float, float, float, float, float, float, float, float, float, float]:
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
    required_columns = ["sqrta", "time_navdata", "time_of_ephemeris", "e", "omega", "cuc", "cus", "crc", "crs", "cic", "cis", "i0",
                        "idot", "omega0", "omegadot"]

    if pd.isna(time) or pd_ephemeris_row[required_columns].isna().any():
        return [np.nan] * 10

    a = pd_ephemeris_row["sqrta"] ** 2  # Semi-major axis

    tk = (time - pd_ephemeris_row["time_of_ephemeris"]).total_seconds()

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
    if pd_ephemeris_row["gnss_id"]=="bei":
        toe = pd_ephemeris_row["time_of_ephemeris"].bei_tow()
    else:
        toe = pd_ephemeris_row["time_of_ephemeris"].tow()
    #omegak = pd_ephemeris_row["omega0"] + (pd_ephemeris_row["omegadot"] - const.OMEGA_E) * tk \
    #         - const.OMEGA_E * pd_ephemeris_row["time_of_ephemeris"].tow()
    omegak = pd_ephemeris_row["omega0"] + (pd_ephemeris_row["omegadot"] - const.OMEGA_E) * tk - const.OMEGA_E * toe

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


def _glo_equations_of_motion(state: np.ndarray,
                        ddx: float, ddy: float, ddz: float) -> np.ndarray:
    """
    Fonction F(t, Y) : dérivées du vecteur d'état en ECI.

    Équation (5) — système différentiel GLONASS :
        dx/dt  = vx
        dy/dt  = vy
        dz/dt  = vz
        dvx/dt = -μ̄·x̄ + (3/2)·C20·μ̄·x̄·ρ²·(1 − 5z̄²) + ddx
        dvy/dt = -μ̄·ȳ + (3/2)·C20·μ̄·ȳ·ρ²·(1 − 5z̄²) + ddy
        dvz/dt = -μ̄·z̄ + (3/2)·C20·μ̄·z̄·ρ²·(3 − 5z̄²) + ddz

    Args:
        state : [x, y, z, vx, vy, vz] en ECI [m, m/s]
        ddx,ddy,ddz : accélérations luni-solaires totales en ECI [m/s²]

    Returns:
        dY/dt : [vx, vy, vz, ax, ay, az]
    """
    # TODO
    MU = 398_600.44e9  # m³/s²  Constante gravitationnelle (PZ-90)
    AE = 6_378_136.0  # m      Rayon équatorial terrestre (PZ-90)
    C20 = -1_082.63e-9  # [-]    Coefficient zonal J2 (= -J2 = +√5·C̄20)

    x, y, z, vx, vy, vz = state

    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    if r < 1e3:
        raise ValueError(f"Rayon trop faible : r = {r} m (vecteur d'état invalide ?)")

    # Variables normalisées (barre)
    mu_bar = MU / r ** 2  # μ/r²   [m/s²/m normalisé]
    x_bar = x / r  # x/r    [-]
    y_bar = y / r
    z_bar = z / r
    rho = AE / r  # ae/r   [-]
    rho2 = rho ** 2
    z_bar2 = z_bar ** 2

    # Terme J2 (aplatissement)
    j2_coeff = 1.5 * C20 * mu_bar * rho2

    # Accélérations [m/s²]
    ax = -mu_bar * x_bar + j2_coeff * x_bar * (1.0 - 5.0 * z_bar2) + ddx
    ay = -mu_bar * y_bar + j2_coeff * y_bar * (1.0 - 5.0 * z_bar2) + ddy
    az = -mu_bar * z_bar + j2_coeff * z_bar * (3.0 - 5.0 * z_bar2) + ddz

    return np.array([vx, vy, vz, ax, ay, az])


def _rk4_step(state: np.ndarray, ddx: float, ddy: float, ddz: float, h: float) -> np.ndarray:
    """
    Un pas d'intégration RK4 (équation 7).

    K1 = F(tn,      Yn)
    K2 = F(tn+h/2,  Yn + h·K1/2)
    K3 = F(tn+h/2,  Yn + h·K2/2)
    K4 = F(tn+h,    Yn + h·K3)
    Y_{n+1} = Yn + h/6·(K1 + 2K2 + 2K3 + K4)

    Note : F ne dépend pas explicitement de t dans ce modèle
           (les accélérations luni-solaires Jx,Jy,Jz sont supposées constantes
            sur l'intervalle de propagation ≤ 15 min).

    Args:
        state : vecteur d'état courant [m, m/s]
        ddx,ddy,ddz : accélérations luni-solaires [m/s²]
        h     : pas de temps [s]

    Returns:
        Nouveau vecteur d'état après le pas h
    """
    K1 = _glo_equations_of_motion(state, ddx, ddy, ddz)
    K2 = _glo_equations_of_motion(state + h * K1 / 2, ddx, ddy, ddz)
    K3 = _glo_equations_of_motion(state + h * K2 / 2, ddx, ddy, ddz)
    K4 = _glo_equations_of_motion(state + h * K3, ddx, ddy, ddz)

    return state + (h / 6.0) * (K1 + 2 * K2 + 2 * K3 + K4)


def _get_glo_sv_state_row(pd_ephemeris_row: pd.Series, time: GnssTimestamp, step_s:float=30) -> tuple[float, float, float]:
    """
    Compute Glonass SV states.
    Computes one satellite position at a specific time using its ephemeris parameters.
    Based on https://gssc.esa.int/navipedia/index.php?title=GLONASS_Satellite_Coordinates_Computation
    :param pd_ephemeris_row: Satellite ephemeris. Use a pd.Series parsed with the BITS ephemeris parser.
    :param time: Time at which the satellite's position should be computed
    :return: (x_ecef, y_ecef, z_ecef) -> Satellite position in ECEF
    """
    C20 = -1_082.63e-9 # TODO
    required_columns = ["time_of_ephemeris", "X", "Y", "Z", "dX", "dY", "dZ", "dX2", "dY2", "dZ2",]

    if pd.isna(time) or pd_ephemeris_row[required_columns].isna().any():
        return [np.nan] * 3

    delta_t = (time - pd_ephemeris_row["time_of_ephemeris"]).total_seconds()

    # Coordinates transformation to an inertial reference frame
    xa, ya, za = space_conversion.ecef_to_eci_position(
        pd_ephemeris_row["X"], pd_ephemeris_row["Y"], pd_ephemeris_row["Z"], pd_ephemeris_row["time_of_ephemeris"])
    dxa, dya, dza = space_conversion.ecef_to_eci_velocity(
        pd_ephemeris_row["X"], pd_ephemeris_row["Y"], pd_ephemeris_row["Z"],
        pd_ephemeris_row["dX"], pd_ephemeris_row["dY"], pd_ephemeris_row["dZ"], pd_ephemeris_row["time_of_ephemeris"])
    ddxa, ddya, ddza = space_conversion.ecef_to_eci_position(
        pd_ephemeris_row["dX2"], pd_ephemeris_row["dY2"], pd_ephemeris_row["dZ2"], pd_ephemeris_row["time_of_ephemeris"])


    # Intégration RK4 par pas de step_s secondes
    Y = np.array([xa, ya, za, dxa, dya, dza]) # Initial state
    remaining = delta_t
    sign = 1.0 if delta_t >= 0 else -1.0

    while abs(remaining) > 1e-9:
        h = sign * min(step_s, abs(remaining))
        Y = _rk4_step(Y, ddxa, ddya, ddza, h)
        #t += h
        remaining -= h

    x_eci, y_eci, z_eci, vx_eci, vy_eci, vz_eci = tuple(Y)

    x_ecef, y_ecef, z_ecef = space_conversion.eci_to_ecef_position(x_eci, y_eci, z_eci, time)

    theta_ge = time.sidereal()
    s = np.sin(theta_ge)
    c = np.cos(theta_ge)

    # ── 2. Vitesse ECEF ───────────────────────────────────────────────────
    # v_ECI − ω×r_ECI  (soustrait la rotation terrestre)

    # Correction : ω×r_ECI = (−ω·y_ECI, +ω·x_ECI, 0)
    #   v_ECEF_in_ECI = v_ECI − ω×r_ECI
    # Reprenons proprement :
    # ω×r = [ω_z·y − ω_y·z, ω_x·z − ω_z·x, ω_y·x − ω_x·y]
    # avec ω = [0, 0, ω_E]  →  ω×r = [−ω·y, +ω·x, 0]
    omega_cross_r = np.array([-const.OMEGA_E * y_eci, const.OMEGA_E * x_eci, 0.0])
    v_ecef_eci = np.array([vx_eci, vy_eci, vz_eci]) - omega_cross_r

    # Rotation inverse pour exprimer en ECEF
    vx_ecef = v_ecef_eci[0] * c + v_ecef_eci[1] * s
    vy_ecef = -v_ecef_eci[0] * s + v_ecef_eci[1] * c
    vz_ecef = v_ecef_eci[2]

    # ── 3. Accélération ECEF ─────────────────────────────────────────────
    # Recalcul de a_ECI depuis les équations du mouvement (éq. 5)
    r = np.sqrt(x_eci ** 2 + y_eci ** 2 + z_eci ** 2)
    mu_bar = const.NU / r ** 2
    x_bar, y_bar, z_bar = x_eci / r, y_eci / r, z_eci / r
    rho2 = (const.RE / r) ** 2
    z_bar2 = z_bar ** 2
    j2 = 1.5 * C20 * mu_bar * rho2

    ax_eci = -mu_bar * x_bar + j2 * x_bar * (1.0 - 5.0 * z_bar2) + dxa
    ay_eci = -mu_bar * y_bar + j2 * y_bar * (1.0 - 5.0 * z_bar2) + dya
    az_eci = -mu_bar * z_bar + j2 * z_bar * (3.0 - 5.0 * z_bar2) + dza

    a_eci = np.array([ax_eci, ay_eci, az_eci])

    # Termes à soustraire pour passer en ECEF :
    #   Coriolis  : 2·ω × v_ECEF  (exprimé dans ECI)
    omega_cross_v = np.array([-const.OMEGA_E * vy_ecef, const.OMEGA_E * vx_ecef, 0.0])  # ω×v_ECEF (approx en ECI)
    coriolis = 2.0 * omega_cross_v

    #   Centrifuge: ω×(ω×r_ECI) = [−ω²·xi, −ω²·yi, 0]
    centrifuge = np.array([-const.OMEGA_E ** 2 * x_eci, -const.OMEGA_E ** 2 * y_eci, 0.0])

    a_ecef_eci = a_eci - coriolis - centrifuge

    # Rotation inverse
    ax_ecef = a_ecef_eci[0] * c + a_ecef_eci[1] * s
    ay_ecef = -a_ecef_eci[0] * s + a_ecef_eci[1] * c
    az_ecef = a_ecef_eci[2]

    return x_ecef, y_ecef, z_ecef, vx_ecef, vy_ecef, vz_ecef, ax_ecef, ay_ecef, az_ecef, None#ek


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
    gps_ephemeris_required_columns = ["time", "time_of_ephemeris", "sqrta", "e", "i0", "idot", "omega0", "omega", "m0", "omegadot",
                                  "deltan", "cuc", "cus", "crc", "crs", "cic", "cis"]
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

    # 1. Calculate satellite coordinates at the emission time in the associated ECEF reference frame (i.e., tied to the
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
        "eccentric_anomaly",
    ]

    pd_glo = pd_gnss[pd_gnss["gnss_id"] == "glo"]
    if not pd_glo.empty and check_dataframe(pd_gnss, glo_ephemeris_required_columns):
        pd_glo.loc[:, cols] = pd_gnss.apply(
            lambda row: _get_glo_sv_state_row(row, row["emission_time"]), axis=1, result_type="expand").to_numpy()
    else:
        pd_glo = pd.DataFrame()

    pd_gps = pd_gnss[pd_gnss["gnss_id"] != "glo"]
    if not pd_gps.empty and check_dataframe(pd_gnss, gps_ephemeris_required_columns):
        pd_gps.loc[:, cols] = pd_gnss.apply(
            lambda row: _get_sv_state_row(row, row["emission_time"]), axis=1, result_type="expand").to_numpy()
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


def retrieve_ephemeris(pd_gnss_raw: pd.DataFrame, pd_ephemeris: pd.DataFrame = None, ephem_filepath: str = None)\
        -> pd.DataFrame:
    """
    Finds the closest ephemeris parameters for each satellite vehicle
    :param pd_gnss_raw: GNSS raw dataframe from BITS parser
    :param pd_ephemeris: ephemeris dataframe from BITS parser
    :param ephem_filepath: Path of a rinex nav file
    :return: GNSS raw dataframe with ephemeris
    """
    gps_ephemeris_required_columns = ["time", "time_of_ephemeris", "sqrta", "e", "i0", "idot", "omega0", "omega", "m0", "omegadot",
                                  "deltan", "cuc", "cus", "crc", "crs", "cic", "cis"]
    glo_ephemeris_required_columns = ["time", "X", "Y", "Z", "dX", "dY", "dZ", "dX2", "dY2", "dZ2", ]

    # Check if ephemeris is already retrieved
    gps_present = check_dataframe(pd_gnss_raw, gps_ephemeris_required_columns, with_warning=False)
    glo_present = check_dataframe(pd_gnss_raw, glo_ephemeris_required_columns, with_warning=False)
    if not gps_present or not glo_present:
        if pd_ephemeris is None:
            if ephem_filepath is None:
                pd_ephemeris = ephemeris_loader(pd_gnss_raw["time"].iloc[0]) # Get ephemeris from the internet
            else:
                pd_ephemeris = rinex_nav(ephem_filepath)
        # Find corresponding ephemeris for each SV from gnss_raw
        merged = pd_gnss_raw.merge(pd_ephemeris, on=['gnss_id', 'sv_id'], suffixes=('', '_navdata'))
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


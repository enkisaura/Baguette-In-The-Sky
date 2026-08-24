"""
Space reference system conversions algorithms.

Conversions are based on the Earth-Centred, Earth-Fixed (ECEF) reference system.

"ECEF is an earth-fixed, i.e. rotating reference system. Its origin is the Earth's centre of mass, the fundamental plane
contains this origin and it is perpendicular to the Earth's Conventional Terrestrial Pole (CTP). Its principal axis is
pointing to the intersection of the mean Greenwich meridian and the equator. Since this coordinate system follows the
diurnal rotation of earth, this is not an inertial reference system.
    -> z: This axis is defined by the Conventional Terrestrial Pole (CTP)
    -> x: This axis is defined as the intersection between the equatorial plane and the mean Greenwich meridian
    -> y: It is orthogonal to the formers ones, so the system is right-handed", source: Navipedia
"""

import numpy as np
import pandas as pd
import pyproj
import warnings

from bits.src import convert
from bits.src import const

# WGS84 ellipsoid parameters
B = const.RE * (1 - const.FLATTENING_E) # Earth semi-minor axis (m)
E2 = 1 - (B ** 2) / (const.RE ** 2) # First eccentricity squared ()
EP2 = (const.RE ** 2 - B ** 2) / (B ** 2) # Second eccentricity squared ()

_wgs_to_ecef_transformer = pyproj.Transformer.from_crs("epsg:4979", "epsg:4978", always_xy=True)
_ecef_to_wgs_transformer = pyproj.Transformer.from_crs("epsg:4978", "epsg:4979", always_xy=True)


def wgs_to_ecef(lat: np.ndarray, lon: np.ndarray, alt: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Converts WGS (EPSG:4326) coordinates to ECEF (EPSG:4978)
    :param lat: latitude (wgs)
    :param lon: longitude (wgs)
    :param alt: altitude (wgs)
    :return: x_ecef, y_ecef, z_ecef
    """
    x_ecef, y_ecef, z_ecef = _wgs_to_ecef_transformer.transform(lon, lat, alt)
    return np.asarray(x_ecef), np.asarray(y_ecef), np.asarray(z_ecef)


def ecef_to_wgs(x_ecef: np.ndarray, y_ecef: np.ndarray, z_ecef: np.ndarray) \
        -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Converts ECEF (EPSG:4978) coordinates to WGS (EPSG:4326)
    :param x_ecef:
    :param y_ecef:
    :param z_ecef:
    :return: lat, lon, alt (wgs)
    """
    lon, lat, alt = _ecef_to_wgs_transformer.transform(x_ecef, y_ecef, z_ecef)
    return np.asarray(lat), np.asarray(lon), np.asarray(alt)


def _rotate_ecef_eci(x, y, z, sidereal_time, to_eci:bool):
    if to_eci:
        sign = 1
    else:
        sign = -1

    x_rot = x * np.cos(sidereal_time) - sign*y * np.sin(sidereal_time)
    y_rot = sign*x * np.sin(sidereal_time) + y * np.cos(sidereal_time)
    z_rot = z

    return x_rot, y_rot, z_rot


def ecef_to_eci(time: np.ndarray|pd.Series|np.datetime64|pd.Timestamp,
                x: np.ndarray|pd.Series|float, y: np.ndarray|pd.Series|float, z: np.ndarray|pd.Series|float,
                vx: np.ndarray|pd.Series|float|None=None, vy: np.ndarray|pd.Series|float|None=None, vz: np.ndarray|pd.Series|float|None=None,
                ax: np.ndarray|pd.Series|float|None=None, ay: np.ndarray|pd.Series|float|None=None, az: np.ndarray|pd.Series|float|None=None
                ):
    """
    Converts ECEF to Earth Centred Inertial (ECI) reference system. ECEF coordinates inputs can contain either positon,
    position + velocity or position + velocity + acceleration. ECI coordinates outputs will correspond to the input.

    "ECI is mainly used for the description of satellite motion. It has its origin in the Earth's centre of mass or
    Geocentre, its fundamental plane is the mean Equator plane (containing the Geocentre) of the epoch J2000.0, and the
    principal axis x is pointing to the mean Vernal equinox of epoch J2000.0.
        -> x: Its origin is the Geocentre, the Earth's centre of mass, and its direction is towards the mean equinox at
        J2000.0
        -> z: This axis is defined by the direction of the earth mean rotation pole at J2000.0
        -> y: It is orthogonal to the formers ones, so the system is right-handed", source: Navipedia

    :param time: reference time of the state (UTC)
    :param x: ECEF position on x axis (m)
    :param y: ECEF position on y axis (m)
    :param z: ECEF position on z axis (m)
    :param vx: ECEF velocity on x axis (m/s)
    :param vy: ECEF velocity on y axis (m/s)
    :param vz: ECEF velocity on z axis (m/s)
    :param ax: ECEF acceleration on x axis (m/s²)
    :param ay: ECEF acceleration on y axis (m/s²)
    :param az: ECEF acceleration on z axis (m/s²)
    :return: ECI coordinates
    """
    velocity_state = vx is not None and vy is not None and vz is not None
    acceleration_state = ax is not None and ay is not None and az is not None

    sidereal_time = convert.time.utc_to_sidereal(time)

    # Position
    x_eci, y_eci, z_eci = _rotate_ecef_eci(x, y, z, sidereal_time, True)

    if not velocity_state:
        if acceleration_state:
            warnings.warn("Need to provide velocity to convert acceleration, returned position state only.")
        return x_eci, y_eci, z_eci

    # Velocity
    vx_eci, vy_eci, vz_eci = _rotate_ecef_eci(vx, vy, vz, sidereal_time, True)
    # Earth rotation corrections
    vx_eci -= const.OMEGA_E * y_eci
    vy_eci += const.OMEGA_E * x_eci

    if not acceleration_state:
        return x_eci, y_eci, z_eci, vx_eci, vy_eci, vz_eci

    # Acceleration
    ax_eci, ay_eci, az_eci = _rotate_ecef_eci(ax, ay, az, sidereal_time, True)
    # Coriolis and centrifugal corrections
    ax_eci = ax_eci + const.OMEGA_E ** 2 * x_eci - 2 * const.OMEGA_E * vy_eci
    ay_eci = ay_eci + const.OMEGA_E ** 2 * y_eci + 2 * const.OMEGA_E * vx_eci

    return x_eci, y_eci, z_eci, vx_eci, vy_eci, vz_eci, ax_eci, ay_eci, az_eci


def eci_to_ecef(time: np.ndarray|pd.Series|np.datetime64|pd.Timestamp,
                 x: np.ndarray|pd.Series|float, y: np.ndarray|pd.Series|float, z: np.ndarray|pd.Series|float,
                 vx: np.ndarray|pd.Series|float|None=None, vy: np.ndarray|pd.Series|float|None=None, vz: np.ndarray|pd.Series|float|None=None,
                 ax: np.ndarray|pd.Series|float|None=None, ay: np.ndarray|pd.Series|float|None=None, az: np.ndarray|pd.Series|float|None=None
                 ):
    """
    Converts Earth Centred Inertial (ECI) to ECEF reference system. ECI coordinates inputs can contain either positon,
    position + velocity or position + velocity + acceleration. ECEF coordinates outputs will correspond to the input.

    "ECI is mainly used for the description of satellite motion. It has its origin in the Earth's centre of mass or
    Geocentre, its fundamental plane is the mean Equator plane (containing the Geocentre) of the epoch J2000.0, and the
    principal axis x is pointing to the mean Vernal equinox of epoch J2000.0.
        -> x: Its origin is the Geocentre, the Earth's centre of mass, and its direction is towards the mean equinox at
        J2000.0
        -> z: This axis is defined by the direction of the earth mean rotation pole at J2000.0
        -> y: It is orthogonal to the formers ones, so the system is right-handed", source: Navipedia

    :param time: reference time of the state (UTC)
    :param x: ECI position on x axis (m)
    :param y: ECI position on y axis (m)
    :param z: ECI position on z axis (m)
    :param vx: ECI velocity on x axis (m/s)
    :param vy: ECI velocity on y axis (m/s)
    :param vz: ECI velocity on z axis (m/s)
    :param ax: ECI acceleration on x axis (m/s²)
    :param ay: ECI acceleration on y axis (m/s²)
    :param az: ECI acceleration on z axis (m/s²)
    :return: ECEF coordinates
    """
    velocity_state = vx is not None and vy is not None and vz is not None
    acceleration_state = ax is not None and ay is not None and az is not None

    sidereal_time = convert.time.utc_to_sidereal(time)

    # Position
    x_ecef, y_ecef, z_ecef = _rotate_ecef_eci(x, y, z, sidereal_time, False)

    if not velocity_state:
        if acceleration_state:
            warnings.warn("Need to provide velocity to convert acceleration, returned position state only.")
        return x_ecef, y_ecef, z_ecef

    # Velocity
    # Earth rotation corrections
    vx_corr = vx + const.OMEGA_E * y
    vy_corr = vy - const.OMEGA_E * x
    # Rotation to ECEF
    vx_ecef, vy_ecef, vz_ecef = _rotate_ecef_eci(vx_corr, vy_corr, vz, sidereal_time, False)

    if not acceleration_state:
        return x_ecef, y_ecef, z_ecef, vx_ecef, vy_ecef, vz_ecef

    # Acceleration
    # Coriolis and centrifugal corrections
    ax = ax - const.OMEGA_E ** 2 * x + 2 * const.OMEGA_E * vy
    ay = ay - const.OMEGA_E ** 2 * y - 2 * const.OMEGA_E * vx
    # Rotation to ECEF
    ax_ecef, ay_ecef, az_ecef = _rotate_ecef_eci(ax, ay, az, sidereal_time, False)

    return x_ecef, y_ecef, z_ecef, vx_ecef, vy_ecef, vz_ecef, ax_ecef, ay_ecef, az_ecef


def rotate_ecef(x: float, y: float, z: float, delta_time: np.ndarray|np.timedelta64) \
        -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Rotate ECEF coordinates over a specified time interval to account for Earth's rotation. This is used to correct for
    Earth's rotation when converting from ECI to ECEF.
    :param x_ecef: X ECEF (m)
    :param y_ecef: Y ECEF (m)
    :param z_ecef: Z ECEF (m)
    :param delta_time: Earth rotation duration
    :return: (x_ecef, y_ecef, z_ecef)
    """
    # Rotation of the earth during a period of delta_time
    rotation_angle = const.OMEGA_E * (delta_time / np.timedelta64(1, "s"))

    return _rotate_ecef_eci(x, y, z, rotation_angle, to_eci=False)


def ecef_to_enu(x_ref: np.ndarray, y_ref: np.ndarray, z_ref: np.ndarray,
                 x_target: np.ndarray, y_target: np.ndarray, z_target: np.ndarray
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Converts ECEF target position(s) into local ENU (East-North-Up) coordinates
    relative to a reference position, using the reference to define the local tangent plane.
    Fully vectorized: reference and target can be arrays of the same length (one
    reference per target), or the reference can be scalars (single shared reference).

    :param x_ref: X coordinate(s) of the reference position in ECEF (m)
    :param y_ref: Y coordinate(s) of the reference position in ECEF (m)
    :param z_ref: Z coordinate(s) of the reference position in ECEF (m)
    :param x_target: X coordinate(s) of the target position in ECEF (m)
    :param y_target: Y coordinate(s) of the target position in ECEF (m)
    :param z_target: Z coordinate(s) of the target position in ECEF (m)
    :return: tuple (e, n, u), East/North/Up coordinates (m) of target relative to reference
    """
    x_ref = np.asarray(x_ref, dtype=np.float64)
    y_ref = np.asarray(y_ref, dtype=np.float64)
    z_ref = np.asarray(z_ref, dtype=np.float64)
    x_target = np.asarray(x_target, dtype=np.float64)
    y_target = np.asarray(y_target, dtype=np.float64)
    z_target = np.asarray(z_target, dtype=np.float64)

    # WGS84 ellipsoid parameters
    a = 6378137.0
    f = 1 / 298.257223563
    b = a * (1 - f)
    e2 = 1 - (b ** 2) / (a ** 2)
    ep2 = (a ** 2 - b ** 2) / (b ** 2)

    # Convert reference point ECEF -> geodetic latitude/longitude (Bowring's method)
    p = np.sqrt(x_ref ** 2 + y_ref ** 2)
    theta = np.arctan2(z_ref * a, p * b)

    lon = np.arctan2(y_ref, x_ref)
    lat = np.arctan2(z_ref + ep2 * b * np.sin(theta) ** 3,
                      p - e2 * a * np.cos(theta) ** 3)

    # Displacement vector from reference to target, in ECEF
    dx = x_target - x_ref
    dy = y_target - y_ref
    dz = z_target - z_ref

    # Rotation from ECEF to ENU (row-wise, vectorized: one rotation per reference point)
    sin_lat, cos_lat = np.sin(lat), np.cos(lat)
    sin_lon, cos_lon = np.sin(lon), np.cos(lon)

    e = -sin_lon * dx + cos_lon * dy
    n = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz
    u = cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz

    return e, n, u


def _ecef_to_enu_transition_matrix(x_ref: np.ndarray, y_ref: np.ndarray, z_ref: np.ndarray) -> np.array:
    """
    x->Est
    y->North
    z->Up
    Basis Change:
        If a vector has coordinates X and X' in two different bases B and B', then:
            X = P * X'

        The new basis B' (e'₁, e'₂, e'₃) is obtained by a rotation of an angle α around the axis e₃. Therefore, we have:
            e'1 = cos(α) e1 + sin(α) e2 ;
            e'2 = –sin(α) e1 + cos(α) e2 ;
            e'3 = e3.

        The change of basis matrix P is written as:
            P = [[cos(α), -sin(α), 0], [sin(α), cos(α), 0], [0, 0, 1]]

    Usage: ENU_data = p.dot(ECEF_data)
    :param approx_ecef:
    :return:
    """
    ρ = np.sqrt(x_ref ** 2 + y_ref ** 2 + z_ref ** 2)

    # I/Rotation along the z axis
    φ = np.atan2(y_ref, x_ref)  # Rotation angle
    p1 = np.array([[np.cos(φ), np.sin(φ), 0], [-np.sin(φ), np.cos(φ), 0], [0, 0, 1]])

    # II/Rotation along the y axis
    θ = np.pi/2 - np.acos(z_ref / ρ)  # Rotation angle
    p2 = np.array([[np.cos(θ), 0, np.sin(θ)], [0, 1, 0], [-np.sin(θ), 0, np.cos(θ)]])

    # III/Switching axis
    p3 = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])

    p = p2.dot(p1)
    p = p3.dot(p)

    return p


def enu_to_ecef(x_ref: float, y_ref: float, z_ref: float, e: np.ndarray, n: np.ndarray, u: np.ndarray) \
        -> np.array:
    """
    Compute ECEF coordinates from East North Up coordinates from an ancre.
    :param ancre_ecef: Reference point coordinates (ECEF meters)
    :param enu_matrix: ENU coordinates (meters) to convert to ECEF. E, N, U must be the matrix's columns.
    :return: ECEF coordinates (meters)
    """
    enu_matrix = np.hstack([e, n, u])
    p = _ecef_to_enu_transition_matrix(x_ref, y_ref, z_ref)
    p = np.linalg.inv(p)
    ecef_matrix = p @ enu_matrix
    return ecef_matrix.transpose()
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
import warnings

from bits.src import convert
from bits.src import const


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
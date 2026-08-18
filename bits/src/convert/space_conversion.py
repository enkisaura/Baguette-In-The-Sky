"""
Functions for space conversions
"""

__authors__ = ("Enki SAURA")
__contact__ = ("esaura@ikosconsulting.com")
__copyright__ = "IKOS"
__date__ = "20/02/2025"
__version__ = "0.0.1"


import pyproj
import numpy as np
import math

def wgs_to_ecef(lat: float, lon: float, alt: float) -> tuple[float, float, float]:
    """
    Converts WGS (EPSG:4326) coordinates to ECEF (EPSG:4978)
    :param lat: latitude (wgs)
    :param lon: longitude (wgs)
    :param alt: altitude (wgs)
    :return: x_ecef, y_ecef, z_ecef
    """
    transformer = pyproj.Transformer.from_crs("epsg:4979", "epsg:4978", always_xy=True)
    x_ecef, y_ecef, z_ecef = transformer.transform(lon, lat, alt)
    return x_ecef, y_ecef, z_ecef


def ecef_to_wgs(x_ecef: float, y_ecef: float, z_ecef: float) -> tuple[float, float, float]:
    """
    Converts ECEF (EPSG:4978) coordinates to WGS (EPSG:4326)
    :param x_ecef:
    :param y_ecef:
    :param z_ecef:
    :return: lat, lon, alt (wgs)
    """
    transformer = pyproj.Transformer.from_crs("epsg:4978", "epsg:4979", always_xy=True)
    lon, lat, alt = transformer.transform(x_ecef, y_ecef, z_ecef)
    return lat, lon, alt

def pz_90_to_ecef(x_pz_90: float, y_pz_90: float, z_pz_90: float) -> tuple[float, float, float]:
    """
    Converts GLONASS PZ90 coordinates to ECEF
    source: https://gssc.esa.int/navipedia/index.php?title=GLONASS_Satellite_Coordinates_Computation
    :param x_pz_90:
    :param y_pz_90:
    :param z_pz_90:
    :return: (x_ecef, y_ecef, z_ecef)
    """
    pz_90 = np.array(
        ([x_pz_90],
        [y_pz_90],
        [z_pz_90])
    )

    A = np.array(
        ([-3, -353, -4],
        [353, -3, 19],
        [4, -19, -3])
    )

    B = np.array(
        ([0.07],
        [0],
        [-0.77])
    )

    C = np.array(
        ([-0.36],
        [0.08],
        [0.18])
    )

    ecef = pz_90 + A.dot(pz_90) + B + C

    return ecef[0][0], ecef[1][0], ecef[2][0]


def _ecef_to_enu_transition_matrix(approx_ecef: tuple[float, float, float]) -> np.array:
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
    ρ = math.sqrt(approx_ecef[0] ** 2 + approx_ecef[1] ** 2 + approx_ecef[2] ** 2)

    # I/Rotation along the z axis
    φ = math.atan2(approx_ecef[1], approx_ecef[0])  # Rotation angle
    p1 = np.array([[math.cos(φ), math.sin(φ), 0], [-math.sin(φ), math.cos(φ), 0], [0, 0, 1]])

    # II/Rotation along the y axis
    θ = math.pi/2 - math.acos(approx_ecef[2] / ρ)  # Rotation angle
    p2 = np.array([[math.cos(θ), 0, math.sin(θ)], [0, 1, 0], [-math.sin(θ), 0, math.cos(θ)]])

    # III/Switching axis
    p3 = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])

    p = p2.dot(p1)
    p = p3.dot(p)

    return p


def ecef_to_enu(ancre_ecef: tuple[float, float, float], ecef_matrix: np.array) -> np.array:
    """
    Compute East North Up coordinates from an ancre.
    :param ancre_ecef: Reference point coordinates (ECEF meters)
    :param ecef_matrix: ECEF coordinates (meters) to convert to ENU. X, Y, Z must be the matrix's columns.
    :return: ENU coordinates (meters)
    """
    p = _ecef_to_enu_transition_matrix(ancre_ecef)
    enu_matrix = p @ ecef_matrix.transpose()
    return enu_matrix.transpose()


def enu_to_ecef(ancre_ecef: tuple[float, float, float], enu_matrix: np.array) -> np.array:
    """
    Compute ECEF coordinates from East North Up coordinates from an ancre.
    :param ancre_ecef: Reference point coordinates (ECEF meters)
    :param enu_matrix: ENU coordinates (meters) to convert to ECEF. E, N, U must be the matrix's columns.
    :return: ECEF coordinates (meters)
    """
    p = _ecef_to_enu_transition_matrix(ancre_ecef)
    p = np.linalg.inv(p)
    ecef_matrix = p @ enu_matrix.transpose()
    return ecef_matrix.transpose()

def enu_to_spheric(enu_matrix: np.array) -> np.array:
    """
    Computes coordinates from the local reference frame ENU to range (m), elevation (rad, angle from the horizon),
    azimuth (rad, angle from north).
    :param enu_matrix: ENU coordinates (meters). E, N, U must be the matrix's columns.
    :return: range (m), elevation (rad, angle from the horizon), azimuth (rad, angle from north)
    """
    # Computing norm
    r = np.linalg.norm(enu_matrix, axis=1)
    # Computing elevation
    el = np.arccos(enu_matrix[:, 2] / r) - (math.pi/2)
    # Computing azimuth
    az = np.arctan2(enu_matrix[:, 0], enu_matrix[:, 1]) + math.pi
    az = np.where(az > math.pi, az - 2 * math.pi, az)
    # Building final matrix
    polar_coords = np.vstack((r, el, az)).T
    return polar_coords

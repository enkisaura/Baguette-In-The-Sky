#!/usr/bin/env python3

"""
Scripts to enable single point positioning (SPP)

Usage:
======
get_position_estimate(pd_raw)

"""

__authors__ = ("Enki SAURA")
__contact__ = ("esaura@ikosconsulting.com")
__copyright__ = "IKOS"
__date__ = "2025-06-06"
__version__ = "0.0.1"

import pandas as pd
import numpy as np
import math
import warnings
from typing import Tuple

from .convert.space_conversion import ecef_to_wgs, ecef_to_enu, enu_to_spheric, enu_to_ecef, rotate_ecef
from .corrections import get_clock_corrections, get_atmospheric_corrections
from .sv_model import get_sv_states
from . import const
from .utils import check_dataframe

class PositionEstimationError(Exception):
    """Exception raised for errors during position estimation."""
    pass


def compute_geometry_matrix(sv_position: np.array, rx_pos: np.array) -> np.array:
    """
    Computes a geometry matrix between two position in ECEF.
    :param sv_position: Satellite vehicule position in ecef (meters) np.Array([[X], [Y], [Z]]) (column)
    :param rx_pos: Receiver position in ecef (meters) np.Array([X, Y, Z]) (line)
    :return: geometry matrix
    """
    # Checking rx_pos dim
    if rx_pos.ndim == 2 and rx_pos.shape[1] == 1:
        rx_pos.transpose()
    if rx_pos.shape[0] == sv_position.shape[0]: # sv_position is transposed
        warnings.warn("While computing geometry matrix, SV position matrix seemed transposed.")
        sv_position.transpose()

    # Doing the math
    rx_pos = np.tile(rx_pos, (sv_position.shape[0], 1))
    sv_rx_range_ecef = rx_pos - sv_position
    sv_rx_range_ecef_norm = np.linalg.norm(sv_rx_range_ecef, axis=1)
    geometry_matrix = sv_rx_range_ecef / sv_rx_range_ecef_norm[:, np.newaxis]
    ones_column = np.ones((geometry_matrix.shape[0], 1))
    geometry_matrix = np.hstack((geometry_matrix, ones_column))
    return geometry_matrix


def get_geometry_matrix(pd_gnss_raw: pd.DataFrame, pd_approx_pos: pd.DataFrame) -> pd.DataFrame:
    """
    Computes geometry matrices using dataframes.
    :param pd_gnss_raw: GNSS raw dataframe from BITS parser
    :param pd_approx_pos: GNSS pvt dataframe (at least "time", "x_rx_m", "y_rx_m", "z_rx_m")
    :return: pd_gnss_pvt like dataframe with geometry matrices
    """
    if {"corr_pr_m"}.issubset(pd_gnss_raw.columns):
        pr_column_name = "corr_pr_m"
    else:
        pr_column_name = "pr_m"
    # Initialization of the output columns
    dict_geometry_matrix = {
        "time": [],
        'pr_m': [],
        'doppler_hz': [],
        "sv_id": [],
        "gnss_id": [],
        "geometry_matrix": [],
    }

    timestamp_list = pd_gnss_raw["time"].unique().tolist()

    for timestamp in timestamp_list:
        if timestamp in pd_approx_pos['time'].values:
            pd_gnss_raw_at_timestamp = pd_gnss_raw[pd_gnss_raw["time"] == timestamp]
            pd_approx_pos_at_timestamp = pd_approx_pos[pd_approx_pos["time"] == timestamp]
            if len(pd_approx_pos_at_timestamp) != 1:
                txt = f"Position estimate has several possibilities at timestamp{timestamp}. Using first instance to compute geometry matrix."
                warnings.warn(txt, UserWarning)
            pd_ser_approx_pos_at_timestamp = pd_approx_pos_at_timestamp.iloc[0]

            dict_geometry_matrix["time"].append(pd_gnss_raw_at_timestamp["time"].iloc[0])
            dict_geometry_matrix[pr_column_name].append(pd_gnss_raw_at_timestamp[pr_column_name].to_numpy())
            dict_geometry_matrix["doppler_hz"].append(pd_gnss_raw_at_timestamp["doppler_hz"].to_numpy())
            dict_geometry_matrix["sv_id"].append(pd_gnss_raw_at_timestamp["sv_id"].tolist())
            dict_geometry_matrix["gnss_id"].append(pd_gnss_raw_at_timestamp["gnss_id"].tolist())

            sv_position_x = pd_gnss_raw_at_timestamp["x_sv_m"].to_numpy()
            sv_position_y = pd_gnss_raw_at_timestamp["y_sv_m"].to_numpy()
            sv_position_z = pd_gnss_raw_at_timestamp["z_sv_m"].to_numpy()
            sv_position = np.array([sv_position_x, sv_position_y, sv_position_z]).transpose()

            rx_pos = np.array([pd_ser_approx_pos_at_timestamp["x_rx_m"], pd_ser_approx_pos_at_timestamp["y_rx_m"],
                      pd_ser_approx_pos_at_timestamp["z_rx_m"]])

            geometry_matrix = compute_geometry_matrix(sv_position, rx_pos)
            dict_geometry_matrix["geometry_matrix"].append(geometry_matrix)

        else:
            txt = f"Cannot build geometry at timestamp {timestamp}, no RX position available"
            warnings.warn(txt)

    pd_geometry_matrix = pd.DataFrame(dict_geometry_matrix)
    return pd_geometry_matrix


def _build_init_pd_gnss_pvt(pd_gnss_raw: pd.DataFrame, init_pvt: Tuple[float, float, float]=(0, 0, 0)) -> pd.DataFrame:
    """
    Builds a pd_gnss_pvt dataframe compatible with pd_gnss_raw with "init_pvt" as positions
    :param pd_gnss_raw: GNSS raw dataframe from BITS parser
    :param init_pvt: position to use in ECEF (meters)
    :return: GNSS pvt dataframe
    """
    dict_init_gnss_pvt = {
        "time": [],
        "x_rx_m": [],
        "y_rx_m": [],
        "z_rx_m": [],
        "b_rx_m": [],
        "vx_rx_mps": [],
        "vy_rx_mps": [],
        "vz_rx_mps": [],
        "vb_rx_mps": [],
    }

    timestamp_list = pd_gnss_raw["time"].unique().tolist()

    for timestamp in timestamp_list:
        dict_init_gnss_pvt["time"].append(timestamp)
        dict_init_gnss_pvt["x_rx_m"].append(float(init_pvt[0]))
        dict_init_gnss_pvt["y_rx_m"].append(float(init_pvt[1]))
        dict_init_gnss_pvt["z_rx_m"].append(float(init_pvt[2]))
        dict_init_gnss_pvt["b_rx_m"].append(0.0)
        dict_init_gnss_pvt["vx_rx_mps"].append(0.0)
        dict_init_gnss_pvt["vy_rx_mps"].append(0.0)
        dict_init_gnss_pvt["vz_rx_mps"].append(0.0)
        dict_init_gnss_pvt["vb_rx_mps"].append(0.0)

    pd_init_gnss_pvt = pd.DataFrame(dict_init_gnss_pvt)
    return pd_init_gnss_pvt


def ordinary_least_square(Y: np.array, G: np.array) -> np.array:
    """
    Performs a simple Ordinary Least Square.
    sources:    https://gssc.esa.int/navipedia/index.php?title=Weighted_Least_Square_Solution_(WLS)
                https://gssc.esa.int/navipedia/index.php?title=Best_Linear_Unbiased_Minimum-Variance_Estimator_(BLUE)
    Y = G @ X
    => X = (Gt@G)^-1@Gt@Y
    :param Y: measurements
    :param G: Geometry matrix
    :return: estimates
    """
    estimate = G.transpose() @ G
    try:
        estimate = np.linalg.inv(estimate)
    except np.linalg.LinAlgError:
        txt = f"Matrix is singular :("
        raise PositionEstimationError(txt)

    estimate = estimate @ G.transpose() @ Y

    return estimate


def compute_position_estimate(pseudorange: np.array, geometry_matrix: np.array) -> np.array:
    """
    Performs a simple Ordinary Least Square to compute position estimate.
    sources:    https://gssc.esa.int/navipedia/index.php?title=Weighted_Least_Square_Solution_(WLS)
                https://gssc.esa.int/navipedia/index.php?title=Best_Linear_Unbiased_Minimum-Variance_Estimator_(BLUE)
    :param pseudorange: Pseudoranges (m). Array with dim==1
    :param geometry_matrix: Geometry matrix built with compute_geometry_matrix. Shape[0] must be the same length as
    pseudorange.
    :return: Position estimate (same length as geometry_matrix.shape[0], usually (x_ecef, y_ecef, z_ecef, b_ecef)
    """
    # Check pseudorange shape
    if pseudorange.shape[1] != 1:
        pseudorange.reshape(-1, 1)

    # Check if the number of satellites in the geometry matrix is the same as the number of pseudoranges
    if pseudorange.shape[0] != geometry_matrix.shape[0]:
        txt = f"Pseudorange matrix is not the same length as the geometry matrix ({pseudorange.shape[0]}, {geometry_matrix.shape[0]})."
        raise PositionEstimationError(txt)

    # Check if number of pseudoranges is enough to compute position (typically at least 4)
    if geometry_matrix.shape[1] > geometry_matrix.shape[0]:
        txt = f"Not enough satellites in view (need at least {geometry_matrix.shape[1]} found {geometry_matrix.shape[0]})."
        raise PositionEstimationError(txt)

    estimate = ordinary_least_square(pseudorange, geometry_matrix)

    return estimate


def compute_speed_estimate(pr_rate: np.array, geometry_matrix: np.array, sv_speed: np.array) -> np.array:
    """
    Performs a simple Ordinary Least Square to compute speed estimate.
    sources:    https://gssc.esa.int/navipedia/index.php?title=Code_Based_Positioning_(SPS)
                https://gssc.esa.int/navipedia/index.php?title=Parameters_adjustment
    :param pr_rate: Pseudorange rates (m/s). Array with dim==1
    :param geometry_matrix: Geometry matrix built with compute_geometry_matrix. Shape[0] must be the same length as
    pr_rate.
    :param sv_speed: Satellite speed in ECEF (m/s)
    :return: Speed estimate (same length as geometry_matrix.shape[0], usually (vx_ecef, vy_ecef, vz_ecef, vb_ecef)
    """
    # Check pseudorange rate shape
    if pr_rate.shape[1] != 1:
        pr_rate.reshape(-1, 1)

    # Check if the number of satellites in the geometry matrix is the same as the number of pseudorange rates
    if pr_rate.shape[0] != geometry_matrix.shape[0]:
        txt = f"Pseudorange rate matrix is not the same length as the geometry matrix ({pr_rate.shape[0]}, {geometry_matrix.shape[0]})."
        raise PositionEstimationError(txt)

    # Check if number of pseudoranges is enough to compute position (typically at least 4)
    if geometry_matrix.shape[1] > geometry_matrix.shape[0]:
        txt = f"Not enough satellites in view (need at least {geometry_matrix.shape[1]} found {geometry_matrix.shape[0]})."
        raise PositionEstimationError(txt)

    sv_relative_speed = np.sum(sv_speed * geometry_matrix[:, :-1], axis=1).reshape(-1, 1)
    corrected_pr_rate = pr_rate + sv_relative_speed

    estimate = ordinary_least_square(corrected_pr_rate, geometry_matrix)

    return estimate


def get_approx_position_estimate(pd_gnss_raw: pd.DataFrame, pd_gnss_approx_pvt: pd.DataFrame=None,
                                 approx_pvt: Tuple[float, float, float]=(0, 0, 0), convergence_tolerance=1e-7,
                                 max_iteration: int=10) -> pd.DataFrame:
    """
    Computes position without any corrections.
    sources:    https://gssc.esa.int/navipedia/index.php?title=Code_Based_Positioning_(SPS)
                https://gssc.esa.int/navipedia/index.php?title=Parameters_adjustment
    :param pd_gnss_raw: GNSS raw dataframe from BITS parser
    :param pd_gnss_approx_pvt: GNSS pvt dataframe with approximate pvt
    :param approx_pvt: Position (ECEF meters) at initialization (default -> centre of earth)
    :param convergence_tolerance: min acceptable position difference between two iterations
    :param max_iteration: GNSS pvt dataframe
    :return: GNSS pvt dataframe
    """
    if {"corr_pr_m"}.issubset(pd_gnss_raw.columns):
        pr_column_name = "corr_pr_m"
    else:
        pr_column_name = "pr_m"
    if pd_gnss_approx_pvt is None:
        # Build the first iteration of GNSS pvt dataframe
        pd_gnss_approx_pvt = _build_init_pd_gnss_pvt(pd_gnss_raw, init_pvt=approx_pvt)
    pd_gnss_approx_pvt["ols_convergence_m"] = -1.0

    for i in range(max_iteration):
        for index, row in pd_gnss_approx_pvt.iterrows(): # Loop over all timestamp
            if row["ols_convergence_m"] == -1.0 or row["ols_convergence_m"] > convergence_tolerance: # Convergence check
                timestamp = row["time"]
                if timestamp in pd_gnss_raw["time"].values:
                    # Get all raw measurements at timestamp
                    pd_gnss_raw_at_timestamp = pd_gnss_raw[pd_gnss_raw["time"] == timestamp].copy()

                    # Build SV position matrix
                    np_sv_position = pd_gnss_raw_at_timestamp[["x_sv_m", "y_sv_m", "z_sv_m"]].to_numpy()

                    # Build RX position matrix
                    np_rx_pos = np.array([row["x_rx_m"], row["y_rx_m"], row["z_rx_m"]])

                    # Compute pr_delta_m
                    pd_gnss_raw_at_timestamp["sv_range_m"] = pd_gnss_raw_at_timestamp.apply(
                        lambda raw_row: np.linalg.norm(np_rx_pos - np.array([raw_row["x_sv_m"], raw_row["y_sv_m"],
                                                                             raw_row["z_sv_m"]])), axis=1)
                    pd_gnss_raw_at_timestamp["pr_delta_m"] = (pd_gnss_raw_at_timestamp[pr_column_name] - row["b_rx_m"]
                                                        - pd_gnss_raw_at_timestamp["sv_range_m"])
                    np_pseudorange = pd_gnss_raw_at_timestamp["pr_delta_m"].to_numpy().reshape(-1, 1)

                    # Compute geometry matrix
                    np_geometry_matrix = compute_geometry_matrix(np_sv_position, np_rx_pos)

                    # Compute delta position estimate
                    try:
                        np_estimate_delta = compute_position_estimate(np_pseudorange, np_geometry_matrix)

                        # Compute position difference since last iteration
                        convergence = np.linalg.norm(np_estimate_delta)

                        # Update position estimate
                        pd_gnss_approx_pvt.at[index, "x_rx_m"] += np_estimate_delta[0][0]
                        pd_gnss_approx_pvt.at[index, "y_rx_m"] += np_estimate_delta[1][0]
                        pd_gnss_approx_pvt.at[index, "z_rx_m"] += np_estimate_delta[2][0]
                        pd_gnss_approx_pvt.at[index, "b_rx_m"] += np_estimate_delta[3][0]
                        pd_gnss_approx_pvt.at[index, "ols_convergence_m"] = abs(convergence)
                    except PositionEstimationError as e:
                        convergence = -1
                        txt = f"Cannot compute position at timestamp {timestamp}: {e}"
                        warnings.warn(txt)
                        pd_gnss_approx_pvt.loc[index, ["x_rx_m", "y_rx_m", "z_rx_m", "b_rx_m"]] = None
                        pd_gnss_approx_pvt.at[index, "ols_convergence_m"] = -1

                    # Compute speed estimate
                    if (convergence != -1 and abs(convergence) < convergence_tolerance
                            and {"vx_sv_mps", "vy_sv_mps", "vz_sv_mps"}.issubset(pd_gnss_raw_at_timestamp.columns)): # Do it only once at the end
                        # Build doppler and satellite speed arrays
                        pr_rate = pd_gnss_raw_at_timestamp["pr_rate_mps"].to_numpy().reshape(-1, 1)
                        sv_speed = pd_gnss_raw_at_timestamp[["vx_sv_mps", "vy_sv_mps", "vz_sv_mps"]].to_numpy()
                        try:
                            np_speed_estimate = compute_speed_estimate(pr_rate, np_geometry_matrix, sv_speed)
                            pd_gnss_approx_pvt.at[index, "vx_rx_mps"] = np_speed_estimate[0][0]
                            pd_gnss_approx_pvt.at[index, "vy_rx_mps"] = np_speed_estimate[1][0]
                            pd_gnss_approx_pvt.at[index, "vz_rx_mps"] = np_speed_estimate[2][0]
                            pd_gnss_approx_pvt.at[index, "vb_rx_mps"] = np_speed_estimate[3][0]
                        except PositionEstimationError as e:
                            txt = f"Cannot compute speed at timestamp {timestamp}: {e}"
                            warnings.warn(txt)
                            pd_gnss_approx_pvt.loc[index, ["vx_rx_mps", "vy_rx_mps", "vz_rx_mps", "vb_rx_mps"]] = None

        print(i)
        if (pd_gnss_approx_pvt['ols_convergence_m'] < convergence_tolerance).all(): # Check if convergence is satisfactory
            break

    # Detect position convergence failures
    for _, row in pd_gnss_approx_pvt[pd_gnss_approx_pvt["ols_convergence_m"] > convergence_tolerance].iterrows():
        txt = f"Position did not converge at timestamp {row['time']}. Convergence = {row['ols_convergence_m']}"
        warnings.warn(txt)

    # Add WGS coordinates
    pd_gnss_approx_pvt[['lat', 'lon', 'alt']] = (pd_gnss_approx_pvt.apply(
        lambda pvt_row: pd.Series(ecef_to_wgs(pvt_row["x_rx_m"], pvt_row["y_rx_m"], pvt_row["z_rx_m"]))
        if pvt_row["ols_convergence_m"]!=-1 else pd.Series((None, None, None)), axis=1))

    return  pd_gnss_approx_pvt


def get_sv_el_az(pd_gnss_raw: pd.DataFrame, pd_gnss_pvt: pd.DataFrame) -> pd.DataFrame:
    """
    Computes elevations and azimuth of satellite vehicles in pd_gnss_raw at estimated position from pd_gnss_pvt
    :param pd_gnss_raw: GNSS raw dataframe
    :param pd_gnss_pvt: GNSS pvt dataframe
    :return: GNSS raw dataframe
    """
    timestamp_list = pd_gnss_raw["time"].unique().tolist()
    pd_gnss_raw["elevation_rad"] = None
    pd_gnss_raw["azimuth_rad"] = None

    for timestamp in timestamp_list:# Loop over all timestamp
        if timestamp in pd_gnss_pvt["time"].values:
            # Get all raw measurements at timestamp
            pd_gnss_raw_at_timestamp = pd_gnss_raw[pd_gnss_raw["time"] == timestamp].copy()
            pd_gnss_pvt_at_timestamp = pd_gnss_pvt[pd_gnss_pvt["time"] == timestamp]
            if len(pd_gnss_pvt_at_timestamp) != 1:
                txt = f"Position estimate has several possibilities at timestamp {timestamp}. Using first instance to compute az & el."
                warnings.warn(txt, UserWarning)
            pd_ser_gnss_pvt_at_timestamp = pd_gnss_pvt_at_timestamp.iloc[0]

            # Build RX & SV position
            arr_rx_position = np.array([pd_ser_gnss_pvt_at_timestamp["x_rx_m"], pd_ser_gnss_pvt_at_timestamp["y_rx_m"],
                                     pd_ser_gnss_pvt_at_timestamp["z_rx_m"]])

            arr_sv_position = pd_gnss_raw_at_timestamp[["x_sv_m", "y_sv_m", "z_sv_m"]].to_numpy()

            # Build sv ecef geometry matrix
            geometry_matrix_ecef = compute_geometry_matrix(arr_sv_position, arr_rx_position)

            # Convert geometry matrix to azimuth and elevation
            geometry_matrix_ecef = geometry_matrix_ecef[:, :-1]
            geometry_matrix_enu = ecef_to_enu(arr_rx_position, geometry_matrix_ecef)
            geometry_matrix_polar = enu_to_spheric(geometry_matrix_enu)

            # Update elevation and azimuth
            pd_gnss_raw_at_timestamp[["elevation_rad", "azimuth_rad"]] = geometry_matrix_polar[:, 1:3]
            pd_gnss_raw_at_timestamp["elevation_rad"] = pd_gnss_raw_at_timestamp["elevation_rad"].apply(
                lambda el: math.pi - el if el > math.pi/2 else el)
            pd_gnss_raw.loc[pd_gnss_raw_at_timestamp.index, ["elevation_rad", "azimuth_rad"]] = \
            pd_gnss_raw_at_timestamp[["elevation_rad", "azimuth_rad"]]
        else:
            txt = f"While computing SV elevation & azimuth, no position found at timestamp {timestamp}"
            warnings.warn(txt)

    return pd_gnss_raw


def _correct_rx_clock(pd_gnss_raw: pd.DataFrame, pd_gnss_pvt: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Use the computed receiver clock offset ("b_rx_m") to correct pseudoranges and timestamps.
    :param pd_gnss_raw: GNSS raw dataframe
    :param pd_gnss_pvt: GNSS pvt dataframe
    :return: corrected GNSS raw dataframe, corrected GNSS pvt dataframe
    """
    # Check for already existing corrections
    if not {"corr_time"}.issubset(pd_gnss_raw.columns):
        pd_gnss_raw["corr_time"] = None
    if not {"corr_time"}.issubset(pd_gnss_pvt.columns):
        pd_gnss_pvt["corr_time"] = None
    if not {"corr_pr_m"}.issubset(pd_gnss_raw.columns):
        pd_gnss_raw["corr_pr_m"] = pd_gnss_raw["pr_m"]

    timestamp_list = pd_gnss_raw["time"].unique().tolist()
    for timestamp in timestamp_list:# Loop over all timestamp
        if timestamp in pd_gnss_pvt["time"].values:
            # Get all raw measurements at timestamp
            pd_gnss_raw_at_timestamp = pd_gnss_raw[pd_gnss_raw["time"] == timestamp].copy()
            pd_gnss_pvt_at_timestamp = pd_gnss_pvt[pd_gnss_pvt["time"] == timestamp].copy()
            if len(pd_gnss_pvt_at_timestamp) != 1:
                txt = f"Position estimate has several possibilities at timestamp {timestamp}. Using first instance to compute az & el."
                warnings.warn(txt, UserWarning)
            pd_ser_gnss_pvt_at_timestamp = pd_gnss_pvt_at_timestamp.iloc[0]
            corrected_timestamp = pd_ser_gnss_pvt_at_timestamp["time"] + pd.Timedelta(seconds=pd_ser_gnss_pvt_at_timestamp["b_rx_m"] / const.C)
            pd_gnss_raw.loc[pd_gnss_raw_at_timestamp.index, ["corr_time"]] = corrected_timestamp
            pd_gnss_raw.loc[pd_gnss_raw_at_timestamp.index, ["corr_pr_m"]] -= pd_ser_gnss_pvt_at_timestamp["b_rx_m"]
            pd_gnss_pvt.loc[pd_gnss_pvt_at_timestamp.index, ["corr_time"]] = corrected_timestamp

    return pd_gnss_raw, pd_gnss_pvt


def get_position_estimate(pd_gnss_raw: pd.DataFrame, pd_ephemeris: pd.DataFrame = None, ephem_filepath: str = None,
                          approx_pvt: Tuple[float, float, float]=(0, 0, 0)) -> pd.DataFrame:
    """
    Computes position estimate using OLS and clock and atmospheric corrections.
    source: https://gssc.esa.int/navipedia/index.php?title=GNSS_Measurements_Modelling
    :param pd_gnss_raw: GNSS raw dataframe from BITS parser
    :param pd_ephemeris: ephemeris dataframe from BITS parser
    :param ephem_filepath: Path of a rinex nav file
    :param approx_pvt: Position (ECEF meters) at initialization (default -> centre of earth)
    :return: GNSS pvt dataframe, corrected GNSS raw dataframe
    """
    raw_required_columns = ["time", "pr_m", "gnss_id", "sv_id"]
    if not check_dataframe(pd_gnss_raw, raw_required_columns):
        warnings.warn("Missing columns in pd_gnss_raw, cannot compute position.")
        return pd_gnss_raw

    # Get satellite vehicle positions
    pd_gnss_raw = get_sv_states(pd_gnss_raw, pd_ephemeris, ephem_filepath=ephem_filepath)

    # Correct satellite clock errors
    pd_gnss_raw = get_clock_corrections(pd_gnss_raw)

    # Get a first position estimate
    pd_gnss_pvt = get_approx_position_estimate(pd_gnss_raw, approx_pvt=approx_pvt, convergence_tolerance=10000)

    # Correct receiver clock and recompute SV states
    pd_gnss_raw, pd_gnss_pvt = _correct_rx_clock(pd_gnss_raw, pd_gnss_pvt)
    pd_gnss_raw = get_sv_states(pd_gnss_raw, pd_ephemeris, ephem_filepath=ephem_filepath)

    # Correct atmospheric error
    pd_gnss_raw = get_sv_el_az(pd_gnss_raw, pd_gnss_pvt)
    pd_gnss_raw = get_atmospheric_corrections(pd_gnss_raw, pd_gnss_pvt)

    # Compute a corrected position estimate
    pd_gnss_pvt = get_approx_position_estimate(pd_gnss_raw, pd_gnss_approx_pvt=pd_gnss_pvt, convergence_tolerance=100)

    # Correct receiver clock and recompute SV states
    pd_gnss_raw, pd_gnss_pvt = _correct_rx_clock(pd_gnss_raw, pd_gnss_pvt)
    pd_gnss_raw = get_sv_states(pd_gnss_raw, pd_ephemeris, ephem_filepath=ephem_filepath)

    # Compute a final position estimate
    pd_gnss_pvt = get_approx_position_estimate(pd_gnss_raw, pd_gnss_approx_pvt=pd_gnss_pvt)

    return pd_gnss_pvt, pd_gnss_raw

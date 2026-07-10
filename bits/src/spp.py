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
from bits.src.convert.space_conversion import ecef_to_wgs, ecef_to_enu, enu_to_spheric, enu_to_ecef, rotate_ecef
from bits.src.corrections import get_clock_corrections, get_atmospheric_corrections
from bits.src.sv_model import get_sv_states
from bits.src import const
from bits.src.utils import check_dataframe
from bits.src.reference_frame_object import GnssTimestamp
from tqdm import tqdm

class PositionEstimationError(Exception):
    """Exception raised for errors during position estimation."""
    pass


def compute_geometry_matrix(sv_position: np.ndarray, rx_pos: np.ndarray, sv_const_list: np.ndarray|None=None) \
        -> np.ndarray:
    """
    Computes a geometry matrix between two position in ECEF.
    :param sv_position: Satellite vehicule position in ecef (meters) np.Array([[X], [Y], [Z]]) (column)
    :param rx_pos: Receiver position in ecef (meters) np.Array([X, Y, Z]) (line)
    :param sv_const_list: List of constellation names associated with "sv_position"
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

    # Building clock bias part of the geometry matrix
    if sv_const_list is None:
        b = np.ones((geometry_matrix.shape[0], 1))
    elif len(sv_const_list) != geometry_matrix.shape[0]:
        txt = f"Please provide one constellation name per satellite. Expected {geometry_matrix.shape[0]}, received {len(sv_const_list)}."
        raise ValueError(txt)
    else:
        # For each specific unique gnss constellation, assign a b column filled with 1 for this constellation, else 0
        unique_gnss_const_list = np.unique(sv_const_list)
        b = np.zeros((geometry_matrix.shape[0], len(unique_gnss_const_list)))
        col_index = np.searchsorted(unique_gnss_const_list, sv_const_list)
        b[np.arange(geometry_matrix.shape[0]), col_index] = 1

    geometry_matrix = np.hstack((geometry_matrix, b))

    return geometry_matrix


def _build_init_pd_gnss_pvt(pd_gnss_raw: pd.DataFrame, init_pvt: tuple[float, float, float]=(0, 0, 0)) -> pd.DataFrame:
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


def weighted_least_square(Y: np.ndarray, G: np.ndarray, W: np.ndarray) \
        -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Performs a Weighted Least Square (WLS) estimation.
    source: https://gssc.esa.int/navipedia/index.php/Weighted_Least_Square_Solution_(WLS)

    Y = G @ X
    => X = (Gt @ W @ G)^-1 @ Gt @ W @ Y

    covX = (Gt @ W @ G)^-1

    :param Y: measurements
    :param G: Geometry matrix
    :param W: Weight matrix (inverse of covariance matrix of measurement noise)
    :return: X, covX, DOP, residuals
    """
    GtW = G.T @ W  # (m, n)
    GtWG = GtW @ G  # (m, m)
    GtWy = GtW @ Y  # (m,)

    try:
        cov_x = np.linalg.inv(GtWG)  # (m, m)
    except np.linalg.LinAlgError:
        txt = f"Matrix is singular :("
        raise PositionEstimationError(txt)

    x_hat = cov_x @ GtWy

    dop = np.sqrt(np.trace(cov_x))

    residuals = Y - G @ x_hat

    return x_hat, cov_x, dop, residuals


def compute_speed_estimate(pr_rate: np.ndarray, geometry_matrix: np.ndarray, sv_speed: np.ndarray,
                           weight_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Performs a simple Ordinary Least Square to compute speed estimate.
    sources:    https://gssc.esa.int/navipedia/index.php?title=Code_Based_Positioning_(SPS)
                https://gssc.esa.int/navipedia/index.php?title=Parameters_adjustment
    :param pr_rate: Pseudorange rates (m/s). Array with dim==1
    :param geometry_matrix: Geometry matrix built with compute_geometry_matrix. Shape[0] must be the same length as
    pr_rate.
    :param sv_speed: Satellite speed in ECEF (m/s)
    :param weight_matrix: Weight matrix (inverse of covariance matrix of measurement noise)
    :return: Speed estimate, covariance matrix, Dilution Of Precision
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

    sv_relative_speed = np.sum(sv_speed * geometry_matrix[:, :3], axis=1).reshape(-1, 1) # TODO sum or norm ?
    corrected_pr_rate = pr_rate + sv_relative_speed

    v_hat, cov_v, dop, residuals = weighted_least_square(corrected_pr_rate, geometry_matrix, weight_matrix)

    return v_hat, cov_v, dop, residuals


def get_approx_position_estimate(pd_gnss_raw: pd.DataFrame, pd_gnss_approx_pvt: pd.DataFrame=None,
                                 approx_pvt: tuple[float, float, float]=(0, 0, 0), convergence_tolerance:float=1e-7,
                                 max_iteration: int=10, weights_column:str="weight") -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Computes position without any corrections.
    sources:    https://gssc.esa.int/navipedia/index.php?title=Code_Based_Positioning_(SPS)
                https://gssc.esa.int/navipedia/index.php?title=Parameters_adjustment
    :param pd_gnss_raw: GNSS raw dataframe from BITS parser
    :param pd_gnss_approx_pvt: GNSS pvt dataframe with approximate pvt
    :param approx_pvt: Position (ECEF meters) at initialization (default -> centre of earth)
    :param convergence_tolerance: Min acceptable position difference between two iterations
    :param max_iteration: Maximum allowed iterations
    :param weights_column: Column name of satellite weights (if any). Weight should be the inverse of measurement noise.
    :return: GNSS pvt dataframe, GNSS raw dataframe (with residuals and geometry matrix)
    """
    # Use corrected pseudorange if corrections already applied
    if {"corr_pr_m"}.issubset(pd_gnss_raw.columns):
        pr_column_name = "corr_pr_m"
    else:
        pr_column_name = "pr_m"

    # Build the first iteration of GNSS pvt dataframe
    if pd_gnss_approx_pvt is None:
        pd_gnss_approx_pvt = _build_init_pd_gnss_pvt(pd_gnss_raw, init_pvt=approx_pvt)
    pvt_time_list = pd_gnss_approx_pvt["time"].tolist()

    # Loop over all timestamp
    approx_pvt_serie_list = []
    raw_pd_list = []
    for raw_time, group in tqdm(pd_gnss_raw.groupby("time"), total=len(pd_gnss_raw["time"].unique()),
                         desc="Computing position"):
        # Find closest RX position initialization
        pvt_closest_time = min(pvt_time_list, key=lambda d: abs(d - raw_time))
        pvt_at_timestamp_serie = pd_gnss_approx_pvt[pd_gnss_approx_pvt["time"] == pvt_closest_time].iloc[0]

        # Compute position and speed estimate
        group_gnss_raw, serie_gnss_approx_pvt = (
            window_approx_position_estimate(group, pvt_at_timestamp_serie,
                                            convergence_tolerance=convergence_tolerance,
                                            max_iteration=max_iteration, weights_column=weights_column,
                                            pr_column_name=pr_column_name))
        raw_pd_list.append(group_gnss_raw)
        approx_pvt_serie_list.append(serie_gnss_approx_pvt)

    # Merge all timestamps
    pd_gnss_raw = pd.concat(raw_pd_list, ignore_index=True)
    pd_gnss_approx_pvt = pd.DataFrame(approx_pvt_serie_list)

    return  pd_gnss_approx_pvt, pd_gnss_raw

def window_approx_position_estimate(group_gnss_raw: pd.DataFrame, serie_gnss_approx_pvt: pd.Series,
                                    convergence_tolerance:float=1e-7, max_iteration: int=10, weights_column:str="weight",
                                    pr_column_name:str="corr_pr_m") -> tuple[pd.DataFrame, pd.Series]:
    """
    Computes position and speed at a specific timestamp without any corrections.

    sources:    https://gssc.esa.int/navipedia/index.php?title=Code_Based_Positioning_(SPS)
                https://gssc.esa.int/navipedia/index.php?title=Parameters_adjustment

    :param group_gnss_raw: GNSS raw dataframe at a specific timestamp
    :param serie_gnss_approx_pvt: GNSS pvt Serie with position at initialization
    :param convergence_tolerance: Min acceptable position difference between two iterations
    :param max_iteration: Maximum allowed iterations
    :param weights_column: Column name of satellite weights (if any). Weight should be the inverse of measurement noise.
    :param pr_column_name: Column name of pseudoranges
    :return: GNSS raw dataframe, GNSS pvt dataframe
    """
    # Build SV position matrix
    np_sv_position = group_gnss_raw[["x_sv_m", "y_sv_m", "z_sv_m"]].to_numpy()

    # Build RX position matrix
    np_rx_pos = np.array(
        [serie_gnss_approx_pvt["x_rx_m"], serie_gnss_approx_pvt["y_rx_m"], serie_gnss_approx_pvt["z_rx_m"]])

    # Build pseudorange
    np_pr = group_gnss_raw[pr_column_name].to_numpy()

    # Build weight matrix
    if weights_column in group_gnss_raw.columns:
        w = group_gnss_raw[weights_column].to_numpy()
    else:
        w = np.ones_like(np_pr)
    np_weight = np.diag(w.ravel())

    # Build GNSS constellations list
    sv_const_list = group_gnss_raw["gnss_id"].to_numpy()

    # Compute position using Gauss-Newton iterative algorithm
    np_rx_pos, np_geometry_matrix, cov, dop, residuals = (
        gauss_newton(np_pr, np_weight, np_rx_pos, np_sv_position, sv_const_list, convergence_tolerance, max_iteration))

    unique_gnss_const_list = np.unique(sv_const_list)

    # Add ECEF coordinates
    serie_gnss_approx_pvt["x_rx_m"] = float(np_rx_pos[0])
    serie_gnss_approx_pvt["y_rx_m"] = float(np_rx_pos[1])
    serie_gnss_approx_pvt["z_rx_m"] = float(np_rx_pos[2])

    # Add clock bias
    if len(unique_gnss_const_list) == 1:
        serie_gnss_approx_pvt["b_rx_m"] = float(np_rx_pos[3])
    else:
        if "b_rx_m" in serie_gnss_approx_pvt:
            serie_gnss_approx_pvt.drop("b_rx_m", inplace=True)
        i=3
        for constellation in unique_gnss_const_list:
            b_name = f"b{constellation}_rx_m"
            serie_gnss_approx_pvt[b_name] = float(np_rx_pos[i])
            i+=1

    # Add sv elevation and azimuth
    group_gnss_raw = get_sv_el_az(group_gnss_raw,  serie_gnss_approx_pvt.to_frame().T)

    # Add WGS coordinates
    if not np.isnan(np_rx_pos).any():
        serie_gnss_approx_pvt['lat'], serie_gnss_approx_pvt['lon'], serie_gnss_approx_pvt['alt'] \
            = ecef_to_wgs(float(np_rx_pos[0]), float(np_rx_pos[1]), float(np_rx_pos[2]))
    else:
        serie_gnss_approx_pvt['lat'], serie_gnss_approx_pvt['lon'], serie_gnss_approx_pvt['alt'] = (None, None, None)

    # Add variance-covariance matrix
    serie_gnss_approx_pvt["cov_xx_rx_m"] = float(cov[0][0])
    serie_gnss_approx_pvt["cov_yx_rx_m"] = float(cov[0][1])
    serie_gnss_approx_pvt["cov_zx_rx_m"] = float(cov[0][2])
    serie_gnss_approx_pvt["cov_yy_rx_m"] = float(cov[1][1])
    serie_gnss_approx_pvt["cov_zy_rx_m"] = float(cov[1][2])
    serie_gnss_approx_pvt["cov_zz_rx_m"] = float(cov[2][2])

    # Clock bias
    if len(unique_gnss_const_list) == 1:
        serie_gnss_approx_pvt["cov_bx_rx_m"] = float(cov[0][3])
        serie_gnss_approx_pvt["cov_by_rx_m"] = float(cov[1][3])
        serie_gnss_approx_pvt["cov_bz_rx_m"] = float(cov[2][3])
        serie_gnss_approx_pvt["cov_bb_rx_m"] = float(cov[3][3])
    else:
        cov_name_list = ["x", "y", "z"]
        i = 3
        for constellation in unique_gnss_const_list:
            cov_name_list.append(f"b{constellation}")
            for j in range(cov.shape[0] - (3+3-i)):
                cov_name = f"cov_{cov_name_list[i]}{cov_name_list[j]}_rx_m"
                serie_gnss_approx_pvt[cov_name] = float(cov[j][i])
            i += 1


    # Add residuals
    group_gnss_raw["residuals_m"] = residuals

    # Add steering vectors
    group_gnss_raw["e_x"] = np_geometry_matrix[:, 0]
    group_gnss_raw["e_y"] = np_geometry_matrix[:, 1]
    group_gnss_raw["e_z"] = np_geometry_matrix[:, 2]
    # Add clock bias
    if len(unique_gnss_const_list) == 1:
        serie_gnss_approx_pvt["e_b"] = np_geometry_matrix[:, 3]
    else:
        i=3
        for constellation in unique_gnss_const_list:
            b_name = f"e_b{constellation}"
            serie_gnss_approx_pvt[b_name] = np_geometry_matrix[:, i]
            i+=1

    # Correct time
    if "corr_pr_m" not in group_gnss_raw.columns:
        group_gnss_raw["corr_pr_m"] = group_gnss_raw["pr_m"]
    b_rx_m = np_geometry_matrix[:, 3:] @ np_rx_pos[3:]
    group_gnss_raw["corr_pr_m"] -= b_rx_m
    # GnssTimestamp object are awful to use with pandas...
    # In a future version, GnssTimestamp will be switched back to pd.Timestamp
    timestamp_column = group_gnss_raw["time"].apply(lambda timestamp: timestamp.timestamp_pd)
    group_gnss_raw["corr_time"] = (timestamp_column + pd.to_timedelta(b_rx_m / const.C, unit="s")).apply(lambda timestamp: GnssTimestamp.from_pd_timestamp(timestamp))

    # Add Dilution Of Precision
    serie_gnss_approx_pvt["DOP"] = float(dop)

    # Compute speed estimate
    if not np.isnan(np_rx_pos).any() and {"vx_sv_mps", "vy_sv_mps", "vz_sv_mps"}.issubset(group_gnss_raw.columns):
        # Build pseudorange rate and satellite speed arrays
        pr_rate = group_gnss_raw["pr_rate_mps"].to_numpy().reshape(-1, 1)
        sv_speed = group_gnss_raw[["vx_sv_mps", "vy_sv_mps", "vz_sv_mps"]].to_numpy()

        # Compute speed using WLS
        try:
            np_speed_estimate, cov, _ , residuals_mps = (
                compute_speed_estimate(pr_rate, np_geometry_matrix, sv_speed, np_weight))
        except PositionEstimationError as e:
            txt = f"Cannot compute speed at timestamp {serie_gnss_approx_pvt['time']}: {e}"
            warnings.warn(txt)
            np_speed_estimate = np.full_like(np_rx_pos, np.nan, dtype=float)
            residuals_mps = np.full_like(pr_rate, np.nan, dtype=float)
            cov = np.full((np_geometry_matrix.shape[1], np_geometry_matrix.shape[1]), np.nan, dtype=float)

        # Add ECEF coordinates
        serie_gnss_approx_pvt["vx_rx_mps"] = float(np_speed_estimate[0][0])
        serie_gnss_approx_pvt["vy_rx_mps"] = float(np_speed_estimate[1][0])
        serie_gnss_approx_pvt["vz_rx_mps"] = float(np_speed_estimate[2][0])

        # Add clock bias
        if len(unique_gnss_const_list) == 1:
            serie_gnss_approx_pvt["vb_rx_mps"] = float(np_speed_estimate[3])
        else:
            if "vb_rx_mps" in serie_gnss_approx_pvt:
                serie_gnss_approx_pvt.drop("vb_rx_mps", inplace=True)
            i = 3
            for constellation in unique_gnss_const_list:
                b_name = f"vb{constellation}_rx_mps"
                serie_gnss_approx_pvt[b_name] = float(np_speed_estimate[i])
                i += 1

        # Add variance-covariance matrix
        serie_gnss_approx_pvt["cov_vxvx_rx_mps"] = float(cov[0][0])
        serie_gnss_approx_pvt["cov_vyvx_rx_mps"] = float(cov[0][1])
        serie_gnss_approx_pvt["cov_vzvx_rx_mps"] = float(cov[0][2])
        serie_gnss_approx_pvt["cov_vyvy_rx_mps"] = float(cov[1][1])
        serie_gnss_approx_pvt["cov_vzvy_rx_mps"] = float(cov[1][2])
        serie_gnss_approx_pvt["cov_vzvz_rx_mps"] = float(cov[2][2])

        # Clock bias
        if len(unique_gnss_const_list) == 1:
            serie_gnss_approx_pvt["cov_vbvx_rx_mps"] = float(cov[0][3])
            serie_gnss_approx_pvt["cov_vbvy_rx_mps"] = float(cov[1][3])
            serie_gnss_approx_pvt["cov_vbvz_rx_mps"] = float(cov[2][3])
            serie_gnss_approx_pvt["cov_vbvb_rx_mps"] = float(cov[3][3])
        else:
            cov_name_list = ["vx", "vy", "vz"]
            i = 3
            for constellation in unique_gnss_const_list:
                cov_name_list.append(f"vb{constellation}")
                for j in range(cov.shape[0] - (3 + 3 - i)):
                    cov_name = f"cov_{cov_name_list[i]}{cov_name_list[j]}_rx_mps"
                    serie_gnss_approx_pvt[cov_name] = float(cov[j][i])
                i += 1

        # Add residuals
        group_gnss_raw["residuals_mps"] = residuals_mps

    return group_gnss_raw, serie_gnss_approx_pvt

def gauss_newton(np_pr:np.ndarray, np_weight:np.ndarray, np_rx_pos:np.ndarray, np_sv_position:np.ndarray,
                 sv_const_list:np.ndarray|None=None, convergence_tolerance:float=1e-7, max_iteration:int=10) \
        -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes position using Gauss Newton method.

    Gauss Newton is an iterative algorithm that computes an estimated divergence between an a priori RX position and the
    real RX position. Since the Geometry matrix depends on RX position, it must be recomputed at each iteration.

    At each iteration:
    d_pseudorange = pseudorange - clock_bias - range(X)
    d_X = WLS(d_pseudorange, G(X), W)
    X = X + dX

    sources:    https://gssc.esa.int/navipedia/index.php?title=Code_Based_Positioning_(SPS)
                https://gssc.esa.int/navipedia/index.php?title=Parameters_adjustment

    :param np_pr: Pseudorange matrix (m)
    :param np_weight: Weight matrix; Weight should be the inverse of variance-covariance matrix of measurement noise.
    :param np_rx_pos: ECEF RX position at initialization vector (m)
    :param np_sv_position: ECEF  SV position vector (m)
    :param sv_const_list: List of constellation names associated with "np_sv_position"
    :param convergence_tolerance: Min acceptable position difference between two iterations
    :param max_iteration: Maximum allowed iterations
    :return: ECEF position (m), geometry matrix, variance-covariance matrix, Dilution Of Precision, residuals
    """
    # Build RX initialization pos
    if sv_const_list is None:
        number_of_clock_bias = 1
    else:
        unique_gnss_const_list = np.unique(sv_const_list)
        number_of_clock_bias = len(unique_gnss_const_list)

    np_rx_pos = np.hstack([np_rx_pos, [0]*number_of_clock_bias]) # Add one clock bias estimate for each constellation


    for i in range(max_iteration):
        # Compute geometry matrix
        np_geometry_matrix = compute_geometry_matrix(np_sv_position, np_rx_pos[:3], sv_const_list)

        # Compute delta pseudorange
        sv_range_np = np.linalg.norm(np_rx_pos[:3] - np_sv_position, axis=1)
        np_delta_pr = np_pr - np_geometry_matrix[:, 3:] @ np_rx_pos[3:] -  sv_range_np
        np_delta_pr = np_delta_pr.reshape(-1, 1)

        # Compute delta position estimate
        try:
            np_estimate_delta, cov, dop, residuals = weighted_least_square(np_delta_pr, np_geometry_matrix, np_weight)

            # Update position estimate
            np_rx_pos += np_estimate_delta.ravel()

            # Check convergence
            if np.linalg.norm(np_estimate_delta) < convergence_tolerance:
                break
        except PositionEstimationError as e:
            txt = f"Cannot compute position: {e}"
            warnings.warn(txt)
            np_rx_pos = np.full_like(np_rx_pos, np.nan, dtype=float)
            cov = np.full((np_geometry_matrix.shape[1], np_geometry_matrix.shape[1]), np.nan, dtype=float)
            dop = np.nan
            residuals = np.full_like(np_delta_pr, np.nan, dtype=float)
            break

    return np_rx_pos, np_geometry_matrix, cov, dop, residuals



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


def get_position_estimate(pd_gnss_raw: pd.DataFrame, pd_ephemeris: pd.DataFrame = None, ephem_filepath: str = None,
                          approx_pvt: tuple[float, float, float]=(0, 0, 0), verbose=False) \
        -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Computes position estimate using OLS and clock and atmospheric corrections.
    source: https://gssc.esa.int/navipedia/index.php?title=GNSS_Measurements_Modelling
    :param pd_gnss_raw: GNSS raw dataframe from BITS parser
    :param pd_ephemeris: ephemeris dataframe from BITS parser
    :param ephem_filepath: Path of a rinex nav file
    :param approx_pvt: Position (ECEF meters) at initialization (default -> centre of earth)
    :return: GNSS pvt dataframe, corrected GNSS raw dataframe
    """
    if verbose:
        print("Computing position estimate...")

    raw_required_columns = ["time", "pr_m", "gnss_id", "sv_id"]
    if not check_dataframe(pd_gnss_raw, raw_required_columns):
        warnings.warn("Missing columns in pd_gnss_raw, cannot compute position.")
        return pd_gnss_raw

    if verbose:
        print("1/9: Finding satellites...")

    # Get satellite vehicle positions
    pd_gnss_raw = get_sv_states(pd_gnss_raw, pd_ephemeris, ephem_filepath=ephem_filepath)

    if verbose:
        print("2/9: Correcting satellite clock...")

    # Correct satellite clock errors
    pd_gnss_raw = get_clock_corrections(pd_gnss_raw)

    if verbose:
        print("3/9: Computing rough position estimate...")

    # Get a first position estimate
    pd_gnss_pvt, pd_gnss_raw = get_approx_position_estimate(pd_gnss_raw, approx_pvt=approx_pvt, convergence_tolerance=10000)

    if verbose:
        print("4/9: Correcting receiver clock...")

    # Recompute SV states
    if verbose:
        print("5/9: Finding satellites, again...")
    pd_gnss_raw = get_sv_states(pd_gnss_raw, pd_ephemeris, ephem_filepath=ephem_filepath)

    if verbose:
        print("6/9: Correcting atmospheric errors...")
    # Correct atmospheric error
    pd_gnss_raw = get_atmospheric_corrections(pd_gnss_raw, pd_gnss_pvt)

    if verbose:
        print("7/9: Computing a better position estimate...")
    # Compute a corrected position estimate
    pd_gnss_pvt, pd_gnss_raw = get_approx_position_estimate(pd_gnss_raw, pd_gnss_approx_pvt=pd_gnss_pvt, convergence_tolerance=100)

    if verbose:
        print("8/9: Correcting receiver clock and finding satellites, again...")
    # Recompute SV states
    pd_gnss_raw = get_sv_states(pd_gnss_raw, pd_ephemeris, ephem_filepath=ephem_filepath)

    if verbose:
        print("9/9: Computing final position estimate...")

    # Compute a final position estimate
    pd_gnss_pvt, pd_gnss_raw = get_approx_position_estimate(pd_gnss_raw, pd_gnss_approx_pvt=pd_gnss_pvt)

    return pd_gnss_pvt, pd_gnss_raw

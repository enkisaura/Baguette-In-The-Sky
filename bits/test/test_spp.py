#!/usr/bin/env python3

"""
Tests for single point positioning functions
"""

__authors__ = ("Enki SAURA")
__contact__ = ("esaura@ikosconsulting.com")
__copyright__ = "IKOS"
__date__ = "2025-06-06"
__version__ = "0.0.1"

import os
from bits.src.sv_model import get_sv_states
from bits.src.parsers import ephemeris, gnss_raw
from bits.src.spp import *
from bits.src.spp import _build_init_pd_gnss_pvt
from bits.src.convert.space_conversion import wgs_to_ecef, rotate_ecef, ecef_to_enu
from bits.src.plotter import plot

required_precision = 1  # m
required_precision_speed = 0.2 # m/s
az_el_required_precision = 1e-2  # rad
gt = (45.7615208,-1.1411692,0)
gt_speed = (0, 0, 0)

test_data_directory_path = os.path.join(os.getcwd(), "bits", "test", "test_data")
ephem_filepath = os.path.join(test_data_directory_path, "gnss_raw", "rinex_v2.rnx")
raw_filepath = os.path.join(test_data_directory_path, "gnss_raw", "rx1_1")
az_el_ephem_filepath = os.path.join(test_data_directory_path, "rinex_nav.rnx")
az_el_skydel_raw_directory_path = os.path.join(test_data_directory_path, "skydel_raw")

pd_ephemeris = ephemeris.rinex_nav(ephem_filepath)
pd_raw = gnss_raw.micdrop_raw(raw_filepath)
pd_gnss_raw = get_sv_states(pd_raw, pd_ephemeris)


def test_geometry_matrix(tolerance=1e-10):
    pd_approx_pos = _build_init_pd_gnss_pvt(pd_gnss_raw)
    pd_geometry_matrix = get_geometry_matrix(pd_gnss_raw, pd_approx_pos)
    assert isinstance(pd_geometry_matrix, pd.DataFrame) and not pd_geometry_matrix.empty
    for _, row in pd_geometry_matrix.iterrows():
        for h in row["geometry_matrix"]:
            h = h[:-1]
            assert abs((np.linalg.norm(h) - 1)) < tolerance, "Geometry matrix is not composed of unit vectors"

def test_approx_pos_estimate():
    tol = 100
    pd_gnss_pvt = get_approx_position_estimate(pd_gnss_raw, convergence_tolerance=tol)
    assert (pd_gnss_pvt['ols_convergence_m'] < tol).all(), "Position estimate did not converge"

def test_pos_estimate():
    gt_ecef = wgs_to_ecef(*gt)
    pd_gt = pd.DataFrame([gt], columns=['lat', 'lon', 'alt'])
    pd_gnss_pvt, _ = get_position_estimate(pd_raw, ephem_filepath=ephem_filepath)
    pd_gnss_pvt["np_rx_m"] = pd_gnss_pvt.apply(lambda row: np.array([row["x_rx_m"], row["y_rx_m"], row["z_rx_m"]]), axis=1)
    pd_gnss_pvt["np_rx_enu_m"] = pd_gnss_pvt.apply(lambda row: ecef_to_enu(gt_ecef, row["np_rx_m"]), axis=1)
    pd_gnss_pvt["h_error_m"] = pd_gnss_pvt.apply(lambda row: np.linalg.norm(row["np_rx_enu_m"][:2]), axis=1)
    m=plot(pd_gnss_pvt, plot_name="Estimates")
    plot(pd_gt, plot_name="Ground truth", m=m)
    mean_error = pd_gnss_pvt['h_error_m'].mean()
    max_error = pd_gnss_pvt['h_error_m'].max()
    txt = f"Position estimate does not meet the expected accuracy. Expected: {required_precision}m, estimated: mean {mean_error}m, max {max_error}m."
    assert (pd_gnss_pvt['h_error_m'] < required_precision).all(), txt
    pd_gnss_pvt["np_vrx_mps"] = pd_gnss_pvt.apply(lambda row: np.array([row["vx_rx_mps"], row["vy_rx_mps"], row["vz_rx_mps"]]),
                                               axis=1)
    pd_gnss_pvt["verror_mps"] = pd_gnss_pvt.apply(lambda row: np.linalg.norm(row["np_vrx_mps"]), axis=1)
    mean_error = pd_gnss_pvt['verror_mps'].mean()
    max_error = pd_gnss_pvt['verror_mps'].max()
    txt = f"Speed estimate does not meet the expected accuracy. Expected: {required_precision_speed}m/s, estimated: mean {mean_error}m/s, max {max_error}m/s."
    assert (pd_gnss_pvt['verror_mps'] < required_precision_speed).all(), txt

def test_azimuth_elevation():
    pd_az_el_raw = pd.DataFrame()
    for filename in os.listdir(az_el_skydel_raw_directory_path):
        raw_filepath = os.path.join(az_el_skydel_raw_directory_path, filename)
        pd_az_el_raw = pd.concat([pd_az_el_raw, gnss_raw.skydel_raw(raw_filepath).iloc[:2]], axis=0)
    pd_az_el_raw = pd_az_el_raw[pd_az_el_raw["gnss_id"] == "gps"].reset_index()
    pd_az_el_pvt = get_approx_position_estimate(pd_az_el_raw, convergence_tolerance=100)
    pd_az_el_raw = get_sv_el_az(pd_az_el_raw, pd_az_el_pvt)
    pd_az_el_raw["el_diff"] = pd_az_el_raw["elevation_rad"] - pd_az_el_raw["Body Elevation (rad)"]
    pd_az_el_raw["az_diff"] = pd_az_el_raw["azimuth_rad"] - pd_az_el_raw["Body Azimuth (rad)"]
    mean_error = pd_az_el_raw["el_diff"].mean()
    max_error = pd_az_el_raw["el_diff"].max()
    txt = f"Elevation estimate does not meet the expected accuracy. Expected: {az_el_required_precision}rad, estimated: mean {mean_error}rad, max {max_error}rad."
    assert (pd_az_el_raw["el_diff"] < az_el_required_precision).all(), txt
    mean_error = pd_az_el_raw["az_diff"].mean()
    max_error = pd_az_el_raw["az_diff"].max()
    txt = f"Azimuth estimate does not meet the expected accuracy. Expected: {az_el_required_precision}rad, estimated: mean {mean_error}rad, max {max_error}rad."
    assert (pd_az_el_raw["az_diff"] < az_el_required_precision).all(), txt

if __name__ == "__main__":
    #test_geometry_matrix()
    #test_approx_pos_estimate()
    test_pos_estimate()
    #test_azimuth_elevation()

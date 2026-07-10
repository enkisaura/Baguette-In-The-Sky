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
from bits.src.parsers import ephemeris, gnss_raw, nmea
from bits.src.spp import *
from bits.src.convert.space_conversion import ecef_to_enu

required_precision = 15  # m
required_precision_speed = 0.2 # m/s
az_el_required_precision = 1e-2  # rad
gt = (45.7615208,-1.1411692,0)
gt_speed = (0, 0, 0)

test_data_directory_path = os.path.join(os.getcwd(), "bits", "test", "test_data")
ephem_filepath = os.path.join(test_data_directory_path, "gnss_raw", "rinex_v2.rnx")
raw_filepath = os.path.join(test_data_directory_path, "gnss_raw", "rx1_1")
az_el_ephem_filepath = os.path.join(test_data_directory_path, "rinex_nav.rnx")
az_el_skydel_raw_directory_path = os.path.join(test_data_directory_path, "skydel_raw")
ephem2_filepath = os.path.join(test_data_directory_path, "TLSG00FRA_R_20261240000_01D_MN.rnx")
raw2_filepath = os.path.join(test_data_directory_path, "gnss_raw", "XXXX00FRA_R_20261241730_00U_01S_MO.rnx")
nmea_filepath = os.path.join(test_data_directory_path, "20261241730_nmea.txt")

pd_ephemeris = ephemeris.rinex_nav(ephem_filepath)
pd_raw = gnss_raw.micdrop_raw(raw_filepath)
pd_gnss_raw = get_sv_states(pd_raw, pd_ephemeris)
pd_ephemeris2 = ephemeris.rinex_nav(ephem2_filepath)
pd_raw2 = gnss_raw.rinex_obs(raw2_filepath)
nmea_pd = nmea.gga(nmea_filepath)

def test_glo_pos_estimate():
    pos_estimate(gnss_id="glo")

def test_gal_pos_estimate():
    pos_estimate(gnss_id="gal")

def test_gps_pos_estimate():
    pos_estimate(gnss_id="gps")

def test_bei_pos_estimate():
    pos_estimate(gnss_id="bei")

def test_multi_constellation_pos_estimate():
    pos_estimate()

def pos_estimate(gnss_id:str|None = None):
    if gnss_id is None:
        constellation_raw_pd = pd_raw2.copy()
    else:
        constellation_raw_pd = pd_raw2[pd_raw2["gnss_id"] == gnss_id].copy()
    pd_gnss_pvt, _ = get_position_estimate(constellation_raw_pd, pd_ephemeris=pd_ephemeris2)

    gt_ecef = (nmea_pd["x_rx_m"].iloc[100], nmea_pd["y_rx_m"].iloc[100], nmea_pd["z_rx_m"].iloc[100])
    pd_gnss_pvt["np_rx_m"] = pd_gnss_pvt.apply(lambda row: np.array([row["x_rx_m"], row["y_rx_m"], row["z_rx_m"]]), axis=1)
    pd_gnss_pvt["np_rx_enu_m"] = pd_gnss_pvt.apply(lambda row: ecef_to_enu(gt_ecef, row["np_rx_m"]), axis=1)
    pd_gnss_pvt["h_error_m"] = pd_gnss_pvt.apply(lambda row: np.linalg.norm(row["np_rx_enu_m"][:2]), axis=1)
    mean_error = pd_gnss_pvt['h_error_m'].mean()
    max_error = pd_gnss_pvt['h_error_m'].max()

    txt = f"Position estimate does not meet the expected accuracy. Expected: {required_precision}m, estimated: mean {mean_error}m, max {max_error}m."
    assert (pd_gnss_pvt['h_error_m'] < required_precision).all(), txt


def test_azimuth_elevation():
    pd_az_el_raw = pd.DataFrame()
    for filename in os.listdir(az_el_skydel_raw_directory_path):
        raw_filepath = os.path.join(az_el_skydel_raw_directory_path, filename)
        pd_az_el_raw = pd.concat([pd_az_el_raw, gnss_raw.skydel_raw(raw_filepath).iloc[:2]], axis=0)
    pd_az_el_raw = pd_az_el_raw[pd_az_el_raw["gnss_id"] == "gps"].reset_index()
    pd_az_el_pvt, _ = get_approx_position_estimate(pd_az_el_raw, convergence_tolerance=100)
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
    test_glo_pos_estimate()
    test_gal_pos_estimate()
    test_gps_pos_estimate()
    test_bei_pos_estimate()
    test_multi_constellation_pos_estimate()
    test_azimuth_elevation()

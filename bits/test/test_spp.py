#!/usr/bin/env python3

"""
Tests for single point positioning functions
"""

__authors__ = ("Enki SAURA")
__contact__ = ("esaura@ikosconsulting.com")
__copyright__ = "IKOS"
__date__ = "2025-06-06"
__version__ = "0.0.1"

from bits.src import parse
from bits.src.single_point_positioning import *
from bits.src.parse.utils import get_example_data_filepath, fast_parse

required_pos_precision = 20 # m
required_speed_precision = 1 # mps

# Ephemeris
ephem_filepath_list = get_example_data_filepath("ephemeris", rover_type="sv")
ephem_list = []
for filepath in ephem_filepath_list:
    ephem_list.append(fast_parse(filepath, parse.ephemeris.rinex))
ephem_df = pd.concat(ephem_list)

# Fixed
# Ground truth
pos_gt = fast_parse(get_example_data_filepath("pvt")[0], parse.pvt.rmc)
# Raw
pos_raw_df = fast_parse(get_example_data_filepath("raw", rover_type="sv")[0], parse.raw.skydel_folder)
pos_raw_df = pos_raw_df[["time", "sv_id", "gnss_id", "prn_id", "pr_m", "pr_rate_mps", "doppler_hz"]] # Only keep measurements
pos_raw_df = pos_raw_df[pos_raw_df["time"].isin(pos_gt["time"])] # Only keep timestamp with an existing ground_truth
# Add missing klobuchar values
pos_raw_df['klo_a0'] = 4.6566e-9
pos_raw_df['klo_a1'] = 1.4901e-8
pos_raw_df['klo_a2'] = -5.9605e-0
pos_raw_df['klo_a3'] = -1.1921e-7
pos_raw_df['klo_b0'] = 8.1920e+4
pos_raw_df['klo_b1'] = 8.1920e+4
pos_raw_df['klo_b2'] = -6.5536e+4
pos_raw_df['klo_b3'] = -5.2429e+5


def test_pos_gal():
    estimate(raw_df=pos_raw_df[pos_raw_df["gnss_id"] == "gal"], gt_df=pos_gt, gnss_id="gal")

def test_pos_gps():
    estimate(raw_df=pos_raw_df[pos_raw_df["gnss_id"] == "gps"], gt_df=pos_gt, gnss_id="gps")

def test_pos_glo():
    estimate(raw_df=pos_raw_df[pos_raw_df["gnss_id"] == "glo"], gt_df=pos_gt, gnss_id="glo")

def test_pos_bei():
    estimate(raw_df=pos_raw_df[pos_raw_df["gnss_id"] == "bei"], gt_df=pos_gt, gnss_id="bei")

def test_pos_multi():
    estimate(raw_df=pos_raw_df, gt_df=pos_gt, gnss_id="bei")

def estimate(raw_df:pd.DataFrame, gt_df:pd.DataFrame, gnss_id:str, verbose:bool = False):
    pvt_df, _ = get_position_estimate(raw_df, pd_ephemeris=ephem_df, verbose=verbose)

    comparison_df = pvt_df.merge(gt_df, on=["time"], suffixes=("", "_gt"))
    x_diff = comparison_df["x_rx_m"] - comparison_df["x_rx_m_gt"]
    y_diff = comparison_df["y_rx_m"] - comparison_df["y_rx_m_gt"]
    z_diff = comparison_df["z_rx_m"] - comparison_df["z_rx_m_gt"]
    vx_diff = comparison_df["vx_rx_mps"] - comparison_df["vx_rx_mps_gt"]
    vy_diff = comparison_df["vy_rx_mps"] - comparison_df["vy_rx_mps_gt"]
    vz_diff = comparison_df["vz_rx_mps"] - comparison_df["vz_rx_mps_gt"]

    report = (
        f"Current precision for position is x{int(x_diff.abs().max())}m, y{int(y_diff.abs().max())}m, z{int(z_diff.abs().max())}m "
        f"(mean = {int(x_diff.abs().mean())}, {int(y_diff.abs().mean())}, {int(z_diff.abs().mean())}), target precision is {required_pos_precision}m"
        f"Current precision for velocity is x{int(vx_diff.abs().max())}mps, y{int(vy_diff.abs().max())}m, z{int(vz_diff.abs().max())}mps "
        f"(mean = {int(vx_diff.abs().mean())}, {int(vy_diff.abs().mean())}, {int(vz_diff.abs().mean())}), target precision is {required_speed_precision}mps")

    if verbose:
        print(f"Precision report for sv model {gnss_id}")
        print(report)

    assert ((x_diff.abs().max() < required_pos_precision and y_diff.abs().max() < required_pos_precision
            and z_diff.abs().max() < required_pos_precision)
            and (vx_diff.abs().max() < required_speed_precision and vy_diff.abs().max() < required_speed_precision
            and vz_diff.abs().max() < required_speed_precision) and len(comparison_df) > 0), \
        f"Precision requirement is not met for {gnss_id}. \n{report}"


if __name__ == "__main__":
    test_pos_gal()
    test_pos_gps()
    test_pos_glo()
    test_pos_bei()


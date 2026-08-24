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
from bits.src.utils import get_example_data_filepath, fast_parse

required_pos_precision = 5 # m
required_speed_precision = 1 # mps

# Fixed
# Ground truth
pos_gt = fast_parse(get_example_data_filepath("pvt")[0], parse.pvt.rmc)

# Ephemeris
ephem_filepath_list = get_example_data_filepath("ephemeris", rover_type="sv")
ephem_list = []
for filepath in ephem_filepath_list:
    ephem_list.append(fast_parse(filepath, parse.ephemeris.rinex))
ephem_df = pd.concat(ephem_list)
# Add missing klobuchar values
klo_cols = ["klo_a0", "klo_a1", "klo_a2", "klo_a3", "klo_b0", "klo_b1", "klo_b2", "klo_b3"]
klo_values = fast_parse(get_example_data_filepath("ephemeris")[0], parse.ephemeris.rinex)[klo_cols].iloc[0]
ephem_df[klo_cols] = klo_values[klo_cols].values

# Raw
pos_raw_df = fast_parse(get_example_data_filepath("raw", rover_type="sv")[0], parse.raw.skydel_folder)
# Only keep measurements
pos_raw_df = pos_raw_df[["time", "sv_id", "gnss_id", "prn_id", "pr_m", "pr_rate_mps", "doppler_hz", "frequency_hz"]]
pos_raw_df = pos_raw_df[pos_raw_df["time"].isin(pos_gt["time"])] # Only keep timestamp with an existing ground_truth

# Circle
receiver_gt = fast_parse(get_example_data_filepath("pvt", rover_type="circular")[0], parse.pvt.rmc)
receiver_raw_df = fast_parse(get_example_data_filepath("raw", rover_type="circular")[0], parse.raw.rinex)


def test_pos_gal():
    estimate(raw_df=pos_raw_df[pos_raw_df["gnss_id"] == "gal"], gt_df=pos_gt, gnss_id="gal")

def test_pos_gps():
    estimate(raw_df=pos_raw_df[pos_raw_df["gnss_id"] == "gps"], gt_df=pos_gt, gnss_id="gps")

def test_pos_glo():
    estimate(raw_df=pos_raw_df[pos_raw_df["gnss_id"] == "glo"], gt_df=pos_gt, gnss_id="glo")

def test_pos_bei():
    estimate(raw_df=pos_raw_df[pos_raw_df["gnss_id"] == "bei"], gt_df=pos_gt, gnss_id="bei")

def test_pos_multi():
    estimate(raw_df=pos_raw_df, gt_df=pos_gt, gnss_id="all")

def test_receiver():
    pvt_df, _ = get_position_estimate(receiver_raw_df, pd_ephemeris=ephem_df, verbose=True)

def estimate(raw_df:pd.DataFrame, gt_df:pd.DataFrame, gnss_id:str, verbose:bool = True):
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
        f"(mean = {int(x_diff.abs().mean())}, {int(y_diff.abs().mean())}, {int(z_diff.abs().mean())}), target precision is {required_pos_precision}m "
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
    test_pos_multi()
    test_receiver()


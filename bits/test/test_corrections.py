#!/usr/bin/env python3

"""
Tests for correction functions

Usage: Used from pytest
======
    python -m pytest -v
"""

__authors__ = ("Enki SAURA")
__contact__ = ("esaura@ikosconsulting.com")
__copyright__ = "IKOS"
__date__ = "2025-06-05"
__version__ = "0.0.1"

import pandas as pd
from bits.src import corrections, parse
from bits.src.parse.utils import get_example_data_filepath, fast_parse
from bits.src.sv_model import get_sv_states

required_precision = 30

# Ground truth
gt_df = fast_parse(get_example_data_filepath("raw", rover_type="sv")[0], parse.raw.skydel_folder)
gt_df = gt_df[gt_df['time'].dt.microsecond == 0]

# Ephemeris
ephem_filepath_list = get_example_data_filepath("ephemeris", rover_type="sv")
ephem_list = []
for filepath in ephem_filepath_list:
    ephem_list.append(fast_parse(filepath, parse.ephemeris.rinex))
ephem_df = pd.concat(ephem_list)
ephem_df = ephem_df[ephem_df['time'].dt.microsecond == 0]
ephem_df = get_sv_states(gt_df, ephem_df)[["time", "time_of_ephemeris", "sv_id", "gnss_id", "prn_id",
                                           "pr_m", "corr_pr_m",
                                           "sqrta", "deltan", "m0", "e",
                                           "x_sv_m", "y_sv_m", "z_sv_m",
                                           "vx_sv_mps", "vy_sv_mps", "vz_sv_mps",
                                           "azimuth_rad", "elevation_rad",
                                           "clock_bias", "clock_drift", "clock_drift_rate"]]

# Raw
# SV will be computed based on a dataframe with the same timestamps and satellites than the Ground truth
raw_df = gt_df.copy()[["time", "sv_id", "gnss_id", "prn_id", "pr_m", "corr_pr_m"]]
# Add missing klobuchar values
raw_df['klo_a0'] = 4.6566e-9
raw_df['klo_a1'] = 1.4901e-8
raw_df['klo_a2'] = -5.9605e-0
raw_df['klo_a3'] = -1.1921e-7
raw_df['klo_b0'] = 8.1920e+4
raw_df['klo_b1'] = 8.1920e+4
raw_df['klo_b2'] = -6.5536e+4
raw_df['klo_b3'] = -5.2429e+5
# Add elevation/azimuth
raw_df = raw_df.merge(ephem_df[["time", "sv_id", "elevation_rad", "azimuth_rad"]], on=["time", "sv_id"])

# PVT
pvt_df = parse.pvt.gga(get_example_data_filepath("pvt")[0])

def test_clock_corrections(verbose=False):
    df = corrections.get_clock_corrections(raw_df, ephem_df)

    comparison_df = df.merge(gt_df, on=["time", "sv_id"], suffixes=("", "_gt"))

    clock_diff = comparison_df["clock_corr_m"] - comparison_df["clock_corr_m_gt"]

    report = (f"Current precision for clock corrections is {int(clock_diff.abs().max())}m"
              f"(mean = {int(clock_diff.abs().mean())}, target precision is {required_precision}m)")

    if verbose:
        print(report)

    assert (clock_diff.abs().max() < required_precision and len(comparison_df) > 0), \
        f"Precision requirement is not met for clock corrections. \n{report}"

def test_klobuchar(verbose=False):
    df = corrections.get_atmospheric_corrections(raw_df, pvt_df)

    comparison_df = df.merge(gt_df, on=["time", "sv_id"], suffixes=("", "_gt"))

    iono_diff = comparison_df["iono_corr_m"] - comparison_df["iono_corr_m_gt"]
    tropo_diff = comparison_df["tropo_corr_m"] - comparison_df["tropo_corr_m_gt"]

    report_iono = (f"Current precision for iono corrections is {int(iono_diff.abs().max())}m"
                   f"(mean = {int(iono_diff.abs().mean())}, target precision is {required_precision}m)")
    report_tropo = (f"Current precision for tropo corrections is {int(tropo_diff.abs().max())}m"
                    f"(mean = {int(tropo_diff.abs().mean())}, target precision is {required_precision}m)")

    if verbose:
        print(report_iono)
        print(report_tropo)

    assert (iono_diff.abs().max() < required_precision and len(comparison_df) > 0), \
        f"Precision requirement is not met for iono corrections. \n{report_iono}"
    assert (tropo_diff.abs().max() < required_precision and len(comparison_df) > 0), \
        f"Precision requirement is not met for tropo corrections. \n{tropo_diff}"


if __name__ == "__main__":
    test_clock_corrections()
    test_klobuchar()
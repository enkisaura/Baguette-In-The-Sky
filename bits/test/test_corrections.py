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
from bits.src.utils import get_example_data_filepath, fast_parse
from bits.src.sv_model import get_sv_states

required_precision = 3

# Ground truth
gt_df = fast_parse(get_example_data_filepath("raw", rover_type="sv")[0], parse.raw.skydel_folder)
gt_df = gt_df[gt_df['time'].dt.microsecond == 0]

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
# SV will be computed based on a dataframe with the same timestamps and satellites than the Ground truth
raw_df = get_sv_states(gt_df, ephem_df).rename(columns={"clock_corr_m": "clock_corr_m_gt",
                                                        "iono_corr_m": "iono_corr_m_gt",
                                                        "tropo_corr_m": "tropo_corr_m_gt"})

# PVT
pvt_df = parse.pvt.gga(get_example_data_filepath("pvt")[0])


def test_clock_corrections(verbose=False):
    df = corrections.get_clock_corrections(raw_df, ephem_df)

    clock_diff = df["clock_corr_m"] - df["clock_corr_m_gt"]

    report = (f"Current precision for clock corrections is {int(clock_diff.abs().max())}m"
              f"(mean = {int(clock_diff.abs().mean())}, target precision is {required_precision}m)")

    if verbose:
        print(report)

    assert (clock_diff.abs().max() < required_precision and len(df) > 0), \
        f"Precision requirement is not met for clock corrections. \n{report}"

def test_klobuchar(verbose=False):
    df = corrections.get_atmospheric_corrections(raw_df, ephem_df, pvt_df)

    iono_diff = df["iono_corr_m"] - df["iono_corr_m_gt"]
    tropo_diff = df["tropo_corr_m"] - df["tropo_corr_m_gt"]

    report_iono = (f"Current precision for iono corrections is {int(iono_diff.abs().max())}m"
                   f"(mean = {int(iono_diff.abs().mean())}, target precision is {required_precision}m)")
    report_tropo = (f"Current precision for tropo corrections is {int(tropo_diff.abs().max())}m"
                    f"(mean = {int(tropo_diff.abs().mean())}, target precision is {required_precision}m)")

    if verbose:
        print(report_iono)
        print(report_tropo)

    assert (iono_diff.abs().max() < required_precision and len(df) > 0), \
        f"Precision requirement is not met for iono corrections. \n{report_iono}"
    assert (tropo_diff.abs().max() < required_precision and len(df) > 0), \
        f"Precision requirement is not met for tropo corrections. \n{tropo_diff}"


if __name__ == "__main__":
    test_clock_corrections()
    test_klobuchar()
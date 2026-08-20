"""
Tests for the "find sv state" functions

Usage: Used from pytest
======
    python -m pytest -v
"""

__authors__ = ("Enki SAURA")
__contact__ = ("esaura@ikosconsulting.com")
__copyright__ = "IKOS"
__date__ = "18/02/2025"
__version__ = "0.0.1"

import pandas as pd
from bits.src.sv_model import get_sv_states
from bits.src import const, convert, parse
from bits.src.parse.utils import get_example_data_filepath, fast_parse

required_precision = 1 # m

# Ephemeris
ephem_filepath_list = get_example_data_filepath("ephemeris", rover_type="sv")
ephem_list = []
for filepath in ephem_filepath_list:
    ephem_list.append(fast_parse(filepath, parse.ephemeris.rinex))
ephem_df = pd.concat(ephem_list)

# Ground truth
gt_df = fast_parse(get_example_data_filepath("raw", rover_type="sv")[0], parse.raw.skydel_folder)
tof = (1e9 * gt_df["corr_pr_m"] / const.C).astype("timedelta64[ns]")
# Compensate for earth's rotation to get SV state at emission
gt_df[["x_sv_m", "y_sv_m", "z_sv_m"]] = pd.Series(convert.space.rotate_ecef(gt_df["x_sv_m"], gt_df["y_sv_m"],
                                                                            gt_df["z_sv_m"], tof))

# Raw
# SV will be computed based on a dataframe with the same timestamps and satellites than the Ground truth
raw_df = gt_df.copy().drop(columns=["x_sv_m", "y_sv_m", "z_sv_m"])

def test_galileo():
    compute_sv_state("gal")

def test_gps():
    compute_sv_state("gps")

def test_glonass():
    compute_sv_state("glo")

def test_beidou():
    compute_sv_state("bei")

def compute_sv_state(gnss_id:str, verbose:bool = False):
    df = raw_df[raw_df["gnss_id"] == gnss_id]
    df = get_sv_states(df, pd_ephemeris=ephem_df)

    comparison_df = df.merge(gt_df, on=["time", "sv_id"], suffixes=("", "_gt"))
    x_diff = comparison_df["x_sv_m"] - comparison_df["x_sv_m_gt"]
    y_diff = comparison_df["y_sv_m"] - comparison_df["y_sv_m_gt"]
    z_diff = comparison_df["z_sv_m"] - comparison_df["z_sv_m_gt"]

    report = (f"Current precision is x{int(x_diff.abs().max())}m, y{int(y_diff.abs().max())}m, z{int(z_diff.abs().max())}m "
              f"(mean = {int(x_diff.abs().mean())}, {int(y_diff.abs().mean())}, {int(z_diff.abs().mean())}), target precision is {required_precision}m")

    if verbose:
        print(f"Precision report for sv model {gnss_id}")
        print(report)

    assert (x_diff.abs().max() < required_precision and y_diff.abs().max() < required_precision
            and z_diff.abs().max() < required_precision) and len(comparison_df) > 0, \
        f"Precision requirement is not met for {gnss_id}. \n{report}"


if __name__ == "__main__":
    test_galileo()
    test_gps()
    test_glonass()
    test_beidou()
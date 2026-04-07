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

import os
import pandas as pd
from bits.src.parsers import ephemeris, gnss_raw
from bits.src.sv_model import get_sv_states
from bits.src.reference_frame_object import GnssTimestamp
from bits.src.convert.space_conversion import rotate_ecef


# Using skydel's sv state references, part 2. of sv_model.get_sv_states worsen the results. Without this part,
# millimetric precision is expected
required_precision = 20  # m

start_time = GnssTimestamp(year=2023, month=9, day=14, hour=12, minute=0, second=0)

test_data_directory_path = os.path.join(os.getcwd(), "bits", "test", "test_data")
gps_ephem_filepath = os.path.join(test_data_directory_path, "rinex_nav.rnx")
gal_ephem_filepath = os.path.join(test_data_directory_path, "SkydelRINEX_S_2023257120_600S_EN.rnx")
skydel_raw_directory_path = os.path.join(test_data_directory_path, "skydel_raw")

pd_gps_ephemeris = ephemeris.rinex_nav(gps_ephem_filepath)
pd_gal_ephemeris = ephemeris.rinex_nav(gal_ephem_filepath)
pd_ephemeris = pd.concat([pd_gps_ephemeris, pd_gal_ephemeris], ignore_index=True)

pd_full_computed = pd.DataFrame()
for filename in os.listdir(skydel_raw_directory_path):
    raw_filepath = os.path.join(skydel_raw_directory_path, filename)

    pd_gnss_raw = gnss_raw.skydel_raw(raw_filepath)

    pd_gnss_raw_to_be_computed = pd_gnss_raw.copy().drop(columns=["x_sv_m", "y_sv_m", "z_sv_m"])

    pd_computed_sv_states = get_sv_states(pd_gnss_raw_to_be_computed, pd_ephemeris)
    pd_computed_sv_states[["x_sv_m", "y_sv_m", "z_sv_m"]] = \
        pd_computed_sv_states.apply(
            lambda row: pd.Series(rotate_ecef(row["x_sv_m"], row["y_sv_m"], row["z_sv_m"], -row["delta_time"])), axis=1)

    pd_computed_sv_states["x_diff"] = pd_computed_sv_states["x_sv_m"] - pd_gnss_raw["x_sv_m"]
    pd_computed_sv_states["y_diff"] = pd_computed_sv_states["y_sv_m"] - pd_gnss_raw["y_sv_m"]
    pd_computed_sv_states["z_diff"] = pd_computed_sv_states["z_sv_m"] - pd_gnss_raw["z_sv_m"]

    pd_full_computed = pd.concat([pd_full_computed, pd_computed_sv_states], axis=0)


def test_gps_state_precision():
    gps_pd_computed = pd_full_computed[pd_full_computed["gnss_id"] == "gps"]

    max_x_diff = gps_pd_computed['x_diff'].abs().max()
    mean_x_diff = gps_pd_computed['x_diff'].abs().mean()
    max_y_diff = gps_pd_computed['y_diff'].abs().max()
    mean_y_diff = gps_pd_computed['y_diff'].abs().mean()
    max_z_diff = gps_pd_computed['z_diff'].abs().max()
    mean_z_diff = gps_pd_computed['z_diff'].abs().mean()
    assert max_x_diff < required_precision and max_y_diff < required_precision and max_z_diff < required_precision, \
        f"Precision requirement is not met for GPS states. " \
        f"Current precision is x{int(max_x_diff)}m, y{int(max_y_diff)}m, z{int(max_z_diff)}m " \
        f"(mean = {int(mean_x_diff)}, {int(mean_y_diff)}, {int(mean_z_diff)}), target precision is {required_precision}m"


def test_galileo_state_precision():
    gal_pd_computed = pd_full_computed[pd_full_computed["gnss_id"] == "gal"]

    max_x_diff = gal_pd_computed['x_diff'].abs().max()
    mean_x_diff = gal_pd_computed['x_diff'].abs().mean()
    max_y_diff = gal_pd_computed['y_diff'].abs().max()
    mean_y_diff = gal_pd_computed['y_diff'].abs().mean()
    max_z_diff = gal_pd_computed['z_diff'].abs().max()
    mean_z_diff = gal_pd_computed['z_diff'].abs().mean()
    assert max_x_diff < required_precision and max_y_diff < required_precision and max_z_diff < required_precision, \
        f"Precision requirement is not met for Galileo states. " \
        f"Current precision is x{int(max_x_diff)}m, y{int(max_y_diff)}m, z{int(max_z_diff)}m " \
        f"(mean = {int(mean_x_diff)}, {int(mean_y_diff)}, {int(mean_z_diff)}), target precision is {required_precision}m"

########################################################################################################################
# The following tests are not yet available. No testdata are available yet for those.
# Beidou has weirdly bad precision
def available_soon_test_beidou_state_precision():
    bei_pd_computed = pd_full_computed.copy()
    bei_pd_computed = bei_pd_computed[bei_pd_computed["gnss_id"] == "bei"]

    max_x_diff = bei_pd_computed['x_diff'].abs().max()
    mean_x_diff = bei_pd_computed['x_diff'].abs().mean()
    max_y_diff = bei_pd_computed['y_diff'].abs().max()
    mean_y_diff = bei_pd_computed['y_diff'].abs().mean()
    max_z_diff = bei_pd_computed['z_diff'].abs().max()
    mean_z_diff = bei_pd_computed['z_diff'].abs().mean()
    assert max_x_diff < required_precision and max_y_diff < required_precision and max_z_diff < required_precision, \
        f"Precision requirement is not met for Beidou states. " \
        f"Current precision is x{int(max_x_diff)}m, y{int(max_y_diff)}m, z{int(max_z_diff)}m " \
        f"(mean = {int(mean_x_diff)}, {int(mean_y_diff)}, {int(mean_z_diff)}), target precision is {required_precision}m"

# Glonass has weirdly veeeeery bad precision
def available_soon_test_glonass_state_precision():
    glo_pd_computed = pd_full_computed.copy()
    glo_pd_computed = glo_pd_computed[glo_pd_computed["gnss_id"] == "glo"]

    max_x_diff = glo_pd_computed['x_diff'].abs().max()
    mean_x_diff = glo_pd_computed['x_diff'].abs().mean()
    max_y_diff = glo_pd_computed['y_diff'].abs().max()
    mean_y_diff = glo_pd_computed['y_diff'].abs().mean()
    max_z_diff = glo_pd_computed['z_diff'].abs().max()
    mean_z_diff = glo_pd_computed['z_diff'].abs().mean()
    assert max_x_diff < required_precision and max_y_diff < required_precision and max_z_diff < required_precision, \
        f"Precision requirement is not met for Glonass states. " \
        f"Current precision is x{int(max_x_diff)}m, y{int(max_y_diff)}m, z{int(max_z_diff)}m " \
        f"(mean = {int(mean_x_diff)}, {int(mean_y_diff)}, {int(mean_z_diff)}), target precision is {required_precision}m"


if __name__ == "__main__":
    test_gps_state_precision()
    test_galileo_state_precision()
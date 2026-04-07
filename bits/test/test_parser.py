#!/usr/bin/env python3

"""
Tests for the parsers functions

Usage: Used from pytest
======
    python -m pytest -v
"""

__authors__ = ("Enki SAURA")
__contact__ = ("esaura@ikosconsulting.com")
__copyright__ = "IKOS"
__date__ = "2025-06-05"
__version__ = "0.0.1"

import os
import pandas as pd
from bits.src.parsers import ephemeris, gnss_raw

test_data_directory_path = os.path.join(os.getcwd(), "bits", "test", "test_data")
rinex2_filepath = os.path.join(test_data_directory_path, "rinex_nav.rnx")
rinex3_filepath = os.path.join(test_data_directory_path, "SkydelRINEX_S_2023257120_600S_EN.rnx")
skydel_raw_directory_path = os.path.join(test_data_directory_path, "skydel_raw")
micdrop_raw_filepath = os.path.join(test_data_directory_path, "gnss_raw", "rx1_1")

def test_rinex_nav_2():
    pd_parsed_ephemeris = ephemeris.rinex_nav(rinex2_filepath)
    assert isinstance(pd_parsed_ephemeris, pd.DataFrame) and not pd_parsed_ephemeris.empty

def test_rinex_nav_3():
    pd_parsed_ephemeris = ephemeris.rinex_nav(rinex3_filepath)
    assert isinstance(pd_parsed_ephemeris, pd.DataFrame) and not pd_parsed_ephemeris.empty

def test_skydel_raw():
    for filename in os.listdir(skydel_raw_directory_path):
        raw_filepath = os.path.join(skydel_raw_directory_path, filename)
        pd_parsed_raw = gnss_raw.skydel_raw(raw_filepath)
        assert isinstance(pd_parsed_raw, pd.DataFrame) and not pd_parsed_raw.empty

def test_micdrop_raw():
    pd_parsed_raw = gnss_raw.micdrop_raw(micdrop_raw_filepath)
    assert isinstance(pd_parsed_raw, pd.DataFrame) and not pd_parsed_raw.empty


if __name__ == "__main__":
    test_rinex_nav_2()
    test_rinex_nav_3()
    test_skydel_raw()
    test_micdrop_raw()
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

import os
import pandas as pd
from bits.src.parsers import ephemeris, gnss_raw
from bits.src import corrections

test_data_directory_path = os.path.join(os.getcwd(), "bits", "test", "test_data")
ephem_filepath = os.path.join(test_data_directory_path, "gnss_raw", "rinex_v2.rnx")
raw_filepath = os.path.join(test_data_directory_path, "gnss_raw", "rx1_1")

def test_clock_corrections():
    pd_ephemeris = ephemeris.rinex_nav(ephem_filepath)
    pd_raw = gnss_raw.micdrop_raw(raw_filepath)
    pd_gnss = corrections.get_clock_corrections(pd_raw, pd_ephemeris)
    assert isinstance(pd_gnss, pd.DataFrame) and not pd_gnss.empty

if __name__ == "__main__":
    test_clock_corrections()
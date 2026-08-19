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

import pandas as pd
from bits.src import parse
from bits.src.parse.utils import get_example_data_filepath, fast_parse


def test_ephem_rinex():
    df = parse.ephemeris.rinex(get_example_data_filepath("ephemeris")[0])
    assert isinstance(df, pd.DataFrame) and not df.empty

def test_raw_rinex():
    df = parse.raw.rinex(get_example_data_filepath("raw")[0])
    assert isinstance(df, pd.DataFrame) and not df.empty

def test_raw_skydel():
    df = parse.raw.skydel_folder(get_example_data_filepath("raw", rover_type="sv")[0])
    assert isinstance(df, pd.DataFrame) and not df.empty

def test_pvt_gga():
    df = parse.pvt.gga(get_example_data_filepath("pvt")[0])
    assert isinstance(df, pd.DataFrame) and not df.empty

def test_pvt_rmc():
    df = parse.pvt.rmc(get_example_data_filepath("pvt")[0])
    assert isinstance(df, pd.DataFrame) and not df.empty


if __name__ == "__main__":
    test_ephem_rinex()
    test_raw_rinex()
    test_raw_skydel()
    test_pvt_gga()
    test_pvt_rmc()
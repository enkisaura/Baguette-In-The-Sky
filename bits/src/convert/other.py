#!/usr/bin/env python3

"""
This is an awesome script that was not commented...

Usage:
======
python other.py

"""

__authors__ = ("Enki SAURA")
__contact__ = ("esaura@ikosconsulting.com")
__copyright__ = "IKOS"
__date__ = "2025-06-05"
__version__ = "0.0.1"

from bits.src.const import C


def doppler_to_pr_rate(doppler, ft=1575420000):
    """
    Converts doppler to pseudorange rate. L1 frequency by default
    :param doppler: doppler shift (Hz)
    :param ft: signal frequency (Hz)
    :return: pseudorange rate (m/s)
    """
    wavelength = C / ft
    pr_rate = -wavelength * doppler

    return pr_rate



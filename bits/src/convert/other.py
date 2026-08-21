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

from bits.src import const
import numpy as np


def doppler_to_pr_rate(doppler_shift:np.ndarray, frequency:np.ndarray) -> np.ndarray:
    """
    Converts doppler to pseudorange rate
    :param doppler_shift: doppler shift (Hz)
    :param frequency: signal frequency (Hz)
    :return: pseudorange rate (m/s)
    """
    doppler_shift = np.asarray(doppler_shift)
    frequency = np.asarray(frequency)

    wavelength = const.C / frequency
    pr_rate = -wavelength * doppler_shift

    return pr_rate



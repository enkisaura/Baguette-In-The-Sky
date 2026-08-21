#!/usr/bin/env python3

"""
Constant to be used in BITS
"""

__authors__ = ("Enki SAURA")
__contact__ = ("esaura@ikosconsulting.com")
__copyright__ = "IKOS"
__date__ = "2025-06-04"
__version__ = "0.0.1"

import numpy as np

C = 299792458  # Speed of light (m/s)
NU = 3.986005e14  # WGS 84 value of the earth's gravitational constant for GPS user (m3/s2)
OMEGA_E = 7.2921151467e-5  # WGS 84 value of the earth's rotation rate (rad/s)
OMEGA_E_BEI = 7.2921150e-5  # BEIDOU value of the earth's rotation rate (rad/s)
F = -4.442807633e-10 # float : Relativistic correction term (s/m^(1/2))
G = 9.80665 # gravitational acceleration (m/s²)
RE = 6378137 # WGS 84 Earth Equatorial Radius (m)
J2 = 0.0010826262 # Oblate Earth Gravity Coefficient

# Mean frequency associated with a gnss band (Hz)
band_frequency = {
    # Galileo
    "e1": 1575.420e6,
    "e6": 1278.750e6,
    "e5": 1191.795e6,
    "e5a": 1176.450e6,
    "e5b": 1207.140e6,
    # GPS
    "l1": 1575.42e6,
    "l2": 1227.60e6,
    "l5": 1176.45e6,
    # Glonass; each SV has a different frequency, please use const.get_glo_freq()
    "g1": 1602e6,
    "dg1": 562.5e3,
    "g2": 1246e6,
    "dg2": 437.5e3,
    "g3": 1201e6,
    "dg3": 437.5e3,
    # Beidou
    "b1c": 1575.42e6,
    "b1i": 1561.098e6,
    "b2a": 1176.45e6,
    "b2b": 1207.14e6,
    "b3": 1268.520e6,
}

# Table used to extrapolate weather parameters for tropospheric corrections
WEATHER_PARAM = {
    "latitude": (15, 30, 45, 60, 75), # (°)
    "P0": (1013.25, 1017.25, 1015.75, 1011.75, 1013), # pressure (mbar)
    "T0": (299.65, 294.15, 283.15, 272.15, 263.65), # temperature (°K)
    "e0": (26.31, 21.79, 11.66, 6.78, 4.11), # water vapour pressure (mbar)
    "beta0": (6.30e-3, 6.05e-3, 5.58e-3, 5.39e-3, 4.53e-3), # temperature "lapse" rate (°K/m)
    "lambda0": (2.77, 3.15, 2.57, 1.81, 1.55), # water vapour "lapse rate" ()
    "deltaP": (0, -3.75, -2.25, -1.75, -0.5), # pressure (mbar)
    "deltaT": (0, 7, 11, 15, 14.5), # temperature (°K)
    "deltae": (0, 8.85, 7.24, 5.36, 3.39), # water vapour pressure (mbar)
    "deltabeta": (0, 0.25e-3, 0.32e-3, 0.81e-3, 0.62e-3), # temperature "lapse" rate (°K/m)
    "deltalambda": (0, 0.33, 0.46, 0.74, 0.3), # water vapour "lapse rate" ()
}
K1 = 77.604 # (K/mbar)
K2 = 382000 # (K²/mbar)
RD = 287.054 # (J/Kg/K)
GM = 9.784 # (m/s²)


def get_omega_e(gnss_id: str | np.ndarray) -> np.ndarray:
    """
    Beidou has a different value of the earth's rotation rate (rad/s) that can result of dozen of meters of errors on
    sv state precision.

    :param gnss_id: constellation id (str: "gps", "gal", "bei", "glo")
    :return: earth's rotation rate (rad/s)
    """
    gnss_id = np.asarray(gnss_id)

    return np.where((gnss_id == "bei"), OMEGA_E_BEI, OMEGA_E)

def get_glo_freq(band:np.ndarray|str, slot:np.ndarray|int) -> np.ndarray:
    """
    With Glonass, each SV is attributed a different frequency.
    sv_freq = central_freq + d_freq * SV_slot

    :param band: name of the band (g1, g2, g3) (str)
    :param slot: sv slot number between -7 to +6 (int)
    :return: sv frequency (float)
    """
    band = np.asarray(band)
    slot = np.asarray(slot)

    central_freq = np.where(band == "g1", band_frequency["g1"], np.nan)
    central_freq = np.where(band == "g2", band_frequency["g2"], central_freq)
    central_freq = np.where(band == "g3", band_frequency["g3"], central_freq)

    dfreq = np.where(band == "g1", band_frequency["dg1"], np.nan)
    dfreq = np.where(band == "g2", band_frequency["dg2"], dfreq)
    dfreq = np.where(band == "g3", band_frequency["dg3"], dfreq)

    sv_freq = central_freq + slot * dfreq

    sv_freq = np.where(np.isin(slot, range(-7, 7)), sv_freq, np.nan) # Ensure slot is valid

    return sv_freq


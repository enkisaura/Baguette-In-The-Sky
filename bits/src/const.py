#!/usr/bin/env python3

"""
Constant to be used in BITS
"""

__authors__ = ("Enki SAURA")
__contact__ = ("esaura@ikosconsulting.com")
__copyright__ = "IKOS"
__date__ = "2025-06-04"
__version__ = "0.0.1"


C = 299792458  # Speed of light (m/s)
NU = 3.986005e14  # WGS 84 value of the earth's gravitational constant for GPS user (m3/s2)
OMEGA_E = 7.292115e-5  # WGS 84 value of the earth's rotation rate (rad/s)
F = -4.442807633e-10 # float : Relativistic correction term (s/m^(1/2))
G = 9.80665 # gravitational acceleration (m/s²)
RE = 6378137 # WGS 84 Earth Equatorial Radius (m)
J2 = 0.0010826262 # Oblate Earth Gravity Coefficient

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
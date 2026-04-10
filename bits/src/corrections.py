#!/usr/bin/env python3

"""
Used to apply further corrections to raw pseudoranges
"""

__authors__ = ("Enki SAURA")
__contact__ = ("esaura@ikosconsulting.com")
__copyright__ = "IKOS"
__date__ = "2025-06-05"
__version__ = "0.0.1"

import pandas as pd
import numpy as np
import math
import warnings
from typing import Literal
from typing import Tuple

from .sv_model import retrieve_ephemeris, compute_eccentric_anomaly
from . import const
from .utils import check_dataframe

# Clock corrections
def compute_satellite_clock_correction(dt, a0, a1, a2) -> float:
    """
    Compute polynomial satellite clock correction.
    source: https://gssc.esa.int/navipedia/index.php/Clock_Modelling
    :param dt: Time from sv time of clock (s)
    :param a0: SV clock bias (s)
    :param a1: SV clock drift (s^-1)
    :param a2: SV clock drift rate (s^-2)
    :return: Polynomial clock correction (s)
    """
    satellite_clock_correction = a0 + a1*dt + np.sign(dt) * a2*(dt**2)
    return satellite_clock_correction

def compute_relativistic_clock_correction(e, sqrta, eccentric_anomaly):
    """
    Compute relativistic satellite clock correction.
    source: https://gssc.esa.int/navipedia/index.php?title=Relativistic_Clock_Correction
    :param e: Eccentricity (dimensionless)
    :param sqrta: Square root of the semi-major axis (sqrt(m))
    :param eccentric_anomaly: Eccentric anomaly, use sv_model.compute_eccentric_anomaly
    :return: Relativistic clock correction (s)
    """
    relativistic_clock_correction = const.F * e * sqrta * np.sin(eccentric_anomaly)
    return relativistic_clock_correction

def get_clock_corrections(pd_gnss_raw: pd.DataFrame, pd_ephemeris: pd.DataFrame = None) -> pd.DataFrame:
    """
    Compute clock corrections using a pd.Dataframe ephemeris from the BITS ephemeris parser for GPS and Galileo
    :param pd_gnss_raw: GNSS raw dataframe from BITS parser
    :param pd_ephemeris: ephemeris dataframe from BITS parser
    :return: raw data with corrected pseudoranges and corresponding clock corrections
    """
    raw_required_columns = ["time", "pr_m", "gnss_id", "sv_id"]
    ephem_in_raw_required_columns_poly = ["time_diff", "clock_bias", "clock_drift", "clock_drift_rate"]
    ephem_in_raw_required_columns_relat_a = ["sqrta", "time_navdata", "toe", "deltan", "m0", "e"]
    ephem_in_raw_required_columns_relat_b = ["e", "sqrta", "eccentric_anomaly"]

    if not check_dataframe(pd_gnss_raw, raw_required_columns):
        warnings.warn("Missing columns in pd_gnss_raw, cannot compute clock correction.")
        return pd_gnss_raw

    pd_gnss = retrieve_ephemeris(pd_gnss_raw, pd_ephemeris) # Get clock correction parameters from ephemeris

    if not(check_dataframe(pd_gnss, ephem_in_raw_required_columns_poly)
           and (check_dataframe(pd_gnss, ephem_in_raw_required_columns_relat_a)
                or check_dataframe(pd_gnss, ephem_in_raw_required_columns_relat_b))):
        warnings.warn("Missing ephemeris data, cannot compute clock correction.")
        return pd_gnss

    # Check for already existing corrections
    if not {"corr_pr_m"}.issubset(pd_gnss.columns):
        pd_gnss["corr_pr_m"] = pd_gnss["pr_m"]
    elif {"clock_corr_m"}.issubset(pd_gnss.columns):
        pd_gnss["corr_pr_m"] -= pd_gnss["clock_corr_m"]

    # 1) Compute satellite clock correction:
    pd_gnss["poly_clock_corr_m"] = pd_gnss.apply(
        lambda row: const.C * compute_satellite_clock_correction(row["time_diff"].total_seconds(), row["clock_bias"],
                                                             row["clock_drift"], row["clock_drift_rate"]), axis=1)

    # 2) Compute relativistic clock corrections
    if "eccentric_anomaly" not in pd_gnss.columns:
        pd_gnss["eccentric_anomaly"] = pd_gnss.apply(
            lambda row: compute_eccentric_anomaly(row, row["time"], ek_iterations=5)[0], axis=1)
    pd_gnss["relat_clock_corr_m"] = pd_gnss.apply(
        lambda row: const.C * compute_relativistic_clock_correction(row["e"], row["sqrta"], row["eccentric_anomaly"]),
        axis=1)

    # 3) Compute group delay
    pd_gnss["tgd_clock_corr_m"] = const.C * pd_gnss["tgd"]

    # 4) Fuse clock corrections
    pd_gnss["clock_corr_m"] = pd_gnss.apply(lambda row: row["poly_clock_corr_m"] + row["relat_clock_corr_m"]
                                                        - row["tgd_clock_corr_m"], axis=1)

    # 5) Correct pseudoranges
    pd_gnss["corr_pr_m"] += pd_gnss["clock_corr_m"]

    return pd_gnss

########################################################################################################################
# Atmospheric corrections
def compute_klobuchar(rx_lat: float, rx_lon: float, tow: float, sv_elevation: float, sv_azimuth: float,
                      alpha: Tuple[float, float, float, float], beta: Tuple[float, float, float, float]) -> float:
    """
    Compute ionospheric delay using Klobuchar's model.
    GPS satellites broadcast the parameters of the Klobuchar ionospheric model for single frequency users. This
    broadcast model is based on an empirical approach and is estimated to reduce about the 50% RMS ionospheric range
    error worldwide.
    source: https://gssc.esa.int/navipedia/index.php?title=Klobuchar_Ionospheric_Model
    Klobuchar, J. A. 1987. Ionospheric time-delay algorithm for single-frequency GPS users. IEEE Transactions on
    Aerospace and Electronic Systems, v.AES-23, n.3, p.325-331.
    :param rx_lat: Receiver's latitude WGS84 (°)
    :param rx_lon: Receiver's longitude WGS84 (°)
    :param tow: GPS time of week (s)
    :param sv_elevation: Elevation of the satellite (rad)
    :param sv_azimuth: Azimuth of the satellite (rad)
    :param alpha: Broadcasted ephemeris parameters alpha
    :param beta: Broadcasted ephemeris parameters beta
    :return: Ionospheric delay (m)
    """
    # Convert to semicircles
    rx_lat = math.radians(rx_lat)# / math.pi
    rx_lon = math.radians(rx_lon)# / math.pi
    #sv_elevation = sv_elevation / math.pi
    #sv_azimuth = sv_azimuth / math.pi

    # 1. Calculate the earth-centred angle (elevation in semicircles).
    earth_centered_angle = (0.0137 / (sv_elevation + 0.11)) - 0.022

    # 2. Compute the latitude of the Ionospheric Pierce Point (IPP)
    ipp_lat = rx_lat + earth_centered_angle * math.cos(sv_azimuth)
    if abs(ipp_lat) > 0.416:
        ipp_lat = math.copysign(0.416, ipp_lat)

    # 3. Compute the longitude of the IPP.
    ipp_lon = rx_lon + ((earth_centered_angle * math.sin(sv_azimuth)) / math.cos(ipp_lat))

    # 4. Find the geomagnetic latitude of the IPP.
    ipp_mag_lat = ipp_lat + 0.064 * math.cos(ipp_lon - 1.617)

    # 5. Find the local time (in seconds) at the ionospheric pierce point.
    t_loc = (43200 * ipp_lon + tow) % 86400

    # 6. Compute the amplitude of ionospheric delay.
    A_i = 0
    for i in range(4):
        A_i += alpha[i] * (ipp_mag_lat ** i)
    if A_i < 0:
        A_i = 0

    # 7. Compute the period of ionospheric delay.
    P_i = 0
    for i in range(4):
        P_i += beta[i] * (ipp_mag_lat ** i)
    if P_i < 72000:
        P_i = 72000

    # 8. Compute the phase of ionospheric delay.
    X_i = 2 * math.pi * (t_loc - 50400) / P_i

    # 9. Compute the slant factor.
    F = 1 + 16 * (0.53 - sv_elevation) ** 3

    # 10. Compute the ionospheric time delay.
    if abs(X_i) < 1.57:
        delay = (5e-9 + A_i * (1 - (X_i ** 2) / 2 + (X_i ** 4) / 24)) * F
    else:
        delay = 5e-9 * F

    return delay * const.C


def compute_nequick():
    """
    not implemented
    https://gssc.esa.int/navipedia/index.php?title=NeQuick_Ionospheric_Model
    :return:
    """
    print("NeQuick not yet implemented")
    return -1


def compute_weather_param(rx_lat: float, day_of_year: int, param_name: Literal["P", "T", "e", "beta", "lambda"]) -> float:
    """
    Compute average and seasonal variations of the weather parameters at the receiver latitude linearly interpolated
    from mean weather data.
    source: https://gssc.esa.int/navipedia/index.php?title=Tropospheric_Delay
    :param rx_lat: Receiver's latitude WGS84 (°)
    :param day_of_year: Number of days since the 1st of january
    :param param_name: Name of the parameter ("P", "T", "e", "beta", "lambda")
    :return: Average weather parameter
    """
    if rx_lat > 0:
        Dmin = 28
    else:
        Dmin = 211

    # Get closest average meteo observation
    param0_name = f"{param_name}0"
    deltaparam_name = f"delta{param_name}"
    if rx_lat <= const.WEATHER_PARAM["latitude"][0]:
        param0 = const.WEATHER_PARAM[param0_name][0]
        deltaparam = const.WEATHER_PARAM[deltaparam_name][0]
    elif rx_lat >= const.WEATHER_PARAM["latitude"][-1]:
        param0 = const.WEATHER_PARAM[param0_name][-1]
        deltaparam = const.WEATHER_PARAM[deltaparam_name][-1]
    else:
        lat_list = const.WEATHER_PARAM["latitude"]
        for i in range(len(lat_list) - 1):
            if lat_list[i] <= rx_lat <= lat_list[i + 1]:
                break

        # Extrapolate average meteo parameter
        start0 = (lat_list[i], const.WEATHER_PARAM[param0_name][i])
        stop0 = (lat_list[i+1], const.WEATHER_PARAM[param0_name][i+1])
        param0 = start0[1] + (stop0[1] - start0[1]) * ((rx_lat - start0[0]) / (stop0[0] - start0[0]))
        deltastart = (lat_list[i], const.WEATHER_PARAM[deltaparam_name][i])
        deltastop = (lat_list[i + 1], const.WEATHER_PARAM[deltaparam_name][i + 1])
        deltaparam = (deltastart[1]
                      + (deltastop[1] - deltastart[1]) * ((rx_lat - deltastart[0]) / (deltastop[0] - deltastart[0])))

    weather_param = param0 - deltaparam * math.cos(2 * math.pi * (day_of_year - Dmin)/365.25)

    return weather_param

def compute_tropo_corrections(rx_lat: float, rx_alt: float, day_of_year: int, sv_elevation: float) -> float:
    """
    Compute tropospheric corrections for a receiver at day "day_of_year" for a satellite at elevation "sv_elevation".
    source: https://gssc.esa.int/navipedia/index.php?title=Tropospheric_Delay
    :param rx_lat: Receiver's latitude WGS84 (°)
    :param rx_alt: Receiver's altitude (m)
    :param day_of_year: Number of days since the 1st of january
    :param sv_elevation: Elevation of the satellite (rad)
    :return: Tropospheric delay (m)
    """
    if rx_alt < 0 or rx_alt>1000:
        rx_alt = 0
    # 1. Compute obliquity factor, valid for satellite elevation angles over 5 degrees
    M = 1.001/math.sqrt(0.002001 + math.sin(sv_elevation)**2)

    # 2. Estimate weather parameters
    w_P = compute_weather_param(rx_lat, day_of_year, "P")
    w_T = compute_weather_param(rx_lat, day_of_year, "T")
    w_e = compute_weather_param(rx_lat, day_of_year, "e")
    w_beta = compute_weather_param(rx_lat, day_of_year, "beta")
    w_lambda = compute_weather_param(rx_lat, day_of_year, "lambda")

    # 3a. Compute hydrostatic component delay
    # Its effect varies with local temperature and atmospheric pressure in quite a predictable manner, besides its
    # variation is less that the 1% in a few hours.
    # The error caused by this component is about 2.3 meters in the zenith direction and 10 meters for lower elevations.
    t_z0_dry = 1e-6 * const.K1 * const.RD * w_P / const.GM
    t_z_dry = (1 - (w_beta * rx_alt/w_T))**(const.G/(const.RD * w_beta)) * t_z0_dry

    # 3b. Compute wet component delay
    # It is caused by the water vapour and condensed water in form of clouds and depends on weather conditions.
    # The error caused by this component is only some tens of centimetres, but this component varies faster than the
    # hydrostatic component and a quite randomly way, being very difficult to model.
    t_z0_wet = (1e-6 * const.K2 * const.RD / ((w_lambda + 1) * const.GM - w_beta * const.RD)) * (w_e/w_T)
    t_z_wet = (1 - (w_beta * rx_alt/w_T))**(((w_lambda+1) * const.G / (const.RD * w_beta)) - 1) * t_z0_wet

    # 4. Corrections, .... assemble !
    tropo_delay = (t_z_dry + t_z_wet) * M

    return tropo_delay


def get_atmospheric_corrections(pd_gnss_raw: pd.DataFrame, pd_gnss_pvt: pd.DataFrame) -> pd.DataFrame:
    """
    Correct pseudoranges from pd_gnss_raw with ionospheric and tropospheric corrections using an approximate position
    from pd_gnss_pvt.
    sources :   https://gssc.esa.int/navipedia/index.php?title=Ionospheric_Delay
                https://gssc.esa.int/navipedia/index.php?title=Tropospheric_Delay
    :param pd_gnss_raw: GNSS raw dataframe from BITS parser
    :param pd_gnss_pvt: GNSS pvt dataframe
    :return: GNSS pvt dataframe with corrected pseudoranges
    """
    raw_required_columns = ["time", "pr_m", "gnss_id", "sv_id", "elevation_rad", "azimuth_rad", "ionospheric_param"]
    pvt_required_columns = ["time", "lat", "lon", "alt"]

    if not check_dataframe(pd_gnss_raw, raw_required_columns):
        warnings.warn("Missing raw data, cannot compute atmospheric correction.")
        return pd_gnss_raw
    if not check_dataframe(pd_gnss_pvt, pvt_required_columns):
        warnings.warn("Missing PVT data, cannot compute atmospheric correction.")
        return pd_gnss_raw

    # Check for already existing corrections
    if not {"corr_pr_m"}.issubset(pd_gnss_raw.columns):
        pd_gnss_raw["corr_pr_m"] = pd_gnss_raw["pr_m"]
    elif {"atm_corr_m"}.issubset(pd_gnss_raw.columns):
        pd_gnss_raw["corr_pr_m"] += pd_gnss_raw["atm_corr_m"]

    # Loop over all timestamps
    timestamp_list = pd_gnss_raw["time"].unique().tolist()
    for timestamp in timestamp_list:
        if timestamp in pd_gnss_pvt["time"].values:
            #  Get raw and pvt at timestamp
            pd_gnss_raw_at_timestamp = pd_gnss_raw[pd_gnss_raw["time"] == timestamp].copy()
            pd_gnss_pvt_at_timestamp = pd_gnss_pvt[pd_gnss_pvt["time"] == timestamp]
            if len(pd_gnss_pvt_at_timestamp) != 1:
                txt = f"Position estimate has several possibilities at timestamp {timestamp}. Using first instance to compute atmospheric corrections."
                warnings.warn(txt, UserWarning)
            pd_ser_gnss_pvt_at_timestamp = pd_gnss_pvt_at_timestamp.iloc[0]

            # 1. Compute ionospheric delays
            pd_gnss_raw_at_timestamp["iono_corr_m"] = pd_gnss_raw_at_timestamp.apply(
                lambda row: compute_klobuchar(pd_ser_gnss_pvt_at_timestamp["lat"],
                                              pd_ser_gnss_pvt_at_timestamp["lon"], timestamp.tow(),
                                              row["elevation_rad"], row["azimuth_rad"],
                                              row["ionospheric_param"][:4], row["ionospheric_param"][4:]), axis=1)
            pd_gnss_raw.loc[pd_gnss_raw_at_timestamp.index, ["iono_corr_m"]] = \
                pd_gnss_raw_at_timestamp["iono_corr_m"]

            # 2. Compute tropospheric delays
            pd_gnss_raw_at_timestamp["tropo_corr_m"] = pd_gnss_raw_at_timestamp.apply(
                lambda row: compute_tropo_corrections(pd_ser_gnss_pvt_at_timestamp["lat"],
                                                      pd_ser_gnss_pvt_at_timestamp["alt"],
                                                      timestamp.timestamp_pd.day_of_year, row["elevation_rad"]), axis=1)
            pd_gnss_raw.loc[pd_gnss_raw_at_timestamp.index, ["tropo_corr_m"]] = \
                pd_gnss_raw_at_timestamp["tropo_corr_m"]

    # 3. Correct pseudoranges
    pd_gnss_raw["atm_corr_m"] = pd_gnss_raw["iono_corr_m"] + pd_gnss_raw["tropo_corr_m"]
    pd_gnss_raw["corr_pr_m"] -= pd_gnss_raw["atm_corr_m"]

    return pd_gnss_raw
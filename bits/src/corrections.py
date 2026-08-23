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
import warnings
from typing import Literal

from bits.src import const, convert, sv_model, utils

# Clock corrections
def compute_satellite_clock_correction(time:np.ndarray[np.datetime64], toe:np.ndarray[np.datetime64],
                                       a0:np.ndarray[np.float64], a1:np.ndarray[np.float64], a2:np.ndarray[np.float64]) \
        -> np.ndarray[np.float64]:
    """
    Compute polynomial satellite clock correction.
    source: https://gssc.esa.int/navipedia/index.php/Clock_Modelling
    :param time: Time at which the satellite's position should be computed in UTC (datetime64)
    :param toe: Ephemeris data reference time in UTC (datetime64)
    :param a0: SV clock bias (s)
    :param a1: SV clock drift (s^-1)
    :param a2: SV clock drift rate (s^-2)
    :return: Polynomial clock correction (s)
    """
    tk = (time - toe) / np.timedelta64(1, 's')
    satellite_clock_correction = a0 + a1*tk + np.sign(tk) * a2 * (tk ** 2)
    return satellite_clock_correction

def compute_relativistic_clock_correction_kepler(time:np.ndarray[np.datetime64], toe:np.ndarray[np.datetime64],
                                                 sqrta:np.ndarray, e:np.ndarray, deltan:np.ndarray, m0:np.ndarray) \
        -> np.ndarray:
    """
    Computes relativistic clock corrections using kepler elements

    :param time: Time at which the satellite's position should be computed in UTC (datetime64)
    :param toe: Ephemeris data reference time in UTC (datetime64)
    :param a: square root of semi-major axis (sqrt(m))
    :param e: eccentricity ()
    :param deltan: mean Motion difference from computed value at reference time (semi-circles/s)
    :param m0: mean anomaly at reference time (semi-circles)
    :return: eccentric anomaly (rad)
    :return: Relativistic clock corrections (s)
    """
    tk = (time - toe) / np.timedelta64(1, 's')

    # Corrected mean motion
    n = sv_model.compute_corrected_mean_motion(sqrta ** 2, deltan)

    # Eccentric anomaly
    ek = sv_model.compute_eccentric_anomaly(tk, e, m0, n)

    F = -2 * np.sqrt(const.NU) / const.C**2

    return F * e * sqrta * np.sin(ek)

def compute_relativistic_clock_correction_state(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                                                vx: np.ndarray, vy: np.ndarray, vz: np.ndarray) -> np.ndarray:
    """
    Computes relativistic clock corrections using SV state

    :param x: Position of the satellite on the x axis in ECEF (m)
    :param y: Position of the satellite on the y axis in ECEF (m)
    :param z: Position of the satellite on the z axis in ECEF (m)
    :param vx: Speed of the satellite on the x axis in ECEF (m/s)
    :param vy: Speed of the satellite on the y axis in ECEF (m/s)
    :param vz: Speed of the satellite on the z axis in ECEF (m/s)
    :return: Relativistic clock corrections (s)
    """
    r_dot_v = x*vx + y*vy + z*vz
    return -2.0 * r_dot_v / const.C**2

def get_clock_corrections(pd_gnss_raw: pd.DataFrame, pd_ephemeris: pd.DataFrame) -> pd.DataFrame:
    """
    Compute clock corrections using a pd.Dataframe ephemeris from the BITS ephemeris parser for GPS and Galileo
    :param pd_gnss_raw: GNSS raw dataframe from BITS parser
    :param pd_ephemeris: ephemeris dataframe from BITS parser
    :return: raw data with corrected pseudoranges and corresponding clock corrections
    """
    raw_required_columns = ["time", "pr_m", "gnss_id", "sv_id"]

    ephem_required_columns = ["time", "time_of_ephemeris", "sv_id"]
    ephem_sv_clock_required_columns = ["clock_bias", "clock_drift", "clock_drift_rate"]
    ephem_relat_kepler_required_columns = ["sqrta", "e", "deltan", "m0"]
    raw_relat_state_required_columns = ["x_sv_m", "y_sv_m", "z_sv_m", "vx_sv_mps", "vy_sv_mps", "vz_sv_mps", ]

    ephem_col_to_get = ["time_of_ephemeris"]

    # Check raw dataframe
    if not utils.check_dataframe(pd_gnss_raw, raw_required_columns):
        warnings.warn("Missing columns in raw, cannot compute clock correction.")
        return pd_gnss_raw

    # Check ephemeris dataframe
    if not utils.check_dataframe(pd_ephemeris, ephem_required_columns):
        warnings.warn("Missing columns in ephemeris, cannot compute clock correction.")
        return pd_gnss_raw

    # Polynomial SV clock corrections
    if not utils.check_dataframe(pd_ephemeris, ephem_sv_clock_required_columns):
        warnings.warn("Missing columns in ephemeris, cannot compute SV clock correction.")
        poly_ok = False
    else:
        poly_ok = True
        ephem_col_to_get += ephem_sv_clock_required_columns

    # Relativistic clock correction
    # Galileo, GPS, Beidou
    if pd_gnss_raw["gnss_id"].isin(["gal", "gps", "bei"]).any():
        if utils.check_dataframe(pd_ephemeris, ephem_relat_kepler_required_columns):
            relat_kepler_ok = True
            ephem_col_to_get += ephem_relat_kepler_required_columns
        else:
            warnings.warn("Missing columns in ephemeris, cannot compute relativistic clock correction.")
            relat_kepler_ok = False
    else:
        relat_kepler_ok = False
    # Glonass
    if pd_gnss_raw["gnss_id"].isin(["glo"]).any():
        if utils.check_dataframe(pd_gnss_raw, raw_relat_state_required_columns):
            relat_state_ok = True
        else:
            warnings.warn("Missing columns in raw, cannot compute relativistic clock correction. Please use bits.get_sv_states(raw_df) first.")
            relat_state_ok = False
    else:
        relat_state_ok = False

    # TGD
    if utils.check_dataframe(pd_ephemeris, ["tgd"]):
        tgd_ok = True
        ephem_col_to_get += ["tgd"]
    else:
        warnings.warn("Missing columns in ephemeris, cannot compute time group delay clock correction.")
        tgd_ok = False

    # Get ephemeris data
    pd_gnss_raw = utils.get_data_from_ephemeris(pd_gnss_raw, pd_ephemeris, ephem_col_to_get)

    # Check for already existing corrections
    if not {"corr_pr_m"}.issubset(pd_gnss_raw.columns):
        pd_gnss_raw["corr_pr_m"] = pd_gnss_raw["pr_m"]
    elif {"clock_corr_m"}.issubset(pd_gnss_raw.columns): # Uncorrect pr so that we don't correct it twice
        pd_gnss_raw["corr_pr_m"] -= pd_gnss_raw["clock_corr_m"]

    pd_gnss_raw["clock_corr_m"] = 0.0

    # 1) Compute satellite clock correction:
    if poly_ok:
        pd_gnss_raw["poly_clock_corr_m"] = (
                const.C * compute_satellite_clock_correction(pd_gnss_raw["time"],pd_gnss_raw["time_of_ephemeris"],
                                                             pd_gnss_raw["clock_bias"], pd_gnss_raw["clock_drift"],
                                                             pd_gnss_raw["clock_drift_rate"]))
        pd_gnss_raw["clock_corr_m"] += pd_gnss_raw["poly_clock_corr_m"]

    # 2) Compute relativistic clock corrections
    glo_mask = pd_gnss_raw["gnss_id"] == "glo"
    if relat_kepler_ok:
        pd_gnss_raw.loc[~glo_mask, "relat_clock_corr_m"] = (
                const.C * compute_relativistic_clock_correction_kepler(pd_gnss_raw["time"],pd_gnss_raw["time_of_ephemeris"],
                                                                       pd_gnss_raw["sqrta"],pd_gnss_raw["e"],
                                                                       pd_gnss_raw["deltan"],pd_gnss_raw["m0"],))
        pd_gnss_raw.loc[~glo_mask, "clock_corr_m"] += pd_gnss_raw.loc[~glo_mask, "relat_clock_corr_m"]

    if relat_state_ok:
        pd_gnss_raw.loc[glo_mask, "relat_clock_corr_m"] = (
                const.C * compute_relativistic_clock_correction_state(pd_gnss_raw["x_sv_m"],pd_gnss_raw["y_sv_m"],
                                                                      pd_gnss_raw["z_sv_m"], pd_gnss_raw["vx_sv_mps"],
                                                                      pd_gnss_raw["vy_sv_mps"], pd_gnss_raw["vz_sv_mps"]))
        pd_gnss_raw.loc[glo_mask, "clock_corr_m"] += pd_gnss_raw.loc[glo_mask, "relat_clock_corr_m"]

    # 3) Time group delay
    if tgd_ok:
        pd_gnss_raw.loc[pd_gnss_raw["tgd"].notna(), "clock_corr_m"] -= pd_gnss_raw["tgd"] * const.C

    # 4) Correct pseudoranges
    pd_gnss_raw["corr_pr_m"] += pd_gnss_raw["clock_corr_m"].fillna(0)

    return pd_gnss_raw.drop(ephem_col_to_get, axis="columns", errors="ignore")


########################################################################################################################
# Atmospheric corrections
def compute_klobuchar(time: np.ndarray[np.datetime64], rx_lat: np.ndarray, rx_lon: np.ndarray,
                      sv_elevation: np.ndarray, sv_azimuth: np.ndarray, signal_freq: np.ndarray,
                      alpha: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
                      beta: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    """
    Compute ionospheric delay using Klobuchar's model.
    GPS satellites broadcast the parameters of the Klobuchar ionospheric model for single frequency users. This
    broadcast model is based on an empirical approach and is estimated to reduce about the 50% RMS ionospheric range
    error worldwide.
    source: https://gssc.esa.int/navipedia/index.php?title=Klobuchar_Ionospheric_Model
    Klobuchar, J. A. 1987. Ionospheric time-delay algorithm for single-frequency GPS users. IEEE Transactions on
    Aerospace and Electronic Systems, v.AES-23, n.3, p.325-331.
    :param time: Time in UTC (datetime64)
    :param rx_lat: Receiver's latitude WGS84 (°)
    :param rx_lon: Receiver's longitude WGS84 (°)
    :param sv_elevation: Elevation of the satellite (rad)
    :param sv_azimuth: Azimuth of the satellite (rad)
    :param signal_freq: Signal frequency (Hz)
    :param alpha: Broadcasted ephemeris parameters alpha
    :param beta: Broadcasted ephemeris parameters beta
    :return: Ionospheric delay (m)
    """
    # Convert to semicircles
    rx_lat = np.radians(rx_lat)
    rx_lon = np.radians(rx_lon)
    sv_elevation = sv_elevation / np.pi

    # 1. Calculate the earth-centred angle (elevation in semicircles).
    earth_centered_angle = (0.0137 / (sv_elevation + 0.11)) - 0.022

    # 2. Compute the latitude of the Ionospheric Pierce Point (IPP)
    ipp_lat = rx_lat + earth_centered_angle * np.cos(sv_azimuth)
    ipp_lat = np.where(abs(ipp_lat) > 0.416, np.copysign(0.416, ipp_lat), ipp_lat)

    # 3. Compute the longitude of the IPP.
    ipp_lon = rx_lon + ((earth_centered_angle * np.sin(sv_azimuth)) / np.cos(ipp_lat))

    # 4. Find the geomagnetic latitude of the IPP.
    ipp_mag_lat = ipp_lat + 0.064 * np.cos(ipp_lon - 1.617)

    # 5. Find the local time (in seconds) at the ionospheric pierce point.
    t_loc = 43200 * ipp_lon + convert.time.utc_to_tow(time, gnss_id="gps") / np.timedelta64(1, "s")
    t_loc = np.where(t_loc >= 86400, t_loc - 86400, t_loc)
    t_loc = np.where(t_loc < 0, t_loc + 86400, t_loc)

    # 6. Compute the amplitude of ionospheric delay.
    A_i = 0
    for i in range(4):
        A_i += alpha[i] * (ipp_mag_lat ** i)
    A_i = np.where(A_i < 0, 0, A_i)

    # 7. Compute the period of ionospheric delay.
    P_i = 0
    for i in range(4):
        P_i += beta[i] * (ipp_mag_lat ** i)
    P_i = np.where(P_i < 72000, 72000, P_i)

    # 8. Compute the phase of ionospheric delay.
    X_i = 2 * np.pi * (t_loc - 50400) / P_i

    # 9. Compute the slant factor.
    F = 1 + 16 * (0.53 - sv_elevation) ** 3

    # 10. Compute the ionospheric time delay.
    delay = np.where(abs(X_i) < 1.57, (5e-9 + A_i * (1 - (X_i ** 2) / 2 + (X_i ** 4) / 24)) * F, 5e-9 * F)

    # 11 Convert to sinal frequency
    delay = (const.band_frequency["l1"]/signal_freq) ** 2 * delay

    return delay * const.C


def compute_nequick():
    """
    not implemented
    https://gssc.esa.int/navipedia/index.php?title=NeQuick_Ionospheric_Model
    :return:
    """
    raise NotImplementedError("NeQuick ionospheric model is not yet implemented. "
                              "Please use bits.corrections.compute_klobuchar() instead.")


def compute_weather_param(rx_lat: np.ndarray, day_of_year: np.ndarray,
                          param_name: Literal["P", "T", "e", "beta", "lambda"]) -> np.ndarray:
    """
    Compute average and seasonal variations of the weather parameters at the receiver latitude linearly interpolated
    from mean weather data.
    source: https://gssc.esa.int/navipedia/index.php?title=Tropospheric_Delay
    :param rx_lat: Receiver's latitude WGS84 (°)
    :param day_of_year: Number of days since the 1st of january
    :param param_name: Name of the parameter ("P", "T", "e", "beta", "lambda")
    :return: Average weather parameter
    """
    Dmin = np.where(rx_lat < 0, 28, 211)
    rx_lat = np.abs(rx_lat)

    # Get closest average meteo observation
    param0_name = f"{param_name}0"
    deltaparam_name = f"delta{param_name}"

    # Extrapolate average meteo parameter
    lat_list = const.WEATHER_PARAM["latitude"]
    lat_index = np.searchsorted(lat_list, rx_lat, side="left")

    lat_left = np.array(lat_list)[lat_index - 1]
    lat_right = np.array(lat_list)[lat_index]

    param_left = np.array(const.WEATHER_PARAM[param0_name])[lat_index - 1]
    param_right = np.array(const.WEATHER_PARAM[param0_name])[lat_index]

    deltaparam_left = np.array(const.WEATHER_PARAM[deltaparam_name])[lat_index - 1]
    deltaparam_right = np.array(const.WEATHER_PARAM[deltaparam_name])[lat_index]

    param0 = param_left + (param_right - param_left) * ((rx_lat - lat_left) / (lat_right - lat_left))
    deltaparam = deltaparam_left + (deltaparam_right - deltaparam_left) * ((rx_lat - lat_left) / (lat_right - lat_left))

    # Default values for lat out of table
    param0 = np.where(rx_lat <= const.WEATHER_PARAM["latitude"][0], const.WEATHER_PARAM[param0_name][0], param0)
    deltaparam = np.where(rx_lat <= const.WEATHER_PARAM["latitude"][0], const.WEATHER_PARAM[deltaparam_name][0], deltaparam)

    param0 = np.where(rx_lat >= const.WEATHER_PARAM["latitude"][-1], const.WEATHER_PARAM[param0_name][-1], param0)
    deltaparam = np.where(rx_lat >= const.WEATHER_PARAM["latitude"][-1], const.WEATHER_PARAM[deltaparam_name][-1], deltaparam)

    weather_param = param0 - deltaparam * np.cos(2 * np.pi * (day_of_year - Dmin)/365.25)

    return weather_param

def compute_tropo_corrections(rx_lat: np.ndarray, rx_alt: np.ndarray, day_of_year: np.ndarray,
                              sv_elevation: np.ndarray) -> np.ndarray:
    """
    Compute tropospheric corrections for a receiver at day "day_of_year" for a satellite at elevation "sv_elevation".
    source: https://gssc.esa.int/navipedia/index.php?title=Tropospheric_Delay
    :param rx_lat: Receiver's latitude WGS84 (°)
    :param rx_alt: Receiver's altitude (m)
    :param day_of_year: Number of days since the 1st of january
    :param sv_elevation: Elevation of the satellite (rad)
    :return: Tropospheric delay (m)
    """
    rx_alt = np.where(rx_alt < 0, 0.0, rx_alt)
    rx_alt = np.where(rx_alt > 1000, 1000.0, rx_alt)

    # 1. Compute obliquity factor, valid for satellite elevation angles over 5 degrees
    M = 1.001/np.sqrt(0.002001 + np.sin(sv_elevation)**2)

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


def get_atmospheric_corrections(pd_gnss_raw: pd.DataFrame, pd_ephemeris: pd.DataFrame, pd_gnss_pvt: pd.DataFrame)\
        -> pd.DataFrame:
    """
    Correct pseudoranges from pd_gnss_raw with ionospheric and tropospheric corrections using an approximate position
    from pd_gnss_pvt.
    sources :   https://gssc.esa.int/navipedia/index.php?title=Ionospheric_Delay
                https://gssc.esa.int/navipedia/index.php?title=Tropospheric_Delay
    :param pd_gnss_raw: GNSS raw dataframe from BITS parser
    :param pd_ephemeris: ephemeris dataframe from BITS parser
    :param pd_gnss_pvt: GNSS pvt dataframe
    :return: GNSS raw dataframe with corrected pseudoranges
    """
    raw_required_columns = ["time", "pr_m", "gnss_id", "sv_id", "elevation_rad", "azimuth_rad", "frequency_hz"]
    pvt_required_columns = ["time", "lat", "lon", "alt"]
    ephem_klo_columns = ["klo_a0", "klo_a1", "klo_a2", "klo_a3", "klo_b0", "klo_b1", "klo_b2", "klo_b3"]

    # Check dataframes
    if not utils.check_dataframe(pd_gnss_raw, raw_required_columns):
        warnings.warn("Missing raw data, cannot compute atmospheric correction.")
        return pd_gnss_raw
    if not utils.check_dataframe(pd_gnss_pvt, pvt_required_columns):
        warnings.warn("Missing PVT data, cannot compute atmospheric correction.")
        return pd_gnss_raw
    if not utils.check_dataframe(pd_ephemeris, ephem_klo_columns):
        warnings.warn("Missing Klobuchar parameters in ephemeris, cannot compute atmospheric correction.")
        return pd_gnss_raw

    # Check for already existing corrections
    if not {"corr_pr_m"}.issubset(pd_gnss_raw.columns):
        pd_gnss_raw["corr_pr_m"] = pd_gnss_raw["pr_m"]
    elif {"atm_corr_m"}.issubset(pd_gnss_raw.columns):
        pd_gnss_raw["corr_pr_m"] += pd_gnss_raw["atm_corr_m"]

    # Get data from pvt and ephemeris
    # Sort
    pd_gnss_raw = pd_gnss_raw.sort_values("time").reset_index(drop=True)
    pd_gnss_pvt = pd_gnss_pvt.sort_values("time").reset_index(drop=True)
    # Get data from PVT
    pd_gnss_raw = pd.merge_asof(
        pd_gnss_raw,
        pd_gnss_pvt[pvt_required_columns],
        on="time",
        direction="nearest",
        suffixes=("_raw", "")
    )
    # Get data from ephemeris
    pd_gnss_raw = utils.get_data_from_ephemeris(pd_gnss_raw, pd_ephemeris, ephem_klo_columns)

    # 1. Compute ionospheric delays
    # Compute Klobuchar corrections
    pd_gnss_raw["iono_corr_m"] = (
        compute_klobuchar(time=pd_gnss_raw["time"], signal_freq=pd_gnss_raw["frequency_hz"],
                          rx_lat=pd_gnss_raw["lat"], rx_lon=pd_gnss_raw["lon"],
                          sv_elevation=pd_gnss_raw["elevation_rad"], sv_azimuth=pd_gnss_raw["azimuth_rad"],
                          alpha=(pd_gnss_raw["klo_a0"], pd_gnss_raw["klo_a1"],
                                 pd_gnss_raw["klo_a2"], pd_gnss_raw["klo_a3"]),
                          beta=(pd_gnss_raw["klo_b0"], pd_gnss_raw["klo_b1"],
                                pd_gnss_raw["klo_b2"], pd_gnss_raw["klo_b3"])))

    # 2. Compute tropospheric delays
    pd_gnss_raw["tropo_corr_m"] = compute_tropo_corrections(rx_lat=pd_gnss_raw["lat"], rx_alt=pd_gnss_raw["alt"],
                                                            day_of_year=convert.time.day_of_year(pd_gnss_raw["time"]),
                                                            sv_elevation=pd_gnss_raw["elevation_rad"])

    # 3. Correct pseudoranges
    pd_gnss_raw["atm_corr_m"] = pd_gnss_raw["iono_corr_m"].fillna(0) + pd_gnss_raw["tropo_corr_m"].fillna(0)
    pd_gnss_raw["corr_pr_m"] -= pd_gnss_raw["atm_corr_m"]

    return pd_gnss_raw.drop(pvt_required_columns[1:]+ephem_klo_columns, axis="columns", errors="ignore")


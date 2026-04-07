# Baguette in the sky

A simple python GNSS library.

![Logo](bits_logo.jpg)

---
## BITS dataframes definition
### BITS ephemeris dataframe
Ephemeris parameters that can be found in a rinex nav file.

| Name | Description | Unit | Type |
|------|-------------|------|------|
| time | Receiver timestamp | | GnssTimestamp |
| toe | Reference time, ephemeris parameters | second | float |
| sqrta | Square root of the semi-major axis | metre^1/2 | float |
| e | Eccentricity | | float |
| i0 | Inclination angle at reference time | radian(semicircles) | float |
| idot | Rate of change of inclination | radian(semicircles)/second | float |
| omega0 | Longitude of ascending node at reference time | radian(semicircles) | float |
| omega | Argument of perigee | radian(semicircles) | float |
| m0 | Mean anomaly at reference time | radian(semicircles) | float |
| omegadot | Rate of change of right ascension | radian(semicircles)/second | float |
| deltan | Mean motion difference from computed value | radian(semicircles)/second | float |
| cuc | Amplitude of the cosine harmonic correction term to the argument of latitude | radian | float |
| cus | Amplitude of the sine harmonic correction term to the argument of latitude | radian | float |
| crc | Amplitude of the cosine harmonic correction term to the orbit radius | metre | float |
| crs | Amplitude of the sine harmonic correction term to the orbit radius | metre | float |
| cic | Amplitude of the cosine harmonic correction term to the angle of inclination | radian | float |
| cis | Amplitude of the sine harmonic correction term to the angle of inclination | radian | float |
| clock_bias | Satellite clock bias | second | float |
| clock_drift | Satellite clock drift | second/second | float |
| clock_drift_rate | Satellite clock drift rate | second/second² | float |
| tgd | Time Group Delay | second | float |
| ionospheric_param | 8 parameters to compute ionospheric corrections | | list[float] |

### BITS raw dataframe
Raw measurements. Can also contain BITS ephemeris dataframe.

| Name | Description | Unit | Type |
|------|-------------|------|------|
| time | Receiver timestamp | | GnssTimestamp |
| corr_time | Corrected receiver timestamp | | GnssTimestamp |
| pr_m | Pseudorange | metre | float |
| corr_pr_m | Corrected pseudorange | metre | float |
| pr_rate_mps | Pseudorange rate | metre/second | float |
| doppler_hz | Doppler shift | 1/second | float |
| sv_id | Satellite PRN number | | int |
| gnss_id | GNSS constellation ID (BITS normalized, cf normalize_gnss_constellation()) | | str |
| x_sv_m | X ECEF coordinate of the satellite | metre | float |
| y_sv_m | Y ECEF coordinate of the satellite | metre | float |
| z_sv_m | Z ECEF coordinate of the satellite | metre | float |
| vx_sv_mps | X ECEF coordinate of the satellite speed | metre/second | float |
| vy_sv_mps | Y ECEF coordinate of the satellite speed | metre/second | float |
| vz_sv_mps | Z ECEF coordinate of the satellite speed | metre/second | float |
| ax_sv_mpss | X ECEF coordinate of the satellite acceleration | metre/second² | float |
| ay_sv_mpss | Y ECEF coordinate of the satellite acceleration | metre/second² | float |
| az_sv_mpss | Z ECEF coordinate of the satellite acceleration | metre/second² | float |
| clock_corr_m | Sum of the clock corrections to be applied to corr_pr_m | metre | float |
| poly_clock_corr_m | Polynomial clock correction | metre | float |
| relat_clock_corr_m | Relativistic clock correction | metre | float |
| tgd_clock_corr_m | Time Group Delay (clock correction) | metre | float |
| atm_corr_m | Sum of the atmospheric corrections to be applied to corr_pr_m | metre | float |
| iono_corr_m | Ionospheric correction | metre | float |
| tropo_corr_m | Tropospheric correction | metre | float |

### BITS PVT dataframe
Computed position speed and time.

| Name | Description | Unit | Type |
|------|-------------|------|------|
| time | Receiver timestamp | | GnssTimestamp |
| corr_time | Corrected receiver timestamp | | GnssTimestamp |
| lat | Latitude of the receiver (WGS 84) | degree | float |
| lon | Longitude of the receiver (WGS 84) | degree | float |
| alt | Altitude of the receiver (WGS 84) | metre | float |
| x_rx_m | X ECEF coordinate of the receiver | metre | float |
| y_rx_m | Y ECEF coordinate of the receiver | metre | float |
| z_rx_m | Z ECEF coordinate of the receiver | metre | float |
| b_rx_m | Receiver remaining clock bias from corr_time | metre | float |
| ols_convergence | Distance from last computed position | metre | float |
| vx_rx_mps | X ECEF coordinate of the receiver speed | metre/second | float |
| vy_rx_mps | Y ECEF coordinate of the receiver speed | metre/second | float |
| vz_rx_mps | Z ECEF coordinate of the receiver speed | metre/second | float |
| vb_rx_mps | Receiver clock drift | metre/second | float |
## Module `bits.src.const`

Constant to be used in BITS

---


---


## Module `bits.src.corrections`

Used to apply further corrections to raw pseudoranges

---

### `compute_klobuchar(rx_lat: float, rx_lon: float, tow: float, sv_elevation: float, sv_azimuth: float, alpha: tuple, beta: tuple) -> float`

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

### `compute_nequick()`

not implemented
https://gssc.esa.int/navipedia/index.php?title=NeQuick_Ionospheric_Model
:return:

### `compute_relativistic_clock_correction(e, sqrta, eccentric_anomaly)`

Compute relativistic satellite clock correction.
source: https://gssc.esa.int/navipedia/index.php?title=Relativistic_Clock_Correction
:param e: Eccentricity (dimensionless)
:param sqrta: Square root of the semi-major axis (sqrt(m))
:param eccentric_anomaly: Eccentric anomaly, use sv_model.compute_eccentric_anomaly
:return: Relativistic clock correction (s)

### `compute_satellite_clock_correction(dt, a0, a1, a2) -> float`

Compute polynomial satellite clock correction.
source: https://gssc.esa.int/navipedia/index.php/Clock_Modelling
:param dt: Time from sv time of clock (s)
:param a0: SV clock bias (s)
:param a1: SV clock drift (s^-1)
:param a2: SV clock drift rate (s^-2)
:return: Polynomial clock correction (s)

### `compute_tropo_corrections(rx_lat: float, rx_alt: float, day_of_year: int, sv_elevation: float) -> float`

Compute tropospheric corrections for a receiver at day "day_of_year" for a satellite at elevation "sv_elevation".
source: https://gssc.esa.int/navipedia/index.php?title=Tropospheric_Delay
:param rx_lat: Receiver's latitude WGS84 (°)
:param rx_alt: Receiver's altitude (m)
:param day_of_year: Number of days since the 1st of january
:param sv_elevation: Elevation of the satellite (rad)
:return: Tropospheric delay (m)

### `compute_weather_param(rx_lat: float, day_of_year: int, param_name: Literal) -> float`

Compute average and seasonal variations of the weather parameters at the receiver latitude linearly interpolated
from mean weather data.
source: https://gssc.esa.int/navipedia/index.php?title=Tropospheric_Delay
:param rx_lat: Receiver's latitude WGS84 (°)
:param day_of_year: Number of days since the 1st of january
:param param_name: Name of the parameter ("P", "T", "e", "beta", "lambda")
:return: Average weather parameter

### `get_atmospheric_corrections(pd_gnss_raw: DataFrame, pd_gnss_pvt: DataFrame) -> DataFrame`

Correct pseudoranges from pd_gnss_raw with ionospheric and tropospheric corrections using an approximate position
from pd_gnss_pvt.
sources :   https://gssc.esa.int/navipedia/index.php?title=Ionospheric_Delay
            https://gssc.esa.int/navipedia/index.php?title=Tropospheric_Delay
:param pd_gnss_raw: GNSS raw dataframe from BITS parser
:param pd_gnss_pvt: GNSS pvt dataframe
:return: GNSS pvt dataframe with corrected pseudoranges

### `get_clock_corrections(pd_gnss_raw: DataFrame, pd_ephemeris: DataFrame = None) -> DataFrame`

Compute clock corrections using a pd.Dataframe ephemeris from the BITS ephemeris parser for GPS and Galileo
:param pd_gnss_raw: GNSS raw dataframe from BITS parser
:param pd_ephemeris: ephemeris dataframe from BITS parser
:return: raw data with corrected pseudoranges and corresponding clock corrections


---


## Module `bits.src.naming`

Normalization of namings to be used in BITS

Inside BITS, GNSS constellations should be referred as:
    GPS -> "gps"
    Galileo -> "gal"
    Glonass -> "glo"
    Beidou -> "bei"

---

### `normalize_gnss_constellation(name: str) -> str`

This function takes a constellation name and converts it to its normalized name, regardless of the format used, and
ignoring case.
:param name: not normalized GNSS constellation name
:return: BITS normalized GNSS constellation name


---


## Module `bits.src.plotter`

Functions to plot positions.

---

### `plot(pd_gnss_pvt: DataFrame, m: folium.map = None, plot_name: str = 'Plot') -> folium.map`

Plots locations on an open street map from a Dataframe. Dataframe must include columns "lat" and "lon".
:param pd_gnss_pvt: Dataframe containing columns "lat" and "lon".
:param m: folium map
:param plot_name: Name of the dataset
:return: folium map

### `plot3d(data)`

Traceurs 3d de satellites

:param data: [liste_x_ecef, liste_y_ecef, liste_z_ecef, liste_noms]
:return:


---


## Module `bits.src.reference_frame_object`

Object constructors to be used in BITS to handle metrics in a reference frame. Solves reference frame ambiguity and ease
conversions.

---

### Class `GnssTimestamp`

BITS' time reference frame object. This constructor must be used to handle time metrics.
Based on Panda's Timestamp, enables basics operations and GPS specific conversions. Use GnssTimestamp.timestamp_pd
for panda's Timestamp functions.

### Panda's Timestamp ###
    It looks like datetime.datetime, it tastes like datetime.datetime, but it is not datetime.datetime, it's
    pd.Timestamp !
    Accuracy down to the nanoseconds, unlike the microsecond accuracy of datetime. Easier to use than np.datetime64
    and takes timezone into account.
    https://pandas.pydata.org/docs/reference/api/pandas.Timestamp.html

### `check_utc(self)`

Ensures that a pandas Timestamp is in UTC.

### `gps_time(self) -> float`

_No documentation provided._

### `gps_week(self) -> int`

_No documentation provided._

### `local_time(self) -> str`

_No documentation provided._

### `pd_timestamp(self) -> Timestamp`

_No documentation provided._

### `sidereal(self) -> float`

_No documentation provided._

### `tow(self) -> float`

_No documentation provided._


---


## Module `bits.src.spp`

Scripts to enable single point positioning (SPP)

Usage:
======
get_position_estimate(pd_raw)

---

### Class `PositionEstimationError`

Exception raised for errors during position estimation.

### `compute_geometry_matrix(sv_position: array, rx_pos: array) -> array`

Computes a geometry matrix between two position in ECEF.
:param sv_position: Satellite vehicule position in ecef (meters) np.Array([[X], [Y], [Z]]) (column)
:param rx_pos: Receiver position in ecef (meters) np.Array([X, Y, Z]) (line)
:return: geometry matrix

### `compute_position_estimate(pseudorange: array, geometry_matrix: array) -> array`

Performs a simple Ordinary Least Square to compute position estimate.
sources:    https://gssc.esa.int/navipedia/index.php?title=Weighted_Least_Square_Solution_(WLS)
            https://gssc.esa.int/navipedia/index.php?title=Best_Linear_Unbiased_Minimum-Variance_Estimator_(BLUE)
:param pseudorange: Pseudoranges (m). Array with dim==1
:param geometry_matrix: Geometry matrix built with compute_geometry_matrix. Shape[0] must be the same length as
pseudorange.
:return: Position estimate (same length as geometry_matrix.shape[0], usually (x_ecef, y_ecef, z_ecef, b_ecef)

### `compute_speed_estimate(pr_rate: array, geometry_matrix: array, sv_speed: array) -> array`

Performs a simple Ordinary Least Square to compute speed estimate.
sources:    https://gssc.esa.int/navipedia/index.php?title=Code_Based_Positioning_(SPS)
            https://gssc.esa.int/navipedia/index.php?title=Parameters_adjustment
:param pr_rate: Pseudorange rates (m/s). Array with dim==1
:param geometry_matrix: Geometry matrix built with compute_geometry_matrix. Shape[0] must be the same length as
pr_rate.
:param sv_speed: Satellite speed in ECEF (m/s)
:return: Speed estimate (same length as geometry_matrix.shape[0], usually (vx_ecef, vy_ecef, vz_ecef, vb_ecef)

### `get_approx_position_estimate(pd_gnss_raw: DataFrame, pd_gnss_approx_pvt: DataFrame = None, approx_pvt: tuple = (0, 0, 0), convergence_tolerance = 1e-07, max_iteration: int = 10) -> DataFrame`

Computes position without any corrections.
sources:    https://gssc.esa.int/navipedia/index.php?title=Code_Based_Positioning_(SPS)
            https://gssc.esa.int/navipedia/index.php?title=Parameters_adjustment
:param pd_gnss_raw: GNSS raw dataframe from BITS parser
:param pd_gnss_approx_pvt: GNSS pvt dataframe with approximate pvt
:param approx_pvt: Position (ECEF meters) at initialization (default -> centre of earth)
:param convergence_tolerance: min acceptable position difference between two iterations
:param max_iteration: GNSS pvt dataframe
:return: GNSS pvt dataframe

### `get_geometry_matrix(pd_gnss_raw: DataFrame, pd_approx_pos: DataFrame) -> DataFrame`

Computes geometry matrices using dataframes.
:param pd_gnss_raw: GNSS raw dataframe from BITS parser
:param pd_approx_pos: GNSS pvt dataframe (at least "time", "x_rx_m", "y_rx_m", "z_rx_m")
:return: pd_gnss_pvt like dataframe with geometry matrices

### `get_position_estimate(pd_gnss_raw: DataFrame, pd_ephemeris: DataFrame = None, ephem_filepath: str = None, approx_pvt: tuple = (0, 0, 0)) -> DataFrame`

Computes position estimate using OLS and clock and atmospheric corrections.
source: https://gssc.esa.int/navipedia/index.php?title=GNSS_Measurements_Modelling
:param pd_gnss_raw: GNSS raw dataframe from BITS parser
:param pd_ephemeris: ephemeris dataframe from BITS parser
:param ephem_filepath: Path of a rinex nav file
:param approx_pvt: Position (ECEF meters) at initialization (default -> centre of earth)
:return: GNSS pvt dataframe, corrected GNSS raw dataframe

### `get_sv_el_az(pd_gnss_raw: DataFrame, pd_gnss_pvt: DataFrame) -> DataFrame`

Computes elevations and azimuth of satellite vehicles in pd_gnss_raw at estimated position from pd_gnss_pvt
:param pd_gnss_raw: GNSS raw dataframe
:param pd_gnss_pvt: GNSS pvt dataframe
:return: GNSS raw dataframe

### `ordinary_least_square(Y: array, G: array) -> array`

Performs a simple Ordinary Least Square.
sources:    https://gssc.esa.int/navipedia/index.php?title=Weighted_Least_Square_Solution_(WLS)
            https://gssc.esa.int/navipedia/index.php?title=Best_Linear_Unbiased_Minimum-Variance_Estimator_(BLUE)
Y = G @ X
=> X = (Gt@G)^-1@Gt@Y
:param Y: measurements
:param G: Geometry matrix
:return: estimates


---


## Module `bits.src.sv_model`

Used to find sv states

---

### `compute_eccentric_anomaly(pd_ephemeris_row: Series, time: GnssTimestamp, ek_iterations = 5)`

Compute eccentric anomaly for a specific satellite vehicle at a specific time.
:param pd_ephemeris_row: Satellite ephemeris. Use a pd.Series parsed with the BITS ephemeris parser.
:param time: Time at which the satellite's position should be computed
:param ek_iterations: Number of iterations to compute the eccentric anomaly
:return: Eccentric anomaly

### `ephemeris_loader(timestamp: GnssTimestamp)`

Loads ephemeris from https://cddis.nasa.gov/.
To be implemented.
:param timestamp: ephemeris time required
:return: ephemeris dataframe (same format as the ephemeris parser)

### `get_sv_states(pd_gnss_raw: DataFrame, pd_ephemeris: DataFrame = None, ephem_filepath: str = None) -> DataFrame`

Compute SV states (positions only) using a pd.Dataframe ephemeris from the BITS ephemeris parser for GPS, Galileo,
Glonass and Beidou.
Based on https://gssc.esa.int/navipedia/index.php?title=Satellite_Coordinates_Computation
:param pd_gnss_raw: BITS raw dataframe
:param pd_ephemeris: BITS ephemeris dataframe
:param ephem_filepath: Path of a rinex nav file
:return: BITS raw dataframe with corresponding sv positions

### `retrieve_ephemeris(pd_gnss_raw: DataFrame, pd_ephemeris: DataFrame = None, ephem_filepath: str = None) -> DataFrame`

Finds the closest ephemeris parameters for each satellite vehicle
:param pd_gnss_raw: GNSS raw dataframe from BITS parser
:param pd_ephemeris: ephemeris dataframe from BITS parser
:param ephem_filepath: Path of a rinex nav file
:return: GNSS raw dataframe with ephemeris


---


## Module `bits.src.utils`

Utilities

---

### `check_dataframe(df: DataFrame, required_columns: list, with_warning: bool = True) -> bool`

Check if all required_columns are in df. If not, returns False.
:param df: Dataframe to check
:param required_columns: Liste of required columns
:param with_warning: Set to False to disable warning
:return: True if no missing columns else False


---


## Module `bits.src.convert.other`

This is an awesome script that was not commented...

Usage:
======
python other.py

---

### `doppler_to_pr_rate(doppler, ft = 1575420000)`

Converts doppler to pseudorange rate. L1 frequency by default
:param doppler: doppler shift (Hz)
:param ft: signal frequency (Hz)
:return: pseudorange rate (m/s)


---


## Module `bits.src.convert.space_conversion`

Functions for space conversions

---

### `ecef_to_eci_position(x_ecef: float, y_ecef: float, z_ecef: float, timestamp: GnssTimestamp) -> tuple`

Converts ECEF coordinates to ECI
source: https://gssc.esa.int/navipedia/index.php?title=GLONASS_Satellite_Coordinates_Computation
:param x_ecef: X ECEF (m)
:param y_ecef: Y ECEF (m)
:param z_ecef: Z ECEF (m)
:param timestamp: Time of the measurements
:return: (x_eci, y_eci, z_eci)

### `ecef_to_eci_velocity(x_ecef: float, y_ecef: float, z_ecef: float, vx_ecef: float, vy_ecef: float, vz_ecef: float, timestamp: GnssTimestamp) -> tuple`

Converts velocity vectors from ECEF to ECI.
source: https://gssc.esa.int/navipedia/index.php?title=GLONASS_Satellite_Coordinates_Computation
:param x_ecef: X ECEF (m)
:param y_ecef: Y ECEF (m)
:param z_ecef: Z ECEF (m)
:param vx_ecef: speed X ECEF (m/s)
:param vy_ecef: speed Y ECEF (m/s)
:param vz_ecef: speed Z ECEF (m/s)
:param timestamp: Time of the measurements
:return: (vx_eci, vy_eci, vz_eci)

### `ecef_to_enu(ancre_ecef: tuple, ecef_matrix: array) -> array`

Compute East North Up coordinates from an ancre.
:param ancre_ecef: Reference point coordinates (ECEF meters)
:param ecef_matrix: ECEF coordinates (meters) to convert to ENU. X, Y, Z must be the matrix's columns.
:return: ENU coordinates (meters)

### `ecef_to_wgs(x_ecef: float, y_ecef: float, z_ecef: float) -> tuple`

Converts ECEF (EPSG:4978) coordinates to WGS (EPSG:4326)
:param x_ecef:
:param y_ecef:
:param z_ecef:
:return: lat, lon, alt (wgs)

### `eci_to_ecef_position(x_eci: float, y_eci: float, z_eci: float, timestamp: GnssTimestamp) -> tuple`

Converts ECI coordinates to ECEF
source: https://gssc.esa.int/navipedia/index.php?title=GLONASS_Satellite_Coordinates_Computation
:param x_eci: X ECI (m)
:param y_eci: Y ECI (m)
:param z_eci: Z ECI (m)
:param timestamp: Time of the measurements
:return: (x_ecef, y_ecef, z_ecef)

### `enu_to_ecef(ancre_ecef: tuple, enu_matrix: array) -> array`

Compute ECEF coordinates from East North Up coordinates from an ancre.
:param ancre_ecef: Reference point coordinates (ECEF meters)
:param enu_matrix: ENU coordinates (meters) to convert to ECEF. E, N, U must be the matrix's columns.
:return: ECEF coordinates (meters)

### `enu_to_spheric(enu_matrix: array) -> array`

Computes coordinates from the local reference frame ENU to range (m), elevation (rad, angle from the horizon),
azimuth (rad, angle from north).
:param enu_matrix: ENU coordinates (meters). E, N, U must be the matrix's columns.
:return: range (m), elevation (rad, angle from the horizon), azimuth (rad, angle from north)

### `pz_90_to_ecef(x_pz_90: float, y_pz_90: float, z_pz_90: float) -> tuple`

Converts GLONASS PZ90 coordinates to ECEF
source: https://gssc.esa.int/navipedia/index.php?title=GLONASS_Satellite_Coordinates_Computation
:param x_pz_90:
:param y_pz_90:
:param z_pz_90:
:return: (x_ecef, y_ecef, z_ecef)

### `rotate_ecef(x_ecef: float, y_ecef: float, z_ecef: float, delta_time: Timedelta) -> tuple`

Rotate ECEF coordinates over a specified time interval to account for Earth's rotation. This is used to correct for
Earth's rotation when converting from ECI to ECEF.
:param x_ecef: X ECEF (m)
:param y_ecef: Y ECEF (m)
:param z_ecef: Z ECEF (m)
:param delta_time: Earth rotation duration
:return: (x_ecef, y_ecef, z_ecef)

### `wgs_to_ecef(lat: float, lon: float, alt: float) -> tuple`

Converts WGS (EPSG:4326) coordinates to ECEF (EPSG:4978)
:param lat: latitude (wgs)
:param lon: longitude (wgs)
:param alt: altitude (wgs)
:return: x_ecef, y_ecef, z_ecef


---


## Module `bits.src.convert.time_conversion`

Functions for time conversions

---

### `count_leap_seconds(dt: Timestamp) -> int`

Counts the number of leap seconds that have occurred up to a given UTC datetime.
:param dt: Time of the measurements
:return: Number of leap seconds

### `gps_time_to_timestamp(gps_time: float) -> Timestamp`

Convert GPS time in seconds to a pandas Timestamp (UTC), accounting for leap seconds.
Precision depends on the size of the float.
:param gps_time:  Time in seconds since the GPS epoch.
:return:  The corresponding UTC timestamp.

### `gps_time_ts_to_utc_ts(gps_time_ts: Timestamp) -> Timestamp`

Converts GPS time to UTC by adding leap seconds.
:param gps_time_ts: GPS time
:return: UTC time

### `gps_week_to_timestamp(gps_week: int, tow: float) -> Timestamp`

Converts GPS time (week, seconds of week) to pandas.Timestamp.
Precision to the nanosecond (ns).
:param gps_week: GPS week number (since January 6, 1980).
:param tow: Seconds elapsed since the beginning of the week.
:return: The corresponding UTC timestamp.

### `timestamp_to_gps_time(ts: Timestamp) -> float`

Convert a UTC timestamp to GPS time (seconds since GPS epoch), accounting for leap seconds.
:param ts: The UTC timestamp to convert.
:return: GPS time in seconds.

### `timestamp_to_gps_tow(ts: Timestamp) -> (<class 'int'>, <class 'float'>)`

Converts a UTC datetime to GPS Time of Week (TOW), considering leap seconds.
Precision to the nanosecond (ns).
:param ts: UTC datetime
:return: (gps week, time of week)

### `utc_to_gmst_radians(timestamp: Timestamp) -> float`

Converts a UTC timestamp to Greenwich Mean Sidereal Time (GMST) in radians.

:param timestamp: pd.Timestamp (must be in UTC)
:return: GMST in radians (0 - 2π)


---


## Module `bits.src.parsers.ephemeris`

Ephemeris parser to be used with baguette in the sky

---

### `rinex_nav(filepath)`

Parse rinex nav into pandas dataframe using georinex.
:param filepath: Path of the rinex nav file
:return: BITS ephemeris dataframe


---


## Module `bits.src.parsers.gnss_raw`

GNSS raw data parser to be used with baguette in the sky

---

### `micdrop_raw(filepath: str) -> DataFrame`

Parse micdrop raw data to pandas Dataframe.
:param filepath: Path of the file
:return: BITS raw dataframe

### `skydel_raw(filepath: str) -> DataFrame`

Parse skydel raw data to pandas Dataframe.
:param filepath: Path of the file
:return: BITS raw dataframe


---

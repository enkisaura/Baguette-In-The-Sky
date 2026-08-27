<h1>
  Baguette in the sky
  <img src="logo_eclipse_de_baguette.png" width="40" valign="middle" alt="Logo">
</h1>
A simple python GNSS library.

---
## BITS dataframes definition
### BITS ephemeris dataframe
Ephemeris parameters. One line per ephemeris time and per SV.

| Name              | Description                                                                  | Unit                       | Type       |
|-------------------|------------------------------------------------------------------------------|----------------------------|------------|
| sv_id             | Satellite id ({gnss_id}{prn_id})                                             |                            | str        |
| gnss_id           | GNSS constellation ID (BITS normalized, cf normalize_gnss_constellation())   |                            | str        |
| prn_id            | Satellite pseudo-random noise code ID                                        |                            | int        |
| time              | Receiver timestamp                                                           |                            | Datetime64 |
| time_of_ephemeris | Reference time, ephemeris parameters                                         |                            | Datetime64 |
| sqrta             | Square root of the semi-major axis                                           | metre^1/2                  | float      |
| e                 | Eccentricity                                                                 |                            | float      |
| i0                | Inclination angle at reference time                                          | radian(semicircles)        | float      |
| idot              | Rate of change of inclination                                                | radian(semicircles)/second | float      |
| omega0            | Longitude of ascending node at reference time                                | radian(semicircles)        | float      |
| omega             | Argument of perigee                                                          | radian(semicircles)        | float      |
| m0                | Mean anomaly at reference time                                               | radian(semicircles)        | float      |
| omegadot          | Rate of change of right ascension                                            | radian(semicircles)/second | float      |
| deltan            | Mean motion difference from computed value                                   | radian(semicircles)/second | float      |
| cuc               | Amplitude of the cosine harmonic correction term to the argument of latitude | radian                     | float      |
| cus               | Amplitude of the sine harmonic correction term to the argument of latitude   | radian                     | float      |
| crc               | Amplitude of the cosine harmonic correction term to the orbit radius         | metre                      | float      |
| crs               | Amplitude of the sine harmonic correction term to the orbit radius           | metre                      | float      |
| cic               | Amplitude of the cosine harmonic correction term to the angle of inclination | radian                     | float      |
| cis               | Amplitude of the sine harmonic correction term to the angle of inclination   | radian                     | float      |
| X                 | Satellite position on X axis (ECEF)                                          | metre                      | float      |
| Y                 | Satellite position on Y axis (ECEF)                                          | metre                      | float      |
| Z                 | Satellite position on Z axis (ECEF)                                          | metre                      | float      |
| dX                | Satellite velocity on X axis (ECEF)                                          | metre/second               | float      |
| dY                | Satellite velocity on Y axis (ECEF)                                          | metre/second               | float      |
| dZ                | Satellite velocity on Z axis (ECEF)                                          | metre/second               | float      |
| dX2               | Luni-solar acceleration on X axis (ECEF)                                     | metre/second²              | float      |
| dY2               | Luni-solar acceleration on Y axis (ECEF)                                     | metre/second²              | float      |
| dZ2               | Luni-solar acceleration on Z axis (ECEF)                                     | metre/second²              | float      |
| clock_bias        | Satellite clock bias                                                         | metre                      | float      |
| clock_drift       | Satellite clock drift                                                        | second/second              | float      |
| clock_drift_rate  | Satellite clock drift rate                                                   | second/second²             | float      |
| tgd               | Time Group Delay                                                             | second                     | float      |
| klo_a0            | Alpha 0 parameter of Klobuchar ionospheric model                             |                            | float      |
| klo_a1            | Alpha 1 parameter of Klobuchar ionospheric model                             |                            | float      |
| klo_a2            | Alpha 2 parameter of Klobuchar ionospheric model                             |                            | float      |
| klo_a3            | Alpha 3 parameter of Klobuchar ionospheric model                             |                            | float      |
| klo_b0            | Beta 0 parameter of Klobuchar ionospheric model                              |                            | float      |
| klo_b1            | Beta 1 parameter of Klobuchar ionospheric model                              |                            | float      |
| klo_b2            | Beta 2 parameter of Klobuchar ionospheric model                              |                            | float      |
| klo_b3            | Beta 3 parameter of Klobuchar ionospheric model                              |                            | float      |
| healthy           | SV is reported healthy and can be used                                       |                            | bool       |

### BITS raw dataframe
Raw measurements. One line per measurement time and per satellite.

| Name               | Description                                                                | Unit          | Type       |
|--------------------|----------------------------------------------------------------------------|---------------|------------|
| sv_id              | Satellite id ({gnss_id}{prn_id})                                           |               | str        |
| gnss_id            | GNSS constellation ID (BITS normalized, cf normalize_gnss_constellation()) |               | str        |
| prn_id             | Satellite pseudo-random noise code ID                                      |               | int        |
| time               | Receiver timestamp                                                         |               | Datetime64 |
| corr_time          | Corrected receiver timestamp                                               |               | Datetime64 |
| pr_m               | Pseudorange                                                                | metre         | float      |
| corr_pr_m          | Corrected pseudorange                                                      | metre         | float      |
| pr_rate_mps        | Pseudorange rate                                                           | metre/second  | float      |
| doppler_hz         | Doppler shift                                                              | 1/second      | float      |
| frequency_hz       | Signal frequency                                                           | 1/second      | float      |
| x_sv_m             | X ECEF coordinate of the satellite                                         | metre         | float      |
| y_sv_m             | Y ECEF coordinate of the satellite                                         | metre         | float      |
| z_sv_m             | Z ECEF coordinate of the satellite                                         | metre         | float      |
| vx_sv_mps          | X ECEF coordinate of the satellite speed                                   | metre/second  | float      |
| vy_sv_mps          | Y ECEF coordinate of the satellite speed                                   | metre/second  | float      |
| vz_sv_mps          | Z ECEF coordinate of the satellite speed                                   | metre/second  | float      |
| ax_sv_mpss         | X ECEF coordinate of the satellite acceleration                            | metre/second² | float      |
| ay_sv_mpss         | Y ECEF coordinate of the satellite acceleration                            | metre/second² | float      |
| az_sv_mpss         | Z ECEF coordinate of the satellite acceleration                            | metre/second² | float      |
| e_x                | Steering vector in the X direction                                         |               | float      |
| e_y                | Steering vector in the Y direction                                         |               | float      |
| e_z                | Steering vector in the Z direction                                         |               | float      |
| e_b                | Steering vector in the b direction                                         |               | float      |
| elevation_rad      | SV elevation                                                               | radian        | float      |
| azimuth_rad        | SV azimuth                                                                 | radian        | float      |
| clock_corr_m       | Sum of the clock corrections applied to corr_pr_m                          | metre         | float      |
| poly_clock_corr_m  | Polynomial clock correction                                                | metre         | float      |
| relat_clock_corr_m | Relativistic clock correction                                              | metre         | float      |
| tgd_clock_corr_m   | Time Group Delay (clock correction)                                        | metre         | float      |
| atm_corr_m         | Sum of the atmospheric corrections applied to corr_pr_m                    | metre         | float      |
| iono_corr_m        | Ionospheric correction                                                     | metre         | float      |
| tropo_corr_m       | Tropospheric correction                                                    | metre         | float      |
| residuals_m        | Pseudorange residual after position estimation                             | metre         | float      |

### BITS PVT dataframe
Position, velocity and time. One line per measurement time.

| Name            | Description                                  | Unit           | Type        |
|-----------------|----------------------------------------------|----------------|-------------|
| time            | Receiver timestamp                           |                | Datetime64  |
| corr_time       | Corrected receiver timestamp                 |                | Datetime64  |
| lat             | Latitude of the receiver (WGS 84)            | degree         | float       |
| lon             | Longitude of the receiver (WGS 84)           | degree         | float       |
| alt             | Altitude of the receiver (WGS 84)            | metre          | float       |
| x_rx_m          | X ECEF coordinate of the receiver            | metre          | float       |
| y_rx_m          | Y ECEF coordinate of the receiver            | metre          | float       |
| z_rx_m          | Z ECEF coordinate of the receiver            | metre          | float       |
| b_rx_m          | Receiver remaining clock bias from corr_time | metre          | float       |
| cov_xx_rx_m     | Variance term of x_rx_m                      | metre²         | float       |
| cov_yy_rx_m     | Variance term of y_rx_m                      | metre²         | float       |
| cov_zz_rx_m     | Variance term of z_rx_m                      | metre²         | float       |
| cov_bb_rx_m     | Variance term of b_rx_m                      | metre²         | float       |
| cov_yx_rx_m     | Covariance term of x_rx_m/y_rx_m             | metre²         | float       |
| cov_zx_rx_m     | Covariance term of x_rx_m/z_rx_m             | metre²         | float       |
| cov_bx_rx_m     | Covariance term of x_rx_m/b_rx_m             | metre²         | float       |
| cov_zy_rx_m     | Covariance term of y_rx_m/z_rx_m             | metre²         | float       |
| cov_by_rx_m     | Covariance term of y_rx_m/b_rx_m             | metre²         | float       |
| cov_bz_rx_m     | Covariance term of z_rx_m/b_rx_m             | metre²         | float       |
| vx_rx_mps       | X ECEF coordinate of the receiver speed      | metre/second   | float       |
| vy_rx_mps       | Y ECEF coordinate of the receiver speed      | metre/second   | float       |
| vz_rx_mps       | Z ECEF coordinate of the receiver speed      | metre/second   | float       |
| vb_rx_mps       | Receiver clock drift                         | metre/second   | float       |
| cov_vxvx_rx_mps | Variance term of vx_rx_mps                   | metre²/second² | float       |
| cov_vyvy_rx_mps | Variance term of vy_rx_mps                   | metre²/second² | float       |
| cov_vzvz_rx_mps | Variance term of vz_rx_mps                   | metre²/second² | float       |
| cov_vbvb_rx_mps | Variance term of vb_rx_mps                   | metre²/second² | float       |
| cov_vyvx_rx_mps | Covariance term of vx_rx_mps/vy_rx_mps       | metre²/second² | float       |
| cov_vzvx_rx_mps | Covariance term of vx_rx_mps/vz_rx_mps       | metre²/second² | float       |
| cov_vbvx_rx_mps | Covariance term of vx_rx_mps/vb_rx_mps       | metre²/second² | float       |
| cov_vzvy_rx_mps | Covariance term of vy_rx_mps/vz_rx_mps       | metre²/second² | float       |
| cov_vbvy_rx_mps | Covariance term of vy_rx_mps/vb_rx_mps       | metre²/second² | float       |
| cov_vbvz_rx_mps | Covariance term of vz_rx_mps/vb_rx_mps       | metre²/second² | float       |
| DOP             | Dilution Of Precision                        |                | float       |

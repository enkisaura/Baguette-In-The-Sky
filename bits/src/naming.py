"""
Normalization of namings to be used in BITS

Inside BITS, GNSS constellations should be referred as:
    GPS -> "gps"
    Galileo -> "gal"
    Glonass -> "glo"
    Beidou -> "bei"
"""

__authors__ = ("Enki SAURA")
__contact__ = ("esaura@ikosconsulting.com")
__copyright__ = "IKOS"
__date__ = "18/02/2025"
__version__ = "0.0.1"

import warnings

# Different possible naming that can be found elsewhere to refer to the GNSS constellations. Case is ignored.
gnss_id_variants = {
    "gps": ["gps", "g", "l", "l1", "l2", "l5", "navstar", "gps-iii", "gps-g", "navstar gps",],
    "glo": [ "glonass", "glo", "r", "r1",],
    "gal": [ "galileo", "gal", "e", "e1", "e2", "e3", "e3a", "e3b",],
    "bei": [ "beidou", "compass", "bds", "bei", "b", "c", "b1", "b2", "b3",],
}


def normalize_gnss_constellation(name: str) -> str:
    """
    This function takes a constellation name and converts it to its normalized name, regardless of the format used, and
    ignoring case.
    :param name: not normalized GNSS constellation name
    :return: BITS normalized GNSS constellation name
    """
    # Convert the input name to lowercase for case-insensitive comparison
    lower_name = name.strip().lower()

    # If the name matches any of the GNSS variants, return its normalized name
    for gnss_id, variants in gnss_id_variants.items():
        if lower_name in variants:
            return gnss_id

    # If no match is found, return the original name
    warn_txt = f"Unknown GNSS constellation {name}"
    warnings.warn(warn_txt)
    return name



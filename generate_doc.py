#!/usr/bin/env python3

"""
This is an awesome script that was not commented...

Usage:
======
python generate_doc.py

"""

__authors__ = ("Enki SAURA")
__contact__ = ("esaura@ikosconsulting.com")
__copyright__ = "IKOS"
__date__ = "2025-06-09"
__version__ = "0.0.1"


import importlib.util
import inspect
import os
import sys
from pathlib import Path
import typing


LIBRARY_PATH = "bits/src"   # 🔁 Remplace par ton dossier source
README_PATH = "README.md"

EPHEMERIS_DATAFRAME = ('| Name | Description | Unit | Type |\n'
                       '|------|-------------|------|------|\n'
                       '| time | Receiver timestamp | | GnssTimestamp |\n'
                       '| toe | Reference time, ephemeris parameters | second | float |\n'
                       '| sqrta | Square root of the semi-major axis | metre^1/2 | float |\n'
                       '| e | Eccentricity | | float |\n'
                       '| i0 | Inclination angle at reference time | radian(semicircles) | float |\n'
                       '| idot | Rate of change of inclination | radian(semicircles)/second | float |\n'
                       '| omega0 | Longitude of ascending node at reference time | radian(semicircles) | float |\n'
                       '| omega | Argument of perigee | radian(semicircles) | float |\n'
                       '| m0 | Mean anomaly at reference time | radian(semicircles) | float |\n'
                       '| omegadot | Rate of change of right ascension | radian(semicircles)/second | float |\n'
                       '| deltan | Mean motion difference from computed value | radian(semicircles)/second | float |\n'
                       '| cuc | Amplitude of the cosine harmonic correction term to the argument of latitude | radian | float |\n'
                       '| cus | Amplitude of the sine harmonic correction term to the argument of latitude | radian | float |\n'
                       '| crc | Amplitude of the cosine harmonic correction term to the orbit radius | metre | float |\n'
                       '| crs | Amplitude of the sine harmonic correction term to the orbit radius | metre | float |\n'
                       '| cic | Amplitude of the cosine harmonic correction term to the angle of inclination | radian | float |\n'
                       '| cis | Amplitude of the sine harmonic correction term to the angle of inclination | radian | float |\n'
                       '| clock_bias | Satellite clock bias | second | float |\n'
                       '| clock_drift | Satellite clock drift | second/second | float |\n'
                       '| clock_drift_rate | Satellite clock drift rate | second/second² | float |\n'
                       '| tgd | Time Group Delay | second | float |\n'
                       '| ionospheric_param | 8 parameters to compute ionospheric corrections | | list[float] |\n')

RAW_DATAFRAME = ('| Name | Description | Unit | Type |\n'
                 '|------|-------------|------|------|\n'
                 '| time | Receiver timestamp | | GnssTimestamp |\n'
                 '| corr_time | Corrected receiver timestamp | | GnssTimestamp |\n'
                 '| pr_m | Pseudorange | metre | float |\n'
                 '| corr_pr_m | Corrected pseudorange | metre | float |\n'
                 '| pr_rate_mps | Pseudorange rate | metre/second | float |\n'
                 '| doppler_hz | Doppler shift | 1/second | float |\n'
                 '| sv_id | Satellite PRN number | | int |\n'
                 '| gnss_id | GNSS constellation ID (BITS normalized, cf normalize_gnss_constellation()) | | str |\n'
                 '| x_sv_m | X ECEF coordinate of the satellite | metre | float |\n'
                 '| y_sv_m | Y ECEF coordinate of the satellite | metre | float |\n'
                 '| z_sv_m | Z ECEF coordinate of the satellite | metre | float |\n'
                 '| vx_sv_mps | X ECEF coordinate of the satellite speed | metre/second | float |\n'
                 '| vy_sv_mps | Y ECEF coordinate of the satellite speed | metre/second | float |\n'
                 '| vz_sv_mps | Z ECEF coordinate of the satellite speed | metre/second | float |\n'
                 '| ax_sv_mpss | X ECEF coordinate of the satellite acceleration | metre/second² | float |\n'
                 '| ay_sv_mpss | Y ECEF coordinate of the satellite acceleration | metre/second² | float |\n'
                 '| az_sv_mpss | Z ECEF coordinate of the satellite acceleration | metre/second² | float |\n'
                 '| clock_corr_m | Sum of the clock corrections to be applied to corr_pr_m | metre | float |\n'
                 '| poly_clock_corr_m | Polynomial clock correction | metre | float |\n'
                 '| relat_clock_corr_m | Relativistic clock correction | metre | float |\n'
                 '| tgd_clock_corr_m | Time Group Delay (clock correction) | metre | float |\n'
                 '| atm_corr_m | Sum of the atmospheric corrections to be applied to corr_pr_m | metre | float |\n'
                 '| iono_corr_m | Ionospheric correction | metre | float |\n'
                 '| tropo_corr_m | Tropospheric correction | metre | float |\n')

PVT_DATAFRAME = ('| Name | Description | Unit | Type |\n'
                 '|------|-------------|------|------|\n'
                 '| time | Receiver timestamp | | GnssTimestamp |\n'
                 '| corr_time | Corrected receiver timestamp | | GnssTimestamp |\n'
                 '| lat | Latitude of the receiver (WGS 84) | degree | float |\n'
                 '| lon | Longitude of the receiver (WGS 84) | degree | float |\n'
                 '| alt | Altitude of the receiver (WGS 84) | metre | float |\n'
                 '| x_rx_m | X ECEF coordinate of the receiver | metre | float |\n'
                 '| y_rx_m | Y ECEF coordinate of the receiver | metre | float |\n'
                 '| z_rx_m | Z ECEF coordinate of the receiver | metre | float |\n'
                 '| b_rx_m | Receiver remaining clock bias from corr_time | metre | float |\n'
                 '| ols_convergence | Distance from last computed position | metre | float |\n'
                 '| vx_rx_mps | X ECEF coordinate of the receiver speed | metre/second | float |\n'
                 '| vy_rx_mps | Y ECEF coordinate of the receiver speed | metre/second | float |\n'
                 '| vz_rx_mps | Z ECEF coordinate of the receiver speed | metre/second | float |\n'
                 '| vb_rx_mps | Receiver clock drift | metre/second | float |\n')


def load_module_from_file(module_name, filepath):
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def format_doc(docstring):
    return inspect.cleandoc(docstring) if docstring else "_No documentation provided._"


def type_to_str(tp):
    if tp is None:
        return "None"
    if hasattr(tp, '__name__'):
        return tp.__name__
    if hasattr(tp, '__origin__'):  # e.g., Union, List, etc.
        origin = type_to_str(tp.__origin__)
        args = ", ".join(type_to_str(a) for a in tp.__args__)
        return f"{origin}[{args}]"
    if isinstance(tp, type):
        return tp.__name__
    return str(tp).replace("typing.", "")


def document_function(func):
    try:
        hints = typing.get_type_hints(func)
    except Exception:
        hints = {}

    sig = inspect.signature(func)
    params = []
    for name, param in sig.parameters.items():
        annotation = hints.get(name, param.annotation)
        ann_str = f": {type_to_str(annotation)}" if annotation != inspect._empty else ""
        default = f" = {param.default!r}" if param.default != inspect._empty else ""
        params.append(f"{name}{ann_str}{default}")

    return_ann = hints.get('return', sig.return_annotation)
    return_str = f" -> {type_to_str(return_ann)}" if return_ann != inspect._empty else ""

    clean_sig = f"({', '.join(params)}){return_str}"
    return f"### `{func.__name__}{clean_sig}`\n\n{format_doc(func.__doc__)}\n"


def document_class(cls):
    lines = [f"### Class `{cls.__name__}`\n", format_doc(cls.__doc__), ""]
    for name, member in inspect.getmembers(cls, predicate=inspect.isfunction):
        if not name.startswith("_"):
            lines.append(document_function(member))
    return "\n".join(lines)


def document_module(module, module_name):
    lines = [f"## Module `{module_name}`\n"]
    lines.append(format_doc(module.__doc__))
    lines.append("\n---\n")

    for name, obj in inspect.getmembers(module):
        if inspect.isfunction(obj) and not name.startswith("_"):
            if obj.__module__ == module.__name__:
                lines.append(document_function(obj))
        elif inspect.isclass(obj) and not name.startswith("_"):
            if obj.__module__ == module.__name__:
                lines.append(document_class(obj))

    lines.append("\n---\n")
    return "\n".join(lines)


def find_python_files(base_dir):
    return [
        path for path in Path(base_dir).rglob("*.py")
        if not path.name.startswith("_") and path.name != "__init__.py"
    ]


def module_name_from_path(path):
    return ".".join(path.with_suffix('').parts)


def main():
    sys.path.insert(0, os.path.abspath("."))

    modules_docs = []

    for filepath in find_python_files(LIBRARY_PATH):
        module_name = module_name_from_path(filepath.relative_to("."))
        try:
            module = load_module_from_file(module_name, str(filepath))
            doc = document_module(module, module_name)
            modules_docs.append(doc)
        except Exception as e:
            print(f"⚠️ Erreur dans {filepath}: {e}")

    with open(README_PATH, "w", encoding="utf-8") as f:
        f.write("# Baguette in the sky\n\n")
        f.write("A simple python GNSS library.\n\n")
        f.write("![Logo](bits_logo.jpg)\n\n")
        f.write("---\n")
        f.write("## BITS dataframes definition\n")
        f.write("### BITS ephemeris dataframe\n")
        f.write("Ephemeris parameters that can be found in a rinex nav file.\n\n")
        f.write(EPHEMERIS_DATAFRAME)
        f.write("\n")
        f.write("### BITS raw dataframe\n")
        f.write("Raw measurements. Can also contain BITS ephemeris dataframe.\n\n")
        f.write(RAW_DATAFRAME)
        f.write("\n")
        f.write("### BITS PVT dataframe\n")
        f.write("Computed position speed and time.\n\n")
        f.write(PVT_DATAFRAME)
        f.write("\n\n".join(modules_docs))

    print(f"✅ README.md généré à partir de `{LIBRARY_PATH}` et ses sous-dossiers.")


if __name__ == "__main__":
    main()

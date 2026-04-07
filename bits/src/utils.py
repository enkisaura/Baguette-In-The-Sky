#!/usr/bin/env python3

"""
Utilities

"""

__authors__ = ("Enki SAURA")
__contact__ = ("esaura@ikosconsulting.com")
__copyright__ = "IKOS"
__date__ = "2025-06-13"
__version__ = "0.0.1"

import pandas as pd
import warnings


def check_dataframe(df: pd.DataFrame, required_columns: list, with_warning: bool=True) -> bool:
    """
    Check if all required_columns are in df. If not, returns False.
    :param df: Dataframe to check
    :param required_columns: Liste of required columns
    :param with_warning: Set to False to disable warning
    :return: True if no missing columns else False
    """
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        if with_warning:
            warnings.warn(f"Missing data : {missing}", UserWarning)
        return False
    return True

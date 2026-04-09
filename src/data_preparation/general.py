import pandas as pd
from datetime import datetime
from typing import cast, List
from numbers import Real
import math
import logging
logger = logging.getLogger(__name__)

# --------------------
# Preprocessing
# --------------------

def filter_df(df: pd.DataFrame, start: datetime, end: datetime) -> pd.DataFrame:
    """
    Filters dataframe based on date range.
    :param df: has aware DatetimeIndex
    :param start: start date (inclusive)
    :param end: end date (exclusive)
    :return: filtered dataframe
    """
    df_copy = df.copy()

    start = start.replace(tzinfo=df_copy.index.tz)
    end = end.replace(tzinfo=df_copy.index.tz)

    if start < min(df_copy.index):
        logging.info(f'Warning: {start} is before the earliest timestamp in df: {min(df_copy.index)}.')
    if end > max(df_copy.index):
        logging.info(f'Warning: {end} is after the latest timestamp in df: {max(df_copy.index)}.')
    df_copy = df_copy[df_copy.index >= start]
    df_copy = df_copy[df_copy.index < end]

    return df_copy

def add_class_col(df: pd.DataFrame, variable: str, col_name: str, boundary: Real, classes: List[str]) -> pd.DataFrame:
    """
    Adds class column to dataframe based on a boundary mirrored cross the freezing point in C.
    Classes must be a list of strings of length three, with elements in descending order by temperature.
    :param df: from collect_data()
    :param variable: variable name
    :param col_name: class column name
    :param boundary: real number that must not be zero
    :param classes: list of exactly length three, with elements in descending order by temperature
    :return: dataframe with added class column
    """
    # check data types
    if not pd.api.types.is_numeric_dtype(df[variable]):
        raise TypeError(f'{variable} column in df must be a numeric type')
    if not isinstance(col_name, str):
        raise TypeError(f'{col_name} must be a string')
    if not isinstance(boundary, Real):
        raise TypeError('boundary must be a number')
    if not isinstance(classes, list):
        raise TypeError('classes must be a list')
    if not all(isinstance(x, str) for x in classes):
        raise TypeError('All elements in classes must be strings')
    # check data values
    if math.isclose(boundary, 0):
        raise ValueError('boundary must not be zero')
    if len(classes) != 3:
        raise ValueError('classes must have exactly 3 elements')
    
    def classify_value(x):
        """Return class label of x based on boundary and classes."""
        if x > abs(boundary):
            return classes[0]
        elif x >= -abs(boundary):
            return classes[1]
        else:
            return classes[2]

    df_copy = df.copy()

    df_copy[col_name] = df_copy[variable].map(classify_value)

    return df_copy

# --------------------
# Input Checking
# --------------------

def validate_time_index(df: pd.DataFrame) -> None:
    """
    Validate time index of df.
    :param df: from preprocessing
    :return: None
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("df index must be DatetimeIndex")
    dt_index = cast(pd.DatetimeIndex, df.index)
    if dt_index.tz is None:
        raise ValueError("df index must be timezone-aware")
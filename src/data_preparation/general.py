import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from typing import cast, List
from pathlib import Path
from numbers import Real
import math
import logging

# --------------------
# Preprocessing
# --------------------

def filter_df(df: pd.DataFrame, date_range: list[datetime]) -> pd.DataFrame:
    """
    Filters dataframe based on date range (inclusive, exclusive)
    :param df: has aware DatetimeIndex
    :param date_range: list of datetimes of length 2
    :return: filtered dataframe
    """
    df_copy = df.copy()

    if date_range is not None:
        date_range = date_range.copy()
        date_range.sort()
        date_range[0] = date_range[0].replace(tzinfo=df_copy.index.tz)
        date_range[1] = date_range[1].replace(tzinfo=df_copy.index.tz)

        # check validity of date_range
        if date_range[0] > max(df.index):
            raise ValueError('The start date must be prior to the last timestamp in df.')
        if date_range[1] <= min(df.index):
            raise ValueError('The end date must be after to the first timestamp in df.')

        if date_range[0] < min(df_copy.index):
            print(f'Warning: {date_range[0]} is before the earliest timestamp in df: {min(df_copy.index)}.')
        if date_range[1] > max(df_copy.index):
            print(f'Warning: {date_range[1]} is after the latest timestamp in df: {max(df_copy.index)}.')
        df_copy = df_copy[df_copy.index >= date_range[0]]
        df_copy = df_copy[df_copy.index < date_range[1]]

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
# Visualization
# --------------------

def plot(df: pd.DataFrame, variable: str, station: str, system: str, form: str,
         y_label=None, start=None, end=None) -> plt.Axes:
    """
    Create a line or scatter plot of variable vs the index.
    Scatter should be chosen if there's any datapoints that are surrounded by NaN.
    If end given but not start, the first timestamp in the df to end will be plotted.
    If start given but not end, the start to the last timestamp in the df will be plotted.
    :param df: from collect_data(), create_timestamp_col(), and convert_nan()
    :param variable: full variable name
    :param station: name of the ISMN station
    :param system: name of the sensor system
    :param form: line or scatter, case-insensitive
    :param y_label: y-label for plot
    :param start: naive datetime.datetime object (inclusive)
    :param end: naive datetime.datetime object (inclusive)
    :return: matplotlib Axes object
    """
    # check input data types
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if not isinstance(variable, str):
        raise TypeError("variable must be a string")
    if not isinstance(station, str):
        raise TypeError("station must be a string")
    if not isinstance(system, str):
        raise TypeError("system must be a string")
    if not isinstance(form, str):
        raise TypeError("form must be a string")
    if y_label is not None:
        if not isinstance(y_label, str):
            raise TypeError("y_label must be a string")
    # check input values
    if df.empty:
        raise ValueError('df must not be empty')
    if variable not in df.columns:
        raise KeyError(f'df missing required column "{variable}".')
    if form.lower() not in ['line', 'scatter']:
        raise ValueError('form must be "line" or "scatter" (case-insensitive).')
    # check input df index
    validate_time_index(df)

    # check start and end independently
    if start is not None:
        if not isinstance(start, datetime):
            raise TypeError(f'start must be a naive datetime.datetime object.')
        start = start.replace(tzinfo=df.index.tz)
        if start < min(df.index):
            raise ValueError(f'{start} must not be before the first timestamp in df: {min(df.index)}.')
        if start >= max(df.index):
            raise ValueError(f'{start} must not be on or after the last timestamp in df: {max(df.index)}.')
    if end is not None:
        if not isinstance(end, datetime):
            raise TypeError(f'end must be a naive datetime.datetime object.')
        end = end.replace(tzinfo=df.index.tz)
        if end > max(df.index):
            raise ValueError(f'{end} must not be after the last timestamp in df: {max(df.index)}.')
        if end <= min(df.index):
            raise ValueError(f'{end} must not be before or on the first timestamp in df: {min(df.index)}.')

    # set date range for plot
    if start is None and end is not None:
        df_slice = df.loc[df.index <= end]
    elif start is not None and end is None:
        df_slice = df.loc[df.index >= start]
    elif start is not None and end is not None:
        # check relation between start and end
        if start == end:
            raise ValueError(f'start and end cannot be the same.')
        if start > end:
            raise ValueError(f'start must be before end.')
        df_slice = df.loc[start:end]
    else: # default to plotting all records
        df_slice = df
    df_slice = df_slice.sort_index()

    # create plot objects
    fig, ax = plt.subplots()
    if form.lower() == 'line':
        ax.plot(df_slice.index, df_slice[variable])
    elif form.lower() == 'scatter':
        ax.scatter(df_slice.index, df_slice[variable])
    else:
        raise ValueError(f'form somehow changed to invalid value from when it was checked to now')
    ax.set_title(f'{station}, {system}')
    if y_label is not None:
        ax.set_ylabel(y_label)
    else:
        ax.set_ylabel(variable)
    ax.set_xlabel('Date')
    ax.tick_params(axis='x', rotation=30)

    if variable == 'soil_temp':
        ax.axhline(y=0, color='k')

    return ax

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

def validate_data_cleaning_input(data_path: Path, date_range: list[datetime]) -> None:
    """
    Validate inputs for data cleaning functions in notebooks.
    :param data_path:
    :param date_range:
    :return: None
    """
    if date_range is not None:
        if not isinstance(date_range, list):
            raise TypeError('date_range must be a list')
        if len(date_range) != 2:
            raise ValueError('date_range must be a list of length 2')
        if not isinstance(date_range[0], datetime) or not isinstance(date_range[1], datetime):
            raise TypeError('date_range must be a list of datetime objects')
    # check input values
    if not data_path.is_dir():
        raise NotADirectoryError(f'{data_path} must be a directory')
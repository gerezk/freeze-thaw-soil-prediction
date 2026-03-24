import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import cast
from pathlib import Path

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

# --------------------
# Visualization
# --------------------

def plot(df: pd.DataFrame, variable: str, station: str, system: str, form: str,
         y_label=None, start=None, end=None) -> None:
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
    :return: None
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
    if form.lower() == 'line':
        plt.plot(df_slice.index, df_slice[variable])
    elif form.lower() == 'scatter':
        plt.scatter(df_slice.index, df_slice[variable])
    else:
        raise ValueError(f'form somehow changed to invalid value from when it was checked to now')
    plt.title(f'{station}, {system}')
    if y_label is not None:
        plt.ylabel(y_label)
    else:
        plt.ylabel(variable)
    plt.xlabel('Date')
    plt.xticks(rotation=30)

    if variable == 'soil_temp':
        plt.axhline(y=0, color='k')

    plt.show()

# --------------------
# Input Checking
# --------------------

def validate_time_index(df: pd.DataFrame):
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
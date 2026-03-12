import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import plotly.express as px
import datetime
from pathlib import Path
import numbers
from typing import cast

# --------------------
# Data Preprocessing
# --------------------

def collect_data(path: Path, max_depth: numbers.Real, short_variable: str, long_variable: str) -> pd.DataFrame:
    """
    Collect long_variable data for a station into a list, excluding data beyond max_depth, then merge into a single df
    The closest depth to max_depth will be selected,
    e.g. if 0.05 and 0.10 exists for max_depth=0.11, only 0.10 will be read.
    The ISMN data filenames must follow this structure:
    CSE_Network_Station_Variablename_depthfrom_depthto_startdate_enddate.stm
    :param path: path to directory for a station
    :param max_depth: max depth in meters, exclusive
    :param short_variable: abbreviated variable name
    :param long_variable: full variable name
    :return: df
    """
    # check input data types
    if not isinstance(path, Path):
        raise TypeError("path must be a pathlib.Path")
    if not isinstance(max_depth, numbers.Real):
        raise TypeError("max_depth must be a real number")
    if not isinstance(short_variable, str):
        raise TypeError("short_variable must be a string")
    if not isinstance(long_variable, str):
        raise TypeError("long_variable must be a string")
    # check input values
    if not path.is_dir():
        raise ValueError('path must point to a directory')
    if max_depth < 0:
        raise ValueError('depth must be zero or positive')

    # find depth closest to max_depth
    depths = []
    for file in path.iterdir():
        filename = file.name
        filename_split = filename.split('_')

        # skip if file extension is not .stm
        if not filename.endswith('.stm'):
            continue
        # skip if file contains wrong variable or soil depth
        if filename_split[3] != short_variable or float(filename_split[4]) >= max_depth:
            continue

        depths.append(float(filename_split[4]))
    if len(depths) == 0:
        raise ValueError(f'No data found for {path.name}, max_depth={max_depth}, variable={long_variable}')
    depths = list(set(depths)) # remove duplicates
    depths.sort()
    closest_depth = depths[-1]

    # collect data
    dfs = []
    col_names = ['UTC_date', 'UTC_time', long_variable, 'ISMN_data_quality', 'provider_data_quality']
    for file in path.iterdir():
        filename = file.name
        filename_split = filename.split('_')

        # skip if file extension is not .stm
        if not filename.endswith('.stm'):
            continue
        # skip if file contains wrong variable or soil depth
        if filename_split[3] != short_variable or float(filename_split[4]) != closest_depth:
            continue

        df = pd.read_csv(file, sep=' ', header=None, skiprows=1, names=col_names)
        dfs.append(df)

    combined_df = pd.concat(dfs, axis=0, ignore_index=True)

    return combined_df

def create_timestamp_col(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a UTC timestamp column.
    ISMN has the date and time in separate columns.
    :param df: from collect_data()
    :return: df with timestamp (datetime64[us, UTC]) index
    """
    # check input values
    if not {'UTC_date', 'UTC_time'}.issubset(df.columns):
        raise KeyError('df must contain UTC_date and UTC_time columns')
    if df.empty:
        raise ValueError('df must not be empty')

    df_copy = df.copy()

    df_copy['UTC_timestamp'] = df_copy['UTC_date'].astype(str) + ' ' + df_copy['UTC_time'].astype(str)
    df_copy['UTC_timestamp'] = pd.to_datetime(df_copy['UTC_timestamp'], format='%Y/%m/%d %H:%M')
    df_copy = df_copy.drop(columns=['UTC_date', 'UTC_time'])
    df_copy.set_index('UTC_timestamp', inplace=True)
    df_copy.index = df_copy.index.tz_localize('UTC')

    return df_copy

def convert_nan(df: pd.DataFrame, long_variable: str) -> pd.DataFrame:
    """
    Create proper nan values in the df.
    ISMN fills nan with -9999.
    provider_data_quality column not used because inconsistent across networks.
    :param df: from collect_data() and create_timestamp_col()
    :param long_variable: full variable name
    :return: df with proper nan values
    """
    # check input values
    if not {'ISMN_data_quality', long_variable}.issubset(df.columns):
        raise KeyError(f'df must contain ISMN_data_quality and {long_variable} columns')
    if df.empty:
        raise ValueError('df must not be empty')
    # check input df index
    validate_time_index(df)

    df_copy = df.copy()

    df_copy.loc[df_copy['ISMN_data_quality'] != 'G', long_variable] = np.nan

    return df_copy

def find_outlier_spikes(df: pd.DataFrame, long_variable: str, threshold: numbers.Real) -> pd.DatetimeIndex:
    """
    Detect single datapoint outliers for column long_variable in df based on threshold.
    A single datapoint is flagged as an outlier if the absolute differences between it and BOTH immediate non-NaN
    neighbors are greater than threshold.
    :param df: after processing with collect_data(), create_timestamp_col(), and convert_nan()
    :param long_variable: full variable name
    :param threshold: number
    :return: pd.DatetimeIndex with timezone containing timestamps of outliers
    """
    # check input data types
    if not isinstance(threshold, numbers.Real):
        raise TypeError("threshold must be a real number")
    # check input values
    if long_variable not in df.columns:
        raise KeyError(f'Missing required column "{long_variable}".')
    if threshold <= 0:
        raise ValueError(f'threshold must be greater than 0.')
    # check input df index
    validate_time_index(df)

    df_copy = df.copy()
    s = df[long_variable]

    # nearest valid neighbor to the left
    prev_valid = s.ffill().shift(1)

    # nearest valid neighbor to the right
    next_valid = s.bfill().shift(-1)

    prev_diff = (s - prev_valid).abs()
    next_diff = (s - next_valid).abs()

    df_copy['outlier'] = (
        (prev_diff > threshold) &
        (next_diff > threshold)
    )

    return df_copy[df_copy['outlier']].index

# --------------------
# NaN Handling
# --------------------

def get_nan_gaps(df: pd.DataFrame, long_variable: str) -> pd.DataFrame:
    """
    Determines nan gaps in the long_variable column of the df.
    Uses the existing UTC_timestamp as the index for start and end timestamps.
    Processing assumes that there's a datapoint every hour.
    :param df: from collect_data(), create_timestamp_col(), and convert_nan()
    :param long_variable: full variable name
    :return: df with
        - start_timestamp: timestamp of the first row in the gap
        - end_timestamp: timestamp of the last row in the gap
        - gap_length_hours: length of the gap in hours
        - prev_(long_variable): value before the gap
        - next_(long_variable): value after the gap
    """
    # check input values
    if long_variable not in df.columns:
        raise KeyError(f'df missing required column "{long_variable}".')
    if df.empty:
        raise ValueError('df must not be empty')
    # check input df index
    validate_time_index(df)

    df_copy = df.copy()

    # Step 1: Fill forward and backward for prev_temp and next_temp
    df_copy[f'prev_{long_variable}'] = df_copy[long_variable].ffill()
    df_copy[f'next_{long_variable}'] = df_copy[long_variable].bfill()

    # Step 2: Identify gaps (NaN in long_variable)
    mask = df_copy[long_variable].isna()

    # Step 3: Create group IDs for consecutive NaNs
    group_ids = (~mask).cumsum()
    group_ids = group_ids.where(~mask, group_ids.shift(1).fillna(0))
    group_ids = group_ids.fillna(0).astype(int)

    # Step 4: Group by group_ids and filter for groups with NaNs
    grouped = df_copy.groupby(group_ids)

    results = []
    for _, group in grouped:
        if group[long_variable].isna().any():
            start_row = group.iloc[0]
            end_row = group.iloc[-1]
            start_ts = start_row.name + datetime.timedelta(hours=1)
            end_ts = end_row.name
            gap_length = len(group) - 1
            prev_val = start_row[f'prev_{long_variable}']
            next_val = end_row[f'next_{long_variable}']
            results.append({
                'start_timestamp': start_ts,
                'end_timestamp': end_ts,
                'gap_length_hours': gap_length,
                f'prev_{long_variable}': prev_val,
                f'next_{long_variable}': next_val
            })

    results = pd.DataFrame(results)
    if not results.empty:
        results = results[results['gap_length_hours'] > 1]

    return results

def add_missed_transitions_col(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a boolean column indicating if an F/T transition occurred during a NaN gap.
    :param df: from get_nan_gaps(); must contain soil_temp data
    :return: df with added boolean column 'possible_transition'
    """
    # check required columns
    required_cols = ['start_timestamp', 'end_timestamp', 'gap_length_hours', 'prev_soil_temp', 'next_soil_temp']
    if not set(required_cols) <= set(df.columns):
        raise KeyError(f'df does not contain all required columns. Required columns: {required_cols}')

    df_copy = df.copy()

    df_copy['possible_transition'] = (
            abs(df_copy['prev_soil_temp'] + df_copy['next_soil_temp'])
            < df_copy[['prev_soil_temp', 'next_soil_temp']].abs().max(axis=1)
    )

    return df_copy

def make_nan_window(df: pd.DataFrame, long_variable: str, start: datetime.datetime, end: datetime.datetime) -> pd.DataFrame:
    """
    Set records between start and end timestamps (inclusive) to np.nan.
    :param df: from collect_data(), create_timestamp_col(), and convert_nan()
    :param long_variable: full variable name
    :param start: naive datetime.datetime object
    :param end: naive datetime.datetime object
    :return: df with specified records set to NaN
    """
    # check data types
    if not isinstance(df, pd.DataFrame):
        raise TypeError('df must be a DataFrame')
    if not isinstance(long_variable, str):
        raise TypeError('long_variable must be a string')
    if not isinstance(start, datetime.datetime):
        raise TypeError('start must be a naive datetime.datetime object')
    if not isinstance(end, datetime.datetime):
        raise TypeError('end must be a naive datetime.datetime object')
    # check values
    if long_variable not in df.columns:
        raise KeyError(f'Missing required column "{long_variable}".')
    # check input df index
    validate_time_index(df)

    # add timezone
    start = start.replace(tzinfo=df.index.tz)
    end = end.replace(tzinfo=df.index.tz)

    # check start and end
    if start not in df.index:
        raise KeyError(f'df must contain data from {start}.')
    if end not in df.index:
        raise KeyError(f'df must contain data from {end}.')

    df_copy = df.copy()
    df_copy.loc[start:end, long_variable] = np.nan

    return df_copy

def make_nan_indices(df: pd.DataFrame, long_variable: str, timestamps: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Set long_variable of rows in df that match timestamps by index to np.nan.
    :param df: from collect_data(), create_timestamp_col(), and convert_nan()
    :param long_variable: full variable name
    :param timestamps: timezone-aware pd.DatetimeIndex
    :return: df with specified records set to NaN
    """
    # check data types
    if not isinstance(df, pd.DataFrame):
        raise TypeError('df must be a DataFrame')
    if not isinstance(long_variable, str):
        raise TypeError('long_variable must be a string')
    if not isinstance(timestamps, pd.DatetimeIndex):
        raise TypeError("timestamps must be a pd.DatetimeIndex")
    # check values
    if long_variable not in df.columns:
        raise KeyError(f'Missing required column "{long_variable}".')
    if timestamps.tz is None:
        raise ValueError("timestamps must be timezone-aware")
    # check input df index
    validate_time_index(df)

    df_copy = df.copy()

    # use default exception throwing if an index in timestamps is not in df.DatetimeIndex
    df_copy.loc[timestamps.to_list(), [long_variable]] = np.nan

    return df_copy

# --------------------
# Visualization
# --------------------

def plot(df: pd.DataFrame, long_variable: str, station: str, form: str, start=None, end=None) -> None:
    """
    Create a line or scatter plot of long_variable vs the index.
    Scatter should be chosen if there's any datapoints that are surrounded by NaN.
    If end given but not start, the first timestamp in the df to end will be plotted.
    If start given but not end, the start to the last timestamp in the df will be plotted.
    :param df: from collect_data(), create_timestamp_col(), and convert_nan()
    :param long_variable: full variable name
    :param station: name of the ISMN station
    :param form: line or scatter
    :param start: naive datetime.datetime object (inclusive)
    :param end: naive datetime.datetime object (inclusive)
    :return: None
    """
    # check data types
    if not isinstance(station, str):
        raise TypeError("station must be a string")
    # check input values
    if long_variable not in df.columns:
        raise KeyError(f'df missing required column "{long_variable}".')
    if form not in ['line', 'scatter']:
        raise ValueError(f'form must be "line" or "scatter"')
    if df.empty:
        raise ValueError('df must not be empty')
    # check input df index
    validate_time_index(df)

    # check start and end independently
    if start is not None:
        if not isinstance(start, datetime.datetime):
            raise TypeError(f'start must be a naive datetime.datetime object.')
        start = start.replace(tzinfo=df.index.tz)
        if start < min(df.index):
            raise ValueError(f'{start} must not be before the first timestamp in df: {min(df.index)}.')
        if start >= max(df.index):
            raise ValueError(f'{start} must not be on or after the last timestamp in df: {max(df.index)}.')
    if end is not None:
        if not isinstance(end, datetime.datetime):
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
    if form == 'line':
        plt.plot(df_slice.index, df_slice[long_variable])
    elif form == 'scatter':
        plt.scatter(df_slice.index, df_slice[long_variable])
    else:
        raise ValueError(f'form somehow changed to invalid value from when it was checked to now')
    plt.title(f'{station}, {long_variable}')
    plt.ylabel(long_variable)
    plt.xlabel('Date')
    plt.xticks(rotation=30)

    if long_variable == 'soil_temp':
        plt.axhline(y=0, color='k')

    plt.show()

def map_stations(path: Path, save_image=False) -> None:
    """
    Show map displaying locations of ISMN stations.
    :param path: path to ISMN_site_survey.csv
    :param save_image: whether to save plot; takes a few seconds if True
    :return: saved plot in ../images if save_image is True
    """
    # check input data type
    if not isinstance(path, Path):
        raise TypeError
    if not isinstance(save_image, bool):
        raise TypeError('save_image must be a bool')

    if not path.exists():
        raise ValueError("path does not exist")

    df = pd.read_csv(path)

    fig = px.scatter_geo(
        df,
        lat="LAT",
        lon="LON",
        color="Region",
        hover_name="ISMN Station Name",
    )
    fig.update_layout(
        margin=dict(l=0, r=120, t=0, b=0),
        legend=dict(
            x=1.02,  # move legend outside the plot
            y=0.5,  # vertical center
            xanchor="left",  # anchor legend's left side at x
            yanchor="middle"
        )
    )
    fig.update_geos(
        fitbounds="locations",
        showcountries=True
    )
    if save_image:
        fig.write_image(Path("../images/map_ISMN_stations.png"))
    fig.show()

# --------------------
# Input Checking
# --------------------

def validate_time_index(df):
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
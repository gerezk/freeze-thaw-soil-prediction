import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import datetime
from pathlib import Path
import pytz
import numbers

def collect_data(path: Path, max_depth: numbers.Real, short_feature: str, long_feature: str) -> pd.DataFrame:
    """
    Collect long_feature data for a station into a list, excluding data beyond max_depth, then merge into a single df
    The closest depth to max_depth will be selected
    e.g. if 0.05 and 0.10 exists for max_depth=0.11, only 0.10 will be read.
    :param path: path to directory for a station
    :param max_depth: max depth in meters, exclusive
    :param short_feature: abbreviated variable name
    :param long_feature: full variable name
    :return: df
    """
    # check data types
    if not isinstance(path, Path):
        raise TypeError('path must be a Path object')
    if not isinstance(max_depth, numbers.Real):
        raise TypeError('max_depth must be a numeric')
    if not isinstance(short_feature, str):
        raise TypeError('short_feature must be a string')
    if not isinstance(long_feature, str):
        raise TypeError('long_feature must be a string')

    # check value of inputs
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
        if filename_split[3] != short_feature or float(filename_split[4]) >= max_depth:
            continue

        depths.append(float(filename_split[4]))
    depths = list(set(depths)) # remove duplicates
    depths.sort()
    closest_depth = depths[-1]

    # collect data
    dfs = []
    col_names = ['UTC_date', 'UTC_time', long_feature, 'ISMN_data_quality', 'provider_data_quality']
    for file in path.iterdir():
        filename = file.name
        filename_split = filename.split('_')

        # skip if file extension is not .stm
        if not filename.endswith('.stm'):
            continue
        # skip if file contains wrong variable or soil depth
        if filename_split[3] != short_feature or float(filename_split[4]) != closest_depth:
            continue

        df = pd.read_csv(file, sep=' ', header=None, skiprows=1, names=col_names)
        dfs.append(df)

    if len(dfs) == 0:
        raise Exception(f'No data found for {path.name}, max_depth={max_depth}, variable={long_feature}')

    combined_df = pd.concat(dfs, axis=0, ignore_index=True)

    return combined_df

def create_timestamp_col(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a UTC timestamp column.
    ISMN has the date and time in separate columns.
    :param df: from collect_data()
    :return: df with timestamp (datetime64[us, UTC]) index
    """
    # check data type
    if not isinstance(df, pd.DataFrame):
        raise TypeError('df must be a pd.DataFrame')

    # check that df contains required columns
    if not {'UTC_date', 'UTC_time'}.issubset(df.columns):
        raise ValueError('df must contain UTC_date and UTC_time')

    df_copy = df.copy()

    df_copy['UTC_timestamp'] = df_copy['UTC_date'].astype(str) + ' ' + df_copy['UTC_time'].astype(str)
    df_copy['UTC_timestamp'] = pd.to_datetime(df_copy['UTC_timestamp'], format='%Y/%m/%d %H:%M')
    df_copy = df_copy.drop(columns=['UTC_date', 'UTC_time'])
    df_copy.set_index('UTC_timestamp', inplace=True)
    df_copy.index = df_copy.index.tz_localize('UTC')

    return df_copy

def convert_nan(df: pd.DataFrame, long_feature: str) -> pd.DataFrame:
    """
    Create proper nan values in the df.
    ISMN fills nan with -9999.
    provider_data_quality column not used because inconsistent across networks.
    :param df: from collect_data() and create_timestamp_col()
    :param long_feature: full variable name
    :return: df with proper nan values
    """
    # check data types
    if not isinstance(df, pd.DataFrame):
        raise TypeError('df must be a pd.DataFrame')
    if not isinstance(long_feature, str):
        raise TypeError('long_feature must be a string')
    if df.index.dtype != 'datetime64[us, UTC]':
        raise Exception(f'Index of df must contain datetime64[us, UTC] data.')

    # check that required columns are present
    if 'ISMN_data_quality' not in df.columns:
        raise ValueError('df must contain ISMN_data_quality')
    if long_feature not in df.columns:
        raise ValueError(f'{long_feature} must be in df.columns')

    df_copy = df.copy()

    df_copy.loc[df_copy['ISMN_data_quality'] != 'G', long_feature] = np.nan

    return df_copy

def report_nan_count(df: pd.DataFrame, long_feature: str) -> None:
    """
    Prints the total number of nan values in the df and percent missing.
    :param df: from collect_data(), create_timestamp_col(), and convert_nan()
    :param long_feature: full variable name
    :return: None
    """
    # check data types
    if not isinstance(df, pd.DataFrame):
        raise TypeError('df must be a pd.DataFrame')
    if not isinstance(long_feature, str):
        raise TypeError('long_feature must be a string')
    if df.index.dtype != 'datetime64[us, UTC]':
        raise Exception(f'Index of df must contain datetime64[us, UTC] data.')

    # check that required column is present
    if long_feature not in df.columns:
        raise ValueError(f'df must contain {long_feature}')

    na_count = df.isnull().sum()[long_feature]
    print(f'There are {na_count} nulls out of {len(df)} datapoints ({round(na_count/len(df),2)}% missing).')

def get_nan_gaps(df: pd.DataFrame, long_feature: str) -> pd.DataFrame:
    """
    Determines nan gaps in the long_feature column of the df.
    Uses the existing UTC_timestamp as the index for start and end timestamps.
    :param df: from collect_data(), create_timestamp_col(), and convert_nan()
    :param long_feature: full variable name
    :return: df with
        - start_timestamp: timestamp of the first row in the gap
        - end_timestamp: timestamp of the last row in the gap
        - gap_length_hours: length of the gap in hours
        - prev_(long_feature): value before the gap
        - next_(long_feature): value after the gap
    """
    # check data types
    if not isinstance(df, pd.DataFrame):
        raise TypeError('df must be a pd.DataFrame')
    if not isinstance(long_feature, str):
        raise TypeError('long_feature must be a string')
    if df.index.dtype != 'datetime64[us, UTC]':
        raise Exception(f'Index of df must contain datetime64[us, UTC] data.')

    # check values
    if long_feature not in df.columns:
        raise Exception(f'Missing required column "{long_feature}".')

    df_copy = df.copy()

    # Step 1: Fill forward and backward for prev_temp and next_temp
    df_copy[f'prev_{long_feature}'] = df_copy[long_feature].ffill()
    df_copy[f'next_{long_feature}'] = df_copy[long_feature].bfill()

    # Step 2: Identify gaps (NaN in long_feature)
    mask = df_copy[long_feature].isna()

    # Step 3: Create group IDs for consecutive NaNs
    group_ids = (~mask).cumsum()
    group_ids = group_ids.where(~mask, group_ids.shift(1).fillna(0))
    group_ids = group_ids.fillna(0).astype(int)

    # Step 4: Group by group_ids and filter for groups with NaNs
    grouped = df_copy.groupby(group_ids)

    results = []
    for _, group in grouped:
        if group[long_feature].isna().any():
            start_row = group.iloc[0]
            end_row = group.iloc[-1]
            start_ts = start_row.name + datetime.timedelta(hours=1)
            end_ts = end_row.name
            gap_length = len(group) - 1
            prev_val = start_row[f'prev_{long_feature}']
            next_val = end_row[f'next_{long_feature}']
            results.append({
                'start_timestamp': start_ts,
                'end_timestamp': end_ts,
                'gap_length_hours': gap_length,
                f'prev_{long_feature}': prev_val,
                f'next_{long_feature}': next_val
            })

    results = pd.DataFrame(results)
    results = results[results['gap_length_hours'] > 1]

    return results

def add_missed_transitions_col(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a boolean column indicating if an F/T transition occurred during a NaN gap.
    :param df: from get_nan_gaps(); must contain soil_temp data
    :return: df with added boolean column 'possible_transition'
    """
    required_cols = ['start_timestamp', 'end_timestamp', 'gap_length_hours', 'prev_soil_temp', 'next_soil_temp']
    if not set(required_cols) <= set(df.columns):
        raise Exception(f'df does not contain all required columns. Required columns: {required_cols}')

    df_copy = df.copy()

    df_copy['possible_transition'] = (
            abs(df_copy['prev_soil_temp'] + df_copy['next_soil_temp'])
            < df_copy[['prev_soil_temp', 'next_soil_temp']].abs().max(axis=1)
    )

    return df_copy

def plot(df: pd.DataFrame, long_feature: str, station: str, form: str, start=None, end=None) -> None:
    """
    Create a line or scatter plot of long_feature vs the index.
    Scatter should be chosen if there's any datapoints that are surrounded by NaN.
    If end given but not start, plot will begin from the earliest timestamp in the df.
    :param df: from collect_data(), create_timestamp_col(), and convert_nan()
    :param long_feature: full variable name
    :param station: name of the ISMN station
    :param form: line or scatter
    :param start: naive datetime.datetime object
    :param end: naive datetime.datetime object
    :return: None
    """
    # check data types
    if not isinstance(df, pd.DataFrame):
        raise TypeError('df must be a pd.DataFrame')
    if not isinstance(long_feature, str):
        raise TypeError('long_feature must be a string')
    if not isinstance(station, str):
        raise TypeError('station must be a string')
    if not isinstance(form, str):
        raise TypeError('form must be a string')
    if df.index.dtype != 'datetime64[us, UTC]':
        raise Exception(f'Index of df must contain datetime64[us, UTC] data.')

    # check values
    if long_feature not in df.columns:
        raise Exception(f'Missing required column "{long_feature}".')
    if form not in ['line', 'scatter']:
        raise Exception(f'form must be "line" or "scatter"')

    # check start and end have correct data type
    if start is not None:
        if type(start) is not datetime.datetime:
            raise Exception(f'start must be a naive datetime.datetime object.')
        start = start.replace(tzinfo=pytz.UTC)
    if end is not None:
        if type(end) is not datetime.datetime:
            raise Exception(f'end must be a naive datetime.datetime object.')
        end = end.replace(tzinfo=pytz.UTC)

    # set date range for plot
    if start is None and end is not None:
        # input check
        if end > max(df.index):
            raise Exception(f'{end} must not be after the last timestamp in df ({df.index[-1]}).')

        df_slice = df.loc[df.index < end]
    elif start is not None and end is not None:
        # input check
        if start == end:
            raise Exception(f'start and end cannot be the same.')
        if start > end:
            raise Exception(f'start must be before end.')
        if start < min(df.index):
            raise Exception(f'{start} must not be before the first timestamp in df ({df.index[0]}).')
        if end > max(df.index):
            raise Exception(f'{end} must not be after the last timestamp in df ({df.index[-1]}).')

        df_slice = df.loc[start:end]
    else: # default to plotting all records
        df_slice = df

    df_slice = df_slice.sort_index()
    if form == 'line':
        plt.plot(df_slice.index, df_slice[long_feature])
    elif form == 'scatter':
        plt.scatter(df_slice.index, df_slice[long_feature])
    else:
        raise ValueError(f'form somehow changed from when it was checked to now.')
    plt.title(f'{station}, {long_feature}')
    plt.ylabel(long_feature)
    plt.xlabel('Date')
    plt.xticks(rotation=30)

    if long_feature == 'soil_temp':
        plt.axhline(y=0, color='k')

    plt.show()

def make_nan_window(df: pd.DataFrame, long_feature: str, start: datetime.datetime, end: datetime.datetime) -> pd.DataFrame:
    """
    Set records between start and end timestamps (inclusive) to np.nan.
    :param df: from collect_data(), create_timestamp_col(), and convert_nan()
    :param long_feature: full variable name
    :param start: naive datetime.datetime object
    :param end: naive datetime.datetime object
    :return: df with specified records set to NaN
    """
    # check data types
    if not isinstance(df, pd.DataFrame):
        raise TypeError('df must be a pd.DataFrame')
    if df.index.dtype != 'datetime64[us, UTC]':
        raise Exception(f'Index of df must contain datetime64[us, UTC] data.')
    if not isinstance(long_feature, str):
        raise TypeError('long_feature must be a string')
    if type(start) is not datetime.datetime:
        raise Exception(f'start must be a datetime.datetime object.')
    if type(end) is not datetime.datetime:
        raise Exception(f'end must be a datetime.datetime object.')

    # add timezone
    start = start.replace(tzinfo=pytz.UTC)
    end = end.replace(tzinfo=pytz.UTC)

    # check values
    if long_feature not in df.columns:
        raise Exception(f'Missing required column "{long_feature}".')
    if start not in df.index:
        raise Exception(f'df must contain data from {start}.')
    if end not in df.index:
        raise Exception(f'df must contain data from {end}.')

    df_copy = df.copy()
    df_copy.loc[start:end, long_feature] = np.nan

    return df_copy

def make_nan_indices(df: pd.DataFrame, long_feature: str, timestamps: pd.Index) -> pd.DataFrame:
    """
    Set long_feature of rows in df that match timestamps by index to np.nan.
    :param df: from collect_data(), create_timestamp_col(), and convert_nan()
    :param long_feature: full variable name
    :param timestamps: index of datetime64[us, UTC]
    :return: df with specified records set to NaN
    """
    # check data types
    if not isinstance(df, pd.DataFrame):
        raise TypeError('df must be a pd.DataFrame')
    if df.index.dtype != 'datetime64[us, UTC]':
        raise Exception(f'Index of df must contain datetime64[us, UTC] data.')
    if not isinstance(long_feature, str):
        raise TypeError('long_feature must be a string')
    if not isinstance(timestamps, pd.Index):
        raise TypeError('timestamps must be a pd.Index')

    # check values
    if long_feature not in df.columns:
        raise Exception(f'Missing required column "{long_feature}".')

    df_copy = df.copy()

    df_copy.loc[timestamps.to_list(), [long_feature]] = np.nan

    return df_copy


def find_outlier_spikes(df: pd.DataFrame, long_feature: str, threshold: numbers.Real) -> pd.Index:
    """
    Detect single datapoint outliers for column long_feature in df based on threshold.
    A single datapoint is flagged as an outlier if the absolute differences between it and BOTH immediate non-NaN
    neighbors are greater than threshold.
    :param df: after processing with collect_data(), create_timestamp_col(), and convert_nan()
    :param long_feature: full variable name
    :param threshold: number
    :return: pd.Index containing timestamps of outliers
    """
    # check data types
    if not isinstance(df, pd.DataFrame):
        raise TypeError('df must be a pd.DataFrame')
    if df.index.dtype != 'datetime64[us, UTC]':
        raise Exception(f'Index of df must contain datetime64[us, UTC] data.')
    if not isinstance(long_feature, str):
        raise TypeError('long_feature must be a string')
    if not isinstance(threshold, numbers.Real):
        raise TypeError('threshold must be a numeric')

    # check values
    if long_feature not in df.columns:
        raise Exception(f'Missing required column "{long_feature}".')
    if threshold <= 0:
        raise Exception(f'threshold must be greater than 0.')

    df_copy = df.copy()
    s = df[long_feature]

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
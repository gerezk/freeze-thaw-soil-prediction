import pandas as pd
import numpy as np
import datetime
from pydantic import validate_call, ConfigDict

from src.data_preparation.general import validate_time_index


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
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
    if df.empty:
        raise ValueError('df must not be empty')
    if long_variable not in df.columns:
        raise KeyError(f'df missing required column "{long_variable}".')
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

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
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

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def make_nan_window(df: pd.DataFrame, long_variable: str, start: datetime.datetime, end: datetime.datetime) -> pd.DataFrame:
    """
    Set records between start and end timestamps (inclusive) to np.nan.
    :param df: from collect_data(), create_timestamp_col(), and convert_nan()
    :param long_variable: full variable name
    :param start: naive datetime.datetime object
    :param end: naive datetime.datetime object
    :return: df with specified records set to NaN
    """
    # check input values
    if long_variable not in df.columns:
        raise KeyError(f'Missing required column "{long_variable}".')
    validate_time_index(df)

    # add timezone
    start = start.replace(tzinfo=df.index.tz)
    end = end.replace(tzinfo=df.index.tz)

    # check start and end
    if start not in df.index:
        raise KeyError(f'df must contain data from {start} (hour must be specified).')
    if end not in df.index:
        raise KeyError(f'df must contain data from {end} (hour must be specified).')

    df_copy = df.copy()
    df_copy.loc[start:end, long_variable] = np.nan

    return df_copy

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def make_nan_indices(df: pd.DataFrame, long_variable: str, timestamps: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Set long_variable of rows in df that match timestamps by index to np.nan.
    :param df: from collect_data(), create_timestamp_col(), and convert_nan()
    :param long_variable: full variable name
    :param timestamps: timezone-aware pd.DatetimeIndex
    :return: df with specified records set to NaN
    """
    # check input values
    if long_variable not in df.columns:
        raise KeyError(f'Missing required column "{long_variable}".')
    if timestamps.tz is None:
        raise ValueError("timestamps must be timezone-aware")
    validate_time_index(df)

    df_copy = df.copy()

    # Vectorized operation to set specified records to NaN
    mask = df.index.normalize().isin(timestamps)
    df_copy.loc[mask, long_variable] = np.nan


    return df_copy
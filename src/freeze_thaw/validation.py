import pandas as pd
from pydantic import validate_call, ConfigDict
from typing import cast
from datetime import datetime


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def validate_time_index(df: pd.DataFrame) -> None:
    """
    Validate timezone-aware datetime index of df.
    :param df: from preprocessing
    :return: None
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("df index must be DatetimeIndex")
    dt_index = cast(pd.DatetimeIndex, df.index)
    if dt_index.tz is None:
        raise ValueError("df index must be timezone-aware")

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def check_df_cols(df: pd.DataFrame, system: str) -> None:
    """
    Check if df contains all required ASCAT or ERA5 columns
    :param df: from collect_data()
    :param system: ASCAT or ERA5, case-insensitive
    :return: None
    """
    if system.upper() == "ASCAT":
        required_cols = {'time', 'backscatter40', 'swath_indicator', 'as_des_pass', 'sat_id'}
    elif system.upper() == "ERA5":
        required_cols = {'time', 'skt', 'stl1', 'stl2', 'swvl1', 'swvl2', 'sd'}
    else:
        raise TypeError("system must be ASCAT or ERA5 (case-insensitive)")

    if not required_cols.issubset(df.columns):
        raise KeyError(f'{system} df must contain all of these columns: {str(required_cols)}')

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def validate_date_range(df: pd.DataFrame, start: datetime | None=None, end: datetime | None=None) -> None:
    """
    Validate start and end datetime based on provided df.
    :param df: pd.Dataframe
    :param start: naive datetime.datetime object (inclusive)
    :param end: naive datetime.datetime object (inclusive)
    :return:
    """
    # validate start and end independently
    if start is not None:
        start = start.replace(tzinfo=df.index.tz)
        if start < min(df.index):
            raise ValueError(f'{start} must not be before the first timestamp in df: {min(df.index)}.')
        if start >= max(df.index):
            raise ValueError(f'{start} must not be on or after the last timestamp in df: {max(df.index)}.')
    if end is not None:
        end = end.replace(tzinfo=df.index.tz)
        if end > max(df.index):
            raise ValueError(f'{end} must not be after the last timestamp in df: {max(df.index)}.')
        if end <= min(df.index):
            raise ValueError(f'{end} must not be before or on the first timestamp in df: {min(df.index)}.')

    # check relation between start and end
    if start is not None and end is not None:
        if start >= end:
            raise ValueError(f'start must be before end.')
import pandas as pd
from datetime import datetime
from pydantic import validate_call, ConfigDict
from pathlib import Path
import logging
logger = logging.getLogger(__name__)

from freeze_thaw.validation import validate_time_index, validate_date_range
from freeze_thaw._internal_functions import classify_value
from freeze_thaw.config import StationName


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def filter_df(df: pd.DataFrame, start: datetime | None=None, end: datetime | None=None) -> pd.DataFrame:
    """
    Filters dataframe based on date range.
    :param df: has timezone-aware DatetimeIndex
    :param start: naive datetime.datetime object (inclusive)
    :param end: naive datetime.datetime object (inclusive)
    :return: filtered dataframe
    """
    # check input data types and values
    if df.empty:
        raise ValueError('df must not be empty')
    validate_time_index(df)
    validate_date_range(df, start, end)

    df_copy = df.copy()

    if start is not None:
        start = start.replace(tzinfo=df_copy.index.tz)
    if end is not None:
        end = end.replace(tzinfo=df_copy.index.tz)

    # set date range
    if start is None and end is not None:
        df_copy = df_copy.loc[df.index <= end]
    elif start is not None and end is None:
        df_copy = df_copy.loc[df.index >= start]
    elif start is not None and end is not None:
        df_copy = df_copy.loc[start:end]
    # else include all records
    df_copy = df_copy.sort_index()

    return df_copy


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def add_class_col(df: pd.DataFrame, variable: str, col_name: str) -> pd.DataFrame:
    """
    Adds a class column to dataframe based on a boundary mirrored cross the freezing point in C.
    Classes must be a list of strings of length three, with elements in descending order by temperature.
    :param df: from collect_data()
    :param variable: variable name
    :param col_name: class column name
    :return: dataframe with added class column
    """
    # check input data types
    if not pd.api.types.is_numeric_dtype(df[variable]):
        raise TypeError(f'{variable} column in df must be a numeric type')

    df_copy = df.copy()

    df_copy[col_name] = df_copy[variable].map(classify_value)

    return df_copy

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def collect_cleaned_data(station_name: StationName,
                         cleaned_data_path: Path,
                         datetimeindex_name: str) -> list[pd.DataFrame]:
    """
    Collects cleaned ISMN, ASCAT, and ERA5 data for given ISMN station.
    :param station_name: Name of the ISMN station
    :param cleaned_data_path: Path to the cleaned data directory
    :param datetimeindex_name: Name of the datetime index column in the cleaned data csv file
    :return: list containing dfs of ISMN, ASCAT, and ERA5 data
    """
    dfs = []
    for file in cleaned_data_path.iterdir():
        if file.is_file():
            file_split = file.stem.split('_')
            if file_split[0] == station_name:
                df = pd.read_csv(file,
                                 index_col=datetimeindex_name,
                                 parse_dates=[datetimeindex_name],
                                 )
                dfs.append(df)
    if len(dfs) != 3:
        raise FileNotFoundError(
            f'{cleaned_data_path} must have exactly 3 files (ISMN, ASCAT, ERA5) for given station, {station_name}.')

    return dfs

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def align_timestamps_then_label(dfs: list[pd.DataFrame],
                                ismn_var_name: str,
                                ascat_key_cols: list[str],
                                era5_key_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Align the three datasets on their DatetimeIndex using an inner join, then label ASCAT and ERA5 records
    according to ISMN soil temperature.
    :param dfs: list of length three containing pd.DataFrames of ISMN, ASCAT, and ERA5 data
    :param ismn_var_name: Variable name for ISMN soil temperature
    :param ascat_key_cols: List of str that mark columns representing ASCAT data
    :param era5_key_cols: List of str that mark columns representing ERA5 data
    :return: ascat_df and era5_df, with each record labelled according to ISMN soil temperature. The ERA5
    df contains the ISMN soil temperatures.
    """
    if len(dfs) != 3:
        raise ValueError("dfs must be a list of length three, representing ISMN, ASCAT, and ERA5 data.")

    # inner join all dfs along DatetimeIndex
    combined_df = dfs[0].join(dfs[1:], how="inner")
    combined_df = combined_df.sort_index()

    # add label based on ISMN temp to each record
    combined_df['class'] = combined_df[ismn_var_name].map(classify_value)

    # split into two dfs
    # avoids SettingWithCopyWarning
    ascat_df = combined_df[
        ascat_key_cols + ["class"]
        ].copy()
    era5_df = combined_df[
        era5_key_cols + [ismn_var_name, "class"]
        ].copy()

    # add pred for ERA5
    era5_df['pred'] = era5_df['stl1'].map(classify_value)

    return ascat_df, era5_df
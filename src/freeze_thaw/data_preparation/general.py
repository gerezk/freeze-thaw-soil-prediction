import pandas as pd
from datetime import datetime
from pydantic import validate_call, ConfigDict
import logging
logger = logging.getLogger(__name__)

from freeze_thaw.data_preparation.validation import validate_time_index
from freeze_thaw.internal_functions import classify_value


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def filter_df(df: pd.DataFrame, start: datetime, end: datetime) -> pd.DataFrame:
    """
    Filters dataframe based on date range.
    :param df: has aware DatetimeIndex
    :param start: start date (inclusive)
    :param end: end date (exclusive)
    :return: filtered dataframe
    """
    # check input data types and values
    if df.empty:
        raise ValueError('df must not be empty')
    validate_time_index(df)
    if start >= end:
        raise ValueError('start must be before end')

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
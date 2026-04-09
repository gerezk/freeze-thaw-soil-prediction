import pandas as pd
import numpy as np
from pathlib import Path
from numbers import Real

from src.data_preparation.general import validate_time_index
from src.constants import constants as c


def collect_data(path: Path, max_depth: Real, short_variable: str, long_variable: str) -> pd.DataFrame:
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
    if df.empty:
        raise ValueError('df must not be empty')
    if not {'UTC_date', 'UTC_time'}.issubset(df.columns):
        raise KeyError('df must contain UTC_date and UTC_time columns')

    df_copy = df.copy()

    df_copy[c.DATETIMEINDEX_NAME] = df_copy['UTC_date'].astype(str) + ' ' + df_copy['UTC_time'].astype(str)
    df_copy[c.DATETIMEINDEX_NAME] = pd.to_datetime(df_copy[c.DATETIMEINDEX_NAME], format='%Y/%m/%d %H:%M')
    df_copy = df_copy.drop(columns=['UTC_date', 'UTC_time'])
    df_copy.set_index(c.DATETIMEINDEX_NAME, inplace=True)
    df_copy.index = df_copy.index.tz_localize('UTC')
    df_copy = df_copy.sort_index()

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
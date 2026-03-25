import pandas as pd
from pathlib import Path

from src.constants import DATETIMEINDEX_NAME
from src.data_cleaning.general import validate_time_index


# --------------------
# Data Preprocessing
# --------------------

def collect_data(data_path: Path, ismn_site_survey_path: Path, station_name: str, system: str) -> pd.DataFrame:
    """
    Collect a single ASCAT or ERA5 csv file. Can handle ASCAT and ERA5 data being mixed in the same directory.
    The filename must follow this format: {gpi}_{LON:3f}_{LAT:3f}_{system}_time_series.csv
    :param data_path: path to directory containing ASCAT or ERA5 csv file
    :param ismn_site_survey_path: path to file containing ISMN station data
    :param station_name: must match exactly the names in ISMNS_site_survey.csv
    :param system: ASCAT or ERA5, case-insensitive
    :return: pandas DataFrame
    """
    # check input data types
    if not isinstance(data_path, Path):
        raise TypeError("path must be a Path object")
    if not isinstance(ismn_site_survey_path, Path):
        raise TypeError("ismn_site_survey_path must be a Path object")
    if not isinstance(system, str):
        raise TypeError("system must be a string")
    # check input values
    if not data_path.is_dir():
        raise NotADirectoryError(f'{data_path} must point to a directory')
    if not ismn_site_survey_path.is_file():
        raise FileNotFoundError(f'File not found at{ismn_site_survey_path}')
    if system.upper() not in ["ASCAT", "ERA5"]:
        raise ValueError("system must be ASCAT or ERA5 (case-insensitive)")

    # get unique key for raw data file (lon, lat)
    ismn_sites = pd.read_csv(ismn_site_survey_path)
    site_info = ismn_sites[ismn_sites.ISMN_Station_Name == station_name]
    lon, lat = format(site_info.LON.item(),'.3f'), format(site_info.LAT.item(),'.3f')

    for file in data_path.iterdir():
        filename = file.name
        filename_split = filename.split('_')

        if file.suffix != '.csv' or len(filename_split) != 6:
            continue

        # str matching only
        if filename_split[1] == lon and filename_split[2] == lat and filename_split[3].lower() == system.lower():
            df = pd.read_csv(file)
            return df

    raise ValueError(f'No data was found for {station_name}, {system} in {data_path}')

def check_df_cols(df: pd.DataFrame, system: str) -> None:
    """
    Check if df contains all required ASCAT or ERA5 columns
    :param df: from collect_data()
    :param system: ASCAT or ERA5, case-insensitive
    :return:
    """
    # check input data type
    if not isinstance(system, str):
        raise TypeError("system must be a string")
    # check input value
    if system.upper() not in ["ASCAT", "ERA5"]:
        raise ValueError("system must be ASCAT or ERA5 (case-insensitive)")

    if system.upper() == "ASCAT":
        required_cols = {'time', 'backscatter40', 'swath_indicator', 'as_des_pass', 'sat_id'}
    else: # ERA5
        required_cols = {'time', 'skt', 'stl1', 'stl2', 'swvl1', 'swvl2', 'sd'}

    if not required_cols.issubset(df.columns):
        raise KeyError(f'{system} df must contain all of these columns: {str(required_cols)}')

def round_nearest_hour_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rounds timestamps to nearest hour then sets as the index. Indirect form of interpolation.
    ASSUMPTION: satellite passes are infrequent enough that duplicate timestamps won't be created
    :param df: from collect_data() and check_df_cols()
    :return: pandas DataFrame with DatetimeIndex of rounded timestamps
    """
    df_copy = df.copy()

    df_copy[DATETIMEINDEX_NAME] = pd.to_datetime(df_copy["time"], utc=True)
    df_copy[DATETIMEINDEX_NAME] = df_copy[DATETIMEINDEX_NAME].dt.round("h")
    df_copy = df_copy.set_index(DATETIMEINDEX_NAME)
    df_copy = df_copy.drop(columns=["time"])
    df_copy = df_copy.sort_index()

    return df_copy

def impute_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform df to have hourly imputed values.
    :param df: from collect_data() and check_df_cols()
    :return: df with hourly imputed values
    """
    validate_time_index(df)

    df_copy = df.copy()
    df_copy = df_copy.sort_index()

    df_copy = (df_copy
               .asfreq('h')
               .interpolate(method='time')
               .round(7)
               )

    if df_copy.isna().any().any():
        raise ValueError('Hourly impute failed; some missing values are present.')

    return df_copy
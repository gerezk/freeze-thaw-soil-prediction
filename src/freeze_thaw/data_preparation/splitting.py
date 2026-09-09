import pandas as pd
from pathlib import Path
from pydantic import validate_call, ConfigDict

from freeze_thaw.config import StationName
from freeze_thaw.data_preparation.general import collect_cleaned_data, align_timestamps_then_label
from freeze_thaw.data_preparation.feature_engineering import prepare_df
from freeze_thaw.config import config as c


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def collect_process_split(station: StationName,
                          cleaned_data_path: Path,
                          datetimeindex_name: str,
                          ismn_long_var_name: str,
                          ascat_key_cols: list[str],
                          era5_key_cols: list[str],
                          train_size: float,
                          label_map: dict[str, int],
                          lagged_features: bool = False,
                          lags: list[int] | None = None) -> (pd.DataFrame, pd.DataFrame):
    """
    Given a StationName, collect the relevant cleaned data, join, label, perform feature engineering,
    then create train and test splits.
    :param station: station name from StationName class in config.py
    :param cleaned_data_path: path to cleaned data
    :param datetimeindex_name: datetime index column name in csv file
    :param ismn_long_var_name: full variable name for ISMN soil temperature
    :param ascat_key_cols: columns to keep from ASCAT data
    :param era5_key_cols: columns to keep from ERA5 data
    :param train_size: decimal fraction size of training data
    :param label_map: mapping of class names to ints
    :param lagged_features: create lagged features or not
    :param lags: list of lags e.g. [1, 3] will create features for the backscatter from one and three datapoints prior
    :return: train and test splits as pd.DataFrame objects
    """
    dfs = collect_cleaned_data(station, cleaned_data_path, datetimeindex_name)
    ascat_df, _ = align_timestamps_then_label(dfs, ismn_long_var_name, ascat_key_cols, era5_key_cols)
    ascat_df = ascat_df.drop(columns=ismn_long_var_name)
    ascat_df = prepare_df(ascat_df, label_encoding=label_map, lagged_features=lagged_features, lags=lags)

    train, test = _train_test_split(ascat_df, train_size)

    return train, test


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def get_test_set(station: StationName,
                 train_size: float,
                 label_map: dict[str, int],
                 *,
                 cleaned_data_path: Path | None = None,
                 datetimeindex_name: str | None = None,
                 ismn_long_var_name: str | None = None,
                 ascat_key_cols: list[str] | None = None,
                 era5_key_cols: list[str] | None = None,
                 lagged_features: bool = False,
                 lags: list[int] | None = None) -> (pd.DataFrame, pd.DataFrame):
    """
    Given a StationName, collect the relevant cleaned data, join, label, perform feature engineering,
    then create and return the test split for both ASCAT and ERA5. The test split will always contain the most recent
    data. Both sets are guaranteed to share the same indices. The ERA5 df will contain the original class labels.
    :param station: station name from StationName class in config.py
    :param train_size: decimal fraction size of training data. If 0.8 is entered,
    the most recent 20% of data will be returned i.e. the test split
    :param label_map: mapping of class names to ints
    :param cleaned_data_path: path to cleaned data
    :param datetimeindex_name: datetime index column name in csv file
    :param ismn_long_var_name: full variable name for ISMN soil temperature
    :param ascat_key_cols: columns to keep from ASCAT data
    :param era5_key_cols: columns to keep from ERA5 data
    :param lagged_features: create lagged features or not
    :param lags: list of lags e.g. [1, 3] will create features for the backscatter from one and three datapoints prior
    :return: test split for both ASCAT and ERA5
    """
    # assign default values if None
    cleaned_data_path = cleaned_data_path or c.CLEANED_DATA_PATH
    datetimeindex_name = datetimeindex_name or c.DATETIMEINDEX_NAME
    ismn_long_var_name = ismn_long_var_name or c.ISMN_LONG_VAR_NAME
    ascat_key_cols = ascat_key_cols or c.ASCAT_KEY_COLS
    era5_key_cols = era5_key_cols or c.ERA5_KEY_COLS

    dfs = collect_cleaned_data(station, cleaned_data_path, datetimeindex_name)
    ascat_df, era5_df = align_timestamps_then_label(dfs, ismn_long_var_name, ascat_key_cols, era5_key_cols)
    ascat_df = ascat_df.drop(columns=ismn_long_var_name)
    ascat_df = prepare_df(ascat_df, label_encoding=label_map, lagged_features=lagged_features, lags=lags)

    # re-align indices
    era5_df = era5_df[(era5_df.index >= ascat_df.index[0]) & (era5_df.index <= ascat_df.index[-1])]
    era5_df = era5_df.dropna(subset=["class"]) # match what's done in prepare_df()

    _, ascat_test = _train_test_split(ascat_df, train_size)
    _, era5_test = _train_test_split(era5_df, train_size)

    # validate that both sets contain identical timestamps in the same order
    if not ascat_test.index.equals(era5_test.index):
        indices_only_in_ascat = ascat_test.index.difference(era5_test.index)
        indices_only_in_era5 = era5_test.index.difference(ascat_test.index)
        raise ValueError("Inconsistent indexes for the ASCAT and ERA5 test sets: \n"
                         "Indices only in ASCAT test set: \n"
                         f"{indices_only_in_ascat} \n"
                         "Indices only in ERA5 test set: \n"
                         f"{indices_only_in_era5}")

    return ascat_test, era5_test


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def _train_test_split(df: pd.DataFrame, train_size: float=0.8) -> (pd.DataFrame, pd.DataFrame):
    """
    Splits df into train and test sets. A custom function was made instead of using sklearn
    since not separating x and y is desired.
    :param df: pd.DataFrame
    :param train_size: float within (0, 1]
    :return: train and test dfs
    """
    if train_size <= 0 or train_size > 1:
        raise ValueError("train_size must be within (0, 1]")

    split = int(len(df) * train_size)

    return df.iloc[:split], df.iloc[split:]
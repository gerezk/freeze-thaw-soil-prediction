import pandas as pd
import numpy as np
from typing import List
from pydantic import validate_call, ConfigDict


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def prepare_df(df: pd.DataFrame, label_encoding: dict[str, int], lagged_features: bool,
               lags: list[int] | None = None) -> pd.DataFrame:
    """
    Prepare df for ingestion by an ML model. Class labels will be converted to integers,
    which is required for model input. Lagged features are first created, then unlabelled rows are dropped.
    :param df: pd.DataFrame
    :param label_encoding: dict mapping c.CLASSES to int
    :param lagged_features: whether to use lagged features
    :param lags: list of lags to create
    :return: processed df
    """
    if lagged_features and lags is None:
        raise ValueError("The variable 'lags' should be a list of ints with a length of at least 1 "
                         "when creating lagged features.")

    df_copy = df.copy()

    if lagged_features:
        df_copy = _create_lagged_features(df_copy, lags)
    df_copy = _cyclical_encoding(df_copy)

    null_count = df_copy[df_copy["class"].isnull()].shape[0]
    print(f"Dropping {null_count} rows with no class label.")
    df_copy = df_copy.dropna(subset=["class"])

    df_copy['class'] = df_copy['class'].map(label_encoding)

    return df_copy


def _create_lagged_features(df: pd.DataFrame, lags: List[int]) -> pd.DataFrame:
    """
    Creates lagged features derived from backscatter40.
    :param df: pd.DataFrame
    :param lags: list of lags to create
    :return: df with lagged features
    """
    if not "backscatter40" in df.columns:
        raise ValueError("backscatter40 is not in the dataframe")
    if not df.index.inferred_type == "datetime64":
        raise ValueError("df index inferred_type must be datetime64")
    if not all(type(x) is int and x > 0 for x in lags):
        raise ValueError("'lags' must contain only positive integers.")

    df_copy = df.copy()

    for lag in lags:
        df_copy[f"backscatter40_lag_{lag}"] = df_copy["backscatter40"].shift(lag)

        lagged_time = df_copy.index.to_series().shift(lag)
        df_copy[f"hours_since_lag_{lag}"] = (
                (df_copy.index.to_series() - lagged_time)
                .dt.total_seconds() / 3600
        )

    # drop rows with incomplete lagged features
    df_copy = df_copy.iloc[max(lags):]

    return df_copy

def _cyclical_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create two columns representing a cyclical encoding for day of the year.
    :param df: pd.DataFrame
    :return: df with cyclical encoding for day of the year
    """
    if not df.index.inferred_type == "datetime64":
        raise ValueError("df index inferred_type must be datetime64")

    df_copy = df.copy()

    doy = df_copy.index.dayofyear

    df_copy["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df_copy["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    return df_copy
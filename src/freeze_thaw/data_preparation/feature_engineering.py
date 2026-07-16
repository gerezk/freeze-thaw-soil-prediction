import pandas as pd
import numpy as np


def create_lagged_features(df: pd.DataFrame, n_lags: int) -> pd.DataFrame:
    """
    Creates linearly spaced lagged features derived from backscatter40, with n_lags controlling how many lags to create.
    For example, n_lags=7 will look at the previous 7 datapoints.
    :param df: pd.DataFrame
    :param n_lags: number of lags
    :return: df with lagged features
    """
    if not "backscatter40" in df.columns:
        raise ValueError("backscatter40 is not in the dataframe")
    if not df.index.inferred_type == "datetime64":
        raise ValueError("df index inferred_type must be datetime64")
    if n_lags <= 0:
        raise ValueError("n_lags must be positive")

    df_copy = df.copy()

    for i in range(n_lags):
        lag = i + 1

        df_copy[f"backscatter40_lag_{lag}"] = df_copy["backscatter40"].shift(lag)

        lagged_time = df_copy.index.to_series().shift(lag)
        df_copy[f"hours_since_lag_{lag}"] = (
                (df_copy.index.to_series() - lagged_time)
                .dt.total_seconds() / 3600
        )

    # drop rows with incomplete lagged features
    df_copy = df_copy.iloc[n_lags:]

    return df_copy

def cyclical_encoding(df: pd.DataFrame) -> pd.DataFrame:
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
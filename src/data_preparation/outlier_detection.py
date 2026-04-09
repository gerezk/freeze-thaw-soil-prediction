import pandas as pd
from numbers import Real

from src.data_preparation.general import validate_time_index


def find_outlier_spikes(df: pd.DataFrame, long_variable: str, threshold: Real) -> pd.DatetimeIndex:
    """
    Detect single datapoint outliers for column long_variable in df based on threshold.
    A single datapoint is flagged as an outlier if the absolute differences between it and BOTH immediate non-NaN
    neighbors are greater than threshold.
    :param df: after processing with collect_data(), create_timestamp_col(), and convert_nan()
    :param long_variable: full variable name
    :param threshold: number
    :return: pd.DatetimeIndex with timezone containing timestamps of outliers
    """
    # check input values
    if long_variable not in df.columns:
        raise KeyError(f'Missing required column "{long_variable}".')
    # check input df index
    validate_time_index(df)

    df_copy = df.copy()
    s = df[long_variable]

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
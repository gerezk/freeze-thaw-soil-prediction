import pandas as pd
from pydantic import validate_call, ConfigDict
from typing import cast


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def validate_time_index(df: pd.DataFrame) -> None:
    """
    Validate time index of df.
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
    :return:
    """
    if system.upper() == "ASCAT":
        required_cols = {'time', 'backscatter40', 'swath_indicator', 'as_des_pass', 'sat_id'}
    else: # ERA5
        required_cols = {'time', 'skt', 'stl1', 'stl2', 'swvl1', 'swvl2', 'sd'}

    if not required_cols.issubset(df.columns):
        raise KeyError(f'{system} df must contain all of these columns: {str(required_cols)}')
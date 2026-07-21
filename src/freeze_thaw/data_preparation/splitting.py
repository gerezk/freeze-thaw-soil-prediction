import pandas as pd
from pydantic import validate_call, ConfigDict


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def train_test_split(df, train_size=0.8) -> (pd.DataFrame, pd.DataFrame):
    """
    Splits df into train and test sets. A custom function was made instead of using sklearn
    since not separating x and y is desired.
    :param df: pd.DataFrame
    :param train_size: float within (0, 1)
    :return: train and test dfs
    """
    if train_size <= 0 or train_size >= 1:
        raise ValueError("train_size must be between 0 and 1")

    split = int(len(df) * train_size)

    return df.iloc[:split], df.iloc[split:]
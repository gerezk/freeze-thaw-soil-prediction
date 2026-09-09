"""
Internal functions used in src.
"""
import pandas as pd

from freeze_thaw.config import config as c


def classify_value_rolling(s: pd.Series,
                           window_hours: int = 72,
                           nan_tolerance: float = 0.3) -> pd.Series:
    """
    Classify each observation based on the values in the preceding window (hours), including the current
    observation. If the current observation is NaN or if window contains a proportion of NaN at or
    above `nan_tolerance`, then classify as None.
    :param s: predicted or actual soil temperatures
    :param window_hours: size of window in hours
    :param nan_tolerance: max proportion that a window can contain NaN. At and above this limit, the current
    observation is labelled as None.
    :return: pd.Series of labels
    """
    result = pd.Series(index=s.index, dtype=object)

    for timestamp in s.index:
        window = s.loc[
            (s.index >= timestamp - pd.Timedelta(hours=window_hours)) &
            (s.index <= timestamp)
        ]

        # label as None if window is too short (at the beginning of the series)
        if (window.index[-1] - window.index[0]).total_seconds() / 3600 != window_hours:
            result.loc[timestamp] = None
            continue

        # label as None if current observation is None
        if pd.isna(s.loc[timestamp]):
            result.loc[timestamp] = None
            continue

        # label as None if proportion of NaN >= nan_tolerance
        nan_fraction = (
            window
            .shift(1) # exclude current observation
            .isna()
            .mean()
        )
        if nan_fraction >= nan_tolerance:
            result.loc[timestamp] = None
            continue

        # drop NaN from window before labelling
        window = window.dropna()

        if (window <= 0).all(): # freeze
            result.loc[timestamp] = c.CLASSES[0]
        elif (window > 0).all(): # thaw
            result.loc[timestamp] = c.CLASSES[2]
        else: # transition
            result.loc[timestamp] = c.CLASSES[1]

    return result


def classify_value_simple(x: float) -> str | None:
    """
    Return class label of x based on boundary and classes, based on the value of x itself. For use in .map().
    """
    if pd.isna(x):
        return None

    boundary = c.CLASS_BOUNDARY

    if x < -boundary:
        return c.CLASSES[0]
    elif x <= boundary:
        return c.CLASSES[1]
    else:
        return c.CLASSES[2]
"""
Internal functions used in src.
"""
from freeze_thaw.config import config as c
import pandas as pd


def classify_value(x):
    """
    Return class label of x based on boundary and classes, with x assummed to be a numeric.
    For use in .map()
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
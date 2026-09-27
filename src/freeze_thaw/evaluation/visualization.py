import pandas as pd
from matplotlib.axes import Axes
from matplotlib.dates import DateFormatter
import numpy as np
from pydantic import validate_call, ConfigDict

from freeze_thaw.config import config as c, StationName
from freeze_thaw.data_understanding.visualization import plot_with_labels


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def create_plot_df(ismn_data: pd.DataFrame,
                   lgb_pred: np.ndarray,
                   era5_pred: pd.Series) -> pd.DataFrame:
    """
    Prepare plot df needed as input for plot_predictions().
    :param ismn_data: pd.DataFrame from StationResults.ismn_data
    :param lgb_pred: np.ndarray from TestResult
    :param era5_pred: pd.Series from ERA5Results
    :return: pd.DataFrame
    """
    plot_df = ismn_data.copy()
    plot_df["lgb_pred"] = lgb_pred
    plot_df["era5_pred"] = era5_pred

    return plot_df


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def plot_predictions(df: pd.DataFrame,
                     station: StationName,
                     method: str,
                     *,
                     ismn_long_var_name: str | None = None) -> Axes:
    """
    Plot predictions for a given station and method.
    :param df: must have a datetime index and  contain the following columns: ("lgb_pred" or "era5_pred") and ismn_long_var_name.
    :param station: name of the ISMN station
    :param method: "lgb_pred" or "era5_pred"
    :param ismn_long_var_name: long variable name for ISMN soil temperature
    :return: Axes object
    """
    ismn_long_var_name = ismn_long_var_name or c.ISMN_LONG_VAR_NAME

    if method not in ("lgb_pred", "era5_pred"):
        raise ValueError("method must be 'lgb_pred' or 'era5_pred'")
    if not {method, ismn_long_var_name}.issubset(df.columns):
        raise ValueError(f"df must have columns: '{method}' and '{ismn_long_var_name}'. It currently only has {df.columns}")

    axes = plot_with_labels(df,
                            ismn_long_var_name,
                            hue=method)

    title_fragment = "LightGBM Model"
    if method == "era5_pred":
        title_fragment = "ERA5 Model"
    axes.set_title(f"{title_fragment} Predictions for {station}")

    # remove the legend title
    legend = axes.legend()
    legend.set_title("")

    # rotate x-ticks
    xticks = axes.get_xticks()
    axes.set_xticks(xticks) # to avoid warning
    axes.set_xticklabels(axes.get_xticklabels(), rotation=45)

    # set the x-axis date format to YYYY-MM
    date_format = DateFormatter('%Y-%m')
    axes.xaxis.set_major_formatter(date_format)

    axes.set_xlabel("Timestamp")

    axes.set_ylabel("ISMN Soil Temperature (\u00B0C)")

    return axes
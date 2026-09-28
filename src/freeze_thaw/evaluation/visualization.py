import pandas as pd
import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.dates import DateFormatter
import numpy as np
from pydantic import validate_call, ConfigDict
from pathlib import Path

from freeze_thaw.config import config as c, StationName
from freeze_thaw.data_understanding.visualization import plot_with_labels
from freeze_thaw.utils import find_repo_root


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


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def plot_metrics(lgb_df: pd.DataFrame, era5_df: pd.DataFrame, save_image: bool = False) -> Axes:
    """
    Plot metrics
    :param lgb_df: pd.DataFrame
    :param era5_df: pd.DataFrame
    :param save_image: bool
    :return:
    """
    key_cols = {"Station", "Accuracy", "Macro F1", "Transition F1"}
    if not key_cols.issubset(lgb_df.columns):
        raise ValueError(f"lgb_df must have columns: {key_cols}. It currently only has {lgb_df.columns}")
    if not key_cols.issubset(era5_df.columns):
        raise ValueError(f"era5_df must have columns: {key_cols}. It currently only has {era5_df.columns}")

    img_dir = find_repo_root() / "images"
    if not img_dir.is_dir():
        try:
            img_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            raise OSError(f'Directory not found at {img_dir} and could not be created.')

    fig, axes = plt.subplots(1, 3, figsize=(12, 5), constrained_layout=True)

    # accuracy
    axes[0].plot(lgb_df["Station"], lgb_df["Accuracy"], color="orange", label="lgb-ASCAT")
    axes[0].plot(era5_df["Station"], era5_df["Accuracy"], color="blue", label="ERA5")
    axes[0].tick_params("x", rotation=45, rotation_mode="xtick")
    axes[0].set_ylim(0, 1)
    axes[0].set_xlabel("Station")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    # macro F1
    axes[1].plot(lgb_df["Station"], lgb_df["Macro F1"], color="orange", label="lgb-ASCAT")
    axes[1].plot(era5_df["Station"], era5_df["Macro F1"], color="blue", label="ERA5")
    axes[1].tick_params("x", rotation=45, rotation_mode="xtick")
    axes[1].set_ylim(0, 1)
    axes[1].set_xlabel("Station")
    axes[1].set_ylabel("Macro F1")
    axes[1].legend()

    # transition F1
    axes[2].plot(lgb_df["Station"], lgb_df["Transition F1"], color="orange", label="lgb-ASCAT")
    axes[2].plot(era5_df["Station"], era5_df["Transition F1"], color="blue", label="ERA5")
    axes[2].tick_params("x", rotation=45, rotation_mode="xtick")
    axes[2].set_ylim(0, 1)
    axes[2].set_xlabel("Station")
    axes[2].set_ylabel("Transition F1")
    axes[2].legend()

    if save_image:
        fig.savefig(img_dir / "metrics.png")

    return axes
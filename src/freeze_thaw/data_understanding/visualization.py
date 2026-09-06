import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import seaborn as sns
from datetime import datetime
from pathlib import Path
import plotly.express as px
from plotly.graph_objects import Figure
from pydantic import validate_call, ConfigDict

from freeze_thaw.validation import validate_time_index, validate_date_range
from freeze_thaw.data_preparation.general import filter_df
from freeze_thaw.utils import find_repo_root
from freeze_thaw.config import StationName, config as c


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def plot_var_vs_time(df: pd.DataFrame, variable: str, form: str, draw_zero_line: bool = False,
         y_label: str | None=None, start: datetime | None=None, end: datetime | None=None) -> plt.Axes:
    """
    Create a line or scatter plot of variable vs the index.
    Scatter should be chosen if there's any datapoints that are surrounded by NaN.
    If end given but not start, the first timestamp in the df to end will be plotted.
    If start given but not end, the start to the last timestamp in the df will be plotted.
    :param df: from collect_data(), create_timestamp_col(), and convert_nan()
    :param variable: name of column to plot on y-axis
    :param form: line or scatter, case-insensitive
    :param draw_zero_line: whether to draw zero line
    :param y_label: y-label for plot
    :param start: naive datetime.datetime object (inclusive)
    :param end: naive datetime.datetime object (inclusive)
    :return: matplotlib Axes object
    """
    # check input values
    if df.empty:
        raise ValueError('df must not be empty')
    if variable not in df.columns:
        raise KeyError(f'df missing required column "{variable}".')
    if form.lower() not in ['line', 'scatter']:
        raise ValueError('form must be "line" or "scatter" (case-insensitive).')

    validate_time_index(df)
    validate_date_range(df, start, end)

    df_copy = df.copy()

    df_copy = filter_df(df_copy, start=start, end=end)

    # create plot objects
    fig, ax = plt.subplots()
    if form.lower() == 'line':
        ax.plot(df_copy.index, df_copy[variable])
    elif form.lower() == 'scatter':
        ax.scatter(df_copy.index, df_copy[variable])
    else:
        raise ValueError(f'form somehow changed to invalid value from when it was checked to now')
    if y_label is not None:
        ax.set_ylabel(y_label)
    else:
        ax.set_ylabel(variable)
    ax.set_xlabel('Date')
    ax.tick_params(axis='x', rotation=30)

    if draw_zero_line:
        ax.axhline(y=0, color='k')

    return ax

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def plot_with_labels(df: pd.DataFrame, variable: str,
                     classes: list[str] | None = None,
                     start: datetime | None=None,
                     end: datetime | None=None,
                     x: str | None=None, hue: str = "class") -> Axes:
    """
    Plots scatterplot of variable vs time with datapoints colored by class label according to ISMN data.
    The x and y-labels are left as empty.
    :param df: pd.DataFrame containing a timezone-aware DatetimeIndex and records classified base on ISMN data
    :param variable: name of column to plot on y-axis
    :param classes: freeze-thaw class labels, must be length 3 in ascending temperature order
    :param start: naive datetime.datetime object (inclusive)
    :param end: naive datetime.datetime object (inclusive)
    :param x: x-axis variable name. If none, use index.
    :param hue: name of categorical column to be used to color data points
    :return: matplotlib Axes object
    """
    if not {hue, variable}.issubset(df.columns):
        raise ValueError(f"df must contain {hue} and '{variable}' columns")
    if x is None:
        validate_time_index(df)
        x = df.index.name
    else:
        if x not in df.columns:
            raise ValueError(f'{x} must be in df columns')

    # set defaults
    classes = classes or c.CLASSES
    start = start or df.index[0]
    end = end or df.index[-1]

    df_copy = df.copy()

    df_copy = filter_df(df_copy, start, end)

    palette = {
        classes[0]: "tab:blue",
        classes[1]: "tab:brown",
        classes[2]: "tab:green",
    }

    ax = sns.scatterplot(data=df_copy, x=x, y=variable,
                         hue=hue, palette=palette)

    return ax

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def plot_temp_differences(df: pd.DataFrame, name: StationName, start: datetime, end: datetime) -> tuple[Axes, Axes]:
    """
    Creates two plots based on the ISMN and ERA5 data.
    The first is a line plot of both the measured soil temperatures according to both methods.
    The second plot is a line plot that shows the difference between the ISMN and ERA5 soil temperatures.
    :param df: cleaned and combined data df
    :param name: station name
    :param start: naive datetime.datetime object (inclusive)
    :param end: naive datetime.datetime object (inclusive)
    :return: two matplotlib Axes objects
    """
    if not {'soil_temp', 'stl1'}.issubset(df.columns):
        raise ValueError("df must contain 'soil_temp' and 'stl1' columns")
    validate_time_index(df)

    df_copy = df.copy()
    df_copy = filter_df(df_copy, start, end)

    diff = df_copy["soil_temp"] - df_copy["stl1"]

    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(12, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]}
    )

    # Top panel
    ax1.plot(df_copy.index, df_copy["stl1"], label="ERA5")
    ax1.plot(df_copy.index, df_copy["soil_temp"], label="ISMN")
    ax1.legend()
    ax1.set_title(name)
    ax1.set_ylabel("Temperature (\u00B0C)")
    ax1.grid(alpha=0.3)

    # Bottom panel
    ax2.plot(df_copy.index, diff, color="tab:red")
    ax2.axhline(0, color="black", ls="--", lw=1)
    ax2.set_ylabel("Difference (ISMN-ERA5)")
    ax2.set_xlabel("Timestamp")
    ax2.grid(alpha=0.3)

    return ax1, ax2


@validate_call
def map_stations(site_survey_path: Path, output_dir: Path, save_image: bool = False) -> Figure:
    """
    Create map displaying locations of ISMN stations.
    :param site_survey_path: relative path from project root to ISMN_site_survey.csv
    :param output_dir: relative path from project root to output folder
    :param save_image: whether to save plot in ../images; takes a few seconds if True
    :return: plotly.graph_objects.Figure
    """
    repo_root = find_repo_root()
    site_survey_path = repo_root / site_survey_path
    output_dir = repo_root / Path(output_dir)

    # check input values
    if not site_survey_path.is_file():
        raise FileNotFoundError(f'File not found at {site_survey_path}')
    if not output_dir.is_dir():
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            raise OSError(f'Directory not found at {site_survey_path} and could not be created.')

    df = pd.read_csv(site_survey_path)

    fig = px.scatter_geo(
        df,
        lat="LAT",
        lon="LON",
        color="Region",
        hover_name="ISMN_Station_Name",
    )
    fig.update_layout(
        margin=dict(l=0, r=120, t=0, b=0),
        legend=dict(
            x=1.02,  # move legend outside the plot
            y=0.5,  # vertical center
            xanchor="left",  # anchor legend's left side at x
            yanchor="middle"
        )
    )
    fig.update_geos(
        fitbounds="locations",
        showcountries=True
    )
    if save_image:
        fig.write_image(output_dir / "map_ISMN_stations.png")

    return fig
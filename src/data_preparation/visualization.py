import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import plotly.express as px
from plotly.graph_objects import Figure

from src.data_preparation.general import validate_time_index


def plot(df: pd.DataFrame, variable: str, station: str, system: str, form: str,
         y_label=None, start=None, end=None) -> plt.Axes:
    """
    Create a line or scatter plot of variable vs the index.
    Scatter should be chosen if there's any datapoints that are surrounded by NaN.
    If end given but not start, the first timestamp in the df to end will be plotted.
    If start given but not end, the start to the last timestamp in the df will be plotted.
    :param df: from collect_data(), create_timestamp_col(), and convert_nan()
    :param variable: full variable name
    :param station: name of the ISMN station
    :param system: name of the sensor system
    :param form: line or scatter, case-insensitive
    :param y_label: y-label for plot
    :param start: naive datetime.datetime object (inclusive)
    :param end: naive datetime.datetime object (inclusive)
    :return: matplotlib Axes object
    """
    # check input data types
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if not isinstance(variable, str):
        raise TypeError("variable must be a string")
    if not isinstance(station, str):
        raise TypeError("station must be a string")
    if not isinstance(system, str):
        raise TypeError("system must be a string")
    if not isinstance(form, str):
        raise TypeError("form must be a string")
    if y_label is not None:
        if not isinstance(y_label, str):
            raise TypeError("y_label must be a string")
    # check input values
    if df.empty:
        raise ValueError('df must not be empty')
    if variable not in df.columns:
        raise KeyError(f'df missing required column "{variable}".')
    if form.lower() not in ['line', 'scatter']:
        raise ValueError('form must be "line" or "scatter" (case-insensitive).')
    # check input df index
    validate_time_index(df)

    # check start and end independently
    if start is not None:
        if not isinstance(start, datetime):
            raise TypeError(f'start must be a naive datetime.datetime object.')
        start = start.replace(tzinfo=df.index.tz)
        if start < min(df.index):
            raise ValueError(f'{start} must not be before the first timestamp in df: {min(df.index)}.')
        if start >= max(df.index):
            raise ValueError(f'{start} must not be on or after the last timestamp in df: {max(df.index)}.')
    if end is not None:
        if not isinstance(end, datetime):
            raise TypeError(f'end must be a naive datetime.datetime object.')
        end = end.replace(tzinfo=df.index.tz)
        if end > max(df.index):
            raise ValueError(f'{end} must not be after the last timestamp in df: {max(df.index)}.')
        if end <= min(df.index):
            raise ValueError(f'{end} must not be before or on the first timestamp in df: {min(df.index)}.')

    # set date range for plot
    if start is None and end is not None:
        df_slice = df.loc[df.index <= end]
    elif start is not None and end is None:
        df_slice = df.loc[df.index >= start]
    elif start is not None and end is not None:
        # check relation between start and end
        if start == end:
            raise ValueError(f'start and end cannot be the same.')
        if start > end:
            raise ValueError(f'start must be before end.')
        df_slice = df.loc[start:end]
    else: # default to plotting all records
        df_slice = df
    df_slice = df_slice.sort_index()

    # create plot objects
    fig, ax = plt.subplots()
    if form.lower() == 'line':
        ax.plot(df_slice.index, df_slice[variable])
    elif form.lower() == 'scatter':
        ax.scatter(df_slice.index, df_slice[variable])
    else:
        raise ValueError(f'form somehow changed to invalid value from when it was checked to now')
    ax.set_title(f'{station}, {system}')
    if y_label is not None:
        ax.set_ylabel(y_label)
    else:
        ax.set_ylabel(variable)
    ax.set_xlabel('Date')
    ax.tick_params(axis='x', rotation=30)

    if variable == 'soil_temp':
        ax.axhline(y=0, color='k')

    return ax

def map_stations(path: Path, save_image=False) -> Figure:
    """
    Create map displaying locations of ISMN stations.
    :param path: path to ISMN_site_survey.csv
    :param save_image: whether to save plot in ../images; takes a few seconds if True
    :return: plotly.graph_objects.Figure
    """
    # check input data type
    if not isinstance(path, Path):
        raise TypeError
    if not isinstance(save_image, bool):
        raise TypeError('save_image must be a bool')

    if not path.exists():
        raise ValueError("path does not exist")

    df = pd.read_csv(path)

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
        images_dir = Path("../images")
        images_dir.mkdir(parents=True, exist_ok=True)
        fig.write_image(images_dir / "map_ISMN_stations.png")

    return fig
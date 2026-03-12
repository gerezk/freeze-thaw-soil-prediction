from pathlib import Path
import pandas as pd

# --------------------
# Data Preprocessing
# --------------------

def collect_data(path: Path, station_name: str, system: str) -> pd.DataFrame:
    """
    Collect a single ASCAT or ERA5 csv file. Can handle ASCAT and ERA5 data being mixed in the same directory.
    The filename must follow this format: {gpi}_{LON:3f}_{LAT:3f}_{station_name}_{system}_time_series.csv
    ASSUMPTION: data files are uniquely identified by {station_name} and {system}.
    :param path: path to directory containing ASCAT or ERA5 csv file
    :param station_name: name of the ISMN station, case-insensitive
    :param system: ASCAT or ERA5, case-insensitive
    :return: pandas DataFrame
    """
    # check input data types
    if not isinstance(path, Path):
        raise TypeError("raw_data_dir_path must be a Path object")
    if not isinstance(station_name, str):
        raise TypeError("station_name must be a string")
    if not isinstance(system, str):
        raise TypeError("system must be a string")
    # check input values
    if not path.is_dir():
        raise ValueError(f'path ({path}) must point to a directory')
    if system.upper() not in ["ASCAT", "ERA5"]:
        raise ValueError("system must be ASCAT or ERA5 (case-insensitive)")

    station_name = station_name.upper()
    system = system.upper()
    for file in path.iterdir():
        filename = file.name
        filename_split = filename.split('_')

        if file.suffix != '.csv' or len(filename_split) != 7:
            continue

        file_station_name = filename_split[3].upper()
        file_system = filename_split[4].upper()
        if station_name == file_station_name and system == file_system:
            df = pd.read_csv(file)
            return df

    raise ValueError(f'No data was found for {station_name}, {system} in {path}')
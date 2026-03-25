import pandas as pd
from pathlib import Path

import src.constants as c

def main(station_name: str, cleaned_data_path: Path) -> pd.DataFrame:
    """

    :param station_name:
    :param cleaned_data_path:
    :return:
    """
    # validate inputs
    if not isinstance(station_name, str):
        raise TypeError("Station name must be a string")
    if not isinstance(cleaned_data_path, Path):
        raise TypeError("Cleaned data path must be a Path object")
    if not cleaned_data_path.is_dir():
        raise NotADirectoryError(f'{cleaned_data_path} must be a directory')

    # find relevant csv files then import as df and append to list
    dfs = []
    for file in cleaned_data_path.iterdir():
        if file.is_file():
            file_split = file.stem.split('_')
            if file_split[0] == station_name:
                df = pd.read_csv(file, index_col=c.DATETIMEINDEX_NAME)
                dfs.append(df)
    if len(dfs) != 3:
        raise FileNotFoundError(f'{cleaned_data_path} must have exactly 3 files (ISMN, ASCAT, ERA5) for given station, {station_name}.')

    # inner join all dfs along DatetimeIndex
    combined_df = pd.merge(dfs[0], dfs[1], left_index=True, right_index=True)
    combined_df = pd.merge(combined_df, dfs[2], left_index=True, right_index=True)
    combined_df = combined_df.sort_index()

    # add label based on ISMN temp to each record
    print(combined_df.head())

    return combined_df

if __name__ == "__main__":
    station = 'Aberdeen-35-WNW'
    data_path = Path("../data/cleaned")
    main(station, data_path)
import pandas as pd
from pathlib import Path

from freeze_thaw.constants import constants as c
from freeze_thaw.internal_functions import classify_value


def main(station_name: str, cleaned_data_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Processing and labeling of cleaned data csv files.
    :param station_name: name of the ISMN station
    :param cleaned_data_path: path to the cleaned data directory
    :return: dfs for ASCAT and ERA5 data
    """
    # find relevant csv files then import as df and append to list
    dfs = []
    for file in cleaned_data_path.iterdir():
        if file.is_file():
            file_split = file.stem.split('_')
            if file_split[0] == station_name:
                df = pd.read_csv(file,
                                 index_col=c.DATETIMEINDEX_NAME,
                                 parse_dates=[c.DATETIMEINDEX_NAME],
                                 )
                dfs.append(df)
    if len(dfs) != 3:
        raise FileNotFoundError(
            f'{cleaned_data_path} must have exactly 3 files (ISMN, ASCAT, ERA5) for given station, {station_name}.')

    # inner join all dfs along DatetimeIndex
    combined_df = pd.merge(dfs[0], dfs[1], left_index=True, right_index=True)
    combined_df = pd.merge(combined_df, dfs[2], left_index=True, right_index=True)
    combined_df = combined_df.sort_index()

    # add label based on ISMN temp to each record
    combined_df['class'] = combined_df[c.ISMN_LONG_VAR_NAME].map(classify_value)

    # split into two dfs
    ascat_df = combined_df[c.ASCAT_KEY_COLS + [c.ISMN_LONG_VAR_NAME, 'class']]
    era5_df = combined_df[c.ERA5_KEY_COLS + [c.ISMN_LONG_VAR_NAME, 'class']]

    # add pred for ERA5
    era5_df['pred'] = era5_df['stl1'].map(classify_value)

    return ascat_df, era5_df

if __name__ == "__main__":
    station = 'Aberdeen-35-WNW'
    data_path = Path("../../data/cleaned")
    main(station, data_path)

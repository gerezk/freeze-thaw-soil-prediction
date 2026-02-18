import pandas as pd
from matplotlib import pyplot as plt
from pathlib import Path
import numpy as np

data_path = Path('data/Data_separate_files_header_20050831_20250322_13091_l3Sc_20260203/SNOTEL/Nenana')
df = pd.read_csv(data_path / 'SNOTEL_SNOTEL_Nenana_sd_0.000000_0.000000_n.s._20050831_20250322.stm',
                 sep=' ',
                 header=None,
                 skiprows=1)
df.columns = ['UTC_date', 'UTC_time', 'snow_depth', 'ISMN_data_quality', 'provider_data_quality']

# merge UTC_date and UTC_time into UTC_timestamp
df['UTC_timestamp'] = df['UTC_date'].astype(str) + ' ' + df['UTC_time'].astype(str)
df['UTC_timestamp'] = pd.to_datetime(df['UTC_timestamp'], format='%Y/%m/%d %H:%M')
df = df.drop(columns=['UTC_date', 'UTC_time'])

# --- Snow-depth ---
# convert snow depths that are flagged as not good (G) into missing values
df.loc[df['ISMN_data_quality'] != 'G', 'snow_depth'] = np.nan

# identify all missing or invalid values
missing_invalid = df['snow_depth'].isna()

# identify consecutive missing/invalid runs
groups = missing_invalid.ne(missing_invalid.shift()).cumsum()

# compute run lengths in hours
run_lengths = (
    df[missing_invalid]
    .groupby(groups[missing_invalid])
    .size()
)

# find runs lasting at least 14 days/336 hours
long_runs = run_lengths[run_lengths >= 336]
if long_runs.empty:
    cleaned_df = df.copy()

# find last such run and its ending timestamp
last_run_id = long_runs.index[-1]

last_run_end_idx = df.index[
    (groups == last_run_id)
].max()

# clean df
cleaned_df = df.loc[last_run_end_idx + 1:].reset_index(drop=True)

# # determine the first non-zero or non-missing record and discard all records prior to it
# first_nonzero_idx = df[df['snow_depth'] > 0].index[0]
# df = df.iloc[first_nonzero_idx:].reset_index(drop=True)

# Impute remaining missing values using linear interpolation
cleaned_df['snow_depth'] = cleaned_df['snow_depth'].interpolate(method='linear')

plt.plot(cleaned_df['UTC_timestamp'], cleaned_df['snow_depth'])
plt.show()
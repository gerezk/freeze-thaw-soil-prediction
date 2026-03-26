from datetime import datetime
from pathlib import Path

"""
Application constants.
Values here should not change during runtime.
"""
# any data outside this range is dropped
# order doesn't matter, but the min value is included in the data and the max value is excluded.
DATE_RANGE = [datetime(2007, 1, 1), datetime(2025, 1, 1)]

# symmetric threshold across freezing point for determining class
THRESHOLD = 1.0

# path variables
SITE_SURVEY_PATH = Path('../ISMN_site_survey.csv')
CLEANED_DATA_PATH = Path('../data/cleaned')

# station names
ABERDEEN_NAME = 'Aberdeen-35-WNW'
JAMESTOWN_NAME = 'Jamestown-38-WSW'
GOBBLERS_KNOB_NAME = 'GobblersKnob'
NENANA_NAME = 'Nenana'
L23_NAME = 'L23'
L38_NAME = 'L38'
NST_07_NAME = 'NST-07'
NST_09_NAME = 'NST-09'
SOD012_NAME = 'SOD012'
SOD103_NAME = 'SOD103'

# key columns to keep for each dataset after DatetimeIndex set
# either a list of strings, or None to keep all columns
ASCAT_KEY_COLS = ['backscatter40', 'swath_indicator', 'as_des_pass', 'sat_id']
ERA5_KEY_COL = ['stl1']
ISMN_KEY_COL = ['soil_temp']

DATETIMEINDEX_NAME = 'UTC_timestamp'
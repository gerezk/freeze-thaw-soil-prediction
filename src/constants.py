from datetime import datetime
from pathlib import Path

"""
Application constants.
Values here should not change during runtime.
"""
# any data outside this range is dropped, exclusive
DATE_RANGE = {datetime(2007, 1, 1), datetime(2025, 1, 1)}

# path variables
SITE_SURVEY_PATH = Path('../ISMN_site_survey.csv')

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
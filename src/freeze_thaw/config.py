"""
Application constants that should not change during runtime.
Changes to the classes, other than the default values will most likely break the program.

Editable values (edited at last line)
---------------
DateRange:
    start               Start of the date range for data pre-processing
    end                 End of the date range for data pre-processing

Constants:
    DATE_RANGE          DateRange instance (see above)
    CLASS_BOUNDARY      Symmetric temperature boundary (°C) around the freezing point
    SITE_SURVEY_PATH    Path to the ISMN site survey CSV, relative to repo root
    CLEANED_DATA_PATH   Path to the cleaned data directory, relative to repo root
    MODEL_PATH          Path to the trained models directory, relative to repo root
    ASCAT_KEY_COLS      Columns to extract from raw ASCAT data
    ERA5_KEY_COLS       Columns to extract from raw ERA5 data
    ISMN_KEY_COLS       Columns to extract from raw ISMN data
    CLASSES             Freeze-thaw class labels, must be length 3 in ascending temperature order
    DATETIMEINDEX_NAME  Name of the datetime index column in the cleaned data csv files
    ISMN_LONG_VAR_NAME  Long variable name for ISMN soil temperature

StationName:
    Enum members can be added or removed to match the ISMN site survey CSV.
    The enum value must exactly match the ISMN_Station_Name column in the CSV.
"""

from datetime import datetime
from pathlib import Path
from pydantic import BaseModel, Field, model_validator
from enum import Enum
import pandas as pd

from freeze_thaw.utils import find_repo_root
REPO_ROOT = find_repo_root()

class DateRange(BaseModel):
    start: datetime
    end: datetime

    @model_validator(mode="after")
    def validate_order(self):
        if self.start >= self.end:
            raise ValueError("start must be before end")
        return self


class Config(BaseModel):
    model_config = {"frozen": True}

    # sets date range for data pre-processing
    DATE_RANGE: DateRange = DateRange(
        start=datetime(2007, 1, 1),
        end=datetime(2025, 1, 1)
    )

    # symmetric boundary across the freezing point in Celsius
    CLASS_BOUNDARY: float = 1.0

    SITE_SURVEY_PATH: Path = REPO_ROOT / "ISMN_site_survey.csv"
    CLEANED_DATA_PATH: Path = REPO_ROOT / "data" / "cleaned"
    MODEL_PATH: Path = REPO_ROOT / "models"

    ASCAT_KEY_COLS: list[str] = Field(default_factory=lambda: [
        'backscatter40', 'swath_indicator', 'as_des_pass', 'sat_id'
    ])
    ERA5_KEY_COLS: list[str] = Field(default_factory=lambda: ['stl1'])
    ISMN_KEY_COLS: list[str] = Field(default_factory=lambda: ['soil_temp'])

    # must be ordered in ascending temperature
    CLASSES: list[str] = Field(
        default_factory=lambda: ['frozen', 'transition', 'thawed'],
        min_length=3,
        max_length=3
    )

    DATETIMEINDEX_NAME: str = 'UTC_timestamp'
    ISMN_LONG_VAR_NAME: str = 'soil_temp'

    @model_validator(mode="after")
    def validate(self):
        if self.CLASS_BOUNDARY <= 0:
            raise ValueError("CLASS_BOUNDARY must be positive.")
        if not self.SITE_SURVEY_PATH.is_file():
            raise FileNotFoundError(f'File not found at {self.SITE_SURVEY_PATH}')
        if self.CLEANED_DATA_PATH.is_file():
            raise FileExistsError(f'{self.CLEANED_DATA_PATH} must point to a directory.')
        if not self.CLEANED_DATA_PATH.exists():
            self.CLEANED_DATA_PATH.mkdir(parents=True, exist_ok=True)
        if not self.MODEL_PATH.exists():
            self.MODEL_PATH.mkdir(parents=True, exist_ok=True)
        return self

    @model_validator(mode="after")
    def validate_stations_in_sync(self):
        csv_names = set(pd.read_csv(self.SITE_SURVEY_PATH)['ISMN_Station_Name'])
        enum_names = {s.value for s in StationName}
        if csv_names != enum_names:
            raise ValueError(f"StationName enum is out of sync with CSV: "
                             f"missing {csv_names - enum_names}, "
                             f"extra {enum_names - csv_names}")
        return self


class StationName(str, Enum):
    """
    Editable values for ISMN stations. Stations can be added or removed.
    Values must be in sync with those in ISMN_site_survey.csv."""
    ABERDEEN = 'Aberdeen-35-WNW'
    JAMESTOWN = 'Jamestown-38-WSW'
    GOBBLERS_KNOB = 'GobblersKnob'
    NENANA = 'Nenana'
    L23 = 'L23'
    L38 = 'L38'
    NST_07 = 'NST-07'
    NST_09 = 'NST-09'
    SOD012 = 'SOD012'
    SOD103 = 'SOD103'

    def __str__(self):
        return self.value

# assign values here if needed
# dateRange = DateRange()
config = Config()
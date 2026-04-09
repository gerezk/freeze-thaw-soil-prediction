from datetime import datetime
from pathlib import Path
from pydantic import BaseModel, Field, model_validator
from enum import Enum

"""
Application constants that should not change during runtime.
Changes to the classes, other than the default values will most likely break the program.
"""

class DateRange(BaseModel):
    start: datetime
    end: datetime


class Constants(BaseModel):
    model_config = {"frozen": True}

    DATE_RANGE: DateRange = DateRange(
        start=datetime(2007, 1, 1),
        end=datetime(2025, 1, 1)
    )

    # symmetric boundary across the freezing point in celsius
    CLASS_BOUNDARY: float = 1.0

    BASE_DIR: Path = Path(__file__).resolve().parent
    SITE_SURVEY_PATH: Path = BASE_DIR / "../ISMN_site_survey.csv"
    CLEANED_DATA_PATH: Path = BASE_DIR / "../data/cleaned"

    ASCAT_KEY_COLS: list[str] = Field(default_factory=lambda: [
        'backscatter40', 'swath_indicator', 'as_des_pass', 'sat_id'
    ])

    ERA5_KEY_COLS: list[str] = Field(default_factory=lambda: ['stl1'])
    ISMN_KEY_COLS: list[str] = Field(default_factory=lambda: ['soil_temp'])

    # must be ordered in descending temperature
    CLASSES: list[str] = Field(
        default_factory=lambda: ['thawed', 'transition', 'frozen'],
        min_length=3,
        max_length=3
    )

    DATETIMEINDEX_NAME: str = 'UTC_timestamp'
    ISMN_LONG_VAR_NAME: str = 'soil_temp'

    @model_validator(mode="after")
    def validate(self):
        if self.DATE_RANGE.start >= self.DATE_RANGE.end:
            raise ValueError("DATE_RANGE.start must be before DATE_RANGE.end.")
        if self.CLASS_BOUNDARY <= 0:
            raise ValueError("CLASS_BOUNDARY must be positive.")
        if not self.SITE_SURVEY_PATH.is_file():
            raise FileNotFoundError(f'File not found at {self.SITE_SURVEY_PATH}')
        if self.CLEANED_DATA_PATH.is_file():
            raise FileExistsError(f'{self.CLEANED_DATA_PATH} must point to a directory.')
        if not self.CLEANED_DATA_PATH.exists():
            self.CLEANED_DATA_PATH.mkdir(parents=True, exist_ok=True)
        return self


class StationName(str, Enum):
    """Editable values for ISMN stations. Stations can be added or removed."""
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
constants = Constants()
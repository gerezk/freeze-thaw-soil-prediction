from dataclasses import dataclass
import pandas as pd
import numpy as np
import lightgbm as lgb
from pydantic import validate_call, ConfigDict
from sklearn.metrics import accuracy_score
from pathlib import Path

from freeze_thaw.evaluation.metrics import calculate_f1_scores
from freeze_thaw.config import StationName, config as c
from freeze_thaw.data_preparation.splitting import get_test_sets


@dataclass(frozen=True)
class TestResult:
    accuracy: float
    macro_f1: float
    transition_f1: float
    y_pred: np.ndarray


@dataclass(frozen=True)
class StationResults:
    y_true: pd.Series
    ismn_data: pd.DataFrame
    lgb_results: TestResult
    era5_results: TestResult


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def process_one_station(station: StationName,
                        train_size: float,
                        label_map: dict[str, int],
                        lags: list[int] | None = None,
                        model_dir: Path | None = None,
                        ismn_long_var_name: str | None = None) -> StationResults:
    """
    For the given station, load lgb model, predict on test set, and return model results and ERA5 predictions.
    :param station: name of the ISMN station
    :param train_size: decimal fraction size of training data
    :param label_map: mapping of class names to int labels
    :param lags: lag configuration
    :param model_dir: path to directory containing the models
    :param ismn_long_var_name: long name of the ISMN key variable
    :return: StationResults
    """
    model_dir = model_dir or c.MODEL_PATH
    ismn_long_var_name = ismn_long_var_name or c.ISMN_LONG_VAR_NAME
    label_type = "simple"
    if lags is not None:
        label_type = "rolling"

    ascat_test, era5_test = get_test_sets(station, train_size=train_size, label_map=label_map, lags=lags)
    y_true = era5_test["class"]

    lgb_model = lgb.Booster(model_file=model_dir / f"{station}_{label_type}_model.txt")
    lgb_results = lgb_pred(ascat_test, lgb_model, label_map)

    era5_accuracy = accuracy_score(y_true, era5_test["pred"])
    era5_macro_f1, era5_transition_f1 = calculate_f1_scores(y_true, era5_test["pred"], list(label_map.keys()))

    era5_results = TestResult(accuracy=era5_accuracy,
                               macro_f1=era5_macro_f1,
                               transition_f1=era5_transition_f1,
                               y_pred=era5_test["pred"])

    ismn_data = era5_test.copy()
    ismn_data = ismn_data.drop(columns=[col for col in ismn_data.columns if col != ismn_long_var_name])

    return StationResults(y_true=y_true,
                          lgb_results=lgb_results,
                          era5_results=era5_results,
                          ismn_data=ismn_data)


# evaluation could be separated into different function
@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def lgb_pred(df: pd.DataFrame,
             model: lgb.Booster,
             label_encoding: dict[str, int],) -> TestResult:
    """
    Predict soil state on df given an lgbm model and output macro and transition state f1 scores.
    :param df: pd.DataFrame from prepare_df()
    :param model: lightgbm.Booster
    :param label_encoding: mapping of c.CLASSES to int labels
    :return: the macro and transition state f1 scores, along with the predictions
    """
    if "class" not in df.columns:
        raise ValueError("'df' must have 'class' column")

    df_copy = df.copy()

    x = df_copy.drop(columns="class")
    y = df_copy["class"]

    predictions = model.predict(x)
    y_pred = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(y, y_pred)
    macro_f1, transition_f1 = calculate_f1_scores(y, y_pred, list(label_encoding.values()))

    # convert back to original class labels
    inv_map = {v: k for k, v in label_encoding.items()}
    y_pred = np.vectorize(inv_map.get)(y_pred)

    return TestResult(accuracy=accuracy,
                      macro_f1=macro_f1,
                      transition_f1=transition_f1,
                      y_pred=y_pred)


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def add_row_to_results(df: pd.DataFrame, station: StationName, results: TestResult) -> pd.DataFrame:
    """
    Add row to results dataframe given the TestResult object.
    :param df: must have columns ["Station", "Accuracy", "Macro F1", "Transition F1"]
    :param station: name of ISMN station
    :param results: TestResult object
    :return: df with new record added
    """
    key_cols = {"Station", "Accuracy", "Macro F1", "Transition F1"}
    if not key_cols.issubset(df.columns):
        raise ValueError(f"df must have columns: {key_cols}. It currently only has {df.columns}")

    df_copy = df.copy()

    df_copy.loc[len(df_copy)] = [station,
                                 results.accuracy,
                                 results.macro_f1,
                                 results.transition_f1]

    return df_copy
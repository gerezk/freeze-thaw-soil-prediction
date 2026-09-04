import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from dataclasses import dataclass
from pydantic import validate_call, ConfigDict
from pathlib import Path
from typing import Iterable

from freeze_thaw.evaluation.metrics import calculate_f1_scores
from freeze_thaw.data_preparation.splitting import collect_process_split
from freeze_thaw.config import StationName, config as c


@dataclass(frozen=True)
class TrainingResult:
    model: lgb.Booster
    fold_macro_f1_scores: list[float]
    fold_transition_f1_scores: list[float]
    oof_macro_f1: float
    oof_transition_f1: float
    oof_predictions: np.ndarray
    oof_probabilities: np.ndarray


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def train_and_save_station_models(train_size: float,
                                  n_splits: int,
                                  label_encoding: dict[str, int],
                                  *,
                                  lagged_features: bool = False,
                                  lags: list[int] | None = None,
                                  params: dict[str, object] | None = None,
                                  stations: Iterable[StationName] = StationName,
                                  cleaned_data_path: Path | None = None,
                                  model_path: Path | None = None,
                                  datetimeindex_name: str | None = None,
                                  ismn_long_var_name: str | None = None,
                                  ascat_key_cols: list[str] | None = None,
                                  era5_key_cols: list[str] | None = None,
                                  overwrite: bool = False) -> None:
    """
    Orchestrator to train and save lightgbm models for all stations with the cleaned, labelled ASCAT data being the input.
    :param train_size: decimal fraction size of training data
    :param n_splits: number of folds for cross-validation
    :param label_encoding: mapping of str classes to int labels
    :param lagged_features: create lagged features or not
    :param lags: list of lags e.g. [1, 3] will create features for the backscatter from one and three datapoints prior
    :param params: parameters to pass to lightgbm.train
    :param stations: StationName class from config.py
    :param cleaned_data_path: absolute path to the cleaned data directory
    :param model_path: absolute path to directory to save the model
    :param datetimeindex_name: name of the datetime index column in the cleaned data csv files
    :param ismn_long_var_name: long variable name for ISMN soil temperature
    :param ascat_key_cols: list of ASCAT key column names
    :param era5_key_cols: list of ERA5 key column names
    :param overwrite: overwrite existing model files
    :return: None
    """
    cleaned_data_path = cleaned_data_path or c.CLEANED_DATA_PATH
    model_path = model_path or c.MODEL_PATH
    datetimeindex_name = datetimeindex_name or c.DATETIMEINDEX_NAME
    ismn_long_var_name = ismn_long_var_name or c.ISMN_LONG_VAR_NAME
    ascat_key_cols = ascat_key_cols or c.ASCAT_KEY_COLS
    era5_key_cols = era5_key_cols or c.ERA5_KEY_COLS

    for station in stations:
        print(f"Training and saving LightGBM model for {station}")
        file_path = model_path / f"{station}_model.txt"
        if file_path.exists() and not overwrite:
            print(f"{file_path.name} already exists. Skipping.")
            continue

        train, _ = collect_process_split(station,
                                            cleaned_data_path,
                                            datetimeindex_name,
                                            ismn_long_var_name,
                                            ascat_key_cols,
                                            era5_key_cols,
                                            train_size,
                                            label_encoding,
                                            lagged_features,
                                            lags)

        train_result = train_model(train, n_splits, label_encoding, params)

        train_result.model.save_model(file_path)

# this function could be broken up so that evaluation is done separately
@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def train_model(df: pd.DataFrame,
                n_splits: int,
                label_encoding: dict[str, int],
                params: dict[str, object] | None = None) -> TrainingResult:
    """
    Train a lightgbm model using time-series cross-validation. Fold-level and out-of-fold metrics are outputted.
    :param df: pd.DataFrame
    :param n_splits: number of folds
    :param label_encoding: mapping of c.CLASSES to int labels
    :param params: parameters to pass to lightgbm.train
    :return: TrainingResult; the model along with the macro and transition state f1 scores at the fold-level
    and out-of-fold.
    """
    if n_splits <= 0:
        raise ValueError("n_splits must be positive")

    tscv = TimeSeriesSplit(n_splits=n_splits)

    if params is None:
        params = {
            'objective': 'multiclass',
            'num_class': len(label_encoding),
            'metric': 'multi_logloss',
            'seed': 42,
            'verbose': -1
        }

    macro_f1_scores = []
    transition_f1_scores = []
    model = None

    x = df.drop(columns="class")
    y = df["class"]

    n_samples = len(df)
    n_classes = len(label_encoding)

    # store predictions in their original row positions.
    oof_predictions = np.full(n_samples, -1, dtype=int)
    oof_probabilities = np.full(
        (n_samples, n_classes),
        np.nan,
        dtype=float
    )

    for train_index, test_index in tscv.split(df):
        x_train = x.iloc[train_index]
        x_test = x.iloc[test_index]
        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index]

        train_data = lgb.Dataset(x_train, label=y_train)
        test_data = lgb.Dataset(x_test, label=y_test)

        model = lgb.train(params, train_data, valid_sets=[test_data], num_boost_round=100)

        predictions = model.predict(x_test, num_iteration=model.best_iteration)
        y_pred = np.argmax(predictions, axis=1)

        # store OOF predictions in their original positions
        oof_predictions[test_index] = y_pred
        oof_probabilities[test_index] = predictions

        # fold-level metrics
        macro_f1, transition_f1 = calculate_f1_scores(y_test, y_pred, list(label_encoding.values()))

        macro_f1_scores.append(macro_f1)
        transition_f1_scores.append(transition_f1)

    # only observations that were actually part of a validation fold
    oof_mask = oof_predictions != -1

    y_oof = y.iloc[oof_mask]
    y_pred_oof = oof_predictions[oof_mask]

    # calculate metrics across all OOF predictions
    oof_macro_f1, oof_transition_f1 = calculate_f1_scores(
        y_oof,
        y_pred_oof,
        list(label_encoding.values())
    )

    return TrainingResult(
        model=model,
        fold_macro_f1_scores=macro_f1_scores,
        fold_transition_f1_scores=transition_f1_scores,
        oof_macro_f1=oof_macro_f1,
        oof_transition_f1=oof_transition_f1,
        oof_predictions=oof_predictions,
        oof_probabilities=oof_probabilities,
    )

# run full_train_pipeline with no lagged features
if __name__ == '__main__':
    label_map = {
        c.CLASSES[0]: 0,
        c.CLASSES[1]: 1,
        c.CLASSES[2]: 2,
    }

    train_and_save_station_models(train_size=0.8,
                                  n_splits=5,
                                  label_encoding=label_map)
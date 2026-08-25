import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from dataclasses import dataclass
from statistics import mean
from pydantic import validate_call, ConfigDict

from freeze_thaw.evaluation.metrics import calculate_f1_scores


@dataclass(frozen=True)
class TrainingResult:
    model: lgb.Booster
    macro_f1_scores: list[float]
    transition_f1_scores: list[float]
    average_macro_f1: float
    average_transition_f1: float


def full_train_pipeline(stations: list[str],
                        n_splits: int,
                        label_encoding: dict[str, int],
                        params: dict[str, str] | None = None) -> list[TrainingResult]:
    """
    Orchestrator to train and save lightgbm models for all stations with the cleaned, labelled ASCAT data being the input.
    :param stations: list of ISMN stations
    :param n_splits:
    :param label_encoding:
    :param params:
    :return:
    """

    # loops over prepare_df and train_model for all stations


# this function could be broken up so that evaluation is done separately
@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def train_model(df: pd.DataFrame,
                n_splits: int,
                label_encoding: dict[str, int],
                params: dict[str, object] | None = None) -> TrainingResult:
    """
    Train a lightgbm model using time-series cross-validation.
    :param df: pd.DataFrame
    :param n_splits: number of folds
    :param label_encoding: mapping of c.CLASSES to int labels
    :param params: parameters to pass to lightgbm.train
    :return: the model along with the macro and transition state f1 scores for each fold
    """
    if n_splits <= 0:
        raise ValueError("n_splits must be positive")

    tscv = TimeSeriesSplit(n_splits=n_splits)

    # Define LightGBM parameters
    if params is None:
        params = {
            'objective': 'multiclass',
            'num_class': len(label_encoding),
            'metric': 'multi_logloss',
        }

    macro_f1_scores = []
    transition_f1_scores = []
   # classification_reports = []
    model = None

    x = df.drop(columns="class")
    y = df["class"]
    for fold, (train_index, test_index) in enumerate(tscv.split(df), 1):
   # for train_index, test_index in tscv.split(df):
        x_train = x.iloc[train_index]
        x_test = x.iloc[test_index]
        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index]

        train_data = lgb.Dataset(x_train, label=y_train)
        test_data = lgb.Dataset(x_test, label=y_test)

        model = lgb.train(params, train_data, valid_sets=[test_data], num_boost_round=100)

      #  model.save_model(c.MODEL_PATH / f"{c.MODEL_NAME}.model")

        predictions = model.predict(x_test, num_iteration=model.best_iteration)
        y_pred = np.argmax(predictions, axis=1)

     #   report = classification_report(y_test, y_pred)
     #   classification_reports.append(report)

        macro_f1, transition_f1 = calculate_f1_scores(y_test, y_pred, list(label_encoding.values()))

        macro_f1_scores.append(macro_f1)
        transition_f1_scores.append(transition_f1)

    return TrainingResult(
        model=model,
        macro_f1_scores=macro_f1_scores,
        transition_f1_scores=transition_f1_scores,
        average_macro_f1=mean(macro_f1_scores),
        average_transition_f1=mean(transition_f1_scores)
    )
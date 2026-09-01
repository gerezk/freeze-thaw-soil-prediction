import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from dataclasses import dataclass
from pydantic import validate_call, ConfigDict

from freeze_thaw.evaluation.metrics import calculate_f1_scores


@dataclass(frozen=True)
class TrainingResult:
    model: lgb.Booster
    fold_macro_f1_scores: list[float]
    fold_transition_f1_scores: list[float]
    oof_macro_f1: float
    oof_transition_f1: float
    oof_predictions: np.ndarray
    oof_probabilities: np.ndarray


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
            'verbose': 0
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
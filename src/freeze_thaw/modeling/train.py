import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from dataclasses import dataclass
from statistics import mean

from freeze_thaw.evaluation.metrics import calculate_f1_scores


@dataclass(frozen=True)
class TrainingResult:
    model: lgb.Booster
    macro_f1_scores: list[float]
    transition_f1_scores: list[float]
    average_macro_f1: float
    average_transition_f1: float

# this function could be broken up
def train_model(df: pd.DataFrame, n_splits: int, label_encoding: dict[str, int]) -> TrainingResult:
    """

    :param df: pd.DataFrame
    :param n_splits: number of folds
    :param label_encoding: mapping of c.CLASSES to int labels
    :return: the model along with the macro and transition state f1 scores for each fold
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)

    # Define LightGBM parameters
    params = {
        'objective': 'multiclass',
        'num_class': 3,
        'metric': 'multi_logloss',
    }

    macro_f1_scores = []
    transition_f1_scores = []
   # classification_reports = []
    model = None

    x = df.drop(columns="class")
    y = df["class"]
    for train_index, test_index in tscv.split(df):
        x_train = x.iloc[train_index]
        x_test = x.iloc[test_index]
        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index]

        train_data = lgb.Dataset(x_train, label=y_train)
        test_data = lgb.Dataset(x_test, label=y_test)

        model = lgb.train(params, train_data, valid_sets=[test_data], num_boost_round=100)

        predictions = model.predict(x_test, num_iteration=model.best_iteration)
        predicted_classes = np.argmax(predictions, axis=1)

     #   report = classification_report(y_test, predicted_classes)
     #   classification_reports.append(report)

        macro_f1, transition_f1 = calculate_f1_scores(y_test, predicted_classes, list(label_encoding.values()))

        macro_f1_scores.append(macro_f1)
        transition_f1_scores.append(transition_f1)

    return TrainingResult(
        model=model,
        macro_f1_scores=macro_f1_scores,
        transition_f1_scores=transition_f1_scores,
        average_macro_f1=mean(macro_f1_scores),
        average_transition_f1=mean(transition_f1_scores)
    )
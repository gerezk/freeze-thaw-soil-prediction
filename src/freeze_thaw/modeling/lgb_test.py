from dataclasses import dataclass
import pandas as pd
import numpy as np
import lightgbm as lgb
from pydantic import validate_call, ConfigDict

from freeze_thaw.evaluation.metrics import calculate_f1_scores


@dataclass(frozen=True)
class TestResult:
    macro_f1: float
    transition_f1: float
    y_pred: np.ndarray


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def lgb_pred(df: pd.DataFrame,
             model: lgb.Booster,
             label_encoding: dict[str, int],) -> TestResult:
    """
    Predict soil state on df given an lgbm model and output macro and transition state f1 scores.
    :param df: pd.DataFrame from prepare_df()
    :param model: lightgbm.Booster
    :param label_encoding: mapping of c.CLASSES to int labels
    :return: the macro and transition state f1 scores
    """
    if "class" not in df.columns:
        raise ValueError("'df' must have 'class' column")

    df_copy = df.copy()

    x = df_copy.drop(columns="class")
    y = df_copy["class"]

    predictions = model.predict(x)
    y_pred = np.argmax(predictions, axis=1)

    macro_f1, transition_f1 = calculate_f1_scores(y, y_pred, list(label_encoding.values()))

    return TestResult(macro_f1=macro_f1,
                      transition_f1=transition_f1,
                      y_pred=y_pred)
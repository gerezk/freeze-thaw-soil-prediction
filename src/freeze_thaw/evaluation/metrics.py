import pandas as pd
import numpy as np
from typing import Union
from sklearn.metrics import confusion_matrix, f1_score
from pydantic import validate_call, ConfigDict


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def formatted_cm(y_true: Union[pd.Series, np.ndarray], y_pred: Union[pd.Series, np.ndarray], classes: list[object]) -> pd.DataFrame:
    """
    Output formatted sklearn confusion matrix. To clarify on mapping, class 0 will map to classes[0].
    :param y_true: Ground truth (correct) target values
    :param y_pred: Estimated targets as returned by a classifier
    :param classes: List of classes
    :return: sklearn confusion matrix in the form of a df
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")

    cm = confusion_matrix(y_true, y_pred, labels=classes)
    cm = pd.DataFrame(cm, index=classes, columns=classes)

    return cm

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def calculate_f1_scores(y_true: Union[pd.Series, np.ndarray], y_pred: Union[pd.Series, np.ndarray],
                        classes: list[object]) -> tuple[float, float]:
    """
    Calculate macro and transition state f1 scores.
    All classes must be present in y_true or y_pred.
    :param y_true: Ground truth (correct) target values
    :param y_pred: Estimated targets as returned by a classifier
    :param classes: List of classes
    :return: macro_f1_score, transition_f1_score
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")

    if not all(cls in set(y_true).union(set(y_pred)) for cls in classes):
        raise ValueError("All classes must be present in either y_true or y_pred.")

    f1_per_class = f1_score(
        y_true,
        y_pred,
        average=None,
        labels=classes,
    )

    macro_f1_score = f1_score(y_true, y_pred, average="macro")
    transition_f1_score = f1_per_class[1]

    return macro_f1_score, transition_f1_score
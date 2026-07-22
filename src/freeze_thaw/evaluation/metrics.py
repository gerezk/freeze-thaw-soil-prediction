import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score


def formatted_cm(y_true: pd.Series, y_pred: pd.Series, classes: list[object]) -> pd.DataFrame:
    """
    Output formatted sklearn confusion matrix. To clarify on mapping, class 0 will map to classes[0].
    :param y_true: Ground truth (correct) target values
    :param y_pred: Estimated targets as returned by a classifier
    :param classes: List of classes
    :return: sklearn confusion matrix in the form of a df
    """
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    cm = pd.DataFrame(cm, index=classes, columns=classes)

    return cm


def calculate_f1_scores(y_true: pd.Series, y_pred: pd.Series, classes: list[object]) -> tuple[float, float]:
    """
    Calculate macro and transition state f1 scores.
    :param y_true: Ground truth (correct) target values
    :param y_pred: Estimated targets as returned by a classifier
    :param classes: List of classes
    :return: macro_f1_score, transition_f1_score
    """
    f1_per_class = f1_score(
        y_true,
        y_pred,
        average=None,
        labels=classes,
    )

    macro_f1_score = f1_score(y_true, y_pred, average="macro")
    transition_f1_score = f1_per_class[1]

    return macro_f1_score, transition_f1_score
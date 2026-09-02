from pydantic import validate_call, ConfigDict
import numpy as np
from itertools import product


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def paired_permutation_test(x: np.typing.ArrayLike, y: np.typing.ArrayLike, alternative: str = "greater") -> (float, float):
    """
    Exact paired sign-flipping permutation test.
    :param x: scores from model 1
    :param y: scores from model 2
    :param alternative: less, greater, or two-sided
    :return: observed mean difference (y - x) and exact permutation p-value
    """
    if alternative not in ("less", "greater", "two-sided"):
        raise ValueError("alternative must be 'less', 'greater' or 'two-sided'")

    differences = y - x
    observed = np.mean(differences)

    permuted = []

    for signs in product([-1, 1], repeat=len(differences)):
        signs = np.asarray(signs)
        permuted.append(np.mean(differences * signs))

    permuted = np.asarray(permuted)

    p_value = np.inf
    if alternative == "greater":
        p_value = np.mean(permuted >= observed)
    elif alternative == "less":
        p_value = np.mean(permuted <= observed)
    elif alternative == "two-sided":
        p_value = np.mean(np.abs(permuted) >= abs(observed))

    return observed, p_value
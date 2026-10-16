"""Expected calibration error with equal-width bins on the predicted default probability (y = 1 default)."""

from __future__ import annotations

import numpy as np


def ece(y_true, prob_default, n_bins: int) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(prob_default, dtype=float)
    if len(y) == 0:
        return float("nan")
    # bins [0,1/b), ..., [(b-1)/b, 1]; the last bin includes 1.0
    idx = np.minimum((p * n_bins).astype(int), n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            total += m.sum() / len(y) * abs(y[m].mean() - p[m].mean())
    return float(total)

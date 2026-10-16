"""Decision threshold (N-07): maximise default-class F1 on out-of-fold TRAIN predictions only."""

from __future__ import annotations

import numpy as np

from fairxai.config import Grid


def f1_at(y, prob, t: float) -> float:
    y = np.asarray(y)
    pred = np.asarray(prob) >= t
    tp = float(np.sum(pred & (y == 1)))
    denom = 2 * tp + float(np.sum(pred & (y == 0))) + float(np.sum(~pred & (y == 1)))
    return 2 * tp / denom if denom else 0.0


def choose_threshold(y_oof, prob_oof, grid: Grid) -> float:
    """Grid start..stop step; ties -> closest to 0.5 (then smaller)."""
    n = round((grid.stop - grid.start) / grid.step) + 1
    ts = [round(grid.start + i * grid.step, 6) for i in range(n)]
    scored = [(round(f1_at(y_oof, prob_oof, t), 12), t) for t in ts]
    best = max(s for s, _ in scored)
    return min((t for s, t in scored if s == best), key=lambda t: (abs(t - 0.5), t))

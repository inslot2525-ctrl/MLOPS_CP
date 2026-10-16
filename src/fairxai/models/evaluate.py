"""CV out-of-fold predictions and classification metrics (default class = positive)."""

from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold


def _fit_fold(model, X, y, tr, va):
    m = clone(model)
    m.fit(X.iloc[tr], y.iloc[tr])
    return va, m.predict_proba(X.iloc[va])[:, 1]


def cv_oof(model, X, y, folds: int, seed: int, n_jobs: int):
    """Stratified shuffled K-fold. Returns (mean per-fold PR-AUC, out-of-fold probabilities)."""
    skf = StratifiedKFold(folds, shuffle=True, random_state=seed)
    results = Parallel(n_jobs=n_jobs)(delayed(_fit_fold)(model, X, y, tr, va) for tr, va in skf.split(X, y))
    oof = np.zeros(len(y))
    aps = []
    for va, p in results:
        oof[va] = p
        aps.append(average_precision_score(y.iloc[va], p))
    return float(np.mean(aps)), oof


def metrics_at(y, prob, threshold: float) -> dict:
    y = np.asarray(y)
    pred = (np.asarray(prob) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y, prob)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }

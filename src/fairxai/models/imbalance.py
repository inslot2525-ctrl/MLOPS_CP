"""Class-imbalance handling (N-21): class_weight / scale_pos_weight; KNN and Naive Bayes run unweighted."""

from __future__ import annotations

import numpy as np

CLASS_WEIGHT_MODELS = {"logreg", "decision_tree", "random_forest", "svm"}


def imbalance_kwargs(model_name: str, y, strategy: str = "class_weight") -> dict:
    if strategy != "class_weight":
        raise ValueError(f"unsupported imbalance_strategy {strategy!r} (SMOTE is experiment-only)")
    if model_name in CLASS_WEIGHT_MODELS:
        return {"class_weight": "balanced"}
    if model_name == "xgboost":
        y = np.asarray(y)
        n_pos = int((y == 1).sum())
        return {"scale_pos_weight": float((y == 0).sum()) / n_pos if n_pos else 1.0}
    return {}

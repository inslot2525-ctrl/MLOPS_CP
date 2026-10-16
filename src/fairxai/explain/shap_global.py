"""Explainer selection (N-23) and the global SHAP summary logged per version (SPEC D.3 step 7)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

TREE = (RandomForestClassifier, DecisionTreeClassifier, XGBClassifier)
KERNEL_BACKGROUND = 100


def split_pipeline(pipe: Pipeline):
    """(preprocessing pipeline, final estimator, transformed feature names)."""
    pre = Pipeline(pipe.steps[:-1])
    est = pipe.steps[-1][1]
    names = list(pipe.named_steps["encode"].get_feature_names_out())
    return pre, est, names


def explainer_method(est) -> tuple[str, str]:
    """(method, output_space) truthfully stated for the model family."""
    if isinstance(est, TREE):
        # TreeExplainer: log-odds for XGBoost's raw margin; probability for sklearn trees
        return "tree_shap", "log_odds" if isinstance(est, XGBClassifier) else "probability"
    if isinstance(est, LogisticRegression):
        return "linear_shap", "log_odds"
    return "kernel_shap", "probability"


def make_explainer(pipe: Pipeline, background: pd.DataFrame | None, seed: int):
    """XGBoost needs no background (path-dependent TreeSHAP, log-odds); every other family does."""
    import shap

    pre, est, names = split_pipeline(pipe)
    method, _ = explainer_method(est)
    if method == "tree_shap" and isinstance(est, XGBClassifier):
        return shap.TreeExplainer(est), names
    if background is None:
        raise ValueError(f"{method} for {type(est).__name__} needs a background sample")
    bg = pre.transform(background)
    if method == "tree_shap":
        return shap.TreeExplainer(est, model_output="probability", data=bg[:KERNEL_BACKGROUND]), names
    if method == "linear_shap":
        return shap.LinearExplainer(est, bg), names
    k = min(KERNEL_BACKGROUND, len(bg))
    centers = KMeans(n_clusters=k, random_state=seed, n_init=1).fit(bg).cluster_centers_
    return shap.KernelExplainer(lambda z: est.predict_proba(z)[:, 1], centers), names


def default_class_values(values) -> np.ndarray:
    """Normalise SHAP output to an (n_rows, n_features) array for the default class."""
    v = values.values if hasattr(values, "values") else values
    v = np.asarray(v)
    if v.ndim == 3:
        v = v[:, :, 1]
    return v


def global_importance(pipe: Pipeline, train_X: pd.DataFrame, n_rows: int, seed: int) -> tuple[dict, str]:
    """Mean |SHAP| per transformed feature on a seeded train sample. Kernel SHAP uses 100 rows (cost)."""
    pre, est, names = split_pipeline(pipe)
    method, _ = explainer_method(est)
    sample = train_X.sample(
        n=min(n_rows if method != "kernel_shap" else KERNEL_BACKGROUND, len(train_X)), random_state=seed
    )
    explainer, _ = make_explainer(
        pipe, train_X.sample(n=min(1000, len(train_X)), random_state=seed + 1), seed
    )
    xs = pre.transform(sample)
    if method == "kernel_shap":
        vals = explainer.shap_values(xs, nsamples=200, silent=True)
    else:
        vals = explainer.shap_values(xs)
    v = default_class_values(vals)
    imp = dict(zip(names, np.abs(v).mean(axis=0).round(8).tolist(), strict=True))
    return dict(sorted(imp.items(), key=lambda kv: -kv[1])), method


def summary_png(importance: dict, path: str, top: int = 20) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    items = list(importance.items())[:top][::-1]
    fig, ax = plt.subplots(figsize=(7, 0.3 * len(items) + 1))
    ax.barh([k for k, _ in items], [v for _, v in items], color="#4C72B0")
    ax.set_xlabel("mean |SHAP value|")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)

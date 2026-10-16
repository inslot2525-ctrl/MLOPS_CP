"""Signal evaluators. Each returns a SignalResult with OK / WARN / ALERT / NA (SPEC D.7). No literals."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from fairxai.config import Config
from fairxai.data.load import FEATURES
from fairxai.data.validate import validate
from fairxai.fairness.report import build_report
from fairxai.models.evaluate import metrics_at
from fairxai.monitoring.drift import DriftResult, prediction_drift

OK, WARN, ALERT, NA = "OK", "WARN", "ALERT", "NA"


@dataclass
class SignalResult:
    kind: str
    status: str
    n_rows: int
    n_labelled: int
    value: float | None = None
    details: dict = field(default_factory=dict)


def na(kind: str, n_rows: int, n_labelled: int, reason: str) -> SignalResult:
    return SignalResult(kind, NA, n_rows, n_labelled, None, {"reason": reason})


def data_quality(window: pd.DataFrame, n_labelled: int, cfg: Config, reference_stats: dict) -> SignalResult:
    n = len(window)
    if n < cfg.base.min_window_rows:
        return na("data_quality", n, n_labelled, f"{n} rows < {cfg.base.min_window_rows}")
    res = validate(window[FEATURES], "batch", cfg, labelled=False, reference_stats=reference_stats)
    status = ALERT if not res.passed else WARN if res.warnings else OK
    return SignalResult(
        "data_quality", status, n, n_labelled, res.stats.get("outlier_fraction"), res.to_dict()
    )


def feature_drift(result: DriftResult | None, n: int, n_labelled: int, cfg: Config) -> SignalResult:
    if result is None or n < cfg.base.min_window_rows:
        return na("feature_drift", n, n_labelled, f"{n} rows < {cfg.base.min_window_rows}")
    share = result.share_drifted
    status = ALERT if share >= cfg.monitoring.dataset_drift_share else WARN if result.drifted_features else OK
    return SignalResult("feature_drift", status, n, n_labelled, share, result.to_dict())


def pred_drift(window: pd.DataFrame, ref_prob: np.ndarray, threshold: float, n_labelled: int, cfg: Config):
    n = len(window)
    if n < cfg.base.min_window_rows:
        return na("prediction_drift", n, n_labelled, f"{n} rows < {cfg.base.min_window_rows}")
    cur = window["probability"].astype(float).to_numpy()
    d = prediction_drift(
        ref_prob, cur, float(np.mean(ref_prob >= threshold)), float(np.mean(cur >= threshold))
    )
    m = cfg.monitoring
    status = WARN if d["ks_p_value"] < m.p_value or d["rate_change"] >= m.prediction_rate_change else OK
    return SignalResult("prediction_drift", status, n, n_labelled, d["rate_change"], d)


def performance(labelled: pd.DataFrame, n_rows: int, threshold: float, reference_pr_auc: float, cfg: Config):
    n_lab = len(labelled)
    if n_lab < cfg.monitoring.min_labelled_rows:
        return na("performance", n_rows, n_lab, f"{n_lab} labelled rows < {cfg.monitoring.min_labelled_rows}")
    y = labelled["label"].astype(int)
    if y.nunique() < 2:
        return na("performance", n_rows, n_lab, "labelled window has one class")
    met = metrics_at(y, labelled["probability"].astype(float), threshold)
    limit = round(reference_pr_auc - cfg.monitoring.perf_drop, cfg.fairness.decimals)
    status = ALERT if round(met["pr_auc"], cfg.fairness.decimals) <= limit else OK
    return SignalResult(
        "performance",
        status,
        n_rows,
        n_lab,
        met["pr_auc"],
        {**met, "reference_pr_auc": reference_pr_auc, "alert_at_or_below": limit},
    )


def live_fairness(labelled: pd.DataFrame, n_rows: int, threshold: float, cfg: Config) -> SignalResult:
    n_lab = len(labelled)
    if n_lab < cfg.monitoring.min_labelled_rows:
        return na("fairness", n_rows, n_lab, f"{n_lab} labelled rows < {cfg.monitoring.min_labelled_rows}")
    rep = build_report(
        labelled, labelled["label"].astype(int), labelled["probability"].astype(float), threshold, cfg
    )
    statuses = [g["status"] for g in rep["gates"].values()]
    if "FAIL" in statuses:
        status = ALERT
    elif "INDETERMINATE" in statuses or rep["warnings"]:
        status = WARN
    else:
        status = OK
    return SignalResult(
        "fairness", status, n_rows, n_lab, None, {"gates": rep["gates"], "warnings": rep["warnings"]}
    )

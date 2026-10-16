"""Per-group rates via Fairlearn MetricFrame and the EO / DI / DP / CAL summaries (SPEC D.4)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from fairlearn.metrics import MetricFrame

from fairxai.config import Fairness
from fairxai.fairness.calibration import ece

PASS, FAIL, INDETERMINATE = "PASS", "FAIL", "INDETERMINATE"


@dataclass
class Summary:
    value: float | None
    status_reason: str | None  # None when computable; otherwise why it is INDETERMINATE
    eligible: list[str]
    excluded: list[str]


def group_table(groups: pd.Series, y_true, y_pred, y_prob, fc: Fairness) -> pd.DataFrame:
    """Rows: group label. Columns: n, n_denominator (rows with Y = f), favorable_rate, tpr, ece, eligibility."""
    f = fc.favorable_label
    y_true = pd.Series(np.asarray(y_true), index=groups.index)
    y_pred = pd.Series(np.asarray(y_pred), index=groups.index)
    y_prob = pd.Series(np.asarray(y_prob, dtype=float), index=groups.index)

    def favorable_rate(yt, yp):
        return float(np.mean(np.asarray(yp) == f)) if len(yp) else float("nan")

    def tpr_f(yt, yp):
        yt, yp = np.asarray(yt), np.asarray(yp)
        m = yt == f
        return float(np.mean(yp[m] == f)) if m.any() else float("nan")

    def n_den(yt, yp):
        return int(np.sum(np.asarray(yt) == f))

    def n_rows(yt, yp):
        return len(yt)

    mf = MetricFrame(
        metrics={"n": n_rows, "n_denominator": n_den, "favorable_rate": favorable_rate, "tpr": tpr_f},
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=groups.rename("group"),
    )
    table = mf.by_group.copy()
    table["ece"] = [ece(y_true[groups == g], y_prob[groups == g], fc.calibration_bins) for g in table.index]
    table["n"] = table["n"].astype(int)
    table["n_denominator"] = table["n_denominator"].astype(int)
    table["eligible_rate"] = (table["n"] >= fc.min_group_rows) & (table["n"] >= fc.min_denominator_rows)
    table["eligible_tpr"] = (table["n"] >= fc.min_group_rows) & (
        table["n_denominator"] >= fc.min_denominator_rows
    )
    table.index = table.index.astype(str)
    return table


def _round(x: float, fc: Fairness) -> float:
    return round(float(x), fc.decimals)


def _spread(table: pd.DataFrame, col: str, elig: str, fc: Fairness) -> Summary:
    ok = table[table[elig]]
    vals = ok[col].dropna()
    excluded = sorted(set(table.index) - set(vals.index))
    if len(vals) < 2:
        return Summary(None, f"fewer than 2 eligible groups for {col}", list(vals.index), excluded)
    return Summary(_round(vals.max() - vals.min(), fc), None, list(vals.index), excluded)


def eo(table: pd.DataFrame, fc: Fairness) -> Summary:
    return _spread(table, "tpr", "eligible_tpr", fc)


def dp(table: pd.DataFrame, fc: Fairness) -> Summary:
    return _spread(table, "favorable_rate", "eligible_rate", fc)


def cal(table: pd.DataFrame, fc: Fairness) -> Summary:
    return _spread(table, "ece", "eligible_rate", fc)


def di(table: pd.DataFrame, fc: Fairness) -> Summary:
    ok = table[table["eligible_rate"]]
    vals = ok["favorable_rate"].dropna()
    excluded = sorted(set(table.index) - set(vals.index))
    if len(vals) < 2:
        return Summary(None, "fewer than 2 eligible groups for favorable_rate", list(vals.index), excluded)
    if vals.max() == 0:
        return Summary(None, "max favorable rate is 0", list(vals.index), excluded)
    return Summary(_round(vals.min() / vals.max(), fc), None, list(vals.index), excluded)


def check_max(s: Summary, limit: float, fc: Fairness) -> str:
    if s.value is None or math.isnan(s.value):
        return INDETERMINATE
    return PASS if s.value <= _round(limit, fc) else FAIL


def check_min(s: Summary, limit: float, fc: Fairness) -> str:
    if s.value is None or math.isnan(s.value):
        return INDETERMINATE
    return PASS if s.value >= _round(limit, fc) else FAIL

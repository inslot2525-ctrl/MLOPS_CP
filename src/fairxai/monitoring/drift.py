"""Feature drift behind one adapter (SPEC D7). Only this module imports Evidently.

`compute_drift()` returns our own dataclass whichever engine ran: Evidently 0.7 (default) or the scipy
contingency (KS for numeric, chi-square for categorical), which gives the same tests by construction.
Both sides get the D.2 EDUCATION/MARRIAGE remap before comparison.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from fairxai.config import Config
from fairxai.data.load import FEATURES
from fairxai.features.pipeline import remap_categories

CATEGORICAL = ["SEX", "EDUCATION", "MARRIAGE"]
NUMERICAL = [c for c in FEATURES if c not in CATEGORICAL]
_VALUE_DRIFT = re.compile(r"ValueDrift\(column=([^,]+),method=(\w+)")


@dataclass
class FeatureDrift:
    test: str
    p_value: float
    drifted: bool


@dataclass
class DriftResult:
    engine: str
    n_reference: int
    n_current: int
    features: dict[str, FeatureDrift] = field(default_factory=dict)

    @property
    def share_drifted(self) -> float:
        return sum(f.drifted for f in self.features.values()) / max(len(self.features), 1)

    @property
    def drifted_features(self) -> list[str]:
        return sorted(k for k, f in self.features.items() if f.drifted)

    def to_dict(self) -> dict:
        return {
            "engine": self.engine,
            "n_reference": self.n_reference,
            "n_current": self.n_current,
            "share_drifted": self.share_drifted,
            "drifted_features": self.drifted_features,
            "features": {k: vars(v) for k, v in self.features.items()},
        }


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    return remap_categories(df[FEATURES].astype(float))


def _scipy(ref: pd.DataFrame, cur: pd.DataFrame, p: float) -> dict[str, FeatureDrift]:
    out: dict[str, FeatureDrift] = {}
    for c in NUMERICAL:
        pv = float(stats.ks_2samp(ref[c], cur[c]).pvalue)
        out[c] = FeatureDrift("ks", pv, pv < p)
    for c in CATEGORICAL:
        # goodness-of-fit of current counts against reference proportions (the same test Evidently runs)
        cats = sorted(set(ref[c].dropna()) | set(cur[c].dropna()))
        k_norm = len(cur) / len(ref)
        f_exp = np.array([(ref[c] == k).sum() * k_norm for k in cats], dtype=float)
        f_obs = np.array([(cur[c] == k).sum() for k in cats], dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            pv = 1.0 if len(cats) < 2 else float(stats.chisquare(f_obs, f_exp)[1])
        pv = 0.0 if np.isnan(pv) else pv  # a category absent from the reference is drift
        out[c] = FeatureDrift("chisquare", pv, pv < p)
    return out


def parse_evidently(snapshot_dict: dict, p: float) -> dict[str, FeatureDrift]:
    out: dict[str, FeatureDrift] = {}
    for m in snapshot_dict.get("metrics", []):
        match = _VALUE_DRIFT.match(str(m.get("metric_name", "")))
        if match:
            col, method = match.groups()
            pv = float(m["value"])
            out[col] = FeatureDrift(method, pv, pv < p)
    return out


def _evidently(ref: pd.DataFrame, cur: pd.DataFrame, p: float) -> dict[str, FeatureDrift]:
    from evidently import DataDefinition, Dataset, Report
    from evidently.presets import DataDriftPreset

    dd = DataDefinition(numerical_columns=NUMERICAL, categorical_columns=CATEGORICAL)
    report = Report([DataDriftPreset(num_method="ks", cat_method="chisquare", threshold=p)])
    snap = report.run(
        Dataset.from_pandas(cur, data_definition=dd), Dataset.from_pandas(ref, data_definition=dd)
    )
    return parse_evidently(snap.dict(), p)


def compute_drift(
    reference: pd.DataFrame, current: pd.DataFrame, cfg: Config, engine: str = "evidently"
) -> DriftResult:
    p = cfg.monitoring.p_value
    ref, cur = _prep(reference), _prep(current)
    feats = _evidently(ref, cur, p) if engine == "evidently" else _scipy(ref, cur, p)
    missing = set(FEATURES) - set(feats)
    if missing:
        raise RuntimeError(f"drift engine {engine} returned no result for {sorted(missing)}")
    return DriftResult(engine, len(ref), len(cur), {k: feats[k] for k in FEATURES})


def prediction_drift(ref_prob, cur_prob, ref_rate: float, cur_rate: float) -> dict:
    """KS on probability_default plus the absolute change in predicted-default rate."""
    pv = float(stats.ks_2samp(np.asarray(ref_prob), np.asarray(cur_prob)).pvalue)
    return {
        "ks_p_value": pv,
        "ref_rate": ref_rate,
        "cur_rate": cur_rate,
        "rate_change": abs(cur_rate - ref_rate),
    }

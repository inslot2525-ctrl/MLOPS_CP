"""fairness_report.json builder: hard checks, advisory warnings, per-group tables (SPEC D.4)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fairxai.config import Config
from fairxai.fairness import metrics as m
from fairxai.fairness.groups import INTERSECTIONAL, group_frame

SCHEMA_VERSION = 1
ATTRS = ("sex", "age_band", "marriage", "education")
HARD_ATTR = {"sex": "sex", "age": "age_band", "inter": INTERSECTIONAL}


def _clean(v):
    if v is None:
        return None
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    v = float(v)
    return None if np.isnan(v) else v


def _attribute(groups: pd.Series, y, y_pred, prob, cfg: Config) -> dict:
    fc = cfg.fairness
    valid = groups.notna() & ~np.isnan(y) & ~np.isnan(prob)
    table = m.group_table(groups[valid].astype(str), y[valid].astype(int), y_pred[valid], prob[valid], fc)
    summaries = {"eo": m.eo(table, fc), "di": m.di(table, fc), "dp": m.dp(table, fc), "cal": m.cal(table, fc)}
    return {
        "n_dropped": int((~valid).sum()),
        "groups": {
            g: {
                "n": int(r.n),
                "n_denominator": int(r.n_denominator),
                "favorable_rate": _clean(r.favorable_rate),
                "tpr": _clean(r.tpr),
                "ece": _clean(r.ece),
                "eligible": {"rate": bool(r.eligible_rate), "tpr": bool(r.eligible_tpr)},
            }
            for g, r in table.iterrows()
        },
        "metrics": {
            k: {"value": s.value, "reason": s.status_reason, "eligible": s.eligible, "excluded": s.excluded}
            for k, s in summaries.items()
        },
        "_summaries": summaries,
    }


def build_report(
    features: pd.DataFrame, y_true, y_prob, threshold: float, cfg: Config, champion_report: dict | None = None
) -> dict:
    fc = cfg.fairness
    y = np.asarray(pd.to_numeric(pd.Series(y_true), errors="coerce"), dtype=float)
    prob = np.asarray(y_prob, dtype=float)
    y_pred = np.where(np.isnan(prob), -1, (prob >= threshold).astype(int))
    gf = group_frame(features.reset_index(drop=True), cfg)
    attrs = {a: _attribute(gf[a], y, y_pred, prob, cfg) for a in (*ATTRS, INTERSECTIONAL) if a in gf}

    gates: dict[str, dict] = {}
    for short, attr in HARD_ATTR.items():
        s = attrs[attr]["_summaries"]
        eo_lim = fc.eo_inter_max if short == "inter" else fc.eo_max
        di_lim = fc.di_inter_min if short == "inter" else fc.di_min
        gates[f"eo_{short}"] = {
            "status": m.check_max(s["eo"], eo_lim, fc),
            "value": s["eo"].value,
            "threshold": eo_lim,
            "reason": s["eo"].status_reason,
        }
        gates[f"di_{short}"] = {
            "status": m.check_min(s["di"], di_lim, fc),
            "value": s["di"].value,
            "threshold": di_lim,
            "reason": s["di"].status_reason,
        }

    warnings: list[dict] = []
    for short, attr in (("sex", "sex"), ("age", "age_band")):
        for metric, limit in (("dp", fc.dp_warn), ("cal", fc.cal_warn)):
            val = attrs[attr]["_summaries"][metric].value
            if val is None or val <= round(limit, fc.decimals):
                continue
            champ = None
            if champion_report:
                champ = champion_report["attributes"][attr]["metrics"][metric]["value"]
            if champ is None or round(val - champ, fc.decimals) > round(fc.warn_worse_margin, fc.decimals):
                warnings.append(
                    {"check": f"{metric}_{short}", "value": val, "threshold": limit, "champion": champ}
                )

    n_dropped = int(
        (np.isnan(y) | np.isnan(prob) | gf["sex"].isna().to_numpy() | gf["age_band"].isna().to_numpy()).sum()
    )
    for a in attrs.values():
        a.pop("_summaries")
    return {
        "schema_version": SCHEMA_VERSION,
        "favorable_label": fc.favorable_label,
        "threshold": threshold,
        "n_rows": len(y),
        "n_dropped": n_dropped,
        "attributes": {a: attrs[a] for a in ATTRS if a in attrs},
        "intersectional": {INTERSECTIONAL: attrs[INTERSECTIONAL]},
        "gates": gates,
        "warnings": warnings,
    }


def metric_dict(report: dict, prefix: str = "fair_") -> dict[str, float]:
    """MLflow metrics fair_eo_sex, fair_di_sex, ... (computable values only)."""
    return {f"{prefix}{k}": v["value"] for k, v in report["gates"].items() if v["value"] is not None}


def groups_csv(report: dict) -> str:
    rows = []
    for attr, block in {**report["attributes"], **report["intersectional"]}.items():
        for g, r in block["groups"].items():
            rows.append(
                {
                    "attribute": attr,
                    "group": g,
                    "n": r["n"],
                    "n_denominator": r["n_denominator"],
                    "favorable_rate": r["favorable_rate"],
                    "tpr": r["tpr"],
                    "ece": r["ece"],
                    "eligible_rate": r["eligible"]["rate"],
                    "eligible_tpr": r["eligible"]["tpr"],
                }
            )
    return pd.DataFrame(rows).to_csv(index=False)

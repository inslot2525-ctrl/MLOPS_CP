"""Monitoring job for one window (SPEC D.7): quality -> drift -> prediction drift -> performance -> fairness
-> write monitoring_results -> evaluate triggers. `python -m fairxai.monitoring.run [--scheduled]`.

Feature drift is a signal to investigate, never evidence of degradation; only labelled data confirms that.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from fairxai.config import Config, load_config
from fairxai.monitoring import performance as sig
from fairxai.monitoring.drift import compute_drift
from fairxai.monitoring.triggers import evaluate
from fairxai.monitoring.windows import labelled_window, prediction_window
from fairxai.notify import notify

KINDS = ("data_quality", "feature_drift", "prediction_drift", "performance", "fairness")


@dataclass
class ModelRef:
    """What monitoring needs about the model whose window is evaluated."""

    version: int
    threshold: float
    eval_pr_auc: float  # performance reference stored at promotion (version tag eval_pr_auc)
    eval_probabilities: np.ndarray  # eval partition scored by this model (prediction-drift reference)


def write_result(conn, r: sig.SignalResult, version: int, window_end) -> None:
    conn.execute(
        "INSERT INTO monitoring_results (kind, status, model_version, window_end, n_rows, n_labelled, value, details)"
        " VALUES (%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (kind, model_version, window_end)"
        " WHERE window_end IS NOT NULL DO UPDATE SET status=EXCLUDED.status, n_rows=EXCLUDED.n_rows,"
        " n_labelled=EXCLUDED.n_labelled, value=EXCLUDED.value, details=EXCLUDED.details, ts=now()",
        (
            r.kind,
            r.status,
            version,
            window_end,
            r.n_rows,
            r.n_labelled,
            r.value,
            json.dumps(r.details, default=str),
        ),
    )


def history(conn, kind: str, version: int, limit: int = 50) -> list[str]:
    rows = conn.execute(
        "SELECT status FROM monitoring_results WHERE kind=%s AND model_version=%s AND window_end IS NOT NULL"
        " ORDER BY window_end DESC LIMIT %s",
        (kind, version, limit),
    ).fetchall()
    return [r[0] for r in rows]


def run_monitoring(
    conn,
    cfg: Config,
    model: ModelRef,
    reference: pd.DataFrame,
    reference_stats: dict,
    scheduled: bool = False,
    engine: str = "evidently",
    do_notify: bool = True,
) -> dict:
    w = prediction_window(conn, cfg.base.window_rows, model.version)
    if w.n_rows == 0:
        return {
            "model_version": model.version,
            "window_end": None,
            "status": "NA",
            "reason": "no predictions",
        }
    lab = labelled_window(conn, cfg.base.window_rows, model.version)
    n_lab_window = len(w.labelled)
    results = [sig.data_quality(w.frame, n_lab_window, cfg, reference_stats)]
    drift = compute_drift(reference, w.frame, cfg, engine) if w.n_rows >= cfg.base.min_window_rows else None
    results.append(sig.feature_drift(drift, w.n_rows, n_lab_window, cfg))
    results.append(sig.pred_drift(w.frame, model.eval_probabilities, model.threshold, n_lab_window, cfg))
    results.append(sig.performance(lab, w.n_rows, model.threshold, model.eval_pr_auc, cfg))
    results.append(sig.live_fairness(lab, w.n_rows, model.threshold, cfg))
    for r in results:
        write_result(conn, r, model.version, w.window_end)
    conn.commit()

    r = cfg.retraining
    decision = evaluate(
        history(conn, "feature_drift", model.version),
        history(conn, "performance", model.version),
        r.drift_persistence,
        r.perf_persistence,
        scheduled,
    )
    summary = {
        "model_version": model.version,
        "window_end": str(w.window_end),
        "n_rows": w.n_rows,
        "n_labelled": len(lab),
        "signals": {x.kind: {"status": x.status, "value": x.value} for x in results},
        "drift_engine": drift.engine if drift else None,
        "trigger": decision.to_dict(),
    }
    if do_notify:
        for msg in decision.warn_events:
            notify("warn", "monitoring", msg, {"model_version": model.version})
        for x in results:
            if x.status == sig.ALERT:
                notify(
                    "warn",
                    "monitoring",
                    f"{x.kind} ALERT",
                    {"model_version": model.version, "value": x.value},
                )
        if decision.request:
            notify("info", "monitoring", f"retrain requested: {decision.reason}", summary)
    return summary


def model_ref_from_registry(version: int, cfg: Config, eval_csv: Path) -> ModelRef:
    import mlflow.sklearn
    from mlflow import MlflowClient

    from fairxai.data.load import FEATURES

    name = cfg.base.mlflow.registered_model
    tags = MlflowClient().get_model_version(name, str(version)).tags
    pipe = mlflow.sklearn.load_model(f"models:/{name}/{version}")
    prob = pipe.predict_proba(pd.read_csv(eval_csv)[FEATURES])[:, 1]
    return ModelRef(version, float(tags["decision_threshold"]), float(tags["eval_pr_auc"]), prob)


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(prog="python -m fairxai.monitoring.run")
    ap.add_argument("--scheduled", action="store_true")
    ap.add_argument("--model-version", type=int, default=None)
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--engine", default="evidently", choices=["evidently", "scipy"])
    a = ap.parse_args(argv)
    import psycopg

    from fairxai.monitoring.windows import latest_model_version

    cfg = load_config()
    d = Path(a.data_dir)
    with psycopg.connect(os.environ["FAIRXAI_DB_URL"]) as conn:
        version = a.model_version or latest_model_version(conn)
        if version is None:
            print(json.dumps({"status": "NA", "reason": "no predictions logged"}))
            return 0
        model = model_ref_from_registry(version, cfg, d / "eval.csv")
        stats = json.loads((d / "reference_stats.json").read_text(encoding="utf-8"))
        out = run_monitoring(conn, cfg, model, pd.read_csv(d / "train.csv"), stats, a.scheduled, a.engine)
    print(json.dumps(out, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Training steps 1-5 of SPEC D.3 (Phase 3): baselines, ranking, tuning, threshold, eval scoring.

Fairness gating, registration and MLflow arrive in Phases 4-5; results are written to local JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import pandas as pd
import yaml

from fairxai.config import Config, load_config
from fairxai.data.load import FEATURES
from fairxai.data.validate import validate
from fairxai.logs import get_logger
from fairxai.models.evaluate import cv_oof, metrics_at
from fairxai.models.threshold import choose_threshold
from fairxai.models.tune import tune_model
from fairxai.models.zoo import make_pipeline

log = get_logger("fairxai.train")


def train_md5(train_csv: Path, dvc_lock: Path | None = Path("dvc.lock")) -> str:
    """MD5 of the train partition from dvc.lock when it matches the file, else hashed from the file."""
    digest = hashlib.md5(train_csv.read_bytes()).hexdigest()
    if dvc_lock and dvc_lock.exists():
        lock = yaml.safe_load(dvc_lock.read_text(encoding="utf-8"))
        for out in lock.get("stages", {}).get("split", {}).get("outs", []):
            if out["path"].endswith("train.csv") and out["md5"] == digest:
                return out["md5"]
    return digest


def data_version_id(base_md5: str, batches: list[tuple[str, str]] | None = None) -> str:
    payload = {"base": base_md5, "batches": sorted([list(b) for b in (batches or [])])}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def rank_baselines(scores: dict[str, float]) -> list[str]:
    """Descending mean PR-AUC; tie-break by model name ascending."""
    return [m for m, _ in sorted(scores.items(), key=lambda kv: (-round(kv[1], 12), kv[0]))]


def split_xy(df: pd.DataFrame, target: str):
    return df[FEATURES], df[target].astype(int)


def run_training(
    cfg: Config, train_df: pd.DataFrame, eval_df: pd.DataFrame, out_dir: Path, base_md5: str
) -> dict:
    t0 = time.time()
    tc, seed = cfg.training, cfg.base.seed
    g1 = validate(train_df, "batch", cfg)
    if not g1.passed:
        raise RuntimeError(f"G1 failed: {g1.errors}")
    X, y = split_xy(train_df, cfg.data.target)
    Xe, ye = split_xy(eval_df, cfg.data.target)
    dvid = data_version_id(base_md5)

    # Step 2: baselines ranked by mean 5-fold PR-AUC on train; OOF probs reused for thresholds.
    cv_scores: dict[str, float] = {}
    baselines: dict[str, dict] = {}
    for name in tc.models:
        ts = time.time()
        params = tc.baseline_params.get(name, {})
        score, oof = cv_oof(make_pipeline(name, params, y, cfg), X, y, tc.cv_folds, seed, tc.n_jobs)
        thr = choose_threshold(y, oof, tc.threshold_grid)
        model = make_pipeline(name, params, y, cfg).fit(X, y)
        # Step 3: reported baseline table (not used for selection)
        baselines[name] = {
            "cv_pr_auc": score,
            "eval": metrics_at(ye, model.predict_proba(Xe)[:, 1], thr),
            "seconds": round(time.time() - ts, 1),
        }
        cv_scores[name] = score
        log.info("baseline %s cv_pr_auc=%.4f (%.0fs)", name, score, baselines[name]["seconds"])
    ranking = rank_baselines(cv_scores)
    shortlist = ranking[: tc.shortlist]

    # Step 4-5 (without fairness gate): tune, fit full train, threshold from OOF, score eval.
    tuned: list[dict] = []
    fitted_models: dict = {}
    for name in shortlist:
        ts = time.time()
        res = tune_model(name, X, y, cfg)
        pipe = make_pipeline(name, res["best_params"], y, cfg)
        _, oof = cv_oof(pipe, X, y, tc.cv_folds, seed, tc.n_jobs)
        thr = choose_threshold(y, oof, tc.threshold_grid)
        fitted = make_pipeline(name, res["best_params"], y, cfg).fit(X, y)
        res["decision_threshold"] = thr
        fitted_models[name] = fitted
        res["eval"] = metrics_at(ye, fitted.predict_proba(Xe)[:, 1], thr)
        res["seconds"] = round(time.time() - ts, 1)
        tuned.append(res)
        log.info(
            "tuned %s cv=%.4f eval_pr_auc=%.4f thr=%.2f",
            name,
            res["best_cv_pr_auc"],
            res["eval"]["pr_auc"],
            thr,
        )
    order = sorted(tuned, key=lambda r: (-round(r["best_cv_pr_auc"], 12), r["model"]))

    deterministic = {
        "profile": cfg.profile,
        "seed": seed,
        "data_version_id": dvid,
        "g1": g1.to_dict(),
        "baseline_ranking": ranking,
        "baseline_cv_pr_auc": cv_scores,
        "baselines": {k: {kk: vv for kk, vv in v.items() if kk != "seconds"} for k, v in baselines.items()},
        "shortlist": shortlist,
        "tuned": [{k: v for k, v in r.items() if k not in ("seconds", "trials")} for r in order],
        "selected_pending_gate": order[0]["model"],
    }
    full = {
        "deterministic": deterministic,
        "trials": {r["model"]: r["trials"] for r in tuned},
        "timing_seconds": {
            "total": round(time.time() - t0, 1),
            "baselines": {k: v["seconds"] for k, v in baselines.items()},
            "tuning": {r["model"]: r["seconds"] for r in tuned},
        },
        "note": "Fairness gate (G3) and registration are Phase 4-5; 'selected_pending_gate' is not a champion.",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(full, indent=2), encoding="utf-8")
    full["_selected_pipeline"] = fitted_models[order[0]["model"]]  # in-memory only, not in results.json
    return full


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m fairxai.models.train")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--track", action="store_true", help="log to MLflow (MLFLOW_TRACKING_URI) and register")
    a = ap.parse_args(argv)
    cfg = load_config(a.profile)
    d = Path(a.data_dir)
    base = train_md5(d / "train.csv")
    out = Path(a.out_dir) if a.out_dir else Path("data/runs") / f"{cfg.profile}_{base[:8]}"
    train_df = pd.read_csv(d / "train.csv")
    res = run_training(cfg, train_df, pd.read_csv(d / "eval.csv"), out, base)
    tracked = {}
    if a.track:
        from fairxai.models.tracking import track_run

        tracked = track_run(cfg, res, base, train_df, Path("tests/fixtures/smoke_set.csv"))
    print(
        json.dumps(
            {
                "results": str(out / "results.json"),
                **tracked,
                **{
                    k: res["deterministic"][k]
                    for k in ("data_version_id", "shortlist", "selected_pending_gate")
                },
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Training pipeline of SPEC D.3: baselines, ranking, tuning, thresholds, gated selection, test report, SHAP.

`python -m fairxai.models.train [--track]`. Exit code 0 for APPROVED, REJECTED and INDETERMINATE.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from fairxai.config import Config, load_config
from fairxai.data.load import FEATURES
from fairxai.data.validate import validate
from fairxai.fairness.report import build_report
from fairxai.logs import get_logger
from fairxai.models.evaluate import cv_oof, metrics_at
from fairxai.models.threshold import choose_threshold
from fairxai.models.tune import tune_model
from fairxai.models.zoo import make_pipeline
from fairxai.registry.promotion import APPROVED, Candidate, Evaluation, decide

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


@dataclass
class ChampionRef:
    """The live champion, re-scored freshly on the same evaluation set (D.3 step 5)."""

    pipeline: object
    threshold: float
    version: int | None = None


def evaluate_on(pipe, threshold: float, df: pd.DataFrame, cfg: Config, champion_report: dict | None = None):
    X, y = split_xy(df, cfg.data.target)
    prob = pipe.predict_proba(X)[:, 1]
    metrics = metrics_at(y, prob, threshold)
    report = build_report(df, y, prob, threshold, cfg, champion_report)
    ev = Evaluation(
        pr_auc=metrics["pr_auc"], n_eval=len(y), n_classes=int(y.nunique()), fairness_report=report
    )
    return ev, metrics, report


def g5_roundtrip(pipe, smoke_csv: Path | None) -> tuple[bool | None, str]:
    """G5 first half: save/load through mlflow.sklearn to a temp dir; finite probabilities in [0,1] on the
    synthetic smoke set; feature names and order equal the serving schema."""
    if smoke_csv is None:
        return None, "no smoke set given"
    import mlflow.sklearn

    smoke = pd.read_csv(smoke_csv)
    with tempfile.TemporaryDirectory() as d:
        mlflow.sklearn.save_model(pipe, f"{d}/m", serialization_format="cloudpickle")
        loaded = mlflow.sklearn.load_model(f"{d}/m")
    names = [str(n) for n in getattr(loaded, "feature_names_in_", [])]
    if names != FEATURES:
        return False, f"model feature names {names} != serving schema"
    p = loaded.predict_proba(smoke[FEATURES])[:, 1]
    ok = bool(np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all())
    return ok, "round-trip OK" if ok else "non-finite or out-of-range probabilities"


def run_training(
    cfg: Config,
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    out_dir: Path,
    base_md5: str,
    test_df: pd.DataFrame | None = None,
    champion: ChampionRef | None = None,
    smoke_csv: Path | None = None,
    shap_rows: int = 1000,
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

    # Step 4: tune the shortlist; fit on full train; threshold from 5-fold OOF train probabilities.
    tuned: list[dict] = []
    fitted: dict = {}
    for name in shortlist:
        ts = time.time()
        res = tune_model(name, X, y, cfg)
        _, oof = cv_oof(make_pipeline(name, res["best_params"], y, cfg), X, y, tc.cv_folds, seed, tc.n_jobs)
        res["decision_threshold"] = choose_threshold(y, oof, tc.threshold_grid)
        fitted[name] = make_pipeline(name, res["best_params"], y, cfg).fit(X, y)
        res["seconds"] = round(time.time() - ts, 1)
        tuned.append(res)
        log.info("tuned %s cv=%.4f (%.0fs)", name, res["best_cv_pr_auc"], res["seconds"])
    order = sorted(tuned, key=lambda r: (-round(r["best_cv_pr_auc"], 12), r["model"]))

    # Step 5: gated selection; the champion (if any) is scored freshly on the same eval set.
    champ_eval = champ_report = champ_metrics = None
    if champion is not None:
        champ_eval, champ_metrics, champ_report = evaluate_on(
            champion.pipeline, champion.threshold, eval_df, cfg
        )
    selection: list[dict] = []
    selected = None
    for r in order:
        name = r["model"]
        ev, metrics, report = evaluate_on(fitted[name], r["decision_threshold"], eval_df, cfg, champ_report)
        g5_ok, g5_detail = g5_roundtrip(fitted[name], smoke_csv)
        gd = decide(Candidate(ev, g1.passed, g5_ok, "suite batch passed", g5_detail), champ_eval, cfg)
        r["eval"] = metrics
        entry = {"model": name, "decision": gd.decision, "gate": gd.to_dict(), "fairness_report": report}
        selection.append(entry)
        log.info(
            "candidate %s eval_pr_auc=%.4f decision=%s %s", name, metrics["pr_auc"], gd.decision, gd.reasons
        )
        if gd.decision == APPROVED:
            selected = entry
            break
    if selected is None:
        selected = selection[0]  # top-ranked model is registered with its (non-approved) decision
    sel_name = selected["model"]
    sel = next(r for r in order if r["model"] == sel_name)
    sel_pipe = fitted[sel_name]

    # Step 6: test report, once, for the selected model. Never read by decide().
    test_block = None
    if test_df is not None:
        _, tm, trep = evaluate_on(sel_pipe, sel["decision_threshold"], test_df, cfg)
        test_block = {"metrics": tm, "fairness_report": trep}

    # Step 7: global SHAP summary on a seeded train sample.
    shap_imp = shap_method = None
    if shap_rows:
        from fairxai.explain.shap_global import global_importance

        shap_imp, shap_method = global_importance(sel_pipe, X, shap_rows, seed)

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
        "champion": None
        if champion is None
        else {"version": champion.version, "threshold": champion.threshold, "eval": champ_metrics},
        "selection": [
            {"model": e["model"], "decision": e["decision"], "reasons": e["gate"]["reasons"]}
            for e in selection
        ],
        "selected": {
            "model": sel_name,
            "decision": selected["decision"],
            "decision_threshold": sel["decision_threshold"],
            "best_params": sel["best_params"],
            "eval": sel["eval"],
        },
    }
    full = {
        "deterministic": deterministic,
        "gate_report": selected["gate"],
        "fairness_report": selected["fairness_report"],
        "champion_fairness_report": champ_report,
        "test": test_block,
        "shap": {"method": shap_method, "importance": shap_imp},
        "trials": {r["model"]: r["trials"] for r in tuned},
        "timing_seconds": {
            "total": round(time.time() - t0, 1),
            "baselines": {k: v["seconds"] for k, v in baselines.items()},
            "tuning": {r["model"]: r["seconds"] for r in tuned},
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(full, indent=2, default=str), encoding="utf-8")
    full["_selected_pipeline"] = sel_pipe  # in-memory only, not in results.json
    return full


def load_champion(registry) -> ChampionRef | None:
    import mlflow.sklearn

    v = registry.alias_version("champion")
    if v is None:
        return None
    thr = float(registry.tags(v)["decision_threshold"])
    return ChampionRef(mlflow.sklearn.load_model(f"models:/{registry.name}/{v}"), thr, v)


def main(argv: list[str] | None = None) -> int:
    # MLflow prints run URLs with emoji; a Windows cp1252 console would crash the run at end_run()
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
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
    smoke = Path("tests/fixtures/smoke_set.csv")
    train_df = pd.read_csv(d / "train.csv")
    registry = champion = None
    if a.track:
        from fairxai.registry.registry import Registry

        registry = Registry(profile=cfg.profile)
        champion = load_champion(registry)
    res = run_training(
        cfg, train_df, pd.read_csv(d / "eval.csv"), out, base, pd.read_csv(d / "test.csv"), champion, smoke
    )
    tracked: dict = {}
    if a.track:
        from fairxai.models.tracking import track_run

        tracked = track_run(cfg, res, base, train_df, smoke, registry)
    det = res["deterministic"]
    summary = {
        "results": str(out / "results.json"),
        "data_version_id": det["data_version_id"],
        "model": det["selected"]["model"],
        "gate_decision": det["selected"]["decision"],
        **tracked,
    }
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

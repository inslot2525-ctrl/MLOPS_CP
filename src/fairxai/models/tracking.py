"""MLflow logging and registration for a finished training run (SPEC D.3 step 8, Phase 4 subset).

Fairness/gate reports and the `candidate` alias arrive in Phase 5; until then versions are tagged
gate_decision=PENDING_GATE and no alias is set.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature

from fairxai.config import Config
from fairxai.data.load import FEATURES
from fairxai.registry.registry import Registry

PENDING_GATE = "PENDING_GATE"


def git_info(cwd: str | Path = ".") -> tuple[str, bool]:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        return run("rev-parse", "HEAD"), bool(run("status", "--porcelain"))
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown", True


def smoke_check(registered_uri: str, in_memory, smoke_csv: Path) -> bool:
    """G5 (second half): registered model must give identical probabilities on the smoke set."""
    smoke = pd.read_csv(smoke_csv)[FEATURES]
    loaded = mlflow.sklearn.load_model(registered_uri)
    a, b = loaded.predict_proba(smoke), in_memory.predict_proba(smoke)
    ok = bool(np.isfinite(a).all() and np.array_equal(a, b))
    return ok


def _flat(prefix: str, d: dict) -> dict:
    return {f"{prefix}.{k}": v for k, v in d.items()}


def track_run(
    cfg: Config,
    results: dict,
    base_md5: str,
    train_df: pd.DataFrame,
    smoke_csv: Path,
    registry: Registry | None = None,
) -> dict:
    """Log everything to MLflow, register the selected model, run the registered-URI smoke check."""
    det = results["deterministic"]
    selected = det["selected_pending_gate"]
    pipeline = results["_selected_pipeline"]
    chosen = next(r for r in det["tuned"] if r["model"] == selected)
    commit, dirty = git_info()
    mlflow.set_experiment(cfg.base.mlflow.experiment)
    X = train_df[FEATURES]

    with mlflow.start_run(run_name=f"train-{det['data_version_id'][:8]}") as run:
        mlflow.set_tags(
            {
                "git_commit": commit,
                "git_dirty": str(dirty).lower(),
                "dvc_train_md5": base_md5,
                "data_version_id": det["data_version_id"],
                "profile": cfg.profile,
            }
        )
        mlflow.log_params({"model": selected, "seed": cfg.base.seed, **_flat("param", chosen["best_params"])})
        mlflow.log_metrics(
            {
                "cv_pr_auc": chosen["best_cv_pr_auc"],
                "decision_threshold": chosen["decision_threshold"],
                **{f"eval_{k}": v for k, v in chosen["eval"].items() if isinstance(v, float)},
            }
        )
        mlflow.log_dict(cfg.model_dump(), "config_snapshot.json")
        mlflow.log_dict(
            {
                m: {
                    "best_params": r["best_params"],
                    "best_cv_pr_auc": r["best_cv_pr_auc"],
                    "n_trials": r["n_trials"],
                }
                for m, r in ((t["model"], t) for t in det["tuned"])
            }
            | {"trials": results["trials"]},
            "optuna_summary.json",
        )
        mlflow.log_dict(det, "results.json")
        rows = [{"model": m, "cv_pr_auc": b["cv_pr_auc"], **b["eval"]} for m, b in det["baselines"].items()]
        mlflow.log_text(
            pd.DataFrame(rows).drop(columns=["confusion_matrix"]).to_csv(index=False), "baselines.csv"
        )
        sample = X.head(5)
        info = mlflow.sklearn.log_model(
            pipeline,
            name="model",
            signature=infer_signature(sample, pipeline.predict_proba(sample)[:, 1]),
            input_example=sample,
            serialization_format="cloudpickle",
        )
        run_id = run.info.run_id
        parent_id = run_id

    for model_name, trials in results["trials"].items():  # one nested run per Optuna trial
        for t in trials:
            with mlflow.start_run(
                run_name=f"{model_name}-trial-{t['number']}", nested=True, parent_run_id=parent_id
            ):
                mlflow.set_tag("model", model_name)
                mlflow.log_params(t["params"])
                if t["value"] is not None:
                    mlflow.log_metric("cv_pr_auc", t["value"])

    registry = registry or Registry()
    version = registry.register_version(
        info.model_uri,
        {
            "gate_decision": PENDING_GATE,
            "decision_threshold": chosen["decision_threshold"],
            "eval_pr_auc": chosen["eval"]["pr_auc"],
            "data_version_id": det["data_version_id"],
            "git_commit": commit,
            "run_id": run_id,
        },
    )
    smoke_ok = smoke_check(f"models:/{registry.name}/{version}", pipeline, smoke_csv)
    registry.client.set_model_version_tag(
        registry.name, str(version), "g5_smoke", "PASS" if smoke_ok else "FAIL"
    )
    if not smoke_ok:
        registry.client.set_model_version_tag(registry.name, str(version), "gate_decision", "REJECTED")
    return {"run_id": run_id, "model_version": version, "g5_smoke_ok": smoke_ok, "model": selected}


def dump(out: dict) -> str:
    return json.dumps(out)

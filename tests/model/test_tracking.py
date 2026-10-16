"""End-to-end tracking on SYNTHETIC data against a real local MLflow (sqlite + tmp artifacts)."""

from pathlib import Path

import mlflow
import pytest
from mlflow import MlflowClient

from fairxai.config import load_config
from fairxai.models.tracking import track_run
from fairxai.models.train import run_training
from fairxai.registry.lineage import resolve
from fairxai.registry.registry import Registry

SMOKE = Path(__file__).resolve().parents[1] / "fixtures" / "smoke_set.csv"


@pytest.mark.slow
def test_run_artifacts_registered_and_lineage(tmp_path):
    from make_synthetic import make_synthetic

    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    cfg = load_config("test")
    tr, ev = make_synthetic(1500, seed=1), make_synthetic(600, seed=2)
    res = run_training(cfg, tr, ev, tmp_path / "out", "synthetic-md5")
    reg = Registry(MlflowClient(uri, uri), event_sink=None)
    out = track_run(cfg, res, "synthetic-md5", tr, SMOKE, reg)

    assert out["g5_smoke_ok"]  # model (an MLflow 3 LoggedModel, not a run artifact) loads from the registry
    arts = {a.path for a in reg.client.list_artifacts(out["run_id"])}
    for needed in ("config_snapshot.json", "optuna_summary.json", "baselines.csv", "results.json"):
        assert needed in arts, needed
    tags = reg.tags(out["model_version"])
    for key in (
        "gate_decision",
        "decision_threshold",
        "data_version_id",
        "git_commit",
        "run_id",
        "eval_pr_auc",
    ):
        assert key in tags, key
    assert tags["gate_decision"] == "PENDING_GATE" and tags["g5_smoke"] == "PASS"
    assert reg.alias_version("candidate") is None  # never set before the Phase 5 gate
    info = resolve(out["model_version"], reg.client)
    assert info["dvc_train_md5"] == "synthetic-md5" and info["run_id"] == out["run_id"]
    children = reg.client.search_runs(
        [reg.client.get_run(out["run_id"]).info.experiment_id], f"tags.mlflow.parentRunId = '{out['run_id']}'"
    )
    assert len(children) == 3 * cfg.training.optuna_trials

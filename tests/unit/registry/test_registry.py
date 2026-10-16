"""Registry tests run against a REAL local MLflow (sqlite backend, tmp artifact dir), not mocks."""

import mlflow
import pytest
from mlflow import MlflowClient
from sklearn.dummy import DummyClassifier

from fairxai.registry.lineage import LineageError, resolve
from fairxai.registry.registry import (
    APPROVED,
    FIXTURE_FORCED,
    NotApproved,
    Registry,
    RegistryInconsistent,
)


@pytest.fixture
def env(tmp_path, monkeypatch):
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    exp = mlflow.create_experiment("t", artifact_location=(tmp_path / "art").as_uri())
    mlflow.set_experiment(experiment_id=exp)
    events = []
    reg = Registry(MlflowClient(uri, uri), name="test-model", profile="base", event_sink=events.append)
    return reg, events


def make_version(reg: Registry, decision: str, full_lineage: bool = True) -> int:
    with mlflow.start_run() as run:
        if full_lineage:
            mlflow.set_tags(
                {"git_commit": "abc", "git_dirty": "false", "dvc_train_md5": "m", "data_version_id": "d"}
            )
            mlflow.log_dict({"a": 1}, "config_snapshot.json")
            mlflow.log_dict({"a": 1}, "optuna_summary.json")
        info = mlflow.sklearn.log_model(DummyClassifier().fit([[0], [1]], [0, 1]), name="model")
    return reg.register_version(info.model_uri, {"gate_decision": decision, "run_id": run.info.run_id})


def test_set_candidate_refuses_non_approved(env):
    reg, _ = env
    for d in ("REJECTED", "INDETERMINATE", "PENDING_GATE"):
        v = make_version(reg, d)
        with pytest.raises(NotApproved):
            reg.set_candidate(v)
    assert reg.alias_version("candidate") is None


def test_set_candidate_ok_and_event(env):
    reg, events = env
    v = make_version(reg, APPROVED)
    reg.set_candidate(v)
    assert reg.alias_version("candidate") == v
    assert events and events[-1]["version"] == v and "last_transition" in reg.tags(v)


def test_promote_refuses_rejected(env):
    reg, _ = env
    v = make_version(reg, "REJECTED")
    with pytest.raises(NotApproved):
        reg.promote_to_champion(v)
    assert reg.alias_version("champion") is None


def test_promote_sequence_previous_and_candidate_removed(env):
    reg, _ = env
    v1, v2 = make_version(reg, APPROVED), make_version(reg, APPROVED)
    reg.set_candidate(v1)
    reg.promote_to_champion(v1)
    assert reg.alias_version("champion") == v1 and reg.alias_version("candidate") is None
    assert reg.rollback_target() is None
    reg.set_candidate(v2)
    reg.promote_to_champion(v2)
    assert reg.alias_version("champion") == v2 and reg.rollback_target() == v1
    assert reg.alias_version("candidate") is None


def test_rollback_to_previous_champion_allowed(env):
    reg, _ = env
    v1, v2 = make_version(reg, APPROVED), make_version(reg, APPROVED)
    reg.promote_to_champion(v1)
    reg.promote_to_champion(v2)
    # a previously held champion stays deployable even if its tag is later downgraded
    reg.client.set_model_version_tag(reg.name, str(v1), "gate_decision", "REJECTED")
    reg.check_deployable(v1)
    reg.promote_to_champion(v1)
    assert reg.alias_version("champion") == v1 and reg.rollback_target() == v2


def test_promote_same_version_is_stable(env):
    reg, _ = env
    v1, v2 = make_version(reg, APPROVED), make_version(reg, APPROVED)
    reg.promote_to_champion(v1)
    reg.promote_to_champion(v2)
    reg.promote_to_champion(v2)
    assert reg.alias_version("champion") == v2 and reg.rollback_target() == v1


def test_fixture_forced_only_under_demo_or_test(env):
    reg, _ = env
    v = make_version(reg, FIXTURE_FORCED)
    with pytest.raises(NotApproved):
        reg.promote_to_champion(v)
    reg.profile = "demo"
    reg.promote_to_champion(v)
    assert reg.alias_version("champion") == v


def test_inconsistency_detected(env, monkeypatch):
    reg, _ = env
    v1, v2 = make_version(reg, APPROVED), make_version(reg, APPROVED)
    reg.promote_to_champion(v1)
    real = reg.client.set_registered_model_alias

    def flaky(name, alias, version):
        if alias == "champion" and str(version) == str(v2):
            return None  # silently dropped write
        return real(name, alias, version)

    monkeypatch.setattr(reg.client, "set_registered_model_alias", flaky)
    with pytest.raises(RegistryInconsistent):
        reg.promote_to_champion(v2)


def test_lineage_resolves_and_raises_on_null(env):
    reg, _ = env
    good = make_version(reg, APPROVED)
    info = resolve(good, reg.client, reg.name)
    assert info["git_commit"] == "abc" and info["data_version_id"] == "d"
    bad = make_version(reg, APPROVED, full_lineage=False)
    with pytest.raises(LineageError):
        resolve(bad, reg.client, reg.name)
    with pytest.raises(LineageError):  # deployment fields absent before deployment
        resolve(good, reg.client, reg.name, require_deployment=True)

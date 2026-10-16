"""Lineage: resolve a registered version back to run, code, data and config (SPEC D.5)."""

from __future__ import annotations

import os
from collections.abc import Callable

from mlflow import MlflowClient

from fairxai.registry.registry import MODEL_NAME

REQUIRED_RUN_TAGS = ("git_commit", "git_dirty", "dvc_train_md5", "data_version_id")
REQUIRED_ARTIFACTS = ("config_snapshot.json", "optuna_summary.json")


class LineageError(Exception):
    """A lineage field is missing."""


def resolve(
    version: int,
    client: MlflowClient | None = None,
    name: str = MODEL_NAME,
    require_deployment: bool = False,
    deployment_lookup: Callable[[int], int | None] | None = None,
) -> dict:
    """Return all lineage fields; raise LineageError if any is null.

    `require_deployment=True` additionally needs image_tag (env IMAGE_TAG) and a deployment_id,
    which exist only once the version has been deployed (Phase 9).
    """
    client = client or MlflowClient()
    mv = client.get_model_version(name, str(version))
    if not mv.run_id:
        raise LineageError(f"version {version} has no run_id")
    run = client.get_run(mv.run_id)
    out: dict = {"version": int(version), "run_id": mv.run_id}
    for key in REQUIRED_RUN_TAGS:
        val = run.data.tags.get(key) or run.data.params.get(key)
        out[key] = val
    artifacts = {a.path for a in client.list_artifacts(mv.run_id)}
    for art in REQUIRED_ARTIFACTS:
        out[art] = art if art in artifacts else None
    if require_deployment:
        out["image_tag"] = os.environ.get("IMAGE_TAG")
        out["deployment_id"] = deployment_lookup(version) if deployment_lookup else None
    missing = [k for k, v in out.items() if v is None or v == ""]
    if missing:
        raise LineageError(f"version {version}: missing lineage fields {missing}")
    return out

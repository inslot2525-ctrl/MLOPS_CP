"""API test fixtures. Models are SYNTHETIC fixtures; DB tests use real Postgres (FAIRXAI_TEST_DB_URL)."""

import os
import uuid

import pytest
from fastapi.testclient import TestClient

from fairxai.config import load_config
from fairxai.data.load import FEATURES
from fairxai.httpapi import sha256_hex
from fairxai.models.zoo import make_pipeline
from serving.app.main import Settings, create_app
from serving.app.model_store import LoadedModel, UnknownVersion

CLIENT_KEY = "test-client-key-" + uuid.uuid4().hex
ADMIN_KEY = "test-admin-key-" + uuid.uuid4().hex
DB_URL = os.environ.get("FAIRXAI_TEST_DB_URL")


@pytest.fixture(scope="session")
def fixture_pipeline():
    from make_synthetic import make_synthetic

    df = make_synthetic(600, seed=3)
    cfg = load_config("test")
    return make_pipeline("logreg", {}, df["default"], cfg).fit(df[FEATURES], df["default"])


class MemoryLog:
    def __init__(self, fail: bool = False) -> None:
        self.fail, self.predictions, self.labels = fail, {}, {}

    def log_prediction(self, request_id, version, slot, features, prediction, probability, latency_ms):
        if self.fail:
            raise RuntimeError("db down")
        self.predictions.setdefault(request_id, {"version": version, "features": features})

    def add_labels(self, labels):
        acc = dup = 0
        unknown = []
        for rid, lab in labels:
            if rid not in self.predictions:
                unknown.append(rid)
            elif rid in self.labels:
                dup += 1
            else:
                self.labels[rid] = lab
                acc += 1
        return acc, dup, unknown


def make_loader(pipeline, known=(1, 2), broken=(3,)):
    def load(version: int) -> LoadedModel:
        if version in broken:
            raise OSError("truncated model file (synthetic fixture)")
        if version not in known:
            raise UnknownVersion(str(version))
        return LoadedModel(version=version, pipeline=pipeline, threshold=0.5, run_id=f"run{version}")

    return load


@pytest.fixture
def make_client(fixture_pipeline):
    def _make(expected: str = "", db=None, fail_log: bool = False):
        settings = Settings(
            slot="blue",
            api_keys=f"client:{sha256_hex(CLIENT_KEY)}",
            admin_sha256=sha256_hex(ADMIN_KEY),
            db_url=None,
            expected_version=expected,
        )
        log = db if db is not None else MemoryLog(fail=fail_log)
        app = create_app(settings, make_loader(fixture_pipeline), load_config("test"), log)
        return TestClient(app), log

    return _make


@pytest.fixture
def row():
    from make_synthetic import make_synthetic

    r = make_synthetic(1, seed=11)[FEATURES].iloc[0].to_dict()
    return {k: int(v) for k, v in r.items()}


C = {"X-API-Key": CLIENT_KEY}
A = {"X-API-Key": ADMIN_KEY}

"""Holds the loaded model; loads a new version and swaps atomically, keeping the old one on any failure."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime


class UnknownVersion(Exception):
    pass


@dataclass
class LoadedModel:
    version: int
    pipeline: object
    threshold: float
    run_id: str = ""
    git_commit: str = ""
    data_version_id: str = ""
    loaded_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    explainer: object | None = None


Loader = Callable[[int], LoadedModel]


def mlflow_loader(model_name: str) -> Loader:
    def load(version: int) -> LoadedModel:
        import mlflow.sklearn
        from mlflow import MlflowClient
        from mlflow.exceptions import MlflowException

        client = MlflowClient()
        try:
            mv = client.get_model_version(model_name, str(version))
        except MlflowException as exc:
            raise UnknownVersion(str(exc)) from exc
        tags = mv.tags
        pipe = mlflow.sklearn.load_model(f"models:/{model_name}/{version}")
        return LoadedModel(
            version=int(version),
            pipeline=pipe,
            threshold=float(tags["decision_threshold"]),
            run_id=mv.run_id or "",
            git_commit=tags.get("git_commit", ""),
            data_version_id=tags.get("data_version_id", ""),
        )

    return load


class ModelStore:
    def __init__(self, loader: Loader) -> None:
        self.loader = loader
        self.current: LoadedModel | None = None
        self.requested: int | None = None
        self.last_error: str | None = None
        self._lock = threading.Lock()

    def load(self, version: int) -> tuple[LoadedModel, int | None]:
        """Raises on failure; the previous model stays loaded."""
        self.requested = version
        new = self.loader(version)  # may raise; nothing swapped yet
        with self._lock:
            previous = self.current.version if self.current else None
            self.current = new
            self.last_error = None
        return new, previous

    def ready(self) -> tuple[bool, str]:
        cur = self.current
        if cur is None:
            return False, self.last_error or "no model loaded"
        if self.requested is not None and cur.version != self.requested:
            return False, f"loaded v{cur.version} but v{self.requested} was requested ({self.last_error})"
        return True, ""

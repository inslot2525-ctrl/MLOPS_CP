"""Model registry: the ONLY module that calls MLflow alias APIs (SPEC D5, D.5).

Aliases: candidate, champion, previous_champion. Versions carry tags (gate_decision, decision_threshold, ...).
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from typing import Any

from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from fairxai.logs import get_logger

log = get_logger("fairxai.registry")

MODEL_NAME = "fairxai-credit-default"
CANDIDATE, CHAMPION, PREVIOUS = "candidate", "champion", "previous_champion"
APPROVED, FIXTURE_FORCED = "APPROVED", "FIXTURE_FORCED"
FIXTURE_PROFILES = {"demo", "test"}


class NotApproved(Exception):
    """The version's gate decision does not allow this transition."""


class RegistryInconsistent(Exception):
    """Aliases do not match after a promotion; manual repair is documented in docs/runbook.md."""


EventSink = Callable[[dict[str, Any]], None]


def db_event_sink(event: dict[str, Any]) -> None:
    """Write the transition to the `events` table; a missing database is logged, never raised."""
    try:
        from fairxai import db

        with db.connect() as conn:
            conn.execute(
                "INSERT INTO events (severity, source, message, details) VALUES ('info','registry',%s,%s)",
                (f"{event['from']} -> {event['to']} (v{event['version']})", json.dumps(event)),
            )
            conn.commit()
    except Exception as exc:  # noqa: BLE001
        log.warning("registry event not written to DB: %s", exc)


def actor() -> str:
    return (
        os.environ.get("FAIRXAI_ACTOR")
        or f"cli:{os.environ.get('USERNAME') or os.environ.get('USER', 'unknown')}"
    )


class Registry:
    def __init__(
        self,
        client: MlflowClient | None = None,
        name: str = MODEL_NAME,
        profile: str | None = None,
        event_sink: EventSink | None = db_event_sink,
    ) -> None:
        self.client = client or MlflowClient()
        self.name = name
        self.profile = profile or os.environ.get("FAIRXAI_PROFILE", "base")
        self.event_sink = event_sink

    # ---- reads -------------------------------------------------------------------------------
    def alias_version(self, alias: str) -> int | None:
        try:
            return int(self.client.get_model_version_by_alias(self.name, alias).version)
        except MlflowException:
            return None

    def tags(self, version: int) -> dict[str, str]:
        return dict(self.client.get_model_version(self.name, str(version)).tags)

    def rollback_target(self) -> int | None:
        return self.alias_version(PREVIOUS)

    # ---- writes ------------------------------------------------------------------------------
    def register_version(self, model_uri: str, tags: dict[str, str]) -> int:
        import mlflow

        mv = mlflow.register_model(model_uri, self.name)
        for k, v in tags.items():
            self.client.set_model_version_tag(self.name, mv.version, k, str(v))
        return int(mv.version)

    def _fixture_allowed(self) -> bool:
        return self.profile in FIXTURE_PROFILES

    def _decision(self, version: int) -> str | None:
        return self.tags(version).get("gate_decision")

    def check_deployable(self, version: int) -> None:
        """Approval guard: APPROVED, or previously held champion/previous_champion (rollback stays possible)."""
        tags = self.tags(version)
        decision = tags.get("gate_decision")
        if decision == APPROVED or tags.get("held_champion") == "true":
            return
        if decision == FIXTURE_FORCED and self._fixture_allowed():
            return
        raise NotApproved(f"version {version} has gate_decision={decision!r}; not deployable")

    def _emit(self, version: int, frm: str, to: str, reason: str) -> None:
        event = {"actor": actor(), "version": version, "from": frm, "to": to, "reason": reason}
        self.client.set_model_version_tag(self.name, str(version), "last_transition", json.dumps(event))
        if self.event_sink:
            self.event_sink(event)

    def set_candidate(self, version: int, reason: str = "approved by gate") -> None:
        decision = self._decision(version)
        if decision != APPROVED and not (decision == FIXTURE_FORCED and self._fixture_allowed()):
            raise NotApproved(f"version {version} has gate_decision={decision!r}; cannot become candidate")
        prev = self.alias_version(CANDIDATE)
        self.client.set_registered_model_alias(self.name, CANDIDATE, str(version))
        self._emit(version, f"candidate:{prev}", f"candidate:{version}", reason)

    def promote_to_champion(self, version: int, reason: str = "G7 passed") -> None:
        self.check_deployable(version)
        old = self.alias_version(CHAMPION)
        if old is not None and old != version:
            self.client.set_registered_model_alias(self.name, PREVIOUS, str(old))  # previous first
        self.client.set_registered_model_alias(self.name, CHAMPION, str(version))
        self.client.set_model_version_tag(self.name, str(version), "held_champion", "true")
        if self.alias_version(CANDIDATE) == version:
            self.client.delete_registered_model_alias(self.name, CANDIDATE)
        expected_prev = old if (old is not None and old != version) else self.alias_version(PREVIOUS)
        problems = []
        if self.alias_version(CHAMPION) != version:
            problems.append("champion")
        if self.alias_version(PREVIOUS) != expected_prev:
            problems.append("previous_champion")
        if self.alias_version(CANDIDATE) == version:
            problems.append("candidate")
        if problems:
            raise RegistryInconsistent(f"aliases inconsistent after promoting v{version}: {problems}")
        self._emit(version, f"champion:{old}", f"champion:{version}", reason)

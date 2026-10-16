"""notify(): always insert into `events` and log; optionally post a webhook (never fails the pipeline)."""

from __future__ import annotations

import json
import os

import httpx

from fairxai.logs import get_logger

log = get_logger("fairxai.notify")


def _insert_event(severity: str, source: str, message: str, details: dict) -> None:
    from fairxai import db

    with db.connect() as conn:
        conn.execute(
            "INSERT INTO events (severity, source, message, details) VALUES (%s,%s,%s,%s)",
            (severity, source, message, json.dumps(details)),
        )
        conn.commit()


def notify(severity: str, source: str, message: str, details: dict | None = None) -> None:
    details = details or {}
    log.log({"error": 40, "warn": 30}.get(severity, 20), "[%s] %s: %s", severity, source, message)
    try:
        _insert_event(severity, source, message, details)
    except Exception as exc:  # noqa: BLE001
        log.error("could not write event: %s", exc)
    url = os.environ.get("NOTIFY_WEBHOOK_URL", "")
    if not url:
        return
    key = "content" if os.environ.get("NOTIFY_WEBHOOK_STYLE", "slack") == "discord" else "text"
    try:
        httpx.post(url, json={key: f"[{severity}] {source}: {message}"}, timeout=5)
    except Exception as exc:  # noqa: BLE001
        log.warning("webhook failed: %s", exc)

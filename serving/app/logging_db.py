"""Prediction and label logging into the `fairxai` database."""

from __future__ import annotations

import json
import threading

import psycopg


class PredictionLog:
    def __init__(self, url: str | None) -> None:
        self.url = url
        self._conn: psycopg.Connection | None = None
        self._lock = threading.Lock()

    def _connection(self) -> psycopg.Connection:
        if not self.url:
            raise RuntimeError("FAIRXAI_DB_URL not set")
        if self._conn is None or self._conn.closed:
            self._conn = psycopg.connect(self.url, autocommit=True, connect_timeout=3)
        return self._conn

    def _run(self, sql: str, params) -> list:
        with self._lock:
            try:
                cur = self._connection().execute(sql, params)
            except psycopg.OperationalError:
                self._conn = None  # reconnect once
                cur = self._connection().execute(sql, params)
            return cur.fetchall() if cur.description else []

    def log_prediction(
        self, request_id, version, slot, features, prediction, probability, latency_ms
    ) -> None:
        self._run(
            "INSERT INTO predictions (request_id, model_version, slot, features, prediction, probability, latency_ms)"
            " VALUES (%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (request_id) DO NOTHING",
            (request_id, version, slot, json.dumps(features), prediction, probability, latency_ms),
        )

    def add_labels(self, labels: list[tuple[str, int]]) -> tuple[int, int, list[str]]:
        """Returns (accepted, duplicates, unknown_request_ids). A repeat label never overwrites."""
        ids = [rid for rid, _ in labels]
        known = {
            str(r[0])
            for r in self._run("SELECT request_id FROM predictions WHERE request_id = ANY(%s)", (ids,))
        }
        accepted = duplicates = 0
        unknown: list[str] = []
        for rid, label in labels:
            if rid not in known:
                unknown.append(rid)
                continue
            rows = self._run(
                "INSERT INTO labels (request_id, label) VALUES (%s,%s) ON CONFLICT (request_id) DO NOTHING "
                "RETURNING request_id",
                (rid, label),
            )
            if rows:
                accepted += 1
            else:
                duplicates += 1
        return accepted, duplicates, unknown

"""Batch ingestion with quarantine directories and a hashed manifest (SPEC D.2, CR-1)."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Protocol

import pandas as pd

from fairxai.config import Config, load_config
from fairxai.data.validate import validate
from fairxai.notify import notify


class BatchStore(Protocol):
    def status(self, sha256: str) -> str | None: ...
    def upsert(self, batch_id: str, sha256: str, status: str, report: dict | None) -> None: ...


class MemoryBatchStore:
    """In-memory store used by unit tests only."""

    def __init__(self) -> None:
        self.rows: dict[str, dict] = {}

    def status(self, sha256: str) -> str | None:
        for r in self.rows.values():
            if r["sha256"] == sha256 and r["status"] == "ACCEPTED":
                return "ACCEPTED"
        return None

    def upsert(self, batch_id: str, sha256: str, status: str, report: dict | None) -> None:
        self.rows[batch_id] = {"sha256": sha256, "status": status, "report": report}


class PgBatchStore:
    """Postgres `data_batches` table."""

    def status(self, sha256: str) -> str | None:
        from fairxai import db

        with db.connect() as conn:
            row = conn.execute(
                "SELECT status FROM data_batches WHERE sha256=%s AND status='ACCEPTED' LIMIT 1", (sha256,)
            ).fetchone()
        return row[0] if row else None

    def upsert(self, batch_id: str, sha256: str, status: str, report: dict | None) -> None:
        from fairxai import db

        with db.connect() as conn:
            conn.execute(
                "INSERT INTO data_batches (batch_id, sha256, status, report) VALUES (%s,%s,%s,%s) "
                "ON CONFLICT (batch_id) DO UPDATE SET status=EXCLUDED.status, report=EXCLUDED.report, ts=now()",
                (batch_id, sha256, status, json.dumps(report) if report else None),
            )
            conn.commit()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ingest_batch(
    path: str | Path,
    batch_id: str,
    store: BatchStore,
    root: str | Path = "data/ingest",
    cfg: Config | None = None,
    reference_stats: dict | None = None,
    labelled: bool = True,
    do_notify: bool = True,
) -> str:
    """Return ACCEPTED, REJECTED or SKIPPED (already accepted hash). A rejected batch never reaches accepted/."""
    cfg = cfg or load_config()
    root = Path(root)
    for sub in ("incoming", "accepted", "rejected", "reports"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    incoming = root / "incoming" / f"{batch_id}.csv"
    shutil.copyfile(path, incoming)
    digest = sha256_file(incoming)
    if store.status(digest) == "ACCEPTED":
        incoming.unlink()
        return "SKIPPED"
    store.upsert(batch_id, digest, "INCOMING", None)
    try:
        df = pd.read_csv(incoming)
        result = validate(df, "batch", cfg, labelled=labelled, reference_stats=reference_stats)
        report = result.to_dict()
    except Exception as exc:  # noqa: BLE001 - an unreadable batch is a rejected batch
        report = {"suite": "batch", "passed": False, "errors": [f"unreadable batch: {exc}"], "warnings": []}
    if report["passed"]:
        shutil.move(incoming, root / "accepted" / f"{batch_id}.csv")
        store.upsert(batch_id, digest, "ACCEPTED", report)
        return "ACCEPTED"
    shutil.move(incoming, root / "rejected" / f"{batch_id}.csv")
    (root / "reports" / f"{batch_id}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    store.upsert(batch_id, digest, "REJECTED", report)
    if do_notify:
        notify("error", "ingest", f"batch {batch_id} rejected", {"errors": report["errors"]})
    return "REJECTED"

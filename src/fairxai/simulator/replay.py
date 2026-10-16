"""Replay stream batches through the gateway (SPEC D.9). Faults are applied by `faults.py` (Phase 9).

request_id = uuid5(namespace(seed), "<batch>:<row>") so a rerun is idempotent (the log ignores repeats).
"""

from __future__ import annotations

import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import pandas as pd

from fairxai.data.load import FEATURES


def request_ids(seed: int, batch: int, n: int) -> list[str]:
    ns = uuid.UUID(int=seed)
    return [str(uuid.uuid5(ns, f"{batch}:{i}")) for i in range(n)]


def batch_path(stream_dir: str | Path, batch: int) -> Path:
    return Path(stream_dir) / f"batch_{batch:04d}.csv"


class RateLimiter:
    def __init__(self, rps: float) -> None:
        self.interval, self.next, self.lock = 1.0 / rps, time.perf_counter(), threading.Lock()

    def wait(self) -> None:
        with self.lock:
            now = time.perf_counter()
            delay = self.next - now
            self.next = max(self.next, now) + self.interval
        if delay > 0:
            time.sleep(delay)


def send_predictions(
    gateway: str, key: str, df: pd.DataFrame, ids: list[str], workers: int, rps: float
) -> dict:
    limiter = RateLimiter(rps)
    rows = df[FEATURES].to_dict(orient="records")
    statuses: list[int] = []

    def one(args):
        rid, row = args
        limiter.wait()
        feats = {k: (float(v) if isinstance(v, float) else int(v)) for k, v in row.items()}
        try:
            r = client.post(
                f"{gateway}/v1/predict",
                json={"request_id": rid, "features": feats},
                headers={"X-API-Key": key},
                timeout=10,
            )
            return r.status_code
        except httpx.HTTPError:
            return 0

    with httpx.Client() as client, ThreadPoolExecutor(workers) as ex:
        statuses = list(ex.map(one, zip(ids, rows, strict=True)))
    return {
        "sent": len(statuses),
        "ok": sum(s == 200 for s in statuses),
        "non_200": len(statuses) - statuses.count(200),
    }


def send_labels(gateway: str, key: str, ids: list[str], labels: list[int], chunk: int = 500) -> dict:
    acc = dup = unknown = 0
    with httpx.Client() as client:
        for i in range(0, len(ids), chunk):
            body = {
                "labels": [
                    {"request_id": r, "default": int(y)}
                    for r, y in zip(ids[i : i + chunk], labels[i : i + chunk], strict=True)
                ]
            }
            r = client.post(f"{gateway}/v1/feedback", json=body, headers={"X-API-Key": key}, timeout=30)
            r.raise_for_status()
            b = r.json()
            acc, dup, unknown = (
                acc + b["accepted"],
                dup + b["duplicates"],
                unknown + len(b["unknown_request_ids"]),
            )
    return {"accepted": acc, "duplicates": dup, "unknown": unknown}

"""Probe traffic for G6/G7 and latency measurement. Probes carry `X-Probe: true` so slots never log them."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import httpx
import numpy as np
import pandas as pd

from fairxai.data.load import FEATURES


@dataclass
class ProbeResult:
    n: int
    statuses: list[int] = field(default_factory=list)
    latencies_s: list[float] = field(default_factory=list)

    @property
    def errors(self) -> int:
        return sum(1 for s in self.statuses if s >= 400 or s == 0)

    @property
    def rate_5xx(self) -> float:
        return sum(1 for s in self.statuses if s >= 500 or s == 0) / max(len(self.statuses), 1)

    @property
    def p95_s(self) -> float:
        return float(np.percentile(self.latencies_s, 95)) if self.latencies_s else float("inf")

    def summary(self) -> dict:
        return {"n": self.n, "errors": self.errors, "rate_5xx": self.rate_5xx, "p95_s": self.p95_s}


def feature_rows(df: pd.DataFrame, n: int) -> list[dict]:
    rows = df[FEATURES].head(n).to_dict(orient="records")
    return [{k: (float(v) if isinstance(v, float) else int(v)) for k, v in r.items()} for r in rows]


def send(
    url: str,
    api_key: str,
    rows: list[dict],
    probe: bool = True,
    rps: float | None = None,
    workers: int = 1,
    endpoint: str = "/v1/predict",
) -> ProbeResult:
    """POST each row; status 0 means a transport error. Optional rate limit in requests per second."""
    headers = {"X-API-Key": api_key}
    if probe:
        headers["X-Probe"] = "true"
    res = ProbeResult(n=len(rows))
    interval = 1.0 / rps if rps else 0.0

    def one(row: dict, client: httpx.Client) -> tuple[int, float]:
        t = time.perf_counter()
        try:
            r = client.post(f"{url}{endpoint}", json={"features": row}, headers=headers, timeout=10)
            return r.status_code, time.perf_counter() - t
        except httpx.HTTPError:
            return 0, time.perf_counter() - t

    with httpx.Client() as client:
        if workers == 1:
            next_t = time.perf_counter()
            for row in rows:
                if interval:
                    time.sleep(max(0.0, next_t - time.perf_counter()))
                    next_t += interval
                s, lat = one(row, client)
                res.statuses.append(s)
                res.latencies_s.append(lat)
        else:
            with ThreadPoolExecutor(workers) as ex:
                for s, lat in ex.map(lambda r: one(r, client), rows):
                    res.statuses.append(s)
                    res.latencies_s.append(lat)
    return res

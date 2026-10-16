"""Prometheus metrics of a serving slot (SPEC D.6)."""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram


class SlotMetrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        r = self.registry
        self.requests = Counter(
            "fairxai_requests_total", "Requests", ["endpoint", "status", "slot", "model_version"], registry=r
        )
        self.latency = Histogram(
            "fairxai_request_latency_seconds",
            "Latency",
            ["endpoint", "slot"],
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 2.0, 5.0),
            registry=r,
        )
        self.predictions = Counter(
            "fairxai_predictions_total", "Predictions", ["prediction", "model_version"], registry=r
        )
        self.model_loaded = Gauge(
            "fairxai_model_loaded", "1 when a model is loaded", ["model_version"], registry=r
        )
        self.load_failures = Counter("fairxai_model_load_failures_total", "Failed loads", registry=r)
        self.log_failures = Counter("fairxai_log_failures_total", "Failed prediction log writes", registry=r)
        self.load_failures.inc(0)
        self.log_failures.inc(0)

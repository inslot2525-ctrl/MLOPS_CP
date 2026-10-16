"""Serving slot (blue or green): FastAPI app implementing the slot contract of SPEC D.6."""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field

import pandas as pd
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import ValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from fairxai.config import Config, load_config
from fairxai.data.load import FEATURES
from fairxai.httpapi import ApiError, KeyRing, error_response, request_id, validation_details
from fairxai.logs import get_logger

from .auth import require
from .logging_db import PredictionLog
from .metrics import SlotMetrics
from .model_store import LoadedModel, ModelStore, UnknownVersion, mlflow_loader
from .schemas import FeedbackRequest, LoadRequest, build_request_models

log = get_logger("fairxai.serving")


@dataclass
class Settings:
    slot: str = field(default_factory=lambda: os.environ.get("SLOT_NAME", "blue"))
    api_keys: str = field(default_factory=lambda: os.environ.get("API_KEYS", ""))
    admin_sha256: str = field(default_factory=lambda: os.environ.get("ADMIN_API_KEY_SHA256", ""))
    db_url: str | None = field(default_factory=lambda: os.environ.get("FAIRXAI_DB_URL"))
    expected_version: str = field(default_factory=lambda: os.environ.get("EXPECTED_MODEL_VERSION", ""))
    image_tag: str = field(default_factory=lambda: os.environ.get("IMAGE_TAG", ""))


def create_app(settings: Settings | None = None, loader=None, cfg: Config | None = None, db=None) -> FastAPI:
    settings = settings or Settings()
    cfg = cfg or load_config()
    model_name = cfg.base.mlflow.registered_model
    keyring = KeyRing(settings.api_keys, settings.admin_sha256)
    store = ModelStore(loader or mlflow_loader(model_name, cfg.base.seed))
    plog = db or PredictionLog(settings.db_url)
    metrics = SlotMetrics()
    _, PredictRequest = build_request_models(cfg)
    max_body = cfg.serving.max_body_bytes

    app = FastAPI(title=f"fairxai-serving-{settings.slot}", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.store, app.state.metrics, app.state.settings = store, metrics, settings

    def try_load(version: int) -> None:
        try:
            m, _ = store.load(version)
            metrics.model_loaded.labels(str(m.version)).set(1)
        except Exception as exc:
            store.last_error = f"{type(exc).__name__}: {exc}"
            metrics.load_failures.inc()
            raise

    @app.on_event("startup")
    def _startup() -> None:
        if settings.expected_version.strip():
            try:
                try_load(int(settings.expected_version))
            except Exception:  # noqa: BLE001
                log.error("startup load of v%s failed: %s", settings.expected_version, store.last_error)

    @app.middleware("http")
    async def _envelope(request: Request, call_next):
        request.state.request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        length = request.headers.get("content-length")
        if length and length.isdigit() and int(length) > max_body:
            return error_response(request, 413, f"body larger than {max_body} bytes")
        body = await request.body()
        if len(body) > max_body:
            return error_response(request, 413, f"body larger than {max_body} bytes")
        start = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id(request)
        path = request.url.path
        if path.startswith(("/v1/", "/admin/")):
            ver = str(store.current.version) if store.current else "none"
            metrics.requests.labels(path, str(response.status_code), settings.slot, ver).inc()
            metrics.latency.labels(path, settings.slot).observe(time.perf_counter() - start)
        return response

    @app.exception_handler(ApiError)
    async def _api_error(request: Request, exc: ApiError):
        return error_response(request, exc.status, exc.message, exc.details, exc.code)

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError):
        return error_response(request, 422, "request validation failed", validation_details(exc.errors()))

    @app.exception_handler(StarletteHTTPException)
    async def _http(request: Request, exc: StarletteHTTPException):
        status = exc.status_code if exc.status_code in (404, 405) else 500
        return error_response(request, 404 if status == 405 else status, str(exc.detail))

    @app.exception_handler(Exception)
    async def _unexpected(request: Request, exc: Exception):
        log.error("internal error on %s: %s", request.url.path, type(exc).__name__)
        return error_response(request, 500, "internal error")

    async def parse(request: Request, model):
        try:
            raw = await request.body()
            data = json.loads(raw or b"null")
        except json.JSONDecodeError as exc:
            raise ApiError(422, "body is not valid JSON", [{"field": "body", "problem": str(exc)}]) from exc
        try:
            return model.model_validate(data)
        except ValidationError as exc:
            raise ApiError(422, "request validation failed", validation_details(exc.errors())) from exc

    def model_or_503() -> LoadedModel:
        if store.current is None:
            raise ApiError(503, "no model loaded in this slot")
        return store.current

    @app.post("/v1/predict")
    async def predict(request: Request):
        require(request, keyring, admin=False)
        req = await parse(request, PredictRequest)
        m = model_or_503()
        start = time.perf_counter()
        feats = req.features.model_dump()
        X = pd.DataFrame([feats])[FEATURES]
        prob = float(m.pipeline.predict_proba(X)[0, 1])
        pred = int(prob >= m.threshold)
        rid = str(req.request_id or request_id(request))
        latency_ms = (time.perf_counter() - start) * 1000
        metrics.predictions.labels(str(pred), str(m.version)).inc()
        headers = {}
        if request.headers.get("X-Probe", "").lower() != "true":
            try:
                plog.log_prediction(rid, m.version, settings.slot, feats, pred, prob, latency_ms)
            except Exception as exc:  # noqa: BLE001 - the prediction is still returned
                metrics.log_failures.inc()
                log.error("prediction log write failed: %s", type(exc).__name__)
                headers["X-Logged"] = "false"
        body = {
            "request_id": rid,
            "model_name": model_name,
            "model_version": m.version,
            "prediction": pred,
            "probability_default": prob,
            "decision_threshold": m.threshold,
        }
        return JSONResponse(body, headers=headers)

    @app.post("/v1/explain")
    async def explain(request: Request):
        require(request, keyring, admin=False)
        raw_k = request.query_params.get("top_k", str(cfg.serving.top_k_default))
        if not raw_k.isdigit() or not 1 <= int(raw_k) <= len(FEATURES):
            raise ApiError(
                422, "invalid top_k", [{"field": "top_k", "problem": f"must be 1-{len(FEATURES)}"}]
            )
        req = await parse(request, PredictRequest)
        m = model_or_503()
        if m.explainer is None:
            raise ApiError(500, "no explainer for the loaded model")
        X = pd.DataFrame([req.features.model_dump()])[FEATURES]
        prob = float(m.pipeline.predict_proba(X)[0, 1])
        try:
            exp = m.explainer.explain(X, int(raw_k))
        except Exception as exc:  # never fall back to another method silently
            log.error("explainer failed: %s", type(exc).__name__)
            raise ApiError(500, "explanation failed") from exc
        return {
            "request_id": str(req.request_id or request_id(request)),
            "model_version": m.version,
            "method": exp["method"],
            "output_space": exp["output_space"],
            "base_value": exp["base_value"],
            "probability_default": prob,
            "contributions": exp["contributions"],
        }

    @app.post("/v1/feedback")
    async def feedback(request: Request):
        require(request, keyring, admin=False)
        req = await parse(request, FeedbackRequest)
        accepted, duplicates, unknown = plog.add_labels([(str(x.request_id), x.default) for x in req.labels])
        return {"accepted": accepted, "duplicates": duplicates, "unknown_request_ids": unknown}

    @app.get("/v1/model")
    async def model_info(request: Request):
        require(request, keyring, admin=False)
        m = model_or_503()
        return {
            "model_name": model_name,
            "model_version": m.version,
            "run_id": m.run_id,
            "loaded_at": m.loaded_at,
            "decision_threshold": m.threshold,
            "git_commit": m.git_commit,
            "data_version_id": m.data_version_id,
            "slot": settings.slot,
        }

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz():
        if store.requested is None and settings.expected_version.strip():
            store.requested = int(settings.expected_version)
        ok, reason = store.ready()
        if ok:
            return {"ready": True, "model_version": store.current.version}
        return JSONResponse({"ready": False, "reason": reason}, status_code=503)

    @app.get("/metrics")
    async def prom():
        return Response(generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST)

    @app.post("/admin/load")
    async def admin_load(request: Request):
        require(request, keyring, admin=True)
        req = await parse(request, LoadRequest)
        previous = store.current.version if store.current else None
        try:
            try_load(req.model_version)
        except UnknownVersion as exc:
            raise ApiError(404, f"unknown model version {req.model_version}") from exc
        except Exception as exc:
            raise ApiError(500, f"model load failed; slot keeps v{previous}") from exc
        if previous is not None and previous != req.model_version:
            metrics.model_loaded.labels(str(previous)).set(0)
        return {"loaded": True, "model_version": req.model_version, "previous_version": previous}

    return app


def app_factory() -> FastAPI:  # uvicorn --factory serving.app.main:app_factory
    return create_app()

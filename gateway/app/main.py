"""Blue-green gateway (SPEC D.6): forwards /v1/* to the active slot; flips on admin request.

The active slot lives in Postgres `gateway_state` and survives restarts. No Docker socket.
"""

from __future__ import annotations

import os
import time
import uuid
from datetime import UTC, datetime

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, ConfigDict, ValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from fairxai.httpapi import ApiError, KeyRing, error_response, request_id, validation_details
from fairxai.logs import get_logger

log = get_logger("fairxai.gateway")
SLOTS = ("blue", "green")
FORWARD_HEADERS = ("x-api-key", "x-probe", "content-type", "x-request-id")


class FlipRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    slot: str
    reason: str = ""


class PgState:
    def __init__(self, url: str) -> None:
        self.url = url

    def get(self) -> tuple[str, str]:
        import psycopg

        with psycopg.connect(self.url, connect_timeout=3) as c:
            row = c.execute("SELECT active, updated_at FROM gateway_state WHERE id").fetchone()
        return (row[0], row[1].isoformat()) if row else ("blue", "")

    def set(self, slot: str) -> str:
        import psycopg

        with psycopg.connect(self.url, connect_timeout=3) as c:
            row = c.execute(
                "INSERT INTO gateway_state (id, active, updated_at) VALUES (true, %s, now()) "
                "ON CONFLICT (id) DO UPDATE SET active = EXCLUDED.active, updated_at = now() RETURNING updated_at",
                (slot,),
            ).fetchone()
            c.commit()
        return row[0].isoformat()


class MemoryState:
    """Test-only state store."""

    def __init__(self, active: str = "blue") -> None:
        self.active, self.updated = active, datetime.now(UTC).isoformat()

    def get(self) -> tuple[str, str]:
        return self.active, self.updated

    def set(self, slot: str) -> str:
        self.active, self.updated = slot, datetime.now(UTC).isoformat()
        return self.updated


def create_app(
    slot_urls: dict[str, str] | None = None,
    admin_sha256: str | None = None,
    state=None,
    transport: httpx.AsyncBaseTransport | None = None,
    api_keys: str | None = None,
) -> FastAPI:
    slot_urls = slot_urls or {
        "blue": os.environ.get("SLOT_BLUE_URL", "http://serving-blue:8000"),
        "green": os.environ.get("SLOT_GREEN_URL", "http://serving-green:8000"),
    }
    # client keys are known only so a client key on an admin route gets 403 rather than 401
    keyring = KeyRing(
        api_keys if api_keys is not None else os.environ.get("API_KEYS", ""),
        admin_sha256 if admin_sha256 is not None else os.environ.get("ADMIN_API_KEY_SHA256", ""),
    )
    state = state or PgState(os.environ["FAIRXAI_DB_URL"])
    reg = CollectorRegistry()
    g_active = Gauge("gateway_active_slot", "1 for the active slot", ["slot"], registry=reg)
    g_flips = Counter("gateway_flips_total", "Slot flips", registry=reg)
    g_errors = Counter("gateway_upstream_errors_total", "Upstream errors", ["slot", "code"], registry=reg)
    g_latency = Histogram("gateway_request_latency_seconds", "Gateway latency", registry=reg)
    g_flips.inc(0)

    app = FastAPI(title="fairxai-gateway", docs_url=None, redoc_url=None, openapi_url=None)
    client = httpx.AsyncClient(transport=transport) if transport else httpx.AsyncClient()

    def active() -> str:
        slot, _ = state.get()
        for s in SLOTS:
            g_active.labels(s).set(1 if s == slot else 0)
        return slot

    async def slot_ready(slot: str) -> tuple[bool, int | None]:
        try:
            r = await client.get(f"{slot_urls[slot]}/readyz", timeout=httpx.Timeout(2.0, connect=1.0))
            body = r.json()
            return r.status_code == 200, body.get("model_version")
        except (httpx.HTTPError, ValueError):
            return False, None

    @app.middleware("http")
    async def _rid(request: Request, call_next):
        request.state.request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        response = await call_next(request)
        response.headers.setdefault("X-Request-ID", request_id(request))
        return response

    @app.exception_handler(ApiError)
    async def _api_error(request: Request, exc: ApiError):
        return error_response(request, exc.status, exc.message, exc.details, exc.code)

    @app.exception_handler(StarletteHTTPException)
    async def _http(request: Request, exc: StarletteHTTPException):
        return error_response(request, 404, str(exc.detail))

    @app.exception_handler(Exception)
    async def _unexpected(request: Request, exc: Exception):
        log.error("gateway internal error: %s", type(exc).__name__)
        return error_response(request, 500, "internal error")

    def require_admin(request: Request) -> None:
        role = keyring.role(request.headers.get("X-API-Key"))
        if role is None:
            raise ApiError(401, "missing or invalid API key")
        if role != "admin":
            raise ApiError(403, "admin key required")

    @app.api_route("/v1/{path:path}", methods=["GET", "POST"])
    async def forward(path: str, request: Request):
        slot = active()
        read_timeout = 15.0 if path.startswith("explain") else 5.0
        headers = {k: v for k, v in request.headers.items() if k.lower() in FORWARD_HEADERS}
        headers["X-Request-ID"] = request_id(request)
        start = time.perf_counter()
        try:
            r = await client.request(
                request.method,
                f"{slot_urls[slot]}/v1/{path}",
                params=request.query_params,
                content=await request.body(),
                headers=headers,
                timeout=httpx.Timeout(read_timeout, connect=1.0),
            )
        except httpx.TimeoutException as exc:
            g_errors.labels(slot, "504").inc()
            raise ApiError(504, f"slot {slot} timed out") from exc
        except httpx.HTTPError as exc:
            g_errors.labels(slot, "502").inc()
            raise ApiError(502, f"slot {slot} unreachable") from exc
        finally:
            g_latency.observe(time.perf_counter() - start)
        if r.status_code >= 500:
            g_errors.labels(slot, str(r.status_code)).inc()
        passthrough = {
            k: v for k, v in r.headers.items() if k.lower() in ("x-request-id", "x-logged", "content-type")
        }
        return Response(r.content, status_code=r.status_code, headers=passthrough)

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz():
        slot = active()
        ok, version = await slot_ready(slot)
        if ok:
            return {"ready": True, "active": slot, "model_version": version}
        return JSONResponse(
            {"ready": False, "active": slot, "reason": f"slot {slot} not ready"}, status_code=503
        )

    @app.get("/metrics")
    async def prom():
        active()
        return Response(generate_latest(reg), media_type=CONTENT_TYPE_LATEST)

    @app.get("/admin/state")
    async def admin_state(request: Request):
        require_admin(request)
        slot, updated = state.get()
        slots = {}
        for s in SLOTS:
            ok, version = await slot_ready(s)
            slots[s] = {"url": slot_urls[s], "ready": ok, "model_version": version}
        return {"active": slot, "slots": slots, "updated_at": updated}

    @app.post("/admin/active")
    async def admin_active(request: Request):
        require_admin(request)
        try:
            req = FlipRequest.model_validate_json(await request.body() or b"{}")
        except ValidationError as exc:
            raise ApiError(422, "request validation failed", validation_details(exc.errors())) from exc
        if req.slot not in SLOTS:
            raise ApiError(
                422, "slot must be blue or green", [{"field": "slot", "problem": "must be blue|green"}]
            )
        ok, _ = await slot_ready(req.slot)
        if not ok:
            raise ApiError(409, f"slot {req.slot} is not ready")
        previous = active()
        ts = state.set(req.slot)
        if previous != req.slot:
            g_flips.inc()
        active()
        log.info("flip %s -> %s (%s)", previous, req.slot, req.reason)
        return {"active": req.slot, "previous": previous, "ts": ts}

    return app


def app_factory() -> FastAPI:  # uvicorn --factory gateway.app.main:app_factory
    return create_app()

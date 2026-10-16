"""Gateway tests: slots are real serving apps (synthetic fixture model) mounted via an ASGI transport."""

import httpx
import pytest
from fastapi.testclient import TestClient

from fairxai.httpapi import sha256_hex
from gateway.app.main import MemoryState, PgState, create_app

from .conftest import ADMIN_KEY, CLIENT_KEY, DB_URL, A, C


class Router(httpx.AsyncBaseTransport):
    """Routes http://blue / http://green to in-process ASGI slot apps; 'down' slots raise ConnectError."""

    def __init__(self, apps: dict) -> None:
        self.transports = {k: httpx.ASGITransport(app=v) for k, v in apps.items()}
        self.down: set[str] = set()

    async def handle_async_request(self, request):
        host = request.url.host
        if host in self.down:
            raise httpx.ConnectError("down", request=request)
        return await self.transports[host].handle_async_request(request)


@pytest.fixture
def stack(make_client):
    blue, _ = make_client()
    green, _ = make_client()
    router = Router({"blue": blue.app, "green": green.app})
    state = MemoryState("blue")
    gw = create_app(
        {"blue": "http://blue", "green": "http://green"},
        sha256_hex(ADMIN_KEY),
        state,
        router,
        api_keys=f"client:{sha256_hex(CLIENT_KEY)}",
    )
    return TestClient(gw), blue, green, router, state


def test_forward_and_ready(stack, row):
    gw, blue, _, _, _ = stack
    assert gw.get("/readyz").status_code == 503  # active slot has no model
    blue.post("/admin/load", json={"model_version": 1}, headers=A)
    assert gw.get("/readyz").json()["model_version"] == 1
    r = gw.post("/v1/predict", json={"features": row}, headers=C)
    assert r.status_code == 200 and r.json()["model_version"] == 1 and r.headers["X-Request-ID"]


def test_flip_refused_when_not_ready_then_allowed(stack, row):
    gw, blue, green, _, state = stack
    blue.post("/admin/load", json={"model_version": 1}, headers=A)
    r = gw.post("/admin/active", json={"slot": "green", "reason": "t"}, headers=A)
    assert r.status_code == 409 and r.json()["error"]["code"] == "slot_not_ready"
    assert state.active == "blue"
    green.post("/admin/load", json={"model_version": 2}, headers=A)
    r = gw.post("/admin/active", json={"slot": "green", "reason": "t"}, headers=A)
    assert r.status_code == 200 and r.json()["previous"] == "blue"
    assert gw.post("/v1/predict", json={"features": row}, headers=C).json()["model_version"] == 2
    assert "gateway_flips_total 1.0" in gw.get("/metrics").text


def test_admin_auth_on_gateway(stack):
    gw, *_ = stack
    assert gw.get("/admin/state").status_code == 401
    assert gw.get("/admin/state", headers=C).status_code == 403
    st = gw.get("/admin/state", headers=A).json()
    assert st["active"] == "blue" and set(st["slots"]) == {"blue", "green"}
    assert gw.post("/admin/active", json={"slot": "purple"}, headers=A).status_code == 422


def test_upstream_unavailable_502(stack, row):
    gw, _, _, router, _ = stack
    router.down.add("blue")
    r = gw.post("/v1/predict", json={"features": row}, headers=C)
    assert r.status_code == 502 and r.json()["error"]["code"] == "upstream_unavailable"
    assert "gateway_upstream_errors_total" in gw.get("/metrics").text


def test_slot_errors_pass_through(stack, row):
    gw, *_ = stack
    assert gw.post("/v1/predict", json={"features": row}).status_code == 401  # slot's 401 forwarded
    assert gw.post("/v1/predict", json={"features": row}, headers=C).status_code == 503


@pytest.mark.skipif(not DB_URL, reason="FAIRXAI_TEST_DB_URL not set")
def test_active_slot_survives_restart(make_client):
    from pathlib import Path

    import psycopg

    with psycopg.connect(DB_URL) as c:
        c.execute((Path(__file__).resolve().parents[2] / "infra/postgres/schema.sql").read_text())
        c.execute("UPDATE gateway_state SET active='blue'")
        c.commit()
    blue, _ = make_client()
    green, _ = make_client()
    green.post("/admin/load", json={"model_version": 2}, headers=A)
    router = Router({"blue": blue.app, "green": green.app})
    urls = {"blue": "http://blue", "green": "http://green"}
    gw1 = TestClient(create_app(urls, sha256_hex(ADMIN_KEY), PgState(DB_URL), router))
    assert gw1.post("/admin/active", json={"slot": "green"}, headers=A).status_code == 200
    gw2 = TestClient(create_app(urls, sha256_hex(ADMIN_KEY), PgState(DB_URL), router))  # "restart"
    assert gw2.get("/admin/state", headers=A).json()["active"] == "green"
    with psycopg.connect(DB_URL) as c:
        c.execute("UPDATE gateway_state SET active='blue'")
        c.commit()

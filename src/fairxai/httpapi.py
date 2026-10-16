"""Shared HTTP helpers for the serving slots and the gateway: key hashing/auth and the error body (SPEC D.6)."""

from __future__ import annotations

import hashlib
import hmac
import uuid

from fastapi import Request
from fastapi.responses import JSONResponse

STATUS_CODES = {
    401: "unauthorized",
    403: "forbidden",
    404: "not_found",
    409: "slot_not_ready",
    413: "payload_too_large",
    422: "validation_error",
    500: "internal_error",
    501: "not_implemented",
    502: "upstream_unavailable",
    503: "model_not_ready",
    504: "upstream_timeout",
}


def sha256_hex(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


class KeyRing:
    """Client keys from API_KEYS=client:sha256hex,...; admin from ADMIN_API_KEY_SHA256. Plaintext never stored."""

    def __init__(self, api_keys: str, admin_sha256: str) -> None:
        self.clients: dict[str, str] = {}
        for item in filter(None, (p.strip() for p in (api_keys or "").split(","))):
            name, _, digest = item.partition(":")
            self.clients[name] = digest.lower()
        self.admin = (admin_sha256 or "").lower()

    @staticmethod
    def _eq(a: str, b: str) -> bool:
        return bool(a) and bool(b) and hmac.compare_digest(a, b)

    def role(self, presented: str | None) -> str | None:
        """'admin', 'client' or None. Every comparison is constant time."""
        if not presented:
            return None
        digest = sha256_hex(presented)
        is_admin = self._eq(digest, self.admin)
        is_client = any(self._eq(digest, d) for d in self.clients.values())
        return "admin" if is_admin else "client" if is_client else None


def request_id(request: Request) -> str:
    rid = getattr(request.state, "request_id", None)
    if not rid:
        rid = str(uuid.uuid4())
        request.state.request_id = rid
    return rid


class ApiError(Exception):
    def __init__(
        self, status: int, message: str, details: list | None = None, code: str | None = None
    ) -> None:
        super().__init__(message)
        self.status, self.message, self.details = status, message, details or []
        self.code = code or STATUS_CODES.get(status, "error")


def error_response(request: Request, status: int, message: str, details=None, code=None) -> JSONResponse:
    rid = request_id(request)
    body = {
        "error": {
            "code": code or STATUS_CODES.get(status, "error"),
            "message": message,
            "request_id": rid,
            "details": details or [],
        }
    }
    return JSONResponse(body, status_code=status, headers={"X-Request-ID": rid})


def validation_details(errors) -> list[dict]:
    out = []
    for e in errors:
        loc = [str(x) for x in e.get("loc", []) if x != "body"]
        out.append({"field": ".".join(loc) or "body", "problem": e.get("msg", "invalid")})
    return out

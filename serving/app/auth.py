"""X-API-Key dependencies for client and admin routes."""

from __future__ import annotations

from fastapi import Request

from fairxai.httpapi import ApiError, KeyRing


def require(request: Request, keyring: KeyRing, admin: bool) -> str:
    role = keyring.role(request.headers.get("X-API-Key"))
    if role is None:
        raise ApiError(401, "missing or invalid API key")
    if admin and role != "admin":
        raise ApiError(403, "admin key required")
    return role

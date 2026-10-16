"""Postgres access for the `fairxai` database."""

from __future__ import annotations

import os
from contextlib import contextmanager

import psycopg


def db_url() -> str:
    url = os.environ.get("FAIRXAI_DB_URL")
    if not url:
        raise RuntimeError("FAIRXAI_DB_URL is not set")
    return url


@contextmanager
def connect():
    with psycopg.connect(db_url()) as conn:
        yield conn


def apply_schema(conn, schema_path: str = "infra/postgres/schema.sql") -> None:
    with open(schema_path, encoding="utf-8") as f:
        conn.execute(f.read())
    conn.commit()

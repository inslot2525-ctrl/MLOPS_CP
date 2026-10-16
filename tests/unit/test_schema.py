"""Schema tests run against a real Postgres when FAIRXAI_TEST_DB_URL is set; otherwise skipped (not faked)."""

import os
from pathlib import Path

import psycopg
import pytest

URL = os.environ.get("FAIRXAI_TEST_DB_URL")
SCHEMA = Path(__file__).resolve().parents[2] / "infra" / "postgres" / "schema.sql"

pytestmark = pytest.mark.skipif(not URL, reason="FAIRXAI_TEST_DB_URL not set")


def test_schema_applies_twice():
    sql = SCHEMA.read_text()
    with psycopg.connect(URL) as conn:
        conn.execute(sql)
        conn.execute(sql)
        conn.commit()


def test_running_index_rejects_second_row():
    sql = SCHEMA.read_text()
    with psycopg.connect(URL) as conn:
        conn.execute(sql)
        conn.execute("DELETE FROM retrain_events")
        conn.execute("INSERT INTO retrain_events (trigger, status) VALUES ('t', 'RUNNING')")
        with pytest.raises(psycopg.errors.UniqueViolation):
            conn.execute("INSERT INTO retrain_events (trigger, status) VALUES ('t', 'RUNNING')")
        conn.rollback()
        conn.execute("DELETE FROM retrain_events")
        conn.commit()

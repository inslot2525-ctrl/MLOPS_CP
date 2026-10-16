"""run_monitoring against real Postgres (FAIRXAI_TEST_DB_URL) with SYNTHETIC predictions."""

import json
import os
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import psycopg
import pytest

from fairxai.config import load_config
from fairxai.data.load import FEATURES
from fairxai.data.split import reference_stats
from fairxai.monitoring.run import ModelRef, history, run_monitoring

URL = os.environ.get("FAIRXAI_TEST_DB_URL")
pytestmark = pytest.mark.skipif(not URL, reason="FAIRXAI_TEST_DB_URL not set")
CFG = load_config("test")
SCHEMA = Path(__file__).resolve().parents[3] / "infra" / "postgres" / "schema.sql"
VERSION = 900_001  # synthetic test versions far from real registry numbers


@pytest.fixture
def conn():
    with psycopg.connect(URL) as c:
        c.execute(SCHEMA.read_text())
        clean(c)
        yield c
        clean(c)


def clean(c):
    for v in (VERSION, VERSION + 1):
        c.execute(
            "DELETE FROM labels WHERE request_id IN (SELECT request_id FROM predictions WHERE model_version=%s)",
            (v,),
        )
        c.execute("DELETE FROM predictions WHERE model_version=%s", (v,))
        c.execute("DELETE FROM monitoring_results WHERE model_version=%s", (v,))
    c.commit()


def insert(c, df, version, start, labelled: bool, shift: bool = False):
    df = df.copy()
    if shift:
        df["LIMIT_BAL"] = (df["LIMIT_BAL"] * 0.4).clip(lower=1)
        for col in ("PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"):
            df[col] = (df[col] + 2).clip(upper=8)
        df["BILL_AMT1"] = df["BILL_AMT1"] * 1.5
    rng = np.random.default_rng(0)
    for i, row in enumerate(df.itertuples(index=False)):
        rid = str(uuid.uuid4())
        feats = {f: (float(getattr(row, f))) for f in FEATURES}
        p = float(np.clip(0.15 + 0.5 * row.default + rng.normal(0, 0.2), 0.001, 0.999))
        c.execute(
            "INSERT INTO predictions (request_id, ts, model_version, slot, features, prediction, probability)"
            " VALUES (%s,%s,%s,'blue',%s,%s,%s)",
            (rid, start + timedelta(milliseconds=i), version, json.dumps(feats), int(p >= 0.5), p),
        )
        if labelled:
            c.execute("INSERT INTO labels (request_id, label) VALUES (%s,%s)", (rid, int(row.default)))
    c.commit()


@pytest.fixture(scope="module")
def ref():
    from make_synthetic import make_synthetic

    train = make_synthetic(3000, seed=5)
    return train, reference_stats(train, CFG), make_synthetic(4000, seed=6)


def model_ref(version=VERSION):
    return ModelRef(version, 0.5, 0.40, np.clip(np.random.default_rng(1).beta(2, 5, 2000), 0, 1))


def test_rows_for_every_signal_and_na_without_labels(conn, ref):
    train, stats, pool = ref
    t0 = datetime.now(UTC) - timedelta(hours=1)
    insert(conn, pool.iloc[:500], VERSION, t0, labelled=False)
    out = run_monitoring(conn, CFG, model_ref(), train, stats, engine="scipy", do_notify=False)
    rows = conn.execute(
        "SELECT kind, status, n_rows, n_labelled FROM monitoring_results WHERE model_version=%s", (VERSION,)
    ).fetchall()
    kinds = {r[0]: r for r in rows}
    assert set(kinds) == {"data_quality", "feature_drift", "prediction_drift", "performance", "fairness"}
    assert kinds["performance"][1] == "NA" and kinds["fairness"][1] == "NA"  # no labels
    assert kinds["performance"][3] == 0 and kinds["feature_drift"][2] == 500
    assert out["trigger"]["request"] is False


def test_same_window_rerun_is_idempotent(conn, ref):
    train, stats, pool = ref
    insert(conn, pool.iloc[:500], VERSION, datetime.now(UTC) - timedelta(hours=1), labelled=True)
    run_monitoring(conn, CFG, model_ref(), train, stats, engine="scipy", do_notify=False)
    run_monitoring(conn, CFG, model_ref(), train, stats, engine="scipy", do_notify=False)
    n = conn.execute("SELECT count(*) FROM monitoring_results WHERE model_version=%s", (VERSION,)).fetchone()[
        0
    ]
    assert n == 5


def test_three_shifted_windows_request_retrain_and_model_change_resets(conn, ref):
    train, stats, pool = ref
    t = datetime.now(UTC) - timedelta(hours=2)
    outs = []
    for k in range(3):
        insert(
            conn,
            pool.iloc[k * 500 : (k + 1) * 500],
            VERSION,
            t + timedelta(minutes=k),
            labelled=False,
            shift=True,
        )
        outs.append(run_monitoring(conn, CFG, model_ref(), train, stats, engine="scipy", do_notify=False))
    assert [o["signals"]["feature_drift"]["status"] for o in outs] == ["ALERT"] * 3
    assert [o["trigger"]["request"] for o in outs] == [False, False, True]  # drift alone twice: no request
    assert history(conn, "feature_drift", VERSION)[:3] == ["ALERT"] * 3
    # a new model version starts with an empty streak
    insert(conn, pool.iloc[1500:2000], VERSION + 1, t + timedelta(minutes=5), labelled=False, shift=True)
    out = run_monitoring(conn, CFG, model_ref(VERSION + 1), train, stats, engine="scipy", do_notify=False)
    assert out["trigger"]["drift_streak"] == 1 and out["trigger"]["request"] is False

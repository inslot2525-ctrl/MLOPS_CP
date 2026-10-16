import logging
import uuid

import pytest

from fairxai.data.load import FEATURES

from .conftest import ADMIN_KEY, CLIENT_KEY, A, C


def loaded(make_client, **kw):
    client, log = make_client(**kw)
    assert client.post("/admin/load", json={"model_version": 1}, headers=A).status_code == 200
    return client, log


def test_readyz_before_load_and_predict_503(make_client, row):
    client, _ = make_client()
    with client:
        r = client.get("/readyz")
        assert r.status_code == 503 and r.json()["ready"] is False
        r = client.post("/v1/predict", json={"features": row}, headers=C)
        assert r.status_code == 503 and r.json()["error"]["code"] == "model_not_ready"


def test_startup_loads_expected_version(make_client):
    client, _ = make_client(expected="2")
    with client:
        r = client.get("/readyz")
        assert r.status_code == 200 and r.json() == {"ready": True, "model_version": 2}


def test_predict_ok_logged_and_request_id(make_client, row):
    client, log = loaded(make_client)
    rid = str(uuid.uuid4())
    r = client.post("/v1/predict", json={"request_id": rid, "features": row}, headers=C)
    body = r.json()
    assert r.status_code == 200 and r.headers["X-Request-ID"]
    assert set(body) == {
        "request_id",
        "model_name",
        "model_version",
        "prediction",
        "probability_default",
        "decision_threshold",
    }
    assert body["prediction"] == int(body["probability_default"] >= body["decision_threshold"])
    assert rid in log.predictions
    # repeated request_id is not logged twice
    client.post("/v1/predict", json={"request_id": rid, "features": row}, headers=C)
    assert len(log.predictions) == 1


def test_probe_not_logged(make_client, row):
    client, log = loaded(make_client)
    r = client.post("/v1/predict", json={"features": row}, headers={**C, "X-Probe": "true"})
    assert r.status_code == 200 and not log.predictions


def test_log_failure_header(make_client, row):
    client, _ = loaded(make_client, fail_log=True)
    r = client.post("/v1/predict", json={"features": row}, headers=C)
    assert r.status_code == 200 and r.headers.get("X-Logged") == "false"
    assert "fairxai_log_failures_total 1.0" in client.get("/metrics").text


@pytest.mark.parametrize(
    ("field", "value", "ok"),
    [
        ("AGE", 18, True),
        ("AGE", 17, False),
        ("AGE", 100, True),
        ("AGE", 101, False),
        ("SEX", 1, True),
        ("SEX", 3, False),
        ("EDUCATION", 6, True),
        ("EDUCATION", 7, False),
        ("MARRIAGE", 0, True),
        ("MARRIAGE", 4, False),
        ("PAY_0", -2, True),
        ("PAY_0", -3, False),
        ("PAY_6", 9, True),
        ("PAY_6", 10, False),
        ("LIMIT_BAL", 1, True),
        ("LIMIT_BAL", 0, False),
        ("LIMIT_BAL", 2_000_000, True),
        ("LIMIT_BAL", 2_000_001, False),
        ("BILL_AMT1", -1_000_000, True),
        ("BILL_AMT1", -1_000_001, False),
        ("PAY_AMT3", 0, True),
        ("PAY_AMT3", -1, False),
        ("AGE", None, False),
        ("AGE", "abc", False),
    ],
)
def test_field_boundaries(make_client, row, field, value, ok):
    client, log = loaded(make_client)
    r = client.post("/v1/predict", json={"features": {**row, field: value}}, headers=C)
    if ok:
        assert r.status_code == 200
    else:
        assert r.status_code == 422
        err = r.json()["error"]
        assert err["code"] == "validation_error" and err["details"][0]["field"] == f"features.{field}"
        assert not log.predictions  # invalid requests are never logged


def test_missing_and_unknown_fields_rejected(make_client, row):
    client, _ = loaded(make_client)
    partial = {k: v for k, v in row.items() if k != "AGE"}
    assert client.post("/v1/predict", json={"features": partial}, headers=C).status_code == 422
    assert client.post("/v1/predict", json={"features": {**row, "X": 1}}, headers=C).status_code == 422
    assert client.post("/v1/predict", json={"features": row, "extra": 1}, headers=C).status_code == 422
    r = client.post("/v1/predict", content=b"{not json", headers={**C, "content-type": "application/json"})
    assert r.status_code == 422


def test_payload_too_large(make_client, row):
    client, _ = loaded(make_client)
    big = {"features": row, "pad": "x" * 70_000}
    r = client.post("/v1/predict", json=big, headers=C)
    assert r.status_code == 413 and r.json()["error"]["code"] == "payload_too_large"


def test_auth(make_client, row):
    client, _ = loaded(make_client)
    assert client.post("/v1/predict", json={"features": row}).status_code == 401
    assert (
        client.post("/v1/predict", json={"features": row}, headers={"X-API-Key": "wrong"}).status_code == 401
    )
    assert client.post("/v1/predict", json={"features": row}, headers=A).status_code == 200
    r = client.post("/admin/load", json={"model_version": 1}, headers=C)
    assert r.status_code == 403 and r.json()["error"]["code"] == "forbidden"
    assert client.post("/admin/load", json={"model_version": 1}).status_code == 401


def test_keys_absent_from_logs_and_bodies(make_client, row, caplog):
    client, _ = loaded(make_client)
    with caplog.at_level(logging.DEBUG):
        bodies = [
            client.post("/v1/predict", json={"features": row}, headers=C).text,
            client.post("/v1/predict", json={"features": row}, headers={"X-API-Key": "nope"}).text,
            client.post("/admin/load", json={"model_version": 99}, headers=A).text,
            client.get("/v1/model", headers=C).text,
        ]
    for text in [*bodies, caplog.text]:
        assert CLIENT_KEY not in text and ADMIN_KEY not in text


def test_unknown_route_and_version_404(make_client):
    client, _ = loaded(make_client)
    assert client.get("/nope").json()["error"]["code"] == "not_found"
    r = client.post("/admin/load", json={"model_version": 99}, headers=A)
    assert r.status_code == 404


def test_failed_load_keeps_previous_model(make_client, row):
    client, _ = loaded(make_client)
    r = client.post("/admin/load", json={"model_version": 3}, headers=A)  # broken artifact fixture
    assert r.status_code == 500 and r.json()["error"]["code"] == "internal_error"
    assert "Traceback" not in r.text
    assert client.post("/v1/predict", json={"features": row}, headers=C).json()["model_version"] == 1
    assert client.get("/readyz").status_code == 503  # requested v3 != loaded v1
    assert "fairxai_model_load_failures_total 1.0" in client.get("/metrics").text
    # loading a good version makes it ready again
    assert client.post("/admin/load", json={"model_version": 2}, headers=A).json()["previous_version"] == 1
    assert client.get("/readyz").json() == {"ready": True, "model_version": 2}


def test_model_info(make_client):
    client, _ = loaded(make_client)
    body = client.get("/v1/model", headers=C).json()
    assert body["model_version"] == 1 and body["slot"] == "blue" and body["run_id"] == "run1"


def test_explain_is_501_until_phase_7(make_client, row):
    client, _ = loaded(make_client)
    assert client.post("/v1/explain", json={"features": row}, headers=C).status_code == 501


def test_feedback_duplicates_and_unknown(make_client, row):
    client, _ = loaded(make_client)
    rid = str(uuid.uuid4())
    client.post("/v1/predict", json={"request_id": rid, "features": row}, headers=C)
    other = str(uuid.uuid4())
    r = client.post(
        "/v1/feedback",
        json={"labels": [{"request_id": rid, "default": 1}, {"request_id": other, "default": 0}]},
        headers=C,
    )
    assert r.json() == {"accepted": 1, "duplicates": 0, "unknown_request_ids": [other]}
    r = client.post("/v1/feedback", json={"labels": [{"request_id": rid, "default": 0}]}, headers=C)
    assert r.json()["duplicates"] == 1
    assert client.post("/v1/feedback", json={"labels": []}, headers=C).status_code == 422
    too_many = [{"request_id": str(uuid.uuid4()), "default": 0} for _ in range(1001)]
    assert client.post("/v1/feedback", json={"labels": too_many}, headers=C).status_code in (413, 422)


def test_metric_names_present_after_traffic(make_client, row):
    client, _ = loaded(make_client)
    client.post("/v1/predict", json={"features": row}, headers=C)
    text = client.get("/metrics").text
    for name in (
        "fairxai_requests_total",
        "fairxai_request_latency_seconds",
        "fairxai_predictions_total",
        "fairxai_model_loaded",
        "fairxai_model_load_failures_total",
        "fairxai_log_failures_total",
    ):
        assert name in text, name


def test_feature_order_is_contract(fixture_pipeline):
    assert [str(c) for c in fixture_pipeline.feature_names_in_] == FEATURES

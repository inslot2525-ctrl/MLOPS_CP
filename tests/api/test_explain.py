"""/v1/explain on tiny SYNTHETIC fixture models of every explainer family (N-23)."""

import math

import pytest
from fastapi.testclient import TestClient

from fairxai.config import load_config
from fairxai.data.load import FEATURES
from fairxai.explain.shap_local import LocalExplainer
from fairxai.httpapi import sha256_hex
from fairxai.models.zoo import make_pipeline
from serving.app.main import Settings, create_app
from serving.app.model_store import LoadedModel

from .conftest import ADMIN_KEY, CLIENT_KEY, A, C, MemoryLog

CFG = load_config("test")
FAMILIES = {
    "xgboost": ({"n_estimators": 20, "max_depth": 3}, "tree_shap", "log_odds"),
    "random_forest": ({"n_estimators": 15, "max_depth": 4}, "tree_shap", "probability"),
    "decision_tree": ({"max_depth": 4}, "tree_shap", "probability"),
    "logreg": ({}, "linear_shap", "log_odds"),
    "naive_bayes": ({}, "kernel_shap", "probability"),
}


@pytest.fixture(scope="module")
def data():
    from make_synthetic import make_synthetic

    return make_synthetic(500, seed=21)


def client_for(family: str, data, explainer_ok: bool = True) -> TestClient:
    params, _, _ = FAMILIES[family]
    X, y = data[FEATURES], data["default"]
    pipe = make_pipeline(family, params, y, CFG).fit(X, y)
    explainer = LocalExplainer(pipe, X.sample(150, random_state=0), 42)
    if not explainer_ok:

        def boom(*a, **k):
            raise RuntimeError("explainer exploded")

        explainer.explain = boom

    def loader(v):
        return LoadedModel(version=v, pipeline=pipe, threshold=0.5, explainer=explainer)

    settings = Settings("blue", f"client:{sha256_hex(CLIENT_KEY)}", sha256_hex(ADMIN_KEY), None, "")
    c = TestClient(create_app(settings, loader, CFG, MemoryLog()))
    assert c.post("/admin/load", json={"model_version": 1}, headers=A).status_code == 200
    return c


def payload(data, i=0):
    return {"features": {k: int(v) for k, v in data[FEATURES].iloc[i].items()}}


@pytest.mark.parametrize("family", list(FAMILIES))
def test_additivity_and_truthful_method(family, data):
    c = client_for(family, data)
    _, method, space = FAMILIES[family]
    for i in (0, 7):
        r = c.post("/v1/explain?top_k=23", json=payload(data, i), headers=C)
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["method"] == method and body["output_space"] == space
        assert len(body["contributions"]) == 23
        assert {x["feature"] for x in body["contributions"]} == set(FEATURES)
        p = body["probability_default"]
        target = math.log(p / (1 - p)) if space == "log_odds" else p
        total = body["base_value"] + sum(x["shap_value"] for x in body["contributions"])
        assert total == pytest.approx(target, abs=1e-4), (family, total, target)
        # sorted by absolute shap value
        vals = {x["feature"]: x["value"] for x in body["contributions"]}
        assert vals == payload(data, i)["features"] and isinstance(vals["PAY_0"], int)
        mags = [abs(x["shap_value"]) for x in body["contributions"]]
        assert mags == sorted(mags, reverse=True)
        # explanation probability equals the predict endpoint's
        pr = c.post("/v1/predict", json=payload(data, i), headers={**C, "X-Probe": "true"}).json()
        assert pr["probability_default"] == pytest.approx(p, abs=1e-12)


def test_top_k_bounds_and_default(data):
    c = client_for("xgboost", data)
    assert len(c.post("/v1/explain", json=payload(data), headers=C).json()["contributions"]) == 10
    assert len(c.post("/v1/explain?top_k=1", json=payload(data), headers=C).json()["contributions"]) == 1
    for bad in ("0", "24", "-1", "abc"):
        r = c.post(f"/v1/explain?top_k={bad}", json=payload(data), headers=C)
        assert r.status_code == 422 and r.json()["error"]["details"][0]["field"] == "top_k"


def test_explain_not_logged_and_validated(data):
    c = client_for("xgboost", data)
    # explain must not create a predictions row: feedback for its request_id is 'unknown'
    rid = "6f1c2f3e-0000-4000-8000-000000000001"
    c.post("/v1/explain", json={"request_id": rid, **payload(data)}, headers=C)
    fb = c.post("/v1/feedback", json={"labels": [{"request_id": rid, "default": 1}]}, headers=C).json()
    assert fb["unknown_request_ids"] == [rid]  # explain did not create a predictions row
    bad = payload(data)
    bad["features"]["AGE"] = 5
    assert c.post("/v1/explain", json=bad, headers=C).status_code == 422
    assert c.post("/v1/explain", json=payload(data)).status_code == 401


def test_explainer_failure_is_500_without_fallback(data):
    c = client_for("xgboost", data, explainer_ok=False)
    r = c.post("/v1/explain", json=payload(data), headers=C)
    assert r.status_code == 500 and r.json()["error"]["code"] == "internal_error"
    assert "exploded" not in r.text  # no internals leaked


def test_background_required_for_non_xgboost(data):
    X, y = data[FEATURES], data["default"]
    pipe = make_pipeline("logreg", {}, y, CFG).fit(X, y)
    with pytest.raises(ValueError, match="background"):
        LocalExplainer(pipe, None, 42)
    xgb = make_pipeline("xgboost", {"n_estimators": 5}, y, CFG).fit(X, y)
    assert LocalExplainer(xgb, None, 42).method == "tree_shap"

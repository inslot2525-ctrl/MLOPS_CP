import numpy as np
import pandas as pd
import pytest

from fairxai.config import Grid, load_config
from fairxai.data.load import FEATURES
from fairxai.features.pipeline import build_pipeline, remap_categories
from fairxai.models.imbalance import imbalance_kwargs
from fairxai.models.threshold import choose_threshold, f1_at
from fairxai.models.train import data_version_id, rank_baselines
from fairxai.models.zoo import make_estimator, make_pipeline

CFG = load_config()
GRID = Grid(start=0.05, stop=0.95, step=0.01)


def test_remap_matches_and_is_idempotent(synthetic_clean):
    a = synthetic_clean.head(5).copy()
    a["EDUCATION"] = [0, 5, 6, 4, 1]
    a["MARRIAGE"] = [0, 3, 1, 2, 0]
    out = remap_categories(a)
    assert list(out.EDUCATION) == [4, 4, 4, 4, 1]
    assert list(out.MARRIAGE) == [3, 3, 1, 2, 3]
    pd.testing.assert_frame_equal(remap_categories(out), out)


def test_pipeline_gives_same_output_for_remapped_values(synthetic_clean):
    X = synthetic_clean[FEATURES]
    y = synthetic_clean["default"]
    pipe = make_pipeline("logreg", {}, y, CFG).fit(X, y)
    a, b = X.head(3).copy(), X.head(3).copy()
    a["EDUCATION"], a["MARRIAGE"] = 5, 0
    b["EDUCATION"], b["MARRIAGE"] = 4, 3
    np.testing.assert_allclose(pipe.predict_proba(a), pipe.predict_proba(b))


def test_unseen_category_raises(synthetic_clean):
    X = synthetic_clean[FEATURES]
    y = synthetic_clean["default"]
    pipe = make_pipeline("random_forest", {"n_estimators": 10}, y, CFG).fit(X, y)
    bad = X.head(2).copy()
    bad["EDUCATION"] = 9
    with pytest.raises(ValueError):
        pipe.predict_proba(bad)


def test_exclude_protected_drops_sex_and_age(synthetic_clean):
    X, y = synthetic_clean[FEATURES], synthetic_clean["default"]
    est = make_estimator("logreg", {}, y, CFG)
    with_p = build_pipeline("logreg", est, True).fit(X, y)
    without = build_pipeline("logreg", est, False).fit(X, y)
    n_with = with_p.named_steps["encode"].transform(X.head(2)).shape[1]
    n_without = without.named_steps["encode"].transform(X.head(2)).shape[1]
    assert n_with - n_without == 2


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("logreg", {"class_weight": "balanced"}),
        ("decision_tree", {"class_weight": "balanced"}),
        ("random_forest", {"class_weight": "balanced"}),
        ("svm", {"class_weight": "balanced"}),
        ("knn", {}),
        ("naive_bayes", {}),
    ],
)
def test_imbalance_per_model(name, expected):
    assert imbalance_kwargs(name, [0, 0, 0, 1]) == expected


def test_xgboost_scale_pos_weight():
    assert imbalance_kwargs("xgboost", [0, 0, 0, 0, 0, 1, 1]) == {"scale_pos_weight": 2.5}


def test_smote_not_supported_in_production():
    with pytest.raises(ValueError):
        imbalance_kwargs("logreg", [0, 1], "smote")


def test_ranking_tiebreak_by_name():
    assert rank_baselines({"xgboost": 0.5, "knn": 0.5, "svm": 0.6, "logreg": 0.4}) == [
        "svm",
        "knn",
        "xgboost",
        "logreg",
    ]


def test_threshold_maximises_f1_and_ties_prefer_half():
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    t = choose_threshold(y, p, GRID)
    assert f1_at(y, p, t) == 1.0
    assert abs(t - 0.5) <= 0.2 and 0.31 <= t <= 0.7
    assert t == min((x for x in np.round(np.arange(0.31, 0.71, 0.01), 2)), key=lambda v: (abs(v - 0.5), v))


def test_threshold_depends_only_on_given_arrays():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    p = np.clip(y * 0.3 + rng.random(500) * 0.7, 0, 1)
    assert choose_threshold(y, p, GRID) == choose_threshold(y.copy(), p.copy(), GRID)


def test_data_version_id_stable_and_batch_sensitive():
    a = data_version_id("abc")
    assert a == data_version_id("abc")
    assert a != data_version_id("abc", [("b1", "h1")])
    assert data_version_id("abc", [("b1", "h1"), ("b2", "h2")]) == data_version_id(
        "abc", [("b2", "h2"), ("b1", "h1")]
    )


def test_g5_roundtrip_passes_for_real_pipeline(synthetic_clean):
    """Regression: the pipeline must expose feature_names_in_ equal to the serving schema."""
    from pathlib import Path

    from fairxai.models.train import g5_roundtrip

    X, y = synthetic_clean[FEATURES].head(500), synthetic_clean["default"].head(500)
    pipe = make_pipeline("xgboost", {"n_estimators": 10}, y, CFG).fit(X, y)
    assert [str(c) for c in pipe.feature_names_in_] == FEATURES
    ok, detail = g5_roundtrip(pipe, Path(__file__).resolve().parents[2] / "fixtures" / "smoke_set.csv")
    assert ok is True, detail

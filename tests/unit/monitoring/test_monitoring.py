import numpy as np
import pandas as pd
import pytest

from fairxai.config import load_config
from fairxai.data.load import FEATURES
from fairxai.data.split import reference_stats
from fairxai.models.evaluate import metrics_at
from fairxai.monitoring import performance as sig
from fairxai.monitoring.drift import DriftResult, FeatureDrift, compute_drift, parse_evidently
from fairxai.monitoring.triggers import evaluate, streak

CFG = load_config()
A, O, W, N = "ALERT", "OK", "WARN", "NA"


# ---------------- triggers ----------------
@pytest.mark.parametrize(
    ("hist", "expected"),
    [([A, A, A], 3), ([A, N, A, A], 3), ([A, O, A, A], 1), ([N, N], 0), ([O, A, A, A], 0), ([W, A], 0)],
)
def test_streak_skips_na_and_stops_at_non_alert(hist, expected):
    assert streak(hist) == expected


def test_drift_alone_never_requests_below_persistence():
    for hist in ([A], [A, A], [A, N, A], [A, A, O, A]):
        d = evaluate(hist, [], 3, 2)
        assert not d.request and d.warn_events  # WARN event, no retrain


def test_drift_persistence_requests():
    assert evaluate([A, A, A], [], 3, 2).trigger == "drift"
    assert evaluate([A, N, A, N, A], [], 3, 2).request  # NA neither extends nor resets


def test_performance_two_windows_requests():
    assert not evaluate([], [A], 3, 2).request
    assert evaluate([], [A, A], 3, 2).trigger == "performance"
    assert not evaluate([], [A, O, A], 3, 2).request


def test_scheduled():
    assert evaluate([], [], 3, 2, scheduled=True).trigger == "scheduled"


# ---------------- drift adapter ----------------
@pytest.fixture(scope="module")
def ref_cur():
    from make_synthetic import make_synthetic

    return make_synthetic(3000, seed=1), make_synthetic(500, seed=2)


def test_no_drift_on_same_distribution(ref_cur):
    ref, cur = ref_cur
    for engine in ("evidently", "scipy"):
        r = compute_drift(ref, cur, CFG, engine)
        assert isinstance(r, DriftResult) and set(r.features) == set(FEATURES)
        assert r.share_drifted < CFG.monitoring.dataset_drift_share, engine


def test_covariate_shift_detected_and_engines_agree(ref_cur):
    ref, cur = ref_cur
    shifted = cur.copy()
    shifted["LIMIT_BAL"] = (shifted["LIMIT_BAL"] * 0.4).clip(lower=1)
    shifted["BILL_AMT1"] = shifted["BILL_AMT1"] * 1.5
    for c in ("PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"):
        shifted[c] = (shifted[c] + 2).clip(upper=8)
    ev, sp = compute_drift(ref, shifted, CFG, "evidently"), compute_drift(ref, shifted, CFG, "scipy")
    assert ev.share_drifted >= CFG.monitoring.dataset_drift_share
    assert ev.drifted_features == sp.drifted_features
    for f in FEATURES:
        assert ev.features[f].test == sp.features[f].test
        assert ev.features[f].p_value == pytest.approx(sp.features[f].p_value, rel=1e-3, abs=1e-9), f


def test_remap_applied_before_comparison(ref_cur):
    ref, cur = ref_cur
    a = cur.copy()
    a.loc[a.EDUCATION == 4, "EDUCATION"] = 5  # 5 is remapped to 4 => no drift on EDUCATION
    r = compute_drift(ref, a, CFG, "scipy")
    assert not r.features["EDUCATION"].drifted


def test_parse_evidently_stub():
    stub = {
        "metrics": [
            {"metric_name": "DriftedColumnsCount(drift_share=0.5)", "value": {"count": 1}},
            {"metric_name": "ValueDrift(column=AGE,method=ks,threshold=0.05)", "value": 0.01},
            {"metric_name": "ValueDrift(column=SEX,method=chisquare,threshold=0.05)", "value": 0.5},
        ]
    }
    out = parse_evidently(stub, 0.05)
    assert out["AGE"] == FeatureDrift("ks", 0.01, True) and out["SEX"] == FeatureDrift(
        "chisquare", 0.5, False
    )


# ---------------- signal thresholds ----------------
def fake_drift(n_drifted: int, n_total: int = 10) -> DriftResult:
    feats = {
        f"f{i}": FeatureDrift("ks", 0.01 if i < n_drifted else 0.9, i < n_drifted) for i in range(n_total)
    }
    return DriftResult("stub", 1000, 500, feats)


def test_feature_drift_share_boundary():
    assert sig.feature_drift(fake_drift(3), 500, 0, CFG).status == A  # exactly 0.30
    assert sig.feature_drift(fake_drift(2), 500, 0, CFG).status == W  # single features drift => WARN
    assert sig.feature_drift(fake_drift(0), 500, 0, CFG).status == O
    assert sig.feature_drift(fake_drift(5), 299, 0, CFG).status == N
    assert sig.feature_drift(None, 300, 0, CFG).status == N


def window_frame(n: int, seed: int = 3) -> pd.DataFrame:
    from make_synthetic import make_synthetic

    df = make_synthetic(n, seed=seed)
    rng = np.random.default_rng(seed)
    df["probability"] = np.clip(0.15 + 0.5 * df["default"] + rng.normal(0, 0.2, n), 0.001, 0.999)
    df["label"] = df["default"]
    return df


def test_data_quality_min_rows_and_alert():
    stats = reference_stats(window_frame(3000, 9), CFG)
    assert sig.data_quality(window_frame(299), 0, CFG, stats).status == N
    assert sig.data_quality(window_frame(300), 0, CFG, stats).status in (O, W)
    bad = window_frame(300)
    bad.loc[:30, "SEX"] = 3
    assert sig.data_quality(bad, 0, CFG, stats).status == A


def test_prediction_drift_rate_change_boundary():
    w = window_frame(500)
    w["probability"] = np.r_[np.full(300, 0.9), np.full(200, 0.1)]  # current rate 0.6
    ref = np.r_[np.full(250, 0.9), np.full(250, 0.1)]  # rate 0.5 -> change exactly 0.10
    r = sig.pred_drift(w, ref, 0.5, 0, CFG)
    assert r.value == pytest.approx(0.1) and r.status == W
    same = sig.pred_drift(w, np.r_[np.full(300, 0.9), np.full(200, 0.1)], 0.5, 0, CFG)
    assert same.status == O
    assert sig.pred_drift(window_frame(299), ref, 0.5, 0, CFG).status == N


def test_performance_boundaries():
    lab = window_frame(500)
    pr = round(metrics_at(lab.label, lab.probability, 0.5)["pr_auc"], 6)
    assert sig.performance(lab, 500, 0.5, pr + 0.05, CFG).status == A  # exactly at reference - 0.05
    assert sig.performance(lab, 500, 0.5, pr + 0.049, CFG).status == O  # just above the limit
    assert sig.performance(lab, 500, 0.5, pr + 0.06, CFG).status == A  # below the limit
    assert sig.performance(lab.head(299), 500, 0.5, pr, CFG).status == N
    one = lab.head(400).assign(label=0)
    assert sig.performance(one, 500, 0.5, pr, CFG).status == N


def test_live_fairness_needs_labels():
    lab = window_frame(500)
    assert sig.live_fairness(lab.head(299), 500, 0.5, CFG).status == N
    assert sig.live_fairness(lab, 500, 0.5, CFG).status in (O, W, A)

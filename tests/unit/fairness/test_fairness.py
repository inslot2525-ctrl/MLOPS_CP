"""Hand-computed fixtures for every D.4 formula, eligibility and boundary rules."""

import numpy as np
import pandas as pd
import pytest

from fairxai.config import load_config
from fairxai.fairness.calibration import ece
from fairxai.fairness.groups import age_band
from fairxai.fairness.report import build_report

CFG = load_config()
BANDS = CFG.fairness.age_bands


def frame(groups: list[tuple[int, int, int, int, int]]) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """groups: (SEX, AGE, n_y0_pred0, n_y0_pred1, n_y1_pred0, n_y1_pred1) built row by row.

    Probabilities: predicted default (1) -> 0.9, predicted non-default (0) -> 0.1; threshold 0.5.
    """
    rows, ys, ps = [], [], []
    for sex, age, a, b, c, d in groups:
        for y, pred, n in ((0, 0, a), (0, 1, b), (1, 0, c), (1, 1, d)):
            for _ in range(n):
                rows.append({"SEX": sex, "AGE": age, "EDUCATION": 2, "MARRIAGE": 1})
                ys.append(y)
                ps.append(0.9 if pred else 0.1)
    return pd.DataFrame(rows), np.array(ys), np.array(ps)


def test_age_band_edges():
    s = age_band(pd.Series([18, 29, 30, 39, 40, 100, 17]), BANDS)
    assert list(s[:6]) == ["18-29", "18-29", "30-39", "30-39", "40+", "40+"]
    assert pd.isna(s[6])


def test_ece_hand_computed():
    # bin 0 (p=0.05): y mean 0.5 -> |0.5-0.05|=0.45 weight .5 ; bin 9 (p=0.95): y mean 1 -> 0.05 weight .5
    assert ece([0, 1, 1, 1], [0.05, 0.05, 0.95, 0.95], 10) == pytest.approx(0.5 * 0.45 + 0.5 * 0.05)
    assert ece([1], [1.0], 10) == pytest.approx(0.0)  # p = 1.0 falls in the last bin


def test_eo_di_hand_computed():
    # favorable = 0 (not default). Y=f rows are y==0.
    # SEX1: y0 -> 160 pred0, 40 pred1 ; y1 -> 20 pred0, 80 pred1  => TPR_f = 160/200 = .8, FR = 180/300 = .6
    # SEX2: y0 -> 140 pred0, 60 pred1 ; y1 -> 10 pred0, 90 pred1  => TPR_f = 140/200 = .7, FR = 150/300 = .5
    df, y, p = frame([(1, 35, 160, 40, 20, 80), (2, 35, 140, 60, 10, 90)])
    rep = build_report(df, y, p, 0.5, CFG)
    sex = rep["attributes"]["sex"]
    assert sex["groups"]["1"]["tpr"] == pytest.approx(0.8) and sex["groups"]["2"][
        "favorable_rate"
    ] == pytest.approx(0.5)
    assert rep["gates"]["eo_sex"]["value"] == pytest.approx(0.1)  # exactly at 0.10 -> PASS
    assert rep["gates"]["eo_sex"]["status"] == "PASS"
    assert rep["gates"]["di_sex"]["value"] == pytest.approx(round(0.5 / 0.6, 6))
    assert rep["gates"]["di_sex"]["status"] == "PASS"  # 0.833333 >= 0.80
    assert sex["metrics"]["dp"]["value"] == pytest.approx(0.1)


def test_eo_just_above_threshold_fails():
    df, y, p = frame([(1, 35, 160, 40, 20, 80), (2, 35, 139, 61, 10, 90)])  # TPR .8 vs .695
    rep = build_report(df, y, p, 0.5, CFG)
    assert rep["gates"]["eo_sex"]["value"] == pytest.approx(0.105)
    assert rep["gates"]["eo_sex"]["status"] == "FAIL"


def test_di_exactly_080_passes_and_below_fails():
    # FR: SEX1 = 250/300... build FR 0.5 vs 0.4 -> DI 0.8
    df, y, p = frame([(1, 35, 150, 50, 0, 100), (2, 35, 120, 80, 0, 100)])  # FR 150/300=.5, 120/300=.4
    rep = build_report(df, y, p, 0.5, CFG)
    assert (
        rep["gates"]["di_sex"]["value"] == pytest.approx(0.8) and rep["gates"]["di_sex"]["status"] == "PASS"
    )
    df, y, p = frame([(1, 35, 150, 50, 0, 100), (2, 35, 119, 81, 0, 100)])
    rep = build_report(df, y, p, 0.5, CFG)
    assert rep["gates"]["di_sex"]["status"] == "FAIL"


def test_intersectional_threshold_015():
    # six cells all identical except one with TPR lower by exactly 0.15 -> PASS; 0.16 -> FAIL
    base = [(s, a, 80, 20, 0, 100) for s in (1, 2) for a in (25, 35, 45)]  # TPR .8
    ok = [*base[:-1], (2, 45, 65, 35, 0, 100)]  # TPR .65
    rep = build_report(*frame(ok), 0.5, CFG)
    assert (
        rep["gates"]["eo_inter"]["value"] == pytest.approx(0.15)
        and rep["gates"]["eo_inter"]["status"] == "PASS"
    )
    bad = [*base[:-1], (2, 45, 64, 36, 0, 100)]
    assert build_report(*frame(bad), 0.5, CFG)["gates"]["eo_inter"]["status"] == "FAIL"


def test_group_of_99_rows_is_ineligible():
    df, y, p = frame([(1, 35, 160, 40, 20, 80), (2, 35, 60, 20, 9, 10)])  # SEX2 has 99 rows
    rep = build_report(df, y, p, 0.5, CFG)
    assert rep["attributes"]["sex"]["groups"]["2"]["eligible"]["rate"] is False
    assert rep["gates"]["eo_sex"]["status"] == "INDETERMINATE"  # one eligible group left
    assert rep["gates"]["di_sex"]["status"] == "INDETERMINATE"


def test_group_with_49_qualified_rows_ineligible_for_tpr_only():
    df, y, p = frame([(1, 35, 160, 40, 20, 80), (2, 35, 40, 9, 20, 81)])  # SEX2: 150 rows, 49 with Y=f
    rep = build_report(df, y, p, 0.5, CFG)
    g2 = rep["attributes"]["sex"]["groups"]["2"]
    assert g2["eligible"]["rate"] is True and g2["eligible"]["tpr"] is False
    assert rep["gates"]["eo_sex"]["status"] == "INDETERMINATE"
    assert rep["gates"]["di_sex"]["status"] in ("PASS", "FAIL")


def test_max_favorable_rate_zero_is_indeterminate():
    df, y, p = frame([(1, 35, 0, 100, 0, 100), (2, 35, 0, 100, 0, 100)])  # everyone predicted default
    rep = build_report(df, y, p, 0.5, CFG)
    assert rep["gates"]["di_sex"]["status"] == "INDETERMINATE"
    assert "max favorable rate is 0" in rep["gates"]["di_sex"]["reason"]


def test_all_same_class_group():
    # SEX2 has no Y=f rows at all -> its TPR is undefined -> excluded; EO INDETERMINATE
    df, y, p = frame([(1, 35, 160, 40, 20, 80), (2, 35, 0, 0, 50, 150)])
    rep = build_report(df, y, p, 0.5, CFG)
    assert rep["attributes"]["sex"]["groups"]["2"]["eligible"]["tpr"] is False
    assert rep["gates"]["eo_sex"]["status"] == "INDETERMINATE"


def test_nan_probabilities_are_dropped_and_counted():
    df, y, p = frame([(1, 35, 160, 40, 20, 80), (2, 35, 140, 60, 10, 90)])
    p = p.astype(float)
    p[:5] = np.nan
    rep = build_report(df, y, p, 0.5, CFG)
    assert rep["n_dropped"] == 5
    assert rep["attributes"]["sex"]["groups"]["1"]["n"] == 295


def test_flipping_favorable_label_changes_results_as_hand_computed():
    df, y, p = frame([(1, 35, 160, 40, 20, 80), (2, 35, 140, 60, 10, 90)])
    flipped = CFG.model_copy(deep=True)
    flipped.fairness.favorable_label = 1
    rep = build_report(df, y, p, 0.5, flipped)
    # f = 1: TPR_f = P(pred=1 | y=1): SEX1 80/100 = .8, SEX2 90/100 = .9 ; FR: SEX1 120/300=.4, SEX2 150/300=.5
    assert rep["gates"]["eo_sex"]["value"] == pytest.approx(0.1)
    assert rep["gates"]["di_sex"]["value"] == pytest.approx(0.8)


def test_advisory_dp_warning_respects_champion():
    df, y, p = frame([(1, 35, 160, 40, 20, 80), (2, 35, 120, 80, 10, 90)])  # FR .6 vs .433 -> DP .1667
    rep = build_report(df, y, p, 0.5, CFG)
    assert any(w["check"] == "dp_sex" for w in rep["warnings"])
    rep2 = build_report(df, y, p, 0.5, CFG, champion_report=rep)  # not worse than champion -> no warning
    assert not any(w["check"] == "dp_sex" for w in rep2["warnings"])


def test_missing_group_value_dropped():
    df, y, p = frame([(1, 35, 160, 40, 20, 80), (2, 35, 140, 60, 10, 90)])
    df.loc[0, "AGE"] = 10  # outside every band
    rep = build_report(df, y, p, 0.5, CFG)
    assert rep["n_dropped"] == 1
    assert rep["attributes"]["age_band"]["n_dropped"] == 1

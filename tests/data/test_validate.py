import numpy as np
import pytest

from fairxai.config import load_config
from fairxai.data.split import reference_stats, split_frame
from fairxai.data.validate import EXPECTATIONS_DIR, SUITES, build_suite, validate

CFG = load_config()


def test_valid_raw_passes(synthetic_30k):
    assert validate(synthetic_30k, "raw_taiwan", CFG).passed


def test_valid_batch_passes(synthetic_clean):
    assert validate(synthetic_clean.head(500), "batch", CFG).passed


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.drop(columns=["AGE"]),
        lambda d: d.assign(SEX=np.where(np.arange(len(d)) % 10 == 0, 3, d.SEX)),
        lambda d: d.assign(LIMIT_BAL=d.LIMIT_BAL.where(np.arange(len(d)) % 20 != 0)),
        lambda d: d.assign(AGE=d.AGE.where(np.arange(len(d)) != 0, 5)),
        lambda d: d.assign(PAY_0=d.PAY_0.where(np.arange(len(d)) != 0, 12)),
        lambda d: d.assign(BILL_AMT1=d.BILL_AMT1.where(np.arange(len(d)) != 0, 5_000_000)),
        lambda d: d.assign(PAY_AMT1=d.PAY_AMT1.where(np.arange(len(d)) != 0, -1)),
        lambda d: d.assign(default=d.default.where(np.arange(len(d)) != 0, 2)),
    ],
    ids=[
        "missing_col",
        "bad_category",
        "null",
        "age_low",
        "pay_high",
        "bill_high",
        "payamt_neg",
        "bad_target",
    ],
)
def test_invalid_batch_fails(synthetic_clean, mutate):
    assert not validate(mutate(synthetic_clean.head(500)), "batch", CFG).passed


def test_batch_too_small_fails(synthetic_clean):
    assert not validate(synthetic_clean.head(99), "batch", CFG).passed


def test_raw_wrong_rowcount_and_duplicate_id(synthetic_30k):
    assert not validate(synthetic_30k.head(29_999), "raw_taiwan", CFG).passed
    dup = synthetic_30k.copy()
    dup.loc[1, "ID"] = dup.loc[0, "ID"]
    assert not validate(dup, "raw_taiwan", CFG).passed


def test_unlabelled_batch_allows_missing_target(synthetic_clean):
    df = synthetic_clean.head(500).drop(columns=["default"])
    assert validate(df, "batch", CFG, labelled=False).passed
    assert not validate(df, "batch", CFG, labelled=True).passed


def test_default_rate_is_warning_only(synthetic_30k):
    df = synthetic_30k.copy()
    df["default"] = 0
    res = validate(df, "raw_taiwan", CFG)
    assert res.passed and res.warnings


def test_outlier_rule_warn_and_fail(synthetic_clean):
    parts = split_frame(synthetic_clean, CFG)
    ref = reference_stats(parts["train"], CFG)
    batch = parts["stream"].head(500).copy()
    assert not validate(batch, "batch", CFG, reference_stats=ref).warnings
    warn = batch.copy()
    warn.loc[warn.index[:60], "BILL_AMT1"] = 1_900_000  # 60/(500*13) ~ 0.9% -> below warn limit
    assert not validate(warn, "batch", CFG, reference_stats=ref).errors
    bad = batch.copy()
    for c in ("LIMIT_BAL", "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "PAY_AMT1"):
        bad[c] = 1_900_000
    res = validate(bad, "batch", CFG, reference_stats=ref)
    assert not res.passed and any("outlier" in e for e in res.errors)


def test_committed_suites_match_config():
    for name in SUITES:
        committed = (EXPECTATIONS_DIR / f"{name}.json").read_text(encoding="utf-8")
        built = build_suite(name, CFG).to_json_dict()
        import json

        assert json.loads(committed) == json.loads(json.dumps(built))

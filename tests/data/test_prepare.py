from pathlib import Path

import pandas as pd
import pytest

from fairxai.config import load_config
from fairxai.data.load import FEATURES, canonicalize
from fairxai.data.prepare import prepare

CFG = load_config()
RAW = Path(__file__).resolve().parents[2] / "data" / "raw" / "credit_default.csv"


def test_aliases_target_and_pay1(synthetic_30k):
    df = synthetic_30k.rename(columns={"default": "default payment next month", "PAY_0": "PAY_1"})
    assert "default" in canonicalize(df, CFG).columns
    assert "PAY_0" in canonicalize(df, CFG).columns


def test_x_columns_renamed_in_order(synthetic_30k):
    xs = synthetic_30k.drop(columns=["ID", "default"])
    xs.columns = [f"X{i}" for i in range(1, 24)]
    xs["ID"] = 1
    xs["Y"] = 0
    out = canonicalize(xs, CFG)
    assert list(out.columns) == ["ID", *FEATURES, "default"]


def test_column_mismatch_fails(synthetic_30k):
    with pytest.raises(ValueError, match="mismatch"):
        canonicalize(synthetic_30k.drop(columns=["AGE"]), CFG)


def test_prepare_drops_id_and_does_not_remap(synthetic_30k, tmp_path):
    raw = tmp_path / "raw.csv"
    df = synthetic_30k.copy()
    df.loc[0, "EDUCATION"] = 5
    df.to_csv(raw, index=False)
    clean = prepare(raw, tmp_path / "clean.csv", CFG)
    assert "ID" not in clean.columns and clean.loc[0, "EDUCATION"] == 5
    assert len(pd.read_csv(tmp_path / "clean.csv")) == len(df)


@pytest.mark.skipif(not RAW.exists(), reason="real raw file not present (DVC pull needed)")
def test_real_raw_contract():
    df = pd.read_csv(RAW)
    assert len(df) == 30000 and df.ID.is_unique

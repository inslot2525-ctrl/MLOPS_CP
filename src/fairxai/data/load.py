"""Canonical Taiwan credit-default column contract and raw-file loading (SPEC D.2)."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from fairxai.config import Config

PAY_COLS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_COLS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAYAMT_COLS = [f"PAY_AMT{i}" for i in range(1, 7)]
FEATURES = ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE", *PAY_COLS, *BILL_COLS, *PAYAMT_COLS]
CONTINUOUS = ["LIMIT_BAL", *BILL_COLS, *PAYAMT_COLS]
RAW_COLUMNS = ["ID", *FEATURES]  # + target


def column_range(col: str, cfg: Config) -> tuple[float | None, float | None]:
    """Hard valid range for a column, from data.yaml."""
    r = cfg.data.ranges
    if re.fullmatch(r"PAY_[0-6]", col):
        lo, hi = r["PAY"]
    elif col.startswith("BILL_AMT"):
        lo, hi = r["BILL_AMT"]
    elif col.startswith("PAY_AMT"):
        lo, hi = r["PAY_AMT"]
    else:
        lo, hi = r[col]
    return lo, hi


def canonicalize(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Apply aliases; after renaming the column set must match exactly or loading fails."""
    target = cfg.data.target
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    if all(re.fullmatch(r"X\d+", c) for c in df.columns[: len(FEATURES)]) and "X1" in df.columns:
        # headerless-style X1..X23 in file order: ID is absent, so map onto FEATURES
        xs = [c for c in df.columns if re.fullmatch(r"X\d+", c)]
        if len(xs) == len(FEATURES):
            df = df.rename(columns=dict(zip(xs, FEATURES, strict=True)))
    for alias in cfg.data.aliases.target:
        if alias in df.columns:
            df = df.rename(columns={alias: target})
    df = df.rename(columns={k: v for k, v in cfg.data.aliases.rename.items() if k in df.columns})
    expected = {*RAW_COLUMNS, target}
    if set(df.columns) != expected:
        missing, extra = sorted(expected - set(df.columns)), sorted(set(df.columns) - expected)
        raise ValueError(f"column set mismatch: missing={missing} extra={extra}")
    return df[[*RAW_COLUMNS, target]]


def read_raw(path: str | Path) -> pd.DataFrame:
    """Read either a Kaggle-style CSV (one header row) or the UCI .xls (two header rows)."""
    path = Path(path)
    if path.suffix.lower() in {".xls", ".xlsx"}:
        head = pd.read_excel(path, header=None, nrows=2)
        # UCI layout: row 0 = X1..X23,Y ; row 1 = real names
        header_row = 1 if str(head.iloc[0, 1]).startswith("X") else 0
        return pd.read_excel(path, header=header_row)
    return pd.read_csv(path)

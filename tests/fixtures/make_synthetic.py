"""SYNTHETIC data generator: seeded, schema-identical to the Taiwan data, never real measurements."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fairxai.data.load import BILL_COLS, PAY_COLS, PAYAMT_COLS


def make_synthetic(n: int = 2000, seed: int = 42, with_id: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pay = {c: rng.choice([-2, -1, 0, 1, 2, 3], size=n, p=[0.1, 0.2, 0.4, 0.15, 0.1, 0.05]) for c in PAY_COLS}
    df = pd.DataFrame(
        {
            "LIMIT_BAL": rng.integers(10_000, 500_000, n),
            "SEX": rng.choice([1, 2], n, p=[0.4, 0.6]),
            "EDUCATION": rng.choice([1, 2, 3, 4], n, p=[0.35, 0.45, 0.15, 0.05]),
            "MARRIAGE": rng.choice([1, 2, 3], n, p=[0.45, 0.5, 0.05]),
            "AGE": rng.integers(21, 70, n),
            **pay,
        }
    )
    for c in BILL_COLS:
        df[c] = rng.integers(0, 150_000, n)
    for c in PAYAMT_COLS:
        df[c] = rng.integers(0, 20_000, n)
    logit = -1.6 + 0.9 * (df.PAY_0 >= 1) + 0.5 * (df.PAY_2 >= 1) - df.LIMIT_BAL / 400_000
    df["default"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    if with_id:
        df.insert(0, "ID", np.arange(1, n + 1))
    return df

"""Print real dataset facts for review (SPEC Phase 2): contract check, age-band and SEX x band sizes."""

from __future__ import annotations

import sys

import pandas as pd

from fairxai.config import load_config
from fairxai.data.validate import validate


def age_band(age: pd.Series, bands: dict[str, list[int]]) -> pd.Series:
    out = pd.Series(index=age.index, dtype="object")
    for label, (lo, hi) in bands.items():
        out[(age >= lo) & (age <= hi)] = label
    return out


def main(path: str = "data/raw/credit_default.csv") -> None:
    cfg = load_config()
    df = pd.read_csv(path)
    res = validate(df, "raw_taiwan", cfg)
    print(f"rows={len(df)} cols={df.shape[1]} validation_passed={res.passed}")
    print("errors:", res.errors, "warnings:", res.warnings)
    t = cfg.data.target
    print(f"default_rate={df[t].mean():.4f}")
    print("AGE min/max:", int(df.AGE.min()), int(df.AGE.max()))
    for c in ("SEX", "EDUCATION", "MARRIAGE"):
        print(f"{c} counts:", df[c].value_counts().sort_index().to_dict())
    print("PAY_0 counts:", df.PAY_0.value_counts().sort_index().to_dict())
    df["age_band"] = age_band(df.AGE, cfg.fairness.age_bands)
    print("age_band sizes:", df.age_band.value_counts(dropna=False).to_dict())
    cells = df.groupby(["SEX", "age_band"]).agg(n=(t, "size"), default_rate=(t, "mean")).round(4)
    print("SEX x age_band:\n" + cells.to_string())


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))

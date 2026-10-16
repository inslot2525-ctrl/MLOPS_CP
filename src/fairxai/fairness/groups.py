"""Group definitions (SPEC D.4): SEX, age_band, SEX x age_band, plus report-only MARRIAGE and EDUCATION."""

from __future__ import annotations

import pandas as pd

from fairxai.config import Config
from fairxai.features.pipeline import remap_categories

GATED = ("sex", "age_band")
REPORT_ONLY = ("marriage", "education")
INTERSECTIONAL = "sex_x_age_band"


def age_band(age: pd.Series, bands: dict[str, list[int]]) -> pd.Series:
    out = pd.Series(pd.NA, index=age.index, dtype="object")
    for label, (lo, hi) in bands.items():
        out[(age >= lo) & (age <= hi)] = label
    return out


def group_frame(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """One column per attribute; values are group labels (str) or NA when the group value is missing."""
    cats = remap_categories(df[["EDUCATION", "MARRIAGE"]]) if {"EDUCATION", "MARRIAGE"} <= set(df) else None
    out = pd.DataFrame(index=df.index)
    out["sex"] = df["SEX"].astype("Int64").astype("string")
    out["age_band"] = age_band(df["AGE"], cfg.fairness.age_bands).astype("string")
    out[INTERSECTIONAL] = out["sex"] + "|" + out["age_band"]
    if cats is not None:
        out["marriage"] = cats["MARRIAGE"].astype("Int64").astype("string")
        out["education"] = cats["EDUCATION"].astype("Int64").astype("string")
    return out

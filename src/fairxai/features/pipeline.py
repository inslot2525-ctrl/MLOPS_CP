"""Saved sklearn Pipeline: category remap (first step), encoding and scaling (SPEC D.2)."""

from __future__ import annotations

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from fairxai.data.load import BILL_COLS, FEATURES, PAY_COLS, PAYAMT_COLS

SCALED_MODELS = {"logreg", "knn", "svm"}
NUMERIC = ["LIMIT_BAL", "AGE", *PAY_COLS, *BILL_COLS, *PAYAMT_COLS]
EDUCATION_CATEGORIES = [1, 2, 3, 4]
MARRIAGE_CATEGORIES = [1, 2, 3]


class CategoryRemap(BaseEstimator, TransformerMixin):
    """EDUCATION {0,5,6} -> 4 and MARRIAGE 0 -> 3. Idempotent; shared by training, serving and drift."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["EDUCATION"] = X["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
        X["MARRIAGE"] = X["MARRIAGE"].replace({0: 3})
        return X


def remap_categories(df):
    return CategoryRemap().transform(df)


def build_pipeline(model_name: str, estimator, include_protected: bool = True) -> Pipeline:
    numeric = NUMERIC if include_protected else [c for c in NUMERIC if c != "AGE"]
    numeric_step = StandardScaler() if model_name in SCALED_MODELS else "passthrough"
    transformers = [
        (
            "edu",
            OneHotEncoder(categories=[EDUCATION_CATEGORIES], handle_unknown="error", sparse_output=False),
            ["EDUCATION"],
        ),
        (
            "mar",
            OneHotEncoder(categories=[MARRIAGE_CATEGORIES], handle_unknown="error", sparse_output=False),
            ["MARRIAGE"],
        ),
        ("num", numeric_step, numeric),
    ]
    if include_protected:
        transformers.insert(0, ("sex", OrdinalEncoder(categories=[[1, 2]], handle_unknown="error"), ["SEX"]))
    return Pipeline(
        [
            ("remap", CategoryRemap()),
            ("encode", ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.0)),
            ("model", estimator),
        ]
    )


def feature_columns() -> list[str]:
    return list(FEATURES)

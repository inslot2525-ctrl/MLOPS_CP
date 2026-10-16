"""The seven model families and their estimator factories."""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from fairxai.config import Config
from fairxai.features.pipeline import build_pipeline
from fairxai.models.imbalance import imbalance_kwargs

_CLASSES = {
    "logreg": LogisticRegression,
    "decision_tree": DecisionTreeClassifier,
    "random_forest": RandomForestClassifier,
    "xgboost": XGBClassifier,
    "knn": KNeighborsClassifier,
    "naive_bayes": GaussianNB,
    "svm": SVC,
}
_SEEDED = {"logreg", "decision_tree", "random_forest", "svm"}


def make_estimator(name: str, params: dict, y, cfg: Config):
    """Estimator with fixed seed and single-thread fits (parallelism is across CV folds)."""
    kw = dict(params)
    kw.update(imbalance_kwargs(name, y, cfg.training.imbalance_strategy))
    seed = cfg.base.seed
    if name in _SEEDED:
        kw["random_state"] = seed
    if name == "random_forest":
        kw["n_jobs"] = 1
    if name == "xgboost":
        kw.update(random_state=seed, n_jobs=1, eval_metric="logloss")
    if name == "svm":
        kw["probability"] = True
    return _CLASSES[name](**kw)


def make_pipeline(name: str, params: dict, y, cfg: Config) -> Pipeline:
    return build_pipeline(name, make_estimator(name, params, y, cfg), cfg.training.include_protected)

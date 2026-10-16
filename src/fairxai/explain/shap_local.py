"""Per-request explanations (SPEC D.6 /v1/explain, N-23).

SHAP values are computed on the encoded features and summed back onto the 23 raw request features
(one-hot columns of EDUCATION/MARRIAGE sum to their source column), which preserves additivity:
base_value + sum(contributions) == model output in `output_space`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from fairxai.data.load import FEATURES
from fairxai.explain.shap_global import default_class_values, explainer_method, make_explainer, split_pipeline

KERNEL_NSAMPLES = 500


def raw_feature_map(pipe: Pipeline) -> list[str]:
    """Source raw feature for every encoded output column, in output order."""
    ct = pipe.named_steps["encode"]
    out: list[str | None] = [None] * sum(s.stop - s.start for s in ct.output_indices_.values())
    for name, _, cols in ct.transformers_:
        if name == "remainder" or name not in ct.output_indices_:
            continue
        sl = ct.output_indices_[name]
        width = sl.stop - sl.start
        cols = list(cols)
        if len(cols) == 1:
            sources = cols * width
        elif len(cols) == width:
            sources = cols
        else:
            raise ValueError(f"cannot map transformer {name}: {len(cols)} inputs -> {width} outputs")
        out[sl] = sources
    if any(s is None for s in out):
        raise ValueError("unmapped encoded column")
    return out  # type: ignore[return-value]


class LocalExplainer:
    """Built once at model load; a failure here fails the load (the slot keeps its previous model)."""

    def __init__(self, pipe: Pipeline, background: pd.DataFrame | None, seed: int) -> None:
        self.pipe = pipe
        self.pre, self.est, _ = split_pipeline(pipe)
        self.method, self.output_space = explainer_method(self.est)
        self.explainer, _ = make_explainer(pipe, background, seed)  # raises if a background is required
        self.sources = raw_feature_map(pipe)
        self.seed = seed

    def _base(self) -> float:
        b = np.asarray(self.explainer.expected_value, dtype=float).ravel()
        return float(b[-1])  # class 1 (default) when the explainer returns one value per class

    def model_output(self, xs: np.ndarray) -> float:
        if self.output_space == "log_odds":
            if hasattr(self.est, "get_booster"):
                return float(self.est.predict(xs, output_margin=True)[0])
            return float(self.est.decision_function(xs)[0])
        return float(self.est.predict_proba(xs)[0, 1])

    def explain(self, row: pd.DataFrame, top_k: int) -> dict:
        xs = self.pre.transform(row[FEATURES])
        if self.method == "kernel_shap":
            np.random.seed(self.seed)
            vals = self.explainer.shap_values(xs, nsamples=KERNEL_NSAMPLES, silent=True)
        else:
            vals = self.explainer.shap_values(xs)
        v = default_class_values(vals)[0]
        per_feature = dict.fromkeys(FEATURES, 0.0)
        for src, s in zip(self.sources, v, strict=True):
            per_feature[src] += float(s)
        raw = row[FEATURES].to_dict(orient="records")[0]  # keeps per-column dtypes (no int -> float upcast)
        contribs = [
            {
                "feature": f,
                "value": raw[f].item() if hasattr(raw[f], "item") else raw[f],
                "shap_value": per_feature[f],
            }
            for f in FEATURES
        ]
        contribs.sort(key=lambda c: (-abs(c["shap_value"]), c["feature"]))
        return {
            "method": self.method,
            "output_space": self.output_space,
            "base_value": self._base(),
            "model_output": self.model_output(xs),
            "contributions": contribs[:top_k],
            "sum_all_contributions": float(sum(per_feature.values())),
        }

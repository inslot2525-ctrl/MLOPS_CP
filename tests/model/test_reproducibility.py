"""Two runs on SYNTHETIC data under profile `test` must agree within 1e-6 (N-27)."""

import json

import pytest

from fairxai.config import load_config
from fairxai.models.train import run_training


def _flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}.{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out


@pytest.mark.slow
def test_two_runs_identical(tmp_path):
    from make_synthetic import make_synthetic

    cfg = load_config("test")
    tr, ev = make_synthetic(1500, seed=1), make_synthetic(600, seed=2)
    a = run_training(cfg, tr, ev, tmp_path / "a", "synthetic-md5")["deterministic"]
    b = run_training(cfg, tr, ev, tmp_path / "b", "synthetic-md5")["deterministic"]
    fa, fb = _flatten(a), _flatten(b)
    assert fa.keys() == fb.keys()
    for k in fa:
        if isinstance(fa[k], float):
            assert fa[k] == pytest.approx(fb[k], abs=1e-6), k
        else:
            assert json.dumps(fa[k]) == json.dumps(fb[k]), k
    assert a["selected_pending_gate"] == b["selected_pending_gate"]
    assert len(a["shortlist"]) == 3 and len(a["baseline_ranking"]) == 7

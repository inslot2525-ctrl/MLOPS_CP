"""Selection loop (D.3 step 5): the first APPROVED model in rank order becomes the candidate."""

import pytest

import fairxai.models.train as train_mod
from fairxai.config import load_config
from fairxai.registry.promotion import GateDecision


@pytest.mark.slow
@pytest.mark.parametrize(
    ("decisions", "expected_index", "expected_decision"),
    [
        (["REJECTED", "APPROVED", "APPROVED"], 1, "APPROVED"),
        (["APPROVED", "REJECTED", "REJECTED"], 0, "APPROVED"),
        (["REJECTED", "INDETERMINATE", "REJECTED"], 0, "REJECTED"),  # none approved -> top-ranked registered
    ],
)
def test_first_approved_in_rank_order(monkeypatch, tmp_path, decisions, expected_index, expected_decision):
    from make_synthetic import make_synthetic

    calls = iter(decisions)
    monkeypatch.setattr(train_mod, "decide", lambda *a, **k: GateDecision(next(calls)))
    cfg = load_config("test")
    res = train_mod.run_training(
        cfg, make_synthetic(800, 1), make_synthetic(400, 2), tmp_path, "x", shap_rows=0
    )
    det = res["deterministic"]
    order = [t["model"] for t in det["tuned"]]
    assert det["selected"]["model"] == order[expected_index]
    assert det["selected"]["decision"] == expected_decision
    # the loop stops at the first APPROVED one
    assert len(det["selection"]) == (expected_index + 1 if expected_decision == "APPROVED" else 3)

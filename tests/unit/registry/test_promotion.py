import itertools
import math
import re
from pathlib import Path

import pytest

from fairxai.config import load_config
from fairxai.registry.promotion import (
    APPROVED,
    DECISION_INDETERMINATE,
    FAIL,
    INDETERMINATE,
    PASS,
    REJECTED,
    WARN,
    Candidate,
    Evaluation,
    aggregate,
    decide,
    gate_g2,
    gate_g3,
)

CFG = load_config()


def fair(status: str = PASS, warnings=()) -> dict:
    return {
        "gates": {
            k: {"status": status} for k in ("eo_sex", "di_sex", "eo_age", "di_age", "eo_inter", "di_inter")
        },
        "warnings": list(warnings),
    }


def cand(pr=0.5, n=4500, classes=2, report=None, g1=True, g5=True) -> Candidate:
    return Candidate(Evaluation(pr, n, classes, report if report is not None else fair()), g1, g5)


def test_g2_floor_boundary():
    assert gate_g2(Evaluation(0.45, 4500, 2), None, CFG)["status"] == PASS
    assert gate_g2(Evaluation(0.449999, 4500, 2), None, CFG)["status"] == FAIL


def test_g2_champion_margin_boundary():
    champ = Evaluation(0.50, 4500, 2)
    assert gate_g2(Evaluation(0.51, 4500, 2), champ, CFG)["status"] == PASS
    assert gate_g2(Evaluation(0.509999, 4500, 2), champ, CFG)["status"] == FAIL


@pytest.mark.parametrize(
    "ev",
    [Evaluation(0.6, 99, 2), Evaluation(0.6, 4500, 1), Evaluation(math.nan, 4500, 2)],
    ids=["few_rows", "one_class", "nan"],
)
def test_g2_indeterminate(ev):
    assert gate_g2(ev, None, CFG)["status"] == INDETERMINATE


def test_g3_statuses():
    assert gate_g3(fair(PASS))["status"] == PASS
    r = fair(PASS)
    r["gates"]["eo_age"]["status"] = INDETERMINATE
    assert gate_g3(r)["status"] == INDETERMINATE
    r["gates"]["di_sex"]["status"] = FAIL
    assert gate_g3(r)["status"] == FAIL
    assert gate_g3(None)["status"] == INDETERMINATE


@pytest.mark.parametrize("statuses", list(itertools.product([PASS, FAIL, INDETERMINATE], repeat=4)))
def test_aggregation_truth_table(statuses):
    gates = {g: {"status": s} for g, s in zip(("G1", "G2", "G3", "G5"), statuses, strict=True)}
    gates["G4"] = {"status": WARN}
    expected = (
        REJECTED if FAIL in statuses else DECISION_INDETERMINATE if INDETERMINATE in statuses else APPROVED
    )
    assert aggregate(gates) == expected


def test_fail_plus_indeterminate_is_rejected():
    r = fair(PASS)
    r["gates"]["eo_sex"]["status"] = INDETERMINATE
    d = decide(cand(pr=0.40, report=r), None, CFG)  # G2 FAIL + G3 INDETERMINATE
    assert d.decision == REJECTED


def test_better_pr_auc_but_unfair_is_rejected():
    r = fair(PASS)
    r["gates"]["eo_sex"]["status"] = FAIL
    d = decide(cand(pr=0.70, report=r), Evaluation(0.50, 4500, 2), CFG)
    assert d.gates["G2"]["status"] == PASS and d.decision == REJECTED


def test_g4_warn_never_blocks():
    d = decide(cand(report=fair(PASS, warnings=[{"check": "dp_sex"}])), None, CFG)
    assert d.gates["G4"]["status"] == WARN and d.decision == APPROVED and d.warnings


@pytest.mark.parametrize(
    ("g1", "g5", "expected"),
    [
        (False, True, REJECTED),
        (True, False, REJECTED),
        (True, None, DECISION_INDETERMINATE),
        (True, True, APPROVED),
    ],
)
def test_g1_and_g5(g1, g5, expected):
    assert decide(cand(g1=g1, g5=g5), None, CFG).decision == expected


def test_digest_is_deterministic_and_input_sensitive():
    a, b = decide(cand(), None, CFG), decide(cand(), None, CFG)
    assert a.inputs_digest == b.inputs_digest
    assert decide(cand(pr=0.51), None, CFG).inputs_digest != a.inputs_digest


SRC = Path(__file__).resolve().parents[3] / "src" / "fairxai"
LITERALS = ("0.10", "0.15", "0.80", "0.45", "0.01", "0.05", "500", "300")


def test_no_duplicate_thresholds():
    """Q-08: these threshold literals must live only in configs/."""
    files = [*sorted((SRC / "fairness").glob("*.py")), SRC / "registry" / "promotion.py"]
    files += [p for p in (SRC / "monitoring" / "triggers.py",) if p.exists()]
    files += sorted((SRC / "retrain").glob("*.py")) if (SRC / "retrain").exists() else []
    for f in files:
        text = f.read_text(encoding="utf-8")
        for lit in LITERALS:
            pattern = rf"(?<![\w.]){re.escape(lit)}(?![\w])" if "." in lit else rf"(?<![\w.]){lit}(?![\w.])"
            assert not re.search(pattern, text), f"{lit} hard-coded in {f.name}"

"""Deterministic promotion decision over G1-G5 (SPEC D10, D.4). Thresholds come only from configs/."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field

from fairxai.config import Config

PASS, FAIL, INDETERMINATE, WARN = "PASS", "FAIL", "INDETERMINATE", "WARN"
APPROVED, REJECTED = "APPROVED", "REJECTED"
DECISION_INDETERMINATE = "INDETERMINATE"
MANDATORY = ("G1", "G2", "G3", "G5")


@dataclass
class Evaluation:
    """What decide() needs about one model scored on the evaluation set."""

    pr_auc: float
    n_eval: int
    n_classes: int
    fairness_report: dict | None = None


@dataclass
class Candidate:
    evaluation: Evaluation
    g1_passed: bool
    g5_passed: bool | None
    g1_detail: str = ""
    g5_detail: str = ""


@dataclass
class GateDecision:
    decision: str
    gates: dict[str, dict] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    warnings: list = field(default_factory=list)
    inputs_digest: str = ""

    def to_dict(self) -> dict:
        return {
            "decision": self.decision,
            "gates": self.gates,
            "reasons": self.reasons,
            "warnings": self.warnings,
            "inputs_digest": self.inputs_digest,
        }


def _r(x: float, cfg: Config) -> float:
    return round(float(x), cfg.fairness.decimals)


def gate_g2(cand: Evaluation, champ: Evaluation | None, cfg: Config) -> dict:
    g = cfg.gates
    for ev, who in ((cand, "candidate"), (champ, "champion")):
        if ev is None:
            continue
        if ev.n_eval < g.min_eval_rows:
            return {"status": INDETERMINATE, "detail": f"evaluation set has {ev.n_eval} rows"}
        if ev.n_classes < 2:
            return {"status": INDETERMINATE, "detail": "evaluation set has one class"}
        if ev.pr_auc is None or math.isnan(ev.pr_auc):
            return {"status": INDETERMINATE, "detail": f"{who} PR-AUC is NaN"}
    if champ is None:
        ok = _r(cand.pr_auc, cfg) >= _r(g.pr_auc_floor, cfg)
        return {
            "status": PASS if ok else FAIL,
            "detail": f"PR-AUC {cand.pr_auc:.6f} vs floor {g.pr_auc_floor}",
        }
    need = _r(champ.pr_auc + g.pr_auc_margin, cfg)
    ok = _r(cand.pr_auc, cfg) >= need
    return {
        "status": PASS if ok else FAIL,
        "detail": f"PR-AUC {cand.pr_auc:.6f} vs champion {champ.pr_auc:.6f} + margin {g.pr_auc_margin}",
    }


def gate_g3(report: dict | None) -> dict:
    if report is None:
        return {"status": INDETERMINATE, "detail": "no fairness report"}
    statuses = {k: v["status"] for k, v in report["gates"].items()}
    if any(s == FAIL for s in statuses.values()):
        status = FAIL
    elif any(s == INDETERMINATE for s in statuses.values()):
        status = INDETERMINATE
    else:
        status = PASS
    return {"status": status, "detail": statuses}


def gate_g4(report: dict | None) -> dict:
    warns = (report or {}).get("warnings", [])
    return {"status": WARN if warns else PASS, "detail": warns}


def aggregate(gates: dict[str, dict]) -> str:
    statuses = [gates[g]["status"] for g in MANDATORY if g in gates]
    if FAIL in statuses:
        return REJECTED
    if INDETERMINATE in statuses:
        return DECISION_INDETERMINATE
    return APPROVED


def decide(candidate: Candidate, champion: Evaluation | None, cfg: Config) -> GateDecision:
    g5 = candidate.g5_passed
    gates = {
        "G1": {"status": PASS if candidate.g1_passed else FAIL, "detail": candidate.g1_detail},
        "G2": gate_g2(candidate.evaluation, champion, cfg),
        "G3": gate_g3(candidate.evaluation.fairness_report),
        "G4": gate_g4(candidate.evaluation.fairness_report),
        "G5": {
            "status": INDETERMINATE if g5 is None else (PASS if g5 else FAIL),
            "detail": candidate.g5_detail,
        },
    }
    decision = aggregate(gates)
    reasons = [f"{k}: {v['status']}" for k, v in gates.items() if v["status"] in (FAIL, INDETERMINATE)]
    digest_src = {
        "candidate": {
            "pr_auc": candidate.evaluation.pr_auc,
            "n": candidate.evaluation.n_eval,
            "g1": candidate.g1_passed,
            "g5": g5,
            "fair": (candidate.evaluation.fairness_report or {}).get("gates"),
        },
        "champion": None if champion is None else {"pr_auc": champion.pr_auc, "n": champion.n_eval},
        "policy": {"gates": cfg.gates.model_dump(), "fairness": cfg.fairness.model_dump()},
    }
    digest = hashlib.sha256(json.dumps(digest_src, sort_keys=True, default=str).encode()).hexdigest()
    return GateDecision(decision, gates, reasons, gates["G4"]["detail"], digest)

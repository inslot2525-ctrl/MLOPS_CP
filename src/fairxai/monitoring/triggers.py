"""Retraining trigger evaluation (SPEC D.7 persistence, N-10). Thresholds come only from configs/.

Rules: drift ALERT alone => WARN event, never a retrain; dataset drift in N consecutive windows => request;
performance ALERT in M consecutive labelled windows => request; scheduled run => request.
Windows with status NA neither extend nor reset a streak. Only rows of the current model version count,
so a model change resets the counters.
"""

from __future__ import annotations

from dataclasses import dataclass

ALERT, NA = "ALERT", "NA"


@dataclass
class TriggerDecision:
    request: bool
    trigger: str | None
    reason: str
    drift_streak: int
    perf_streak: int
    warn_events: list[str]

    def to_dict(self) -> dict:
        return vars(self)


def streak(statuses_newest_first: list[str]) -> int:
    """Length of the leading ALERT run, skipping NA windows."""
    n = 0
    for s in statuses_newest_first:
        if s == NA:
            continue
        if s != ALERT:
            break
        n += 1
    return n


def evaluate(
    drift_history: list[str],
    perf_history: list[str],
    drift_needed: int,
    perf_needed: int,
    scheduled: bool = False,
) -> TriggerDecision:
    """`*_history`: statuses of this model version's windows, newest first."""
    d, p = streak(drift_history), streak(perf_history)
    warns = []
    if drift_history and drift_history[0] == ALERT and d < drift_needed:
        warns.append(f"feature drift ALERT ({d} consecutive window(s)); no retrain on drift alone")
    if p >= perf_needed:
        return TriggerDecision(
            True, "performance", f"performance ALERT in {p} consecutive windows", d, p, warns
        )
    if d >= drift_needed:
        return TriggerDecision(True, "drift", f"dataset drift in {d} consecutive windows", d, p, warns)
    if scheduled:
        return TriggerDecision(True, "scheduled", "scheduled run", d, p, warns)
    return TriggerDecision(False, None, "no trigger", d, p, warns)

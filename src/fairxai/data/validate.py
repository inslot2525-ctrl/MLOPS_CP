"""Data validation. The only module that imports Great Expectations (SPEC D6, D.2)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import great_expectations as gx
import great_expectations.expectations as gxe
import pandas as pd

from fairxai.config import Config, load_config
from fairxai.data.load import CONTINUOUS, FEATURES, RAW_COLUMNS, column_range

SUITES = ("raw_taiwan", "batch")
EXPECTATIONS_DIR = Path(__file__).resolve().parents[3] / "pipelines" / "expectations"


@dataclass
class ValidationResult:
    suite: str
    passed: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "suite": self.suite,
            "passed": self.passed,
            "errors": self.errors,
            "warnings": self.warnings,
            "stats": self.stats,
        }


def build_suite(name: str, cfg: Config, labelled: bool = True) -> gx.ExpectationSuite:
    """Build a suite from data.yaml so thresholds stay in one place."""
    if name not in SUITES:
        raise ValueError(f"unknown suite {name}")
    target = cfg.data.target
    if name == "raw_taiwan":
        columns = [*RAW_COLUMNS, target]
    else:
        columns = [*FEATURES, *([target] if labelled else [])]
    exps: list = [gxe.ExpectTableColumnsToMatchSet(column_set=columns, exact_match=True)]
    if name == "raw_taiwan":
        exps.append(gxe.ExpectTableRowCountToEqual(value=cfg.data.expected_rows))
        exps.append(gxe.ExpectColumnValuesToBeUnique(column="ID"))
    else:
        exps.append(gxe.ExpectTableRowCountToBeBetween(min_value=cfg.data.min_batch_rows))
    for col in columns:
        if col == "ID":
            continue
        exps.append(gxe.ExpectColumnValuesToNotBeNull(column=col))
        if col == target:
            exps.append(gxe.ExpectColumnValuesToBeInSet(column=col, value_set=[0, 1]))
        else:
            lo, hi = column_range(col, cfg)
            exps.append(gxe.ExpectColumnValuesToBeBetween(column=col, min_value=lo, max_value=hi))
    return gx.ExpectationSuite(name=name if labelled else f"{name}_unlabelled", expectations=exps)


def export_suites(cfg: Config | None = None, out_dir: Path = EXPECTATIONS_DIR) -> None:
    cfg = cfg or load_config()
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in SUITES:
        suite = build_suite(name, cfg)
        (out_dir / f"{name}.json").write_text(json.dumps(suite.to_json_dict(), indent=2), encoding="utf-8")


def _run_ge(df: pd.DataFrame, suite: gx.ExpectationSuite) -> list[tuple[str, bool]]:
    ctx = gx.get_context(mode="ephemeral")
    asset = ctx.data_sources.add_pandas("fairxai").add_dataframe_asset("frame")
    batch = asset.add_batch_definition_whole_dataframe("whole").get_batch(batch_parameters={"dataframe": df})
    result = batch.validate(suite)
    out = []
    for r in result.results:
        cfgd = r.expectation_config
        col = cfgd.kwargs.get("column")
        label = f"{cfgd.type}" + (f"[{col}]" if col else "")
        out.append((label, bool(r.success)))
    return out


def outlier_fraction(df: pd.DataFrame, ref: dict) -> float:
    flagged = 0
    for c in CONTINUOUS:
        b = ref["columns"][c]
        flagged += int(((df[c] < b["lo"]) | (df[c] > b["hi"])).sum())
    return flagged / (len(df) * len(CONTINUOUS))


def validate(
    df: pd.DataFrame,
    suite: str,
    cfg: Config | None = None,
    labelled: bool = True,
    reference_stats: dict | None = None,
) -> ValidationResult:
    """Hard-check failure => passed False. Warnings are recorded and never block."""
    cfg = cfg or load_config()
    ge_suite = build_suite(suite, cfg, labelled)
    errors = [f"{label} failed" for label, ok in _run_ge(df, ge_suite) if not ok]
    warnings: list[str] = []
    stats: dict = {"n_rows": len(df)}
    tgt = cfg.data.target
    if suite == "raw_taiwan" and tgt in df.columns:
        rate = float(df[tgt].mean())
        stats["default_rate"] = rate
        lo, hi = cfg.data.default_rate_warn
        if not lo <= rate <= hi:
            warnings.append(f"default rate {rate:.4f} outside [{lo}, {hi}]")
    if reference_stats is not None and not errors:
        o, f_ref = cfg.data.outlier, reference_stats["f_ref"]
        frac = outlier_fraction(df, reference_stats)
        stats["outlier_fraction"] = frac
        stats["f_ref"] = f_ref
        if frac > f_ref + o.fail_add:
            errors.append(f"outlier fraction {frac:.4f} > f_ref+{o.fail_add}")
        elif frac > max(o.warn_factor * f_ref, f_ref + o.warn_add):
            warnings.append(f"outlier fraction {frac:.4f} above warn limit")
    return ValidationResult(suite, not errors, errors, warnings, stats)

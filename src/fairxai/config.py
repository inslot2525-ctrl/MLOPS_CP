"""Typed, validated configuration loader. The only module that reads configs/*.yaml."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict

CONFIG_DIR = Path(os.environ.get("FAIRXAI_CONFIG_DIR", Path(__file__).resolve().parents[2] / "configs"))


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Paths(_M):
    data_dir: str
    clock_offset_file: str


class MlflowCfg(_M):
    experiment: str
    registered_model: str


class Base(_M):
    seed: int
    paths: Paths
    mlflow: MlflowCfg
    window_rows: int
    min_window_rows: int


class Outlier(_M):
    iqr_multiplier: float
    warn_factor: float
    warn_add: float
    fail_add: float


class Aliases(_M):
    target: list[str]
    rename: dict[str, str]


class Data(_M):
    expected_rows: int
    target: str
    drop_columns: list[str]
    aliases: Aliases
    ranges: dict[str, list[float | None]]
    outlier: Outlier
    default_rate_warn: list[float]
    min_batch_rows: int


class Grid(_M):
    start: float
    stop: float
    step: float


class Training(_M):
    models: list[str]
    cv_folds: int
    tuning_cv_folds: int
    shortlist: int
    optuna_trials: int
    optuna_timeout_s: int
    imbalance_strategy: str
    n_jobs: int
    include_protected: bool
    threshold_grid: Grid
    baseline_params: dict[str, dict[str, Any]]
    search_spaces: dict[str, dict[str, dict[str, Any]]]


class Fairness(_M):
    favorable_label: int
    calibration_bins: int
    decimals: int
    age_bands: dict[str, list[int]]
    eo_max: float
    di_min: float
    eo_inter_max: float
    di_inter_min: float
    dp_warn: float
    cal_warn: float
    warn_worse_margin: float
    min_group_rows: int
    min_denominator_rows: int


class G6(_M):
    rows: int
    p95_s: float


class G7(_M):
    min_requests: int
    min_seconds: int
    max_seconds: int
    max_5xx_rate: float
    p95_s: float


class Approval(_M):
    model_promotion: bool


class Gates(_M):
    pr_auc_floor: float
    pr_auc_margin: float
    min_eval_rows: int
    g6: G6
    g7: G7
    auto_deploy: bool
    approval: Approval


class Monitoring(_M):
    schedule: str | None
    p_value: float
    dataset_drift_share: float
    prediction_rate_change: float
    perf_drop: float
    min_labelled_rows: int


class Retraining(_M):
    drift_persistence: int
    perf_persistence: int
    cooldown_minutes: int
    max_runs_per_week: int
    min_new_rows: int
    min_new_rows_scheduled: int
    new_data_weight: int
    holdout_percent: int


class Serving(_M):
    top_k_default: int
    max_body_bytes: int
    predict_p95_target_s: float
    explain_p95_target_s: float


class Simulator(_M):
    seed: int
    workers: int
    max_rps: int
    label_delay_batches: int
    faults: dict[str, dict[str, Any]]


class Config(_M):
    base: Base
    data: Data
    training: Training
    fairness: Fairness
    gates: Gates
    monitoring: Monitoring
    retraining: Retraining
    serving: Serving
    simulator: Simulator
    profile: str = "base"


_SECTIONS = {
    "base": Base,
    "data": Data,
    "training": Training,
    "fairness": Fairness,
    "gates": Gates,
    "monitoring": Monitoring,
    "retraining": Retraining,
    "serving": Serving,
    "simulator": Simulator,
}


def _merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(profile: str | None = None, config_dir: Path | None = None) -> Config:
    """Load all YAML files, overlay configs/profiles/<profile>.yaml, validate (unknown keys fail)."""
    cdir = Path(config_dir) if config_dir else CONFIG_DIR
    profile = profile or os.environ.get("FAIRXAI_PROFILE", "base")
    raw = {s: yaml.safe_load((cdir / f"{s}.yaml").read_text(encoding="utf-8")) for s in _SECTIONS}
    if profile != "base":
        overlay = yaml.safe_load((cdir / "profiles" / f"{profile}.yaml").read_text(encoding="utf-8")) or {}
        unknown = set(overlay) - set(_SECTIONS)
        if unknown:
            raise ValueError(f"profile {profile}: unknown sections {sorted(unknown)}")
        raw = _merge(raw, overlay)
    return Config(**raw, profile=profile)

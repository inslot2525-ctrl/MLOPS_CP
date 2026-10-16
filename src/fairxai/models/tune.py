"""Optuna tuning (N-11): TPE seed 42, bounded trials and time, in-memory storage."""

from __future__ import annotations

import optuna

from fairxai.config import Config
from fairxai.models.evaluate import cv_oof
from fairxai.models.zoo import make_pipeline

optuna.logging.set_verbosity(optuna.logging.WARNING)


def suggest(trial: optuna.Trial, space: dict) -> dict:
    params: dict = {}
    for name, s in space.items():
        kind = s["type"]
        if kind == "int":
            params[name] = trial.suggest_int(name, s["low"], s["high"], step=s.get("step", 1))
        elif kind == "float":
            params[name] = trial.suggest_float(name, s["low"], s["high"], log=s.get("log", False))
        elif kind == "cat":
            params[name] = trial.suggest_categorical(name, s["choices"])
        else:
            raise ValueError(f"unknown search-space type {kind!r} for {name}")
    return params


def tune_model(name: str, X, y, cfg: Config) -> dict:
    t = cfg.training
    space = t.search_spaces[name]

    def objective(trial: optuna.Trial) -> float:
        params = suggest(trial, space)
        score, _ = cv_oof(
            make_pipeline(name, params, y, cfg), X, y, t.tuning_cv_folds, cfg.base.seed, t.n_jobs
        )
        return score

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=cfg.base.seed))
    study.optimize(objective, n_trials=t.optuna_trials, timeout=t.optuna_timeout_s)
    return {
        "model": name,
        "best_params": study.best_params,
        "best_cv_pr_auc": float(study.best_value),
        "n_trials": len(study.trials),
        "trials": [{"number": tr.number, "value": tr.value, "params": tr.params} for tr in study.trials],
    }

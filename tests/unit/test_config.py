import shutil
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from fairxai.config import CONFIG_DIR, load_config


def test_base_loads():
    cfg = load_config("base")
    assert cfg.base.seed == 42
    assert cfg.fairness.favorable_label == 0
    assert cfg.retraining.cooldown_minutes == 60


def test_profile_overlay():
    assert load_config("demo").retraining.cooldown_minutes == 5
    assert load_config("test").training.optuna_trials == 3
    assert load_config("test").training.shortlist == 3  # untouched keys survive the overlay


def test_unknown_key_rejected(tmp_path: Path):
    d = tmp_path / "configs"
    shutil.copytree(CONFIG_DIR, d)
    p = d / "gates.yaml"
    raw = yaml.safe_load(p.read_text())
    raw["surprise"] = 1
    p.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValidationError):
        load_config("base", config_dir=d)


def test_unknown_profile_section_rejected(tmp_path: Path):
    d = tmp_path / "configs"
    shutil.copytree(CONFIG_DIR, d)
    (d / "profiles" / "bad.yaml").write_text("nonsense: {a: 1}\n")
    with pytest.raises(ValueError):
        load_config("bad", config_dir=d)

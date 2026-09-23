"""Tests for needle/api/config.py: Config.

Loads tests/conf_tests/config.yaml directly (no LAW, no torch), exercising the
resolved MainConfig returned by `Config()`.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import DictConfig

from needle.api.config import load_config

CONF_TESTS_DIR = Path(__file__).parent.parent / "conf_tests"
CONFIG_PATH = CONF_TESTS_DIR / "config.yaml"


def test_missing_file_raises_file_not_found_error(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.yaml"

    with pytest.raises(FileNotFoundError, match="Config file not found"):
        load_config(missing)


def test_config_returns_a_resolved_dictconfig() -> None:
    cfg = load_config(CONFIG_PATH)

    assert isinstance(cfg, DictConfig)


def test_config_estimators_are_populated() -> None:
    cfg = load_config(CONFIG_PATH)

    assert set(cfg.estimators.keys()) == {"model_A", "model_B"}
    assert cfg.estimators["model_A"].model == "mock_transformer"
    assert cfg.estimators["model_A"].expands.folds.num == 2


def test_config_is_resolved_with_overrides_populated() -> None:
    cfg = load_config(CONFIG_PATH)

    estimator_cfg = cfg.estimators["model_A"]
    assert cfg._resolved is True
    assert estimator_cfg.dataset_override is not None
    assert estimator_cfg.datamodule_override is not None


def test_config_accepts_space_separated_override_string() -> None:
    cfg = load_config(CONFIG_PATH, overrides="estimators.model_A.model=simple_mlp")

    assert cfg.estimators["model_A"].model == "simple_mlp"


def test_config_accepts_override_list() -> None:
    cfg = load_config(CONFIG_PATH, overrides=["estimators.model_A.model=simple_mlp"])

    assert cfg.estimators["model_A"].model == "simple_mlp"

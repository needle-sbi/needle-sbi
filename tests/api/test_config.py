"""Tests for needle/api/config.py: Config.

Loads tests/conf_tests/config.yaml directly (no LAW, no torch), exercising the
attribute-forwarding, dot-notation `.get()`, and `__repr__` behavior documented on `Config`.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import DictConfig

from needle.api.config import Config

CONF_TESTS_DIR = Path(__file__).parent.parent / "conf_tests"
CONFIG_PATH = CONF_TESTS_DIR / "config.yaml"


def test_missing_file_raises_file_not_found_error(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.yaml"

    with pytest.raises(FileNotFoundError, match="Config file not found"):
        Config(missing)


def test_config_path_is_resolved_absolute() -> None:
    cfg = Config(CONFIG_PATH)

    assert cfg.config_path == CONFIG_PATH.resolve()
    assert cfg.config_path.is_absolute()


def test_config_wraps_a_resolved_dictconfig() -> None:
    cfg = Config(CONFIG_PATH)

    assert isinstance(cfg, Config)
    assert isinstance(cfg.config, DictConfig)


def test_attribute_access_forwards_to_resolved_config() -> None:
    cfg = Config(CONFIG_PATH)

    assert cfg.estimators is cfg.config.estimators
    assert set(cfg.estimators.keys()) == {"model_A", "model_B"}


def test_get_resolves_nested_dot_notation_key() -> None:
    cfg = Config(CONFIG_PATH)

    assert cfg.get("estimators.model_A.model") == "mock_transformer"
    assert cfg.get("estimators.model_A.expands.folds") == 2


def test_get_returns_default_for_missing_key() -> None:
    cfg = Config(CONFIG_PATH)

    assert cfg.get("estimators.does_not_exist.model", default="fallback") == "fallback"
    assert cfg.get("nonexistent.path") is None


def test_config_is_resolved_with_overrides_populated() -> None:
    cfg = Config(CONFIG_PATH)

    estimator_cfg = cfg.config.estimators["model_A"]
    assert cfg.config._resolved is True
    assert estimator_cfg.dataset_override is not None
    assert estimator_cfg.datamodule_override is not None


def test_repr_shows_wrapper_and_nested_config_structure() -> None:
    cfg = Config(CONFIG_PATH)

    text = repr(cfg)

    assert text.startswith(f"Config('{cfg.config_path}')")
    assert "estimators:" in text
    assert "model_A:" in text
    assert "model_B:" in text
    assert "mock_transformer" in text

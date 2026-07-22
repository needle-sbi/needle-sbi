"""Real CLI-level compatibility checks for the law and b2luigi backends.

These tests shell out to the actual entry points end users invoke - ``law run``,
``python3 -m luigi``, and the ``needle`` console script - rather than calling task
methods (``.law_run()``, ``b2luigi.process()``) directly from Python. This is the
level at which regressions in ``law.cfg`` indexing, luigi CLI parameter wiring, or
``needle/cli.py``'s argument translation would actually surface.

Training uses the small ``model_A`` estimator from ``tests/conf_tests/config.yaml``
(mock transformer, 2 folds, 1 epoch) pointed at the parquet file bundled under
``examples/fair_universe_demo/test_data/`` via a plain relative/absolute filesystem
path - never ``FAIR_UNIVERSE_DATA`` or OmegaConf's ``${oc.env:...}`` interpolation -
so these tests run the same locally and in CI with no extra setup.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import omegaconf
import pytest

from tests.conftest import MainConfigFactory

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ESTIMATOR = "model_A"


def _write_model_a_config(config_factory: MainConfigFactory, tmp_path: Path, data_path: Path) -> Path:
    """Build the tests/conf_tests config with model_A's dataset pointed at a real parquet file."""
    config = config_factory()
    dataset_config = config.estimators[_ESTIMATOR].dataset_override
    assert dataset_config
    dataset_config.paths = str(data_path)
    config._resolved = True
    config_file = tmp_path / "config.yaml"
    omegaconf.OmegaConf.save(config, config_file, resolve=True)
    return config_file


def _needle_bin() -> str:
    """Path to the ``needle`` console script installed alongside the current interpreter."""
    candidate = Path(sys.executable).parent / "needle"
    return str(candidate) if candidate.exists() else "needle"


def _assert_ensemble_trained(results_path: Path, n_folds: int = 2) -> None:
    base = results_path / f"est__{_ESTIMATOR}" / "syst__nominal" / "ensem__0"
    assert (base / ".done").exists(), f"missing ensemble marker under {base}"
    for fold_index in range(n_folds):
        ckpt = base / f"fold__{fold_index}" / "model.ckpt"
        assert ckpt.exists(), f"missing checkpoint at {ckpt}"


# ---------------------------------------------------------------------------
# `law run` — the plain shell command, using a fully-qualified task path so the
# test doesn't depend on LAW_HOME/law.cfg indexing being set up.
# ---------------------------------------------------------------------------


@pytest.mark.law
def test_law_run_cli_ensemble_task(
    config_factory: MainConfigFactory,
    tmp_path: Path,
    fair_universe_demo_parquet: Path,
) -> None:
    config_file = _write_model_a_config(config_factory, tmp_path, fair_universe_demo_parquet)
    results_path = tmp_path / "results"

    cmd = [
        "law",
        "run",
        "needle.tasks.law.ensemble.EnsembleTask",
        "--estimator",
        _ESTIMATOR,
        "--systematic",
        "nominal",
        "--ensemble",
        "0",
        "--config-file",
        str(config_file),
        "--results-path",
        str(results_path),
        "--workers",
        "1",
        "--local-scheduler",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180, cwd=_REPO_ROOT)
    assert result.returncode == 0, result.stdout + result.stderr
    _assert_ensemble_trained(results_path)


@pytest.mark.law
def test_needle_run_cli_law_backend(
    config_factory: MainConfigFactory,
    tmp_path: Path,
    fair_universe_demo_parquet: Path,
) -> None:
    """``needle run --backend law`` shells out to ``law run`` under the hood."""
    config_file = _write_model_a_config(config_factory, tmp_path, fair_universe_demo_parquet)
    results_path = tmp_path / "results"

    env = dict(os.environ, LAW_HOME=str(_REPO_ROOT), LAW_CONFIG_FILE=str(_REPO_ROOT / "law.cfg"))
    cmd = [
        _needle_bin(),
        "run",
        "EnsembleTask",
        "--backend",
        "law",
        "--config",
        str(config_file),
        "--results-path",
        str(results_path),
        "--param",
        f"estimator={_ESTIMATOR}",
        "--param",
        "systematic=nominal",
        "--param",
        "ensemble=0",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180, cwd=_REPO_ROOT, env=env)
    assert result.returncode == 0, result.stdout + result.stderr
    _assert_ensemble_trained(results_path)


# ---------------------------------------------------------------------------
# `python3 -m luigi` — plain luigi CLI compatibility for the b2luigi backend.
# Marked `slow` (not `b2luigi`) so `pytest -m b2luigi` stays limited to the fast,
# structural, no-training tests in test_b2luigi_tasks.py.
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_python_m_luigi_cli_ensemble_task(
    config_factory: MainConfigFactory,
    tmp_path: Path,
    fair_universe_demo_parquet: Path,
) -> None:
    config_file = _write_model_a_config(config_factory, tmp_path, fair_universe_demo_parquet)
    results_path = tmp_path / "results"

    cmd = [
        sys.executable,
        "-m",
        "luigi",
        "--module",
        "needle.tasks.b2luigi.ensemble",
        "EnsembleTask",
        "--estimator",
        _ESTIMATOR,
        "--systematic",
        "nominal",
        "--ensemble",
        "0",
        "--config-file",
        str(config_file),
        "--results-path",
        str(results_path),
        "--local-scheduler",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180, cwd=_REPO_ROOT)
    assert result.returncode == 0, result.stdout + result.stderr
    _assert_ensemble_trained(results_path)


@pytest.mark.slow
def test_needle_run_cli_b2luigi_backend(
    config_factory: MainConfigFactory,
    tmp_path: Path,
    fair_universe_demo_parquet: Path,
) -> None:
    """``needle run --backend b2luigi`` drives the same DAG through ``b2luigi.process()``."""
    config_file = _write_model_a_config(config_factory, tmp_path, fair_universe_demo_parquet)
    results_path = tmp_path / "results"

    cmd = [
        _needle_bin(),
        "run",
        "EnsembleTask",
        "--backend",
        "b2luigi",
        "--config",
        str(config_file),
        "--results-path",
        str(results_path),
        "--param",
        f"estimator={_ESTIMATOR}",
        "--param",
        "systematic=nominal",
        "--param",
        "ensemble=0",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180, cwd=_REPO_ROOT)
    assert result.returncode == 0, result.stdout + result.stderr
    _assert_ensemble_trained(results_path)

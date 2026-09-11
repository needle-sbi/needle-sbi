"""Tests for needle/api/train.py: train_single().

Real (slow) training test, mirroring tests/trainings/test_fold_task.py's
in-process b2luigi execution pattern.
"""

from __future__ import annotations

from pathlib import Path

import omegaconf
import pytest

from needle.api.train import train_single
from tests.conftest import MainConfigFactory


@pytest.mark.slow
def test_train_single_writes_flat_output(
    config_factory: MainConfigFactory,
    tmp_path: Path,
    fair_universe_demo_parquet: Path,
) -> None:
    estimator_name = list(config_factory().estimators.keys())[0]
    config = config_factory()
    dataset_config = config.estimators[estimator_name].dataset_override
    assert dataset_config
    dataset_config.paths = str(fair_universe_demo_parquet)
    config._resolved = True
    config_file = tmp_path / "config.yaml"
    omegaconf.OmegaConf.save(config, config_file, resolve=True)

    outputs = train_single(config_file=config_file, estimator=estimator_name, results_path=tmp_path)

    assert outputs["ckpt"] == tmp_path / "model.ckpt"
    assert outputs["model_config"] == tmp_path / "model_config.yaml"
    assert outputs["input_models"] == tmp_path / "input_models.json"
    assert outputs["ckpt"].exists()
    assert outputs["model_config"].exists()
    assert outputs["input_models"].exists()

    # single=True writes flat under results_path - no est__/syst__/ensem__/fold__ nesting.
    assert not (tmp_path / f"est__{estimator_name}").exists()

"""
Test the execution of the k-fold training Tasks
"""
from pathlib import Path

import omegaconf
import pytest

from needle.law_tasks.ensemble import EnsembleTask
from tests.conftest import MainConfigFactory


@pytest.mark.law
def test_kfold_training(
    config_factory: MainConfigFactory,
    tmp_path: Path,
    fair_universe_sample: str | Path,
):
    fair_universe_sample = Path(fair_universe_sample)
    if fair_universe_sample.is_dir():
        fair_universe_sample = fair_universe_sample / "*.parquet"
    estimator_name = list(config_factory().estimators.keys())[0]
    config = config_factory()
    dataset_config = config.estimators[estimator_name].dataset_override
    assert dataset_config
    dataset_config.paths = str(fair_universe_sample)
    config._resolved = True
    config_tmp_file = tmp_path / "config.yaml"
    omegaconf.OmegaConf.save(config, config_tmp_file, resolve=True)

    ensemble = EnsembleTask(
        config_file=config_tmp_file,
        estimator=estimator_name,
        systematic="nominal",
        results_path=tmp_path,
    )
    assert ensemble.law_run()

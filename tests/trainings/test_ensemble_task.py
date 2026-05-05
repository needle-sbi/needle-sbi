"""
Test the execution of the k-fold training Tasks

Note:
    This Task is called EnsembleTask but it wraps the k-fold Training Tasks, which is why we
    use if in this test. Once the EnsembleTask has more functionality than just calling several
    FoldTasks, we can refactor this test.
"""
from pathlib import Path

import omegaconf
import pytest

from law_tasks.ensemble import EnsembleTask
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
    config.estimators[estimator_name].dataset_override.paths = fair_universe_sample
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

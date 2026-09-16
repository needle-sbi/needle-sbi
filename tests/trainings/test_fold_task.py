"""
Test the execution of the k-fold training Tasks
"""
from pathlib import Path

import luigi
import omegaconf
import pytest

from needle.tasks.law.ensemble import EnsembleTask
from tests.conftest import MainConfigFactory


@pytest.mark.law
def test_kfold_training(
    config_factory: MainConfigFactory,
    tmp_path: Path,
    fair_universe_demo_parquet: Path,
):
    estimator_name = list(config_factory().estimators.keys())[0]
    config = config_factory()
    dataset_config = config.estimators[estimator_name].dataset_override
    assert dataset_config
    dataset_config.paths = str(fair_universe_demo_parquet)
    config._resolved = True
    config_tmp_file = tmp_path / "config.yaml"
    omegaconf.OmegaConf.save(config, config_tmp_file, resolve=True)

    ensemble = EnsembleTask(
        config_file=config_tmp_file,
        estimator=estimator_name,
        systematic="nominal",
        results_path=tmp_path,
    )
    # luigi.build() takes the task instance directly, unlike `.law_run()` which
    # resolves the task by family name through luigi's global task registry - and
    # that registry lookup becomes ambiguous once both the law and b2luigi backends
    # (whose marker tasks share unnamespaced family names, e.g. "EnsembleTask") have
    # been imported into the same process, as happens when pytest collects both
    # tests/trainings/test_b2luigi_tasks.py and this file in one session.
    result = luigi.build([ensemble], local_scheduler=True, workers=1, detailed_summary=True)
    assert result.status == luigi.execution_summary.LuigiStatusCode.SUCCESS

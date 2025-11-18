"""
Test the execution of the k-fold training Tasks

Note:
    This Task is called EnsembleTask but it wraps the k-fold Training Tasks, which is why we
    use if in this test. Once the EnsembleTask has more functionality than just calling several
    FoldTasks, we can refactor this test.
"""
from pathlib import Path

import pytest
import omegaconf

from law_tasks.ensemble import EnsembleTask
from orchestrator.config import MainConfig


@pytest.mark.law
def test_kfold_training(
    config: MainConfig,
    tmp_path: Path,
    fair_universe_sample: str,
):
    config.n_folds = 2
    config.datasets.paths = fair_universe_sample
    config_tmp_file = tmp_path / "config.yaml"
    omegaconf.OmegaConf.save(config, config_tmp_file)

    ensemble = EnsembleTask(
        config_file=config_tmp_file,
        rel_results_path=tmp_path,
    )
    assert ensemble.law_run()

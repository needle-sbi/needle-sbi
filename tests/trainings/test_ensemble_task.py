"""
Test the execution of the k-fold training Tasks

NOTE This Task is called EnsembleTask but it wraps the k-fold Training Tasks, which is why we
use if in this test. Once the EnsembleTask has more functionality than just calling several
FoldTasks, we can refactor this test.
"""
from pathlib import Path

import pytest

from law_tasks.ensemble import EnsembleTask
from ml.utils.config import MLConfig


@pytest.mark.law
def test_kfold_training(
    config_yaml: str,
    tmp_path: Path,
):
    config = MLConfig.from_yaml(config_yaml)
    config.n_folds = 2
    config.to_yaml(config_yaml)

    ensemble = EnsembleTask()
    ensemble.config_yaml = config_yaml  # type: ignore
    ensemble.results_dir_path = tmp_path  # type: ignore
    assert ensemble.law_run()

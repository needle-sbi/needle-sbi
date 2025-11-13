"""
Test the execution of the k-fold training Tasks

NOTE This Task is called EnsembleTask but it wraps the k-fold Training Tasks, which is why we
use if in this test. Once the EnsembleTask has more functionality than just calling several
FoldTasks, we can refactor this test.
"""
from pathlib import Path

import pytest

from law_tasks.ensemble import EnsembleTask
from orchestrator.config import MainConfig


@pytest.mark.law
def test_kfold_training(
    config: MainConfig,
    tmp_path: Path,
):
    config.n_folds = 2

    ensemble = EnsembleTask()
    ensemble.config = config
    ensemble.rel_results_path = tmp_path  # type: ignore
    assert ensemble.law_run()

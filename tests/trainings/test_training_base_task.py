"""Test a simple run of the whole pipeline
"""
import os
from importlib.util import find_spec
from pathlib import Path

import pytest

from law_tasks.training_base import TrainingBaseTask
from ml.utils.config import MLConfig

if find_spec("preprocessor"):
    pytest_plugins = ["preprocessor.tests.conftest"]


@pytest.mark.parametrize("use_dataset_chunk", [True, False])
def test_training_base_task_single_epoch_dummy_array(
    config_yaml: str,
    use_dataset_chunk: bool,
):
    config = MLConfig.from_yaml(config_yaml)
    config.dataset_parallelization_threshold = 1 if use_dataset_chunk else int(1e10)
    config.to_yaml(config_yaml)
    training_base = TrainingBaseTask()
    training_base.config_yaml = config_yaml  # type: ignore
    training_base.run()


@pytest.mark.parametrize("use_dataset_chunk", [True, False])
def test_training_base_task_single_epoch_fair_universe(
    tmp_path: Path,
    fair_universe_sample: str,
    use_dataset_chunk: bool,
):
    config = MLConfig()
    config.files_to_load = [fair_universe_sample]
    config.features_columns = ["PRI_lep_pt"]
    config.labels_columns = ["PRI_n_jets"]
    config.max_number_events = 10_000
    config.total_epoch = 1
    config.batch_size = 128
    config.num_workers = 2
    config.dataset_parallelization_threshold = 1 if use_dataset_chunk else int(1e10)
    config_yaml = os.path.abspath(tmp_path / "config.yaml")
    config.to_yaml(config_yaml)

    training_base = TrainingBaseTask()
    training_base.config_yaml = config_yaml  # type: ignore
    training_base.law_run()


def test_training_base_task_require_run_implementation_in_subclass():
    with pytest.raises(TypeError):

        class SubClassWrong(TrainingBaseTask):
            pass

    class SubClassCorrect(TrainingBaseTask):
        def run(self):
            pass

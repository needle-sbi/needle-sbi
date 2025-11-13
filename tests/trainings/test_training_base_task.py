"""Test a simple run of the whole pipeline
"""
from importlib.util import find_spec

import pytest

from law_tasks.training_base import TrainingBaseTask
from orchestrator.config import MainConfig

if find_spec("preprocessor"):
    pytest_plugins = ["preprocessor.tests.conftest"]


@pytest.mark.parametrize("datasets", ["simple", "fair_universe"])
def test_training_base_task_single_epoch(
    config: MainConfig,
    simple_sample: str,
    fair_universe_sample: str,
    datasets: str,
):
    if datasets == "simple":
        config.datasets.paths = [simple_sample]
        config.datasets.features_columns = ["Lepton.pt"]
        config.datasets.labels_columns = ["Jet.eta"]
    elif datasets == "fair_universe":
        config.datasets.paths = [fair_universe_sample]
    else:
        raise ValueError("Invalid dataset")

    training_base = TrainingBaseTask()
    training_base.config = config
    training_base.run()


def test_training_base_task_require_run_implementation_in_subclass():
    with pytest.raises(TypeError):

        class SubClassWrong(TrainingBaseTask):
            pass

    class SubClassCorrect(TrainingBaseTask):
        def run(self):
            pass

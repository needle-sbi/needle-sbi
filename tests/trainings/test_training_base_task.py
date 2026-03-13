"""Test a simple run of the whole pipeline
"""
from importlib.util import find_spec
from pathlib import Path

import omegaconf
import pytest

pytest.skip("WIP module. Has not been adapted to new config schema", allow_module_level=True)

from law_tasks.training_base import TrainingBaseTask
from orchestrator.config import MainConfig

if find_spec("preprocessor"):
    pytest_plugins = ["preprocessor.tests.conftest"]


@pytest.mark.parametrize("datasets", ["simple", "fair_universe"])
def test_training_base_task_single_epoch(
    config: MainConfig,
    simple_sample: str,
    fair_universe_sample: str,
    tmp_path: Path,
    datasets: str,
):
    estimator_name = list(config.estimators.keys())[0]
    estimator_cfg = config.estimators[estimator_name]

    if datasets == "simple":
        estimator_cfg.dataset_override.paths = simple_sample
        estimator_cfg.dataset_override.features_columns = ["Lepton.pt"]
        estimator_cfg.dataset_override.labels_columns = ["Jet.eta"]
    elif datasets == "fair_universe":
        estimator_cfg.dataset_override.paths = fair_universe_sample
    else:
        raise ValueError("Invalid dataset")

    config_tmp_file = tmp_path / "config.yaml"
    config._resolved = True
    omegaconf.OmegaConf.save(config, config_tmp_file)

    training_base = TrainingBaseTask(
        config_file=config_tmp_file,
        rel_results_path=tmp_path,
    )
    assert training_base.law_run()


def test_training_base_task_require_run_implementation_in_subclass():
    with pytest.raises(TypeError):

        class SubClassWrong(TrainingBaseTask):
            pass

    class SubClassCorrect(TrainingBaseTask):
        def run(self):
            pass

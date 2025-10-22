"""Test a simple run of the whole pipeline
"""
import pytest
import importlib
import os
from pathlib import Path

from ml.utils.config import MLConfig
from law_tasks.training_base import TrainingBaseTask

if importlib.util.find_spec("preprocessor"):  # type: ignore
    pytest_plugins = ["preprocessor.tests.conftest"]


@pytest.mark.parametrize("use_dataset_chunk", [True, False])
def test_training_base_task_single_epoch_dummy_array(
        request: pytest.FixtureRequest,
        tmp_path: Path,
        use_dataset_chunk: bool,
):
    pytest.importorskip("preprocessor", reason="Could not import 'preprocessor'")
    from preprocessor.tests.conftest import ArrayField

    try:
        make_parquet_file = request.getfixturevalue("make_parquet_file")
    except pytest.FixtureLookupError:
        pytest.skip("Fixture 'make_parquet_file' from preprocessor could not be found")
    
    template = ArrayField(dtype=float, shape=(10000, 1, 1))
    file = make_parquet_file(
        columns={
            "Lepton": {"pt": template},
            "Jet": {"eta": template},
        },
        file_name="simple",
    )

    config = MLConfig()
    config.files_to_load = [file]
    config.total_epoch = 1
    config.features_columns = ["Lepton.pt"]
    config.labels_columns = ["Jet.eta"]
    config.dataset_parallelization_threshold = 1 if use_dataset_chunk else int(1e10)
    config_path = os.path.abspath(tmp_path / "config.yaml")
    config.to_yaml(config_path)

    training_base = TrainingBaseTask()
    training_base.config_yaml = config_path  # type: ignore
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
    config_path = os.path.abspath(tmp_path / "config.yaml")
    config.to_yaml(config_path)

    training_base = TrainingBaseTask()
    training_base.config_yaml = config_path  # type: ignore
    training_base.run()

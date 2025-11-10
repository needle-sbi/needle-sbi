import os
from pathlib import Path
from typing import Any, Dict

import law

from ml.data import PaddedDatasetBase, PaddedTorchDataset
from ml.utils import MLConfig
from orchestrator import TrainingBase
from preprocessor.ingestion.formatter import Ingestor
from preprocessor.utils.logging import ColorFormatter

logger = ColorFormatter.get_logger("orchestrator")


class TrainingBaseTask(law.Task):
    def __init_subclass__(cls, **kwargs):
        if "run" not in cls.__dict__:
            raise TypeError(
                "Classes inheriting from TrainingBaseTask must implement their "
                "own 'run()', as law will otherwise not resolve dependencies correctly."
            )

    config_yaml = law.Parameter(
        description="Path to the YAML configuration file for the training.",
        default="config.yaml",
    )
    results_dir_path = law.Parameter(
        description="Directory where the training results will be saved.",
        default="runs",
        significant=False,
    )

    def output(self) -> Dict[str, Any]:
        return {
            "logs": law.LocalDirectoryTarget(f"{self.results_dir}/tensorboard_logs"),
            "outputs": law.LocalFileTarget(f"{self.results_dir}/training_output.json"),
        }

    @property
    def results_dir(self) -> Path:
        return os.path.abspath(self.results_dir_path)  # type: ignore

    @property
    def config(self) -> MLConfig:
        return MLConfig.from_yaml(str(self.config_yaml), strict=True)

    @property
    def features_ingestor(self) -> Ingestor:
        return Ingestor(
            self.config.files_to_load,
            format=self.config.filetype,
            columns=self.config.features_columns,
            reader_kwargs=self.config.dak_reader_kwargs,
            max_number_events=self.config.max_number_events,
        )

    @property
    def labels_ingestor(self) -> Ingestor:
        return Ingestor(
            self.config.files_to_load,
            format=self.config.filetype,
            columns=self.config.labels_columns,
            reader_kwargs=self.config.dak_reader_kwargs,
            max_number_events=self.config.max_number_events,
        )

    @property
    def training_dataset(self) -> PaddedDatasetBase:
        if self.features_ingestor.length < self.config.dataset_parallelization_threshold:
            dataset = PaddedDatasetBase(
                features=self.features_ingestor,
                labels=self.labels_ingestor,
            )
        else:
            dataset = PaddedTorchDataset(
                features=self.features_ingestor,
                labels=self.labels_ingestor,
                shuffle_partitions=self.config.shuffle_partitions,
                shuffle_events=self.config.shuffle_events,
                random_seed=self.config.random_seed,
            )
        return dataset

    @property
    def validation_dataset(self) -> PaddedDatasetBase:
        """Placeholder for actual implementations of the validation dataset."""
        return self.training_dataset

    def warn_if_device_is_cpu(self):
        if self.config.device == "cpu":
            logger.warning("Running on CPU. This may be slow for large datasets.")

    def run(self):
        """Simple implementation for testing.

        Should be overridden in derived classes.
        """
        self.warn_if_device_is_cpu()
        training_base = TrainingBase(
            training_dataset=self.training_dataset,
            validation_dataset=self.validation_dataset,
            config=self.config,
            tensor_board_log_dir=self.output()["logs"].path,
        )
        outputs = training_base.train()
        outputs.to_json(self.output()["outputs"].path)


if __name__ == "__main__":
    training_base = TrainingBaseTask()
    training_base.law_run()

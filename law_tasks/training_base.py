import json
from typing import Dict, Any

import law

from ml.utils import MLConfig
from ml.data import ParticleChunked, ParticleBase
from preprocessor.ingestion.formatter import Ingestor
from preprocessor.utils.logging import ColorFormatter
from orchestrator import TrainingBase

logger = ColorFormatter.get_logger("orchestrator")


class TrainingBaseTask(law.Task):

    config_yaml = law.Parameter(
        description="Path to the YAML configuration file for the training.",
        default="config.yaml"
    )
    results_dir = law.Parameter(
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
    def dataset(self) -> ParticleBase | ParticleChunked:
        if self.features_ingestor.length < self.config.dataset_parallelization_threshold:
            dataset = ParticleBase(
                features=self.features_ingestor,
                labels=self.labels_ingestor,
            )
        else:
            dataset = ParticleChunked(
                features=self.features_ingestor,
                labels=self.labels_ingestor,
                shuffle_partitions=self.config.shuffle_partitions,
                shuffle_events=self.config.shuffle_events,
                random_seed=self.config.random_seed,
            )
        return dataset

    def warn_if_device_is_cpu(self):
        if self.config.device == "cpu":
            logger.warning("Running on CPU. This may be slow for large datasets.")

    def run(self):
        self.warn_if_device_is_cpu()
        training_base = TrainingBase(
            dataset=self.dataset,
            config=self.config,
            tensor_board_log_dir=self.output()["logs"].path,
        )
        outputs = training_base.train()

        with open(self.output()["outputs"].path, "w") as f:
            json.dump(outputs, f, indent=4)


if __name__ == "__main__":
    training_base = TrainingBaseTask()
    training_base.run()

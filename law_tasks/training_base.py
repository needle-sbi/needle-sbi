import json
import logging

import law

from ml.utils import MLConfig
from ml.data import ParticleChunked, ParticleBase
from preprocessor.ingestion.formatter import Ingestor
from orchestrator import TrainingBase

logger = logging.getLogger("orchestrator")


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

    def output(self):
        return {
            "logs": law.LocalDirectoryTarget(f"{self.results_dir}/tensorboard_logs"),
            "outputs": law.LocalFileTarget(f"{self.results_dir}/training_output.json"),
        }

    def run(self):
        self.config = MLConfig.from_yaml(self.config_yaml, strict=True)  # type: ignore

        if self.config.device == "cpu":
            logger.warning("Running on CPU. This may be slow for large datasets.")
        
        features_ingestor = Ingestor(
            self.config.files_to_load,
            format=self.config.filetype,
            columns=self.config.features_columns,
            reader_kwargs=self.config.dak_reader_kwargs,
            max_number_events=self.config.max_number_events,
        )
        labels_ingestor = Ingestor(
            self.config.files_to_load,
            format=self.config.filetype,
            columns=self.config.labels_columns,
            reader_kwargs=self.config.dak_reader_kwargs,
            max_number_events=self.config.max_number_events,
        )
        if features_ingestor.length < self.config.dataset_parallelization_threshold:
            dataset = ParticleBase(
                features=features_ingestor,
                labels=labels_ingestor,
            )
        else:
            dataset = ParticleChunked(
                features=features_ingestor,
                labels=labels_ingestor,
                shuffle_partitions=self.config.shuffle_partitions,
                shuffle_events=self.config.shuffle_events,
                random_seed=self.config.random_seed,
            )
        training_base = TrainingBase(
            dataset=dataset,
            config=self.config,
            tensor_board_log_dir=self.output()["logs"].path,
        )
        outputs = training_base.train()

        with open(self.output()["outputs"].path, "w") as f:
            json.dump(outputs, f, indent=4)

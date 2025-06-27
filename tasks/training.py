import os
import json
import logging

import law

from ml.config import MLConfig
from preprocessor.ingestion.formatter import Ingestor
from orchestrator.train import TrainingBase

logger = logging.getLogger(__name__)


class TrainingBaseTask(law.Task):

    config_yaml = law.Parameter(
        description="Path to the YAML configuration file for the training.",
        default="ml/config/config.yaml"
    )
    results_dir = law.Parameter(
        description="Directory where the training results will be saved.",
        default="runs",
        significant=False,
    )

    def output(self):
        return {
            "logs": law.LocalDirectoryTarget(f"{self.results_dir}/tensorboard_logs"),
            "outputs": law.LocalFileTarget(f"{self.results_dir}/outputs"),
        }

    def run(self):
        self.config = MLConfig.from_yaml(self.config_yaml, strict=True)  # type: ignore

        if self.config.device == "cpu":
            logger.warning("Running on CPU. This may be slow for large datasets.")

        training_base = TrainingBase(
            features_ingestor=Ingestor.read(
                self.config.files_to_load,
                format=self.config.filetype,
                columns=self.config.features_columns,
                reader_kwargs=self.config.dak_reader_kwargs,
                max_number_events=self.config.max_number_events,
            ),
            labels_ingestor=Ingestor.read(
                self.config.files_to_load,
                format=self.config.filetype,
                columns=self.config.labels_columns,
                reader_kwargs=self.config.dak_reader_kwargs,
                max_number_events=self.config.max_number_events,
            ),
            config=self.config,
            tensor_board_log_dir=self.output()["logs"].path,
        )
        outputs = training_base.train()

        with open(os.path.join(self.output()["outputs"], "training_output.json"), "w") as f:
            json.dump(outputs, f, indent=4)

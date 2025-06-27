import time
import os
import json

import law

from ml.config import MLConfig
from preprocessor.ingestion.formatter import Ingestor
from orchestrator.train import TrainingBase


class TrainingBaseTask(law.Task):

    config_yaml = law.Parameter(
        description="Path to the YAML configuration file for the training.",
        default="ml/config/config.yaml"
    )

    def output(self):
        time_stamp = int(time.time())
        base_dir = os.path.join("training_output", f"{time_stamp}")
        return {
            "logs": law.LocalDirectoryTarget(f"{base_dir}/tensorboard_logs"),
            "outputs": law.LocalFileTarget(f"{base_dir}/outputs"),
        }

    def run(self):
        self.config = MLConfig.from_yaml(self.config_yaml, strict=True)  # type: ignore
        training_base = TrainingBase(
            features_ingestor=Ingestor.read(
                self.config.files_to_load,
                format=self.config.filetype,
                columns=self.config.features_columns,
                reader_kwargs=self.config.dak_reader_kwargs,
            ),
            labels_ingestor=Ingestor.read(
                self.config.files_to_load,
                format=self.config.filetype,
                columns=self.config.labels_columns,
                reader_kwargs=self.config.dak_reader_kwargs,
            ),
            config=self.config,
        )
        outputs = training_base.train()

        with open(os.path.join(self.output()["outputs"], "training_output.json"), "w") as f:
            json.dump(outputs, f, indent=4)

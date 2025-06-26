import law

from ml.config import MLConfig
from preprocessor.ingestion.formatter import Ingestor
from orchestrator.train import TrainingBase


class TrainingBaseTask(law.Task):

    config_yaml = law.Parameter(
        help="Path to the YAML configuration file for the training.",
        default="ml/config/config.yaml"
    )

    def run(self):
        self.config = MLConfig.from_yaml(self.config_yaml, strict=True)  # type: ignore
        training_base = TrainingBase(
            ingestor=Ingestor.from_parquet(self.config.files_to_load),
            config=self.config
        )
        training_base.train()

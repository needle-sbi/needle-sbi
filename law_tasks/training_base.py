import os
from pathlib import Path
from typing import Any, Dict

import law
import lightning
from lightning.pytorch.loggers import MLFlowLogger

from ml.lightning.mock_transformer import MockTransformerModule
from ml.lightning.padded_datamodule import PaddedDataModule
from ml.utils import MLConfig
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

    def warn_if_device_is_cpu(self):
        if self.config.device == "cpu":
            logger.warning("Running on CPU. This may be slow for large datasets.")

    def run(self):
        """Simple implementation for testing.

        Should be overridden in derived classes.
        """
        self.warn_if_device_is_cpu()
        mlflow_logger = MLFlowLogger(experiment_name="base")
        model = MockTransformerModule(
            config=self.config,
            tensor_board_log_dir=self.output().get("logs"),
        )
        data_module = PaddedDataModule(
            config=self.config,
        )
        trainer = lightning.Trainer(
            max_epochs=self.config.total_epoch,
            logger=mlflow_logger,
        )
        trainer.fit(model=model, datamodule=data_module)
        model.results.to_json(self.output()["outputs"].path)


if __name__ == "__main__":
    training_base = TrainingBaseTask()
    training_base.law_run()

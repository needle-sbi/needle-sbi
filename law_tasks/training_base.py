import os
from pathlib import Path
from typing import Any, Dict

import hydra
import law
import lightning
from lightning.pytorch.loggers import MLFlowLogger
from omegaconf import OmegaConf

from orchestrator.config import MainConfig
from orchestrator.results import TrainingResults
from preprocessor.utils.logging import ColorFormatter

logger = ColorFormatter.get_logger("orchestrator")


class TrainingBaseTask(law.Task):
    def __init_subclass__(cls, **kwargs):
        if "run" not in cls.__dict__:
            raise TypeError(
                "Classes inheriting from TrainingBaseTask must implement their "
                "own 'run()', as law will otherwise not resolve dependencies correctly."
            )

    config_path = law.Parameter(
        description="Path to the hydra conf folder for the training.",
        default="conf",
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
    def config(self) -> MainConfig:
        with hydra.initialize(config_path=str(Path("..") / str(self.config_path))):
            return OmegaConf.structured(hydra.compose(config_name="config"))

    def run(self):
        """Simple implementation for testing.

        Must be overridden in derived classes.
        """
        mlflow_logger = MLFlowLogger(experiment_name="base")

        model: lightning.LightningModule = hydra.utils.instantiate(
            self.config.models,
            dataset_config=self.config.datasets,
        )
        data_module: lightning.LightningDataModule = hydra.utils.instantiate(
            self.config.datamodules,
            dataset_config=self.config.datasets,
        )
        trainer: lightning.Trainer = hydra.utils.instantiate(self.config.trainers, logger=mlflow_logger)

        trainer.fit(model=model, datamodule=data_module)

        results = TrainingResults(float(trainer.callback_metrics["val_loss"]))
        results.to_json(self.output()["outputs"].path)


if __name__ == "__main__":
    training_base = TrainingBaseTask()
    training_base.law_run()

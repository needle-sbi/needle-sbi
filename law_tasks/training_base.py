import os
from pathlib import Path
from typing import Any, Dict

import hydra
import law
import lightning
from lightning.pytorch.loggers import MLFlowLogger

from law_tasks.mixins import HydraMixin
from orchestrator.registry import (
    validate_datamodule_config,
    validate_model_config,
    validate_trainer_config,
)
from orchestrator.results import TrainingResults
from preprocessor.utils.logging import ColorFormatter

logger = ColorFormatter.get_logger("orchestrator")


class TrainingBaseTask(law.Task, HydraMixin):
    def __init_subclass__(cls, **kwargs):
        if "run" not in cls.__dict__:
            raise TypeError(
                "Classes inheriting from TrainingBaseTask must implement their "
                "own 'run()', as law will otherwise not resolve dependencies correctly."
            )

    rel_results_path = law.Parameter(
        description="Directory where the training results will be saved.",
        default="runs",
        significant=False,
    )

    def output(self) -> Dict[str, Any]:
        if not os.path.isdir(self.abs_results_path):
            os.makedirs(self.abs_results_path, exist_ok=True)
        return {
            "logs": law.LocalDirectoryTarget(f"{self.abs_results_path}/tensorboard_logs"),
            "outputs": law.LocalFileTarget(f"{self.abs_results_path}/training_output.json"),
        }

    @property
    def abs_results_path(self) -> Path:
        return os.path.abspath(self.rel_results_path)  # type: ignore

    def run(self):
        """Simple implementation for testing.

        Must be overridden in derived classes.
        """
        mlflow_logger = MLFlowLogger(experiment_name="base")
        validated_model_config = validate_model_config(self.config.models)
        validated_datamodule_config = validate_datamodule_config(self.config.datamodules)
        validated_trainer_config = validate_trainer_config(self.config.trainers)

        model: lightning.LightningModule = hydra.utils.instantiate(
            validated_model_config,
            dataset_config=self.config.datasets,
        )
        data_module: lightning.LightningDataModule = hydra.utils.instantiate(
            validated_datamodule_config,
            dataset_config=self.config.datasets,
        )
        trainer: lightning.Trainer = hydra.utils.instantiate(validated_trainer_config, logger=mlflow_logger)

        trainer.fit(model=model, datamodule=data_module)

        results = TrainingResults(float(trainer.callback_metrics["val_loss"]))
        results.to_json(self.output()["outputs"].path)


if __name__ == "__main__":
    training_base = TrainingBaseTask()
    training_base.law_run()

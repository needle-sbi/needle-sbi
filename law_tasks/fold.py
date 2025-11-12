"""
Task for a single fold of the training.
"""

import os
from pathlib import Path

import hydra
import law
import lightning
import luigi
from lightning.pytorch.loggers import MLFlowLogger, TensorBoardLogger

from law_tasks.mixins import HydraMixin
from orchestrator.results import FoldResults
from preprocessor.utils import ColorFormatter

logger = ColorFormatter.get_logger("fold")


class FoldTask(law.Task, HydraMixin):
    fold_index = luigi.IntParameter(
        description="K-Fold index",
        significant=True,
    )

    rel_results_path = law.Parameter(
        description="Directory where the fold training results will be saved.",
        default="runs",
        significant=False,
    )

    def output(self):
        return {
            "logs": law.LocalDirectoryTarget(f"{self.abs_results_path}/fold_{self.fold_index}/tensorboard_logs"),
            "outputs": law.LocalFileTarget(f"{self.abs_results_path}/fold_{self.fold_index}/training_output.json"),
        }

    @property
    def abs_results_path(self) -> Path:
        return os.path.abspath(self.rel_results_path)  # type: ignore

    def run(self):
        mlflow_logger = MLFlowLogger(save_dir=self.output()["logs"].path, experiment_name="mlflow")
        tensorboard_logger = TensorBoardLogger(save_dir=self.output()["logs"].path, name="tensorboard")

        model: lightning.LightningModule = hydra.utils.instantiate(
            self.config.models,
            dataset_config=self.config.datasets,
        )
        data_module: lightning.LightningDataModule = hydra.utils.instantiate(
            self.config.datamodules,
            dataset_config=self.config.datasets,
            fold_index=self.fold_index,
        )
        trainer: lightning.Trainer = hydra.utils.instantiate(
            self.config.trainers,
            logger=[mlflow_logger, tensorboard_logger],
        )

        trainer.fit(model=model, datamodule=data_module)

        results = FoldResults(
            best_validation_loss=float(trainer.callback_metrics["val_loss"]),
            n_folds=self.config.n_folds,
            fold_index=self.fold_index,  # type: ignore
        )
        self.output()["outputs"].touch()
        results.to_json(self.output()["outputs"].path)

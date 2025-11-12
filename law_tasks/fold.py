"""
Task for a single fold of the training.
"""

import hydra
import law
import lightning
import luigi
from lightning.pytorch.loggers import MLFlowLogger

from law_tasks.training_base import TrainingBaseTask
from orchestrator.results import FoldResults
from preprocessor.utils import ColorFormatter

logger = ColorFormatter.get_logger("fold")


class FoldTask(TrainingBaseTask):
    fold_index = luigi.IntParameter(
        description="K-Fold index",
        significant=True,
    )

    def output(self):
        return {
            "logs": law.LocalDirectoryTarget(f"{self.results_dir}/fold_{self.fold_index}/tensorboard_logs"),
            "outputs": law.LocalFileTarget(f"{self.results_dir}/fold_{self.fold_index}/training_output.json"),
        }

    def run(self):
        """Simple implementation for testing.

        Should be overridden in derived classes.
        """
        mlflow_logger = MLFlowLogger(experiment_name="base")

        model: lightning.LightningModule = hydra.utils.instantiate(
            self.config.models,
            dataset_config=self.config.datasets,
        )
        data_module: lightning.LightningDataModule = hydra.utils.instantiate(
            self.config.datamodules,
            dataset_config=self.config.datasets,
            fold_index=self.fold_index,
        )
        trainer: lightning.Trainer = hydra.utils.instantiate(self.config.trainers, logger=mlflow_logger)

        trainer.fit(model=model, datamodule=data_module)

        results = FoldResults(
            best_validation_loss=float(trainer.callback_metrics["val_loss"]),
            n_folds=self.config.n_folds,
            fold_index=self.fold_index,  # type: ignore
        )
        results.to_json(self.output()["outputs"].path)

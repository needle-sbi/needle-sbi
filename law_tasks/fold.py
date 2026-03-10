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
from omegaconf import OmegaConf

from law_tasks.mixins import HydraMixin
from orchestrator.results import FoldResults
from orchestrator.config import EstimatorConfig, SystematicConfig
from preprocessor.utils import ColorFormatter

logger = ColorFormatter.get_logger("fold")


class FoldTask(law.Task, HydraMixin):
    rel_results_path = law.Parameter(
        description="Directory where the fold training results will be saved.",
        default="runs",
        significant=False,
    )
    estimator: str = law.Parameter(
        description="Name of the estimator (must be included in config).",
        significant=True,
    )  # type: ignore
    systematic: str = law.Parameter(
        description="Name of the systematic uncertainty.",
        significant=True,
    )  # type: ignore
    ensemble: int = luigi.IntParameter(
        description="Index of the ensemble (type: int).",
        default=0,
        significant=True,
    )  # type: ignore
    fold_index: int = luigi.IntParameter(
        description="Index of the cross-validation fold (type: int)",
        default=0,
        significant=True,
    )  # type: ignore

    def output(self):
        return {
            "logs": law.LocalDirectoryTarget(f"{self.abs_results_path}/fold_{self.fold_index}/tensorboard_logs"),
            "outputs": law.LocalFileTarget(f"{self.abs_results_path}/fold_{self.fold_index}/training_output.json"),
        }

    @property
    def abs_results_path(self) -> Path:
        return os.path.abspath(self.rel_results_path)  # type: ignore

    @property
    def estimator_config(self) -> EstimatorConfig:
        return self.config.estimators[self.estimator]

    @property
    def systematic_config(self) -> SystematicConfig:
        """Populate the entries of the Systematic with the values from the Estimator and update them
        with potential overrides
        """
        return OmegaConf.merge(
            OmegaConf.to_container(
                self.estimator_config.expands.systematics[self.systematic],
                resolve=False,
            ),
            self.estimator_config,
        )  # type: ignore

    def run(self):
        mlflow_logger = MLFlowLogger(save_dir=self.output()["logs"].path, experiment_name="mlflow")
        tensorboard_logger = TensorBoardLogger(save_dir=self.output()["logs"].path, name="tensorboard")

        model_config = self.systematic_config.model_override
        datamodule_config = self.systematic_config.datamodule_override
        dataset_config = self.systematic_config.dataset_override
        trainer_config = self.systematic_config.trainer_override

        model: lightning.LightningModule = hydra.utils.instantiate(
            model_config,
            dataset_config=dataset_config,
        )
        data_module: lightning.LightningDataModule = hydra.utils.instantiate(
            datamodule_config,
            dataset_config=dataset_config,
            fold_index=self.fold_index,
            n_folds=self.estimator_config.expands.folds,
        )
        trainer: lightning.Trainer = hydra.utils.instantiate(
            trainer_config,
            logger=[mlflow_logger, tensorboard_logger],
        )

        trainer.fit(model=model, datamodule=data_module)

        results = FoldResults(
            best_validation_loss=float(trainer.callback_metrics["val_loss"]),
            n_folds=self.estimator_config.expands.folds,
            fold_index=self.fold_index,  # type: ignore
        )
        self.output()["outputs"].touch()
        results.to_json(self.output()["outputs"].path)

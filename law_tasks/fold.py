"""
Task for a single fold of the training.
"""

import os
from pathlib import Path
from typing import Any, Dict, List

import hydra
import law
import lightning
import luigi
import mlflow
import torch
from lightning.pytorch.loggers import MLFlowLogger
from omegaconf import OmegaConf

from law_tasks.mixins import HydraMixin
from law_tasks.training_base import TrainingBase
from orchestrator.config import EstimatorConfig, SystematicConfig
from orchestrator.config_utils import hydra_check_if_arg_supported
from orchestrator.results import FoldResults
from preprocessor.utils import ColorFormatter

logger = ColorFormatter.get_logger("fold")


class FoldTask(HydraMixin, law.Task, TrainingBase):
    results_path: str = law.Parameter(
        description="Directory where the fold training results will be saved.",
        significant=False,
    )  # type: ignore
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

    @property
    def abs_results_path(self) -> Path:
        return os.path.abspath(self.results_path)  # type: ignore

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

    def requires(self) -> List[Any]:
        if not self.estimator_config.requires:
            return []

        from law_tasks import EstimatorTask

        return [
            EstimatorTask(
                config_file=str(self.config_file),
                estimator=dependency,
            )
            for dependency in self.estimator_config.requires
        ]

    def output(self) -> Dict[str, Any]:
        """Define all output targets for this task"""
        base = law.LocalDirectoryTarget(self.abs_results_path)
        return {
            "dir": base,
            "ckpt": base.child("model.ckpt", type="f"),
            "model_config": base.child("model_config.yaml", type="f"),
            "metrics": base.child("metrics", type="d"),
            "outputs": base.child("fold_results.json", type="f"),  # Add this!
        }

    @property
    def mlflow_logger(self) -> MLFlowLogger:
        return MLFlowLogger(
            experiment_name=f"est={self.estimator}_syst={self.systematic}_ens={self.ensemble}_fold={self.fold_index}",
            save_dir=os.path.join(self.output()["metrics"].path, "mlflow"),
            log_model=False,
        )

    def run(self):
        torch.set_float32_matmul_precision("high")

        model_config = self.systematic_config.model_override
        datamodule_config = self.systematic_config.datamodule_override
        dataset_config = self.systematic_config.dataset_override
        trainer_config = self.systematic_config.trainer_override

        # 1. Load model
        if hydra_check_if_arg_supported(model_config, "dataset_config"):
            model: lightning.LightningModule = hydra.utils.instantiate(model_config, dataset_config)
        else:
            model = hydra.utils.instantiate(model_config)

        # 2. Load datamodule
        folds_api_arguments = ["fold_index", "n_folds"]
        data_module_supports_folds: bool = all(
            hydra_check_if_arg_supported(datamodule_config, p) for p in folds_api_arguments
        )  # TODO could be also solved with appropriate Protocol

        if not data_module_supports_folds:
            raise TypeError(
                "The datamodule does not support the API for cross-fold validation. "
                f"Your datamodule must accept the arguments: {folds_api_arguments} (type=int)"
            )

        if hydra_check_if_arg_supported(datamodule_config, "dataset_config"):
            data_module: lightning.LightningDataModule = hydra.utils.instantiate(
                datamodule_config,
                dataset_config=dataset_config,
                fold_index=self.fold_index,
                n_folds=self.estimator_config.expands.folds,
            )
        else:
            data_module = hydra.utils.instantiate(
                datamodule_config,
                fold_index=self.fold_index,
                n_folds=self.estimator_config.expands.folds,
            )

        # 3. Load trainer
        trainer: lightning.Trainer = hydra.utils.instantiate(
            trainer_config,
            logger=self.mlflow_logger,
        )
        trainer.fit(model=model, datamodule=data_module)

        # 4. Record metrics
        checkpoint_path = Path(self.output()["ckpt"].path)
        trainer.save_checkpoint(checkpoint_path)

        with mlflow.start_run(run_id=self.mlflow_logger.run_id):
            mlflow.pytorch.log_model(pytorch_model=model, name="model")
            mlflow.log_artifact(str(checkpoint_path), artifact_path="checkpoints")

        with open(Path(self.output()["model_config"].path), "w") as f:
            OmegaConf.save(model_config, f)

        metrics = {
            "best_val_loss": float(trainer.callback_metrics.get("val_loss", 0.0)),
        }

        fold_results = FoldResults(
            best_validation_loss=metrics["best_val_loss"],
            fold_index=self.fold_index,
            n_folds=self.estimator_config.expands.folds,
        )
        fold_results.to_json(self.output()["outputs"].path)

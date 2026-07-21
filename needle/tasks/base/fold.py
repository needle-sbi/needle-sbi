from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Type
from urllib.parse import urlencode

import hydra
import lightning
import luigi
import mlflow
import torch
from lightning.pytorch.loggers import MLFlowLogger
from omegaconf import OmegaConf

from needle.tasks.mixins.hydra import HydraParamsMixin
from needle.utils.config_schema import EstimatorConfig, SystematicConfig
from needle.utils.config_utils import hydra_check_if_arg_supported, hydra_instantiate
from needle.utils.logging import ColorFormatter
from needle.utils.results import FoldResults

logger = ColorFormatter.get_logger("fold")


class BaseFoldTask(HydraParamsMixin, luigi.Task):
    """Backend-agnostic base for FoldTask.

    Contains all training logic. Subclasses override ``output()`` and
    ``_estimator_task_class()`` to bind to a specific backend.
    """

    results_path: str = luigi.Parameter(
        description="Root directory where results are saved.",
        significant=False,
    )  # type: ignore
    estimator: str = luigi.Parameter(
        description="Name of the estimator (must be included in config).",
        significant=True,
    )  # type: ignore
    systematic: str = luigi.Parameter(
        description="Name of the systematic uncertainty.",
        default="nominal",
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
        return Path(
            os.path.join(
                os.path.abspath(self.results_path),
                f"est__{self.estimator}",
                f"syst__{self.systematic}",
                f"ensem__{self.ensemble}",
                f"fold__{self.fold_index}",
            )
        )

    @property
    def estimator_config(self) -> EstimatorConfig:
        return self.config.estimators[self.estimator]

    @property
    def systematic_config(self) -> SystematicConfig:
        return OmegaConf.merge(
            OmegaConf.to_container(
                self.estimator_config.expands.systematics[self.systematic],
                resolve=False,
            ),
            self.estimator_config,
        )  # type: ignore

    @property
    def input_model_paths(self) -> Dict[str, str]:
        """Collect checkpoint paths from dependency estimators (transfer learning)."""
        model_paths_dict: Dict[str, str] = {}

        for estimator_task in self.requires():
            for systematic_task in estimator_task.requires():
                for ensemble_task in systematic_task.requires():
                    for fold_task in ensemble_task.requires():
                        key = urlencode(
                            {
                                "est": estimator_task.estimator,
                                "syst": systematic_task.systematic,
                                "ensem": ensemble_task.ensemble,
                                "fold": fold_task.fold_index,
                            }
                        )
                        path: str = fold_task.output_as_dict(fold_task.output())["ckpt"].path  # type: ignore
                        model_paths_dict[key] = path

        return model_paths_dict

    @staticmethod
    def output_as_dict(fold_output: Dict[str, Any]) -> Dict[str, Any]:
        """Return the output dict as-is (base implementation for local/b2luigi backends).

        The LAW subclass overrides this to unwrap remote ``TargetCollection`` objects.
        """
        return fold_output

    def output(self) -> Dict[str, Any]:
        """Base output using ``luigi.LocalTarget``. Backends may override with richer targets."""
        base = str(self.abs_results_path)
        return {
            "ckpt": luigi.LocalTarget(os.path.join(base, "model.ckpt")),
            "model_config": luigi.LocalTarget(os.path.join(base, "model_config.yaml")),
            "outputs": luigi.LocalTarget(os.path.join(base, "fold_results.json")),
            "input_models": luigi.LocalTarget(os.path.join(base, "input_models.json")),
        }

    def _estimator_task_class(self) -> Type[luigi.Task]:
        raise NotImplementedError("Backend subclass must implement _estimator_task_class()")

    def requires(self) -> List[Any]:
        if not self.estimator_config.requires:
            return []

        EstimatorTask = self._estimator_task_class()
        return [
            EstimatorTask(
                config_file=str(self.config_file),
                hydra_overrides=self.hydra_overrides,
                estimator=dependency,
                results_path=self.results_path,
            )
            for dependency in self.estimator_config.requires
        ]

    @property
    def mlflow_logger(self) -> MLFlowLogger:
        experiment_name = urlencode(
            {
                "est": self.estimator,
                "syst": self.systematic,
                "ens": self.ensemble,
                "fold": self.fold_index,
            }
        )
        return MLFlowLogger(
            experiment_name=experiment_name,
            save_dir=os.path.join(self.results_path, "metrics"),  # type: ignore
            log_model=False,
        )

    def run(self) -> None:
        torch.set_float32_matmul_precision("high")

        model_config = self.systematic_config.model_override
        datamodule_config = self.systematic_config.datamodule_override
        dataset_config = self.systematic_config.dataset_override
        trainer_config = self.systematic_config.trainer_override

        # 1. Load model
        model: lightning.LightningModule = hydra_instantiate(
            model_config,
            dataset_config=dataset_config,
            input_models=self.input_model_paths,
        )

        # 2. Load datamodule
        folds_api_arguments = ["fold_index", "n_folds"]
        data_module_supports_folds: bool = all(
            hydra_check_if_arg_supported(datamodule_config, p) for p in folds_api_arguments
        )

        if not data_module_supports_folds:
            logger.warning(
                "The datamodule does not support the API for cross-fold validation. "
                f"Your datamodule must accept the arguments: {folds_api_arguments} (type=int)"
            )

        data_module: lightning.LightningDataModule = hydra_instantiate(
            datamodule_config,  # type: ignore
            dataset_config=dataset_config,
            input_models=self.input_model_paths,
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

        with open(self.output()["input_models"].path, "w") as f:
            json.dump(self.input_model_paths, f)

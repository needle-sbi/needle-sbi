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

logger = ColorFormatter.get_logger("training")


class BaseTrainingTask(HydraParamsMixin, luigi.Task):
    """Backend-agnostic base for the leaf training task.

    Runs the PyTorch Lightning training loop for a single fold.
    Backends override ``output()``, ``output_as_dict()``, and ``_estimator_task_class()``
    to bind to a specific execution backend.
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
    single: bool = luigi.BoolParameter(
        description="Train this estimator directly, ignoring `requires` and `expands` entirely "
        "(no upstream estimators, no systematic/ensemble/fold fan-out). Writes flat output "
        "directly under results_path with no est__/syst__/ensem__/fold__ nesting, so it can "
        "never collide with a real DAG run's outputs in the same results_path.",
        default=False,
        significant=False,
    )  # type: ignore

    @property
    def abs_results_path(self) -> Path:
        if self.single:
            return Path(os.path.abspath(self.results_path))

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
        """Override Estimator config with Systematic fields

        Returns:
            SystematicConfig: Container with overridden fields for the current Systematics config
        """
        return OmegaConf.merge(
            OmegaConf.to_container(
                self.estimator_config.expands.systematics[self.systematic],
                resolve=False,
            ),
            self.estimator_config,
        )  # type: ignore

    @property
    def input_model_paths(self) -> Dict[str, str]:
        """Collect checkpoint paths from all upstream estimators."""
        model_paths_dict: Dict[str, str] = {}
        training_task: BaseTrainingTask

        for estimator_task in self.requires():
            for systematic_task in estimator_task.requires():
                for ensemble_task in systematic_task.requires():
                    for fold_task in ensemble_task.requires():
                        (training_task,) = fold_task.requires()
                        key = urlencode(
                            {
                                "est": estimator_task.estimator,
                                "syst": systematic_task.systematic,
                                "ensem": ensemble_task.ensemble,
                                "fold": fold_task.fold_index,
                            }
                        )
                        path: str = training_task.output_as_dict(training_task.output())["ckpt"].path  # type: ignore
                        model_paths_dict[key] = path

        return model_paths_dict

    @staticmethod
    def output_as_dict(task_output: Dict[str, Any]) -> Dict[str, Any]:
        """Return output dict as-is. Law backend overrides to unwrap remote TargetCollection."""
        return task_output

    def output(self) -> Dict[str, Any]:  # type: ignore
        """Base implementation of the return files as plain luigi Targets

        Returns:
            Dict[str, Any]: Output dict with .ckpt, model_config
        """
        base = str(self.abs_results_path)
        return {
            "ckpt": luigi.LocalTarget(os.path.join(base, "model.ckpt")),
            "model_config": luigi.LocalTarget(os.path.join(base, "model_config.yaml")),
            "input_models": luigi.LocalTarget(os.path.join(base, "input_models.json")),
        }

    def _estimator_task_class(self) -> Type[luigi.Task]:
        raise NotImplementedError("Backend subclass must implement _estimator_task_class()")

    def requires(self) -> List[Any]:
        if self.single or not self.estimator_config.requires:
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
        if self.single:
            experiment_name = urlencode({"est": self.estimator, "mode": "single"})
        else:
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

        if self.single:
            model_config = self.estimator_config.model_override
            datamodule_config = self.estimator_config.datamodule_override
            dataset_config = self.estimator_config.dataset_override
            trainer_config = self.estimator_config.trainer_override
        else:
            model_config = self.systematic_config.model_override
            datamodule_config = self.systematic_config.datamodule_override
            dataset_config = self.systematic_config.dataset_override
            trainer_config = self.systematic_config.trainer_override

        model: lightning.LightningModule = hydra_instantiate(
            model_config,
            dataset_config=dataset_config,
            input_models=self.input_model_paths,
        )

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
            fold_index=0 if self.single else self.fold_index,
            n_folds=1 if self.single else self.estimator_config.expands.folds,
        )

        trainer: lightning.Trainer = hydra.utils.instantiate(
            trainer_config,
            logger=self.mlflow_logger,
        )
        trainer.fit(model=model, datamodule=data_module)

        checkpoint_path = Path(self.output()["ckpt"].path)
        trainer.save_checkpoint(checkpoint_path)

        with mlflow.start_run(run_id=self.mlflow_logger.run_id):
            mlflow.pytorch.log_model(pytorch_model=model, name="model")  # type: ignore
            mlflow.log_artifact(str(checkpoint_path), artifact_path="checkpoints")

        with open(Path(self.output()["model_config"].path), "w") as f:
            OmegaConf.save(model_config, f)

        with open(self.output()["input_models"].path, "w") as f:
            json.dump(self.input_model_paths, f)

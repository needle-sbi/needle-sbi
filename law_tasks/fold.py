"""
Task for a single fold of the training.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import urlencode

import hydra
import law
import lightning
import luigi
import mlflow
import torch
from lightning.pytorch.loggers import MLFlowLogger
from omegaconf import OmegaConf

from law_tasks.mixins import HydraMixin
from law_tasks.workflows import (
    HTCondorWorkflow,
    LocalWorkflow,
    SlurmWorkflow,
    check_batch_system,
)
from orchestrator.config import EstimatorConfig, SystematicConfig
from orchestrator.config_utils import hydra_check_if_arg_supported, hydra_instantiate
from orchestrator.results import FoldResults
from preprocessor.utils import ColorFormatter

logger = ColorFormatter.get_logger("fold")

FoldTaskOutput = Dict[str, law.LocalFileTarget] | Dict[str, law.TargetCollection]


class FoldTask(
    HydraMixin,
    LocalWorkflow,
    HTCondorWorkflow,
    SlurmWorkflow,
):
    results_path: str = law.Parameter(
        description="Root directory where results are saved.",
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
        return Path(
            os.path.join(
                os.path.abspath(self.results_path),
                f"est__{self.estimator}",
                f"syst__{self.systematic}",
                f"ensem__{self.ensemble}",
                f"fold__{self.ensemble}",
            )
        )

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

    @property
    def input_model_paths(self) -> Dict[str, str]:
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
                        path = fold_task.output()["ckpt"].path
                        model_paths_dict[key] = path

        return model_paths_dict

    def create_branch_map(self) -> Dict[int, None]:  # type: ignore
        return {0: None}

    def requires(self) -> List[Any]:
        if not self.estimator_config.requires:
            return []

        from law_tasks import EstimatorTask  # Avoid circular imports

        return [
            EstimatorTask(
                config_file=str(self.config_file),
                hydra_overrides=self.hydra_overrides,
                estimator=dependency,
                results_path=self.results_path,
            )
            for dependency in self.estimator_config.requires
        ]

    def output(self) -> Dict[str, Any]:
        """Define all output targets for this task

        Important:
            If using this method in another Task, beware that for remote jobs, the output of this
            method will be wrapped in a Dict of Lists to account for each potential branch of the
            workflow. To avoid encountering this problem, use the `output_as_dict` method instead,
            which flattens the remote output to the same shape as the local version.
        """
        check_batch_system(system=str(self.workflow))  # type: ignore

        base = law.LocalDirectoryTarget(self.abs_results_path)
        return {
            "dir": base,
            "ckpt": base.child("model.ckpt", type="f"),
            "model_config": base.child("model_config.yaml", type="f"),
            "metrics": base.child("metrics", type="d"),
            "outputs": base.child("fold_results.json", type="f"),
            "input_models": base.child("input_models.json", type="f"),
        }

    @staticmethod
    def output_as_dict(fold_output: FoldTaskOutput) -> Dict[str, law.LocalTarget]:
        """Unpack local and remote inputs

         1. Local is simply the Dict defined in the output method of the FoldTask
         2. Remote is instead a DotDict with 'collection' and 'jobs' fields.

        Example:
            print(super().input())
            [
                DotDict(
                    {
                        "jobs": law.LocalFileTarget(),
                        "collection": law.TargetCollection(len=1)
                    }
                )
            ]

        Returns:
            Dict[str, str]: Properly formatted output Dict with key:Target pairs
        """
        remote_collection: law.TargetCollection | None = fold_output.get("collection")  # type: ignore

        if remote_collection:
            if len(remote_collection) != 1:
                raise NotImplementedError(
                    "Currently the usage of branches in FoldTask is not supported. Instead, folds "
                    "have to be their own Task instance required by EnsembleTask."
                )
            return remote_collection[0]
        else:
            return fold_output  # type: ignore

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
        model: lightning.LightningModule = hydra_instantiate(
            model_config,  # type: ignore
            dataset_config=dataset_config,
            input_models=self.input_model_paths,
        )

        # 2. Load datamodule
        folds_api_arguments = ["fold_index", "n_folds"]
        data_module_supports_folds: bool = all(
            hydra_check_if_arg_supported(datamodule_config, p) for p in folds_api_arguments
        )  # TODO could be also solved with appropriate Protocol

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

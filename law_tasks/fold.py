"""
Task for a single fold of the training.
"""

import os
import json 
from pathlib import Path
from typing import Dict, Any

import hydra
import law
import lightning
import luigi
from omegaconf import OmegaConf
from lightning.pytorch.loggers import Logger, MLFlowLogger

from law_tasks.mixins import HydraMixin
from law_tasks.training_base import TrainingBase
from orchestrator.config import EstimatorConfig, SystematicConfig
from orchestrator.results import FoldResults 
from preprocessor.utils import ColorFormatter

logger = ColorFormatter.get_logger("fold")


class FoldTask(law.Task, TrainingBase, HydraMixin):
    results_path = law.Parameter(
        description="Directory where the fold training results will be saved.",
        default="runs",
        significant=False,
    )
    workflow = luigi.ChoiceParameter(
        default="local",
        choices=["local", "htcondor"],
        significant=False,
        description="Execution mode: local (run in-process) or htcondor (submit to cluster)"
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

    #@property
    #def abs_results_path(self) -> Path:
    #    return os.path.abspath(self.rel_results_path)  # type: ignore

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

    def requires(self):
        if not self.estimator_config.requires:
            return []

        from law_tasks import EstimatorTask

        return [
            EstimatorTask.req(
                self,
                config_file=self.config_file,
                estimator=dependency,
            )
            for dependency in self.estimator_config.requires
        ]

    def output(self) -> Dict[str, Any]:
        """Define all output targets for this task"""
        #base_path = f"{self.abs_results_path}/{self.estimator}_{self.systematic}_ens{self.ensemble}_fold{self.fold_index}"
        fold_dir = os.path.join(
            str(self.abs_results_path),
            self.estimator,
            self.systematic,
            f"ensemble_{self.ensemble}",
            f"fold_{self.fold_index}",
        )

        base = law.LocalDirectoryTarget(fold_dir)
        return {
            "dir": base,
            "ckpt": base.child("model.ckpt", type="f"),
            "model_config": base.child("model_config.yaml", type="f"),
            "metrics": base.child("metrics.json", type="f"),
            "outputs": base.child("fold_results.json", type="f"),  # Add this!
        }
    
    #@property
    #def lightning_logger(self) -> Logger:
    #    return MLFlowLogger(
    #        experiment_name=self.output()["dir"],
    #        save_dir=self.output()["metrics"],
    #        log_model=True,
    #    )

    def run(self):
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
                
        # Create logger directly with correct paths
        mlflow_dir = Path(self.output()["dir"].path) / "mlflow"
        mlflow_dir.mkdir(parents=True, exist_ok=True)
        mlflow_logger = MLFlowLogger(
            experiment_name=f"{self.estimator}_{self.systematic}_ens{self.ensemble}_fold{self.fold_index}",
            save_dir=str(mlflow_dir),  #str(self.output()["dir"].path),  # Use dir, not logs
            log_model=False,
        )
        # Now train
        trainer: lightning.Trainer = hydra.utils.instantiate(
            trainer_config,
            logger=mlflow_logger,
        )
        trainer.fit(model=model, datamodule=data_module)

        # Save checkpoint to the output target
        checkpoint_path = Path(self.output()["ckpt"].path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Manually log the model to MLflow with a clean artifact path
        import mlflow
        with mlflow.start_run(run_id=mlflow_logger.run_id):
            mlflow.pytorch.log_model(
                pytorch_model=model,
                artifact_path="model",  # Clean path without special characters
                registered_model_name=f"{self.estimator}_{self.systematic}_fold{self.fold_index}",
            )
            # Also log the checkpoint file
            mlflow.log_artifact(str(checkpoint_path), artifact_path="checkpoints")
        
        # Save model config
        model_config_path = Path(self.output()["model_config"].path)
        with open(model_config_path, 'w') as f:
            OmegaConf.save(model_config, f)
        
        # Extract metrics from trainer
        metrics = {
            "best_val_loss": float(trainer.callback_metrics.get("val_loss", 0.0)),
            "final_train_loss": float(trainer.callback_metrics.get("train_loss", 0.0)),
        }
        
        # Save metrics
        metrics_path = Path(self.output()["metrics"].path)
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Create FoldResults object
        fold_results = FoldResults(
            best_validation_loss=metrics["best_val_loss"],
            fold_index=self.fold_index,
            n_folds=self.estimator_config.expands.folds,
            #final_train_loss=metrics["final_train_loss"],
        )

        # Save fold results using the SerializableDataclass method
        fold_results.to_json(self.output()["outputs"].path)
        
        # Save fold results (this is what EnsembleTask will read)
        #self.output()["outputs"].dump(fold_results)
        
        logger.info(f"Fold {self.fold_index} completed successfully")

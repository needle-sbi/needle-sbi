import os
from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict

import law
from lightning.pytorch.loggers import Logger, MLFlowLogger

from preprocessor.utils.logging import ColorFormatter

logger = ColorFormatter.get_logger("orchestrator")


class TrainingBase:
    rel_results_path = law.Parameter(
        description="Directory where the training results will be saved.",
        default="runs",
        significant=False,
    )

    @property
    def abs_results_path(self) -> Path:
        """Get absolute path to results directory.
        
        Uses __file__ to find workspace root so it works correctly on HTCondor
        worker nodes where cwd is the scratch directory.
        """
        results_path = Path(self.rel_results_path)  # type: ignore
        
        # If already absolute, use as-is
        if results_path.is_absolute():
            return results_path
        
        # Resolve relative to workspace root using module location
        training_base_file = Path(__file__)  # law_tasks/training_base.py
        law_tasks_dir = training_base_file.parent  # law_tasks/
        workspace_root = law_tasks_dir.parent  # orchestrator/
        
        return (workspace_root / results_path).resolve()

    def output(self) -> Dict[str, Any]:
        if not os.path.isdir(self.abs_results_path):
            os.makedirs(self.abs_results_path, exist_ok=True)

        base = law.LocalDirectoryTarget(self.abs_results_path)

        return {
            "dir": base,
            "ckpt": base.child("model.ckpt", type="f"),
            "model_config": base.child("model_config.yaml", type="f"),
            "metrics": base.child("metrics.json", type="f"),
            "logs": base.child("logs.json"),
            "metadata": base.child("metadata.json", type="f"),
        }

    @property
    def lightning_logger(self) -> Logger:
        return MLFlowLogger(
            experiment_name=self.output()["dir"],
            save_dir=self.output()["logs"],
            log_model=True,
        )

    @abstractmethod
    def run(self) -> None:
        """Abstract method. Must be overridden in derived classes."""

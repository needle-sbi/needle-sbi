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
        return os.path.abspath(self.rel_results_path)  # type: ignore

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

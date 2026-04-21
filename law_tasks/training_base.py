import logging
import os
import warnings
from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict

import law

from preprocessor.utils.logging import ColorFormatter

logger = ColorFormatter.get_logger("orchestrator")


class TipFilter(logging.Filter):
    """Suppress advertisement for litmodels checkpointing. Will eventually be fixed by the Lightning
    team.

    Ref: https://github.com/Lightning-AI/pytorch-lightning/issues/21294
    """

    def filter(self, record):
        return "💡 Tip" not in record.getMessage()


logging.getLogger("lightning.pytorch.utilities.rank_zero").addFilter(TipFilter())
logging.getLogger("lightning.pytorch.trainer.connectors.accelerator_connector").setLevel(logging.ERROR)
logging.getLogger("lightning.pytorch.trainer.connectors.signal_connector").setLevel(logging.ERROR)
logging.getLogger("lightning.pytorch.loggers.mlflow").setLevel(logging.ERROR)
logging.getLogger("mlflow.utils.environment").setLevel(logging.ERROR)
logging.getLogger("mlflow.models.model").setLevel(logging.ERROR)
logging.getLogger("mlflow").setLevel(logging.WARNING)
warnings.filterwarnings("once", message="The '*' does not have many workers*")


class TrainingBase:
    """Base class providing common training task functionality.

    Mixin that provides standard result path handling and output structure for training tasks.
    Filters unnecessary logging from PyTorch Lightning to reduce noise.
    """

    results_path = law.Parameter(
        description="Directory where the training results will be saved.",
        default="runs",
        significant=False,
    )

    @property
    def abs_results_path(self) -> Path:
        """Get the absolute path to the results directory.

        Returns:
            Path: Absolute path to results directory.
        """
        return os.path.abspath(self.results_path)  # type: ignore

    def output(self) -> Dict[str, Any]:
        if not os.path.isdir(self.abs_results_path):
            os.makedirs(self.abs_results_path, exist_ok=True)

        base = law.LocalDirectoryTarget(self.abs_results_path)

        return {
            "dir": base,
            "ckpt": base.child("model.ckpt", type="f"),
            "model_config": base.child("model_config.yaml", type="f"),
            "metrics": base.child("metrics.json", type="f"),
            "logs": base.child("logs.json", type="f"),
            "metadata": base.child("metadata.json", type="f"),
        }

    @abstractmethod
    def run(self) -> None:
        """Abstract method. Must be overridden in derived classes."""

from pathlib import Path
from typing import Any, Optional, Tuple, Union

import torch

from needle.utils.logging import ColorFormatter

logger = ColorFormatter.get_logger("needle")


class Model:
    """NEEDLE ensemble model wrapper"""

    model: Any

    def __init__(self, snapshot_path: Union[str, Path], device: Optional[str] = None):
        self.snapshot_path = Path(snapshot_path)

        if not self.snapshot_path.exists():
            raise FileNotFoundError(f"Snapshot not found: {snapshot_path}")

        raise NotImplementedError(
            "The pseudo-model evaluation backend (needle.evaluation) was removed and has not "
            "been reimplemented yet. Loading a trained DAGSnapshot for inference is currently "
            "unsupported."
        )

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Simple prediction without uncertainty"""
        return self.model.eval(x)

    def predict_with_uncertainty(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prediction with uncertainty estimates"""
        return self.model.eval_with_uncertainty(x)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Allow model(x) syntax"""
        return self.predict(x)


def model(snapshot_path: Union[str, Path], device: Optional[str] = None) -> Model:
    """Load NEEDLE model from snapshot"""
    return Model(snapshot_path, device)


__all__ = [
    "Model",
    "model",
]

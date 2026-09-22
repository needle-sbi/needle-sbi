from pathlib import Path
from typing import Dict, Literal, Optional, Tuple, Type, Union

import torch

from needle.evaluation.pseudo_model import NEEDLE as PseudoModelSerial
from needle.evaluation.pseudo_model_parallel import NEEDLEParallel as PseudoModelParallel
from needle.evaluation.pseudo_model_vectorized import NEEDLEVectorized as PseudoModelVectorised
from needle.utils.logging import ColorFormatter

logger = ColorFormatter.get_logger("needle")

#: Available strategies for evaluating the pseudo-model DAG, mapped to their implementation class.
Strategy = Literal["serial", "parallel", "vectorized"]
_STRATEGIES: Dict[Strategy, Type] = {
    "serial": PseudoModelSerial,
    "parallel": PseudoModelParallel,
    "vectorized": PseudoModelVectorised,
}


class Model:
    """NEEDLE ensemble model wrapper"""

    def __init__(
        self,
        snapshot_path: Union[str, Path],
        device: Optional[str] = None,
        strategy: Strategy = "serial",
    ):
        self.snapshot_path = Path(snapshot_path)

        if not self.snapshot_path.exists():
            raise FileNotFoundError(f"Snapshot not found: {snapshot_path}")

        if strategy not in _STRATEGIES:
            raise ValueError(f"Unknown strategy '{strategy}'. Must be one of {list(_STRATEGIES.keys())}")

        pseudo_model_cls = _STRATEGIES[strategy]
        self.model = pseudo_model_cls(snapshot_path=str(snapshot_path), device=device)
        logger.info(f"Loaded NEEDLE model from {snapshot_path} using '{strategy}' strategy")

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Simple prediction without uncertainty"""
        return self.model.eval(x)

    def predict_with_uncertainty(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prediction with uncertainty estimates"""
        return self.model.eval_with_uncertainty(x)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Allow model(x) syntax"""
        return self.predict(x)


def model(snapshot_path: Union[str, Path], device: Optional[str] = None, strategy: Strategy = "serial") -> Model:
    """Load NEEDLE model from snapshot"""
    return Model(snapshot_path, device, strategy)


__all__ = [
    "Model",
    "model",
]

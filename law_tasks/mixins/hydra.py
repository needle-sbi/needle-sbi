from pathlib import Path

import hydra
import law
from omegaconf import OmegaConf

from orchestrator.config import MainConfig


class HydraMixin:
    config_path = law.Parameter(
        description="Path to config folder",
        default="conf",
        significant=True,
    )

    @property
    def config(self) -> MainConfig:
        with hydra.initialize(
            config_path=str(Path("../..") / str(self.config_path)),
            version_base=None,
        ):
            return OmegaConf.structured(hydra.compose(config_name="config"))

from pathlib import Path

import hydra
from omegaconf import OmegaConf

from orchestrator.config import MainConfig


class HydraMixin:
    config_path: str | Path

    @property
    def config(self) -> MainConfig:
        with hydra.initialize(config_path=str(Path("../..") / str(self.config_path))):
            return OmegaConf.structured(hydra.compose(config_name="config"))

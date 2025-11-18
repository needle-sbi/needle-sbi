from pathlib import Path

import hydra
import law
from omegaconf import OmegaConf

from orchestrator.config import MainConfig


class HydraMixin:
    config_file = law.Parameter(
        description="Path to config folder",
        default="conf/config.yaml",
        significant=True,
    )

    _config: MainConfig

    @property
    def config(self) -> MainConfig:
        if hasattr(self, "_config"):
            return self._config

        config_file = Path(str(self.config_file)).resolve()

        print(f"DEBUG {self=} {config_file=}")

        with hydra.initialize_config_dir(
            config_dir=str(config_file.parent),
            version_base=None,
        ):
            self._config = OmegaConf.structured(hydra.compose(config_name=str(config_file.stem)))
            return self._config

    @config.setter
    def config(self, new_config: MainConfig):
        self._config = new_config

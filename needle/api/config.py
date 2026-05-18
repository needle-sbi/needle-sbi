from pathlib import Path
from typing import Union

from omegaconf import OmegaConf

from needle.utils.config_schema import MainConfig
from needle.utils.config_utils import initialize_hydra_config
from needle.utils.logging import ColorFormatter

logger = ColorFormatter.get_logger("needle")


class Config:
    """Configuration manager - reuses needle's Hydra infrastructure"""

    def __init__(self, config_path: Union[str, Path]):
        self.config_path = Path(config_path).resolve()

        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        # Use the same config loader as LAW tasks (but without LAW dependency)
        self.config: MainConfig = initialize_hydra_config(
            config_dir=str(self.config_path.parent),
            config_name=self.config_path.stem,
        )

        logger.info(f"Loaded and resolved config from {config_path}")
        logger.info(f"Config resolved: {self.config._resolved}")

    def get(self, key: str, default=None):
        """Get config value by dot-notation key"""
        return OmegaConf.select(OmegaConf.structured(self.config), key, default=default)

    def __getattr__(self, name):
        """Allow attribute access to config"""
        if name in ["config", "config_path"]:
            return object.__getattribute__(self, name)
        return getattr(self.config, name)


def config(config_path: Union[str, Path]) -> Config:
    """Load configuration using needle's Hydra infrastructure"""
    return Config(config_path)


__all__ = [
    "Config",
    "config",
]

from functools import cache
from pathlib import Path
from typing import List

import luigi

from needle.utils.config_schema import MainConfig
from needle.utils.config_utils import initialize_hydra_config
from needle.utils.logging import ColorFormatter

_DEFAULT_CONFIG = str(Path.cwd() / "conf" / "config.yaml")

logger = ColorFormatter.get_logger("dag")


class HydraParamsMixin:
    """Backend-agnostic mixin for loading Hydra configs into Luigi tasks.

    Uses ``luigi.Parameter`` directly (LAW re-exports this unchanged, so subclasses
    using either backend are unaffected).

    Attributes:
        config_file: Path to the Hydra config YAML.
        hydra_overrides: Space-separated ``key=value`` overrides forwarded to Hydra.
    """

    config_file: str = luigi.Parameter(
        description="Path to the Hydra config file",
        default=_DEFAULT_CONFIG,
        significant=False,
    )  # type: ignore
    hydra_overrides: str = luigi.Parameter(
        description="Overrides to be passed to hydra. Type str. Format: 'key1=value1 key2=value2'",
        significant=False,
        default="",
    )  # type: ignore

    _config: MainConfig

    @property
    def config(self) -> MainConfig:
        """Load and cache the Hydra configuration from file."""
        overrides: List[str] = self.hydra_overrides.split() if self.hydra_overrides else []

        if hasattr(self, "_config"):
            return self._config

        config_file = Path(str(self.config_file)).resolve()
        self._config = initialize_hydra_config(
            config_dir=str(config_file.parent),
            config_name=str(config_file.stem),
            overrides=overrides,
        )
        return self._config

    @config.setter
    def config(self, new_config: MainConfig) -> None:
        self._config = new_config

    @cache
    def print_config_path_once(self) -> None:
        logger.info(f"Using config from path: {self.config_file}")

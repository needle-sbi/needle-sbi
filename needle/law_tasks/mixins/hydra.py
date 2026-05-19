from functools import cache
from pathlib import Path
from typing import List

_DEFAULT_CONFIG = str(Path.cwd() / "conf" / "config.yaml")

import law

from needle.utils.config_schema import MainConfig
from needle.utils.config_utils import initialize_hydra_config
from needle.utils.logging import ColorFormatter

logger = ColorFormatter.get_logger("dag")


class HydraMixin:
    """Mix-In class for loading Hydra configs

    Note:
        Adds:
            - attr: `config_file` (law.Parameter): The config file to use by the instance.
            - prop: `config` (dataclass): The config object as a dataclass created by OmegaConf.

    Example:
        Define your inherited Task:
        >>> class MyClass(HydraMixin, law.Task):
        ...     pass

        The law parameter is only an attribute and cannot be set in __init__ due to how Law works.
        But you can change it after instantiation:

        >>> my_class = MyClass()
        >>> my_class.config_file = </path/to/conf/config.yaml>   # full file path (can be relative)

        You can now change individual values inside the config, for example:

        >>> my_class.config.datasets.paths = <new_paths>

        This will only apply to the current instance of the class, and would need to be repeated in
        downstream Tasks if they also use the HydraMixin. It is not possible to send python objects
        between Tasks. An alternative would be to save the updated config to file and send the path
        to downstream Tasks.
    """

    config_file: str = law.Parameter(
        description="Path to the Hydra config file",
        default=_DEFAULT_CONFIG,
        significant=False,
    )  # type: ignore
    hydra_overrides: str = law.Parameter(
        description="Overrides to be passed to hydra. Type str. Format: 'key1=value1 key2=value2'",
        significant=False,
        default="",
    )  # type: ignore

    _config: MainConfig

    @property
    def config(self) -> MainConfig:
        """Load and cache the Hydra configuration from file.

        Resolves config file path, applies overrides, and caches result for performance.
        Can be overridden after instantiation by setting the property directly.

        Returns:
            MainConfig: Configuration object as OmegaConf DictConfig.
        """
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
    def config(self, new_config: MainConfig):
        self._config = new_config

    @cache
    def print_config_path_once(self) -> None:
        logger.info(f"Using config from path: {self.config_file}")

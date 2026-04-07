from functools import cache
from pathlib import Path

import law

from orchestrator.config import MainConfig
from orchestrator.config_utils import initialize_hydra_config
from preprocessor.utils.logging import ColorFormatter

logger = ColorFormatter.get_logger("hydra")


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

    config_file = law.Parameter(
        description="Path to config folder",
        default="conf/config.yaml",
        significant=True,
    )

    _config: MainConfig

    @property
    def config(self) -> MainConfig:
        """Load the config from the `HydraMixin.config_file` and convert to an instance of MainConfig.

        Returns:
            MainConfig
        """

        if hasattr(self, "_config"):
            return self._config

        config_file = Path(str(self.config_file)).resolve()
        self._config = initialize_hydra_config(
            config_dir=str(config_file.parent),
            config_name=str(config_file.stem),
        )
        return self._config

    @config.setter
    def config(self, new_config: MainConfig):
        self._config = new_config

    @cache
    def print_config_path_once(self) -> None:
        logger.info(f"Using config from path: {self.config_file}")

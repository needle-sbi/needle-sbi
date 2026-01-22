from pathlib import Path
from typing import cast

import hydra
import law
from omegaconf import OmegaConf

from orchestrator.config import MainConfig


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
        if hasattr(self, "_config"):
            return self._config

        config_file = Path(str(self.config_file)).resolve()

        with hydra.initialize_config_dir(
            config_dir=str(config_file.parent),
            version_base=None,
        ):
            cfg_dict = hydra.compose(config_name=str(config_file.stem))
            cfg_defaults = OmegaConf.structured(MainConfig)
            self._config = cast(MainConfig, OmegaConf.merge(cfg_defaults, cfg_dict))
            return self._config

    @config.setter
    def config(self, new_config: MainConfig):
        self._config = new_config

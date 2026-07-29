from pathlib import Path
from typing import Union

from omegaconf import OmegaConf

from needle.utils.config_schema import MainConfig
from needle.utils.config_utils import initialize_hydra_config
from needle.utils.logging import ColorFormatter

logger = ColorFormatter.get_logger("needle")


class Config:
    """Load and resolve a NEEDLE Hydra config outside of a LAW task.

    Wraps :func:`needle.utils.config_utils.initialize_hydra_config` to generate a fully-resolved
    :class:`~needle.utils.config_schema.MainConfig` with defaults applied, ``*_override`` fields
    populated and dependency graph validated.

    The ``config_path`` must point at a Hydra config file (e.g. ``conf/config.yaml``).

    The resolved MainConfig is exposed through direct attribute access on this ``Config``
    instance itself. The ``__getattr__`` forwards anything to the underlying ``MainConfig``).

    Attributes:
        config_path: Resolved, absolute path to the loaded config file.
        config: The fully-resolved ``MainConfig`` instance.

    Raises:
        FileNotFoundError: If `config_path` does not exist.

    Examples:
        Load a config and inspect it via attribute access:

        ```python
        from needle.api import Config

        cfg = Config("conf/config.yaml")
        print(cfg.results_path)
        print(list(cfg.estimators.keys()))
        ```

        Look up a nested value by dot-notation key, with a default if missing:

        ```python
        n_folds = cfg.get("estimators.model_A.expands.folds", default=1)
        ```

        Access the raw, fully-resolved dataclass directly:

        ```python
        estimator_cfg = cfg.estimators["model_A"]
        ```
    """

    def __init__(self, config_path: Union[str, Path]):
        self.config_path = Path(config_path).resolve()

        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        # Use the same config loader as LAW tasks (but without LAW dependency)
        self.config: MainConfig = initialize_hydra_config(
            config_dir=str(self.config_path.parent),
            config_name=self.config_path.stem,
        )

    def get(self, key: str, default=None):
        """Look up a config value by dot-notation key, e.g. ``"estimators.model_A.model"``.

        Args:
            key: Dot-notation path into the resolved config.
            default: Value returned if `key` does not resolve to anything.

        Returns:
            The resolved value at `key`, or `default` if it is missing.
        """
        return OmegaConf.select(OmegaConf.structured(self.config), key, default=default)

    def __getattr__(self, name):
        """Forward attribute access to the underlying resolved `MainConfig`.

        Allows ``cfg.results_path`` / ``cfg.estimators`` etc. as shorthand for
        ``cfg.config.results_path`` / ``cfg.config.estimators``.
        """
        if name in ["config", "config_path"]:
            return object.__getattribute__(self, name)

        return getattr(self.config, name)

    def __repr__(self) -> str:
        """Show the wrapped config's full nested structure, e.g. when printed in a notebook.

        Renders as ``Config(<config_path>)`` followed by the resolved config dumped as
        indented YAML, so the wrapper stays visible while the thing that actually
        matters -- the nested structure of the config itself -- is shown directly
        rather than hidden behind a generic ``<Config object at 0x...>``.
        """
        yaml_str = OmegaConf.to_yaml(self.config).rstrip("\n")
        indented = "\n".join(f"  {line}" for line in yaml_str.splitlines())
        return f"Config('{self.config_path}'):\n{indented}"


__all__ = [
    "Config",
]

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Union

from needle.utils.config_schema import MainConfig
from needle.utils.config_utils import initialize_hydra_config


def load_config(
    config_file: Union[str, Path] = "conf/config.yaml",
    overrides: Optional[Union[str, Sequence[str]]] = None,
) -> MainConfig:
    """Load and the Hydra config from file.

    This is a thin wrapper around :func:`needle.utils.config_utils.initialize_hydra_config`. It
    returns the resolved :class:`~needle.utils.config_schema.MainConfig` as an OmegaConf ``DictConfig``,
    with defaults applied, the ``*_override`` fields populated and the DAG validated.

    Important:
        Does not merge fields (e.g. estimator defaults merged into systematics), since that is handled
        by the corresponding Task.

    Args:
        config_file: Path to a Hydra config file. Either a composable Hydra config or a resolved
            config cached from a previous run. Defaults to ``conf/config.yaml``.
        overrides: Hydra override strings, either a space-separated string as accepted by
            ``--hydra-overrides`` on the CLI or a list of ``"key=value"`` strings.

    Returns:
        MainConfig: The fully-resolved config with all fields populated.

    Raises:
        FileNotFoundError: If `config_file` does not exist.

    Examples:
        >>> from needle import load_config
        >>> cfg = load_config("conf/config.yaml")
        >>> cfg.results_path
        >>> list(cfg.estimators.keys())
    """
    config_path = Path(config_file).resolve()

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    override_list: List[str] = overrides.split() if isinstance(overrides, str) else list(overrides or [])

    return initialize_hydra_config(
        config_dir=str(config_path.parent),
        config_name=config_path.stem,
        overrides=override_list,
    )


__all__ = [
    "load_config",
]

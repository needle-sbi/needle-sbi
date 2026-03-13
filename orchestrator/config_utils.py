import graphlib
from pathlib import Path
from typing import List, Literal, Mapping, cast

import hydra
from hydra.errors import ConfigCompositionException
from omegaconf import DictConfig, OmegaConf

from orchestrator.config import MainConfig


def validate_graph(self: "MainConfig") -> None:
    """Ensure that the graph spelled in the Config is a Directed Acyclic Graph and that all
    dependencies are resolved.

    Args:
        config (MainConfig): The instance of the MainConfig to test. DictConfig would also work
            as a type but will raise static typechecker warnings.

    Raises:
        ValueError: If a dependency mentioned with the `requires` keyword is missing. For example
            if "model_B" depends on a non-existing "model_A"

    Returns:
        MainConfig: Same instance of MainConfig with no side-effects. This is a pure function that
            only performs validation.
    """
    estimators = set(self.estimators)

    graph = {}
    for name, estimator in self.estimators.items():
        if not estimator.requires:
            continue

        deps = set(estimator.requires)
        missing = deps - estimators

        if missing:
            raise ValueError(f"{name} depends on undefined estimators {missing}")

        graph[name] = deps

    list(graphlib.TopologicalSorter(graph).static_order())
    return None


def initialize_hydra_config(
    config_dir: str,
    config_name: str,
    overrides: List[str] | None = None,
) -> MainConfig:
    """Initialize the hydra config from the corresponding directory

    Args:
        config_dir (str): Absolute path to the config directory
        config_name (str): Name of the config file (.e.g 'config')
        overrides (List[str] | None, optional): Hydra overrides. Defaults to None. May not include any
            of the groups listed in `resolve_config`.

    Returns:
        MainConfig: Partially resolved instance of MainConfig. The fields for SystematicConfig
            must be merged manually with the entries in EstimatorConfig.
    """
    with hydra.initialize_config_dir(
        config_dir=config_dir,
        version_base=None,
    ):
        cfg_as_dict: DictConfig = OmegaConf.merge(
            OmegaConf.structured(MainConfig),
            hydra.compose(config_name=config_name, overrides=overrides),
        )  # type: ignore
        cfg_as_dict = resolve_defaults(cfg_as_dict, Path(config_dir))
        cfg: MainConfig = cast(MainConfig, cfg_as_dict)
        validate_graph(cfg)
        return cfg


def resolve_defaults(
    cfg: DictConfig,
    cfg_dir: Path,
    node: Literal["estimators", "systematics"] = "estimators",
) -> DictConfig:
    """Resolve the default fields in the hydra config.

    This method mimics the usual hydra behavior of the 'defaults' field, but extends it to nested fields
    inside the config. Meaning fields like 'dataset' are looked up in the group 'datasets' and the
    values are added to 'dataset_override'. This in turn is also compatible with overriding a value
    inside the group by directly assigning 'dataset_override' afterwards.

    Note that further nesting like Systematics that also provide the `*_override` keyword will not
    have all keywords automatically, but have to be merged with the main field.

    Groups that are registered:
        - "datasets": Resolves the "dataset" field and populates "dataset_override"
        - "datamodules": Resolves the "datamodule" field and populates "datamodule_override"
        - "models": Resolves the "model" field and populates "model_override"
        - "trainers": Resolves the "trainer" field and populates "trainer_override"

    Args:
        cfg (DictConfig): The config object to resolve

    Returns:
        DictConfig: A config object with the fields resolved to the corresponding group

    """

    DEFAULT_GROUPS: Mapping[str, str] = {
        "dataset": "datasets",
        "datamodule": "datamodules",
        "model": "models",
        "trainer": "trainers",
    }

    def _load_group(group: str, name: str) -> DictConfig:
        try:
            return hydra.compose(overrides=[f"+{group}={name}"])[group]
        except ConfigCompositionException as e:
            msg = f"Cannot resolve config group '{group}={name}'."

            if cfg_dir and (cfg_dir / (group + ".yaml")).exists():
                options = [p.stem for p in (cfg_dir / group).glob("*.yaml")]
                msg += f" Available options: {', '.join(options)}"

            raise ValueError(msg) from e

    if cfg.get("_resolved"):
        return cfg

    estimators: DictConfig = cfg.get(node, {})

    for _, est_cfg in estimators.items():
        for field, group in DEFAULT_GROUPS.items():
            group_member = est_cfg.get(field)

            if group_member is None:
                continue

            override_key = f"{field}_override"
            group_cfg = _load_group(group, group_member)
            base_cfg = est_cfg.get(override_key)

            if base_cfg:
                est_cfg[override_key] = OmegaConf.merge(base_cfg, group_cfg)
            else:
                est_cfg[override_key] = group_cfg

    return cfg

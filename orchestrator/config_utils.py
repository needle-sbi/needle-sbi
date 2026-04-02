import graphlib
import inspect
from pathlib import Path
from typing import Any, List, Literal, Mapping, cast

import hydra
from hydra.errors import ConfigCompositionException
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import LightningDataModule as LegacyDataModule
from pytorch_lightning import LightningModule as LegacyModule
from pytorch_lightning import Trainer as LegacyTrainer

from orchestrator.config import MainConfig
from preprocessor.utils import ColorFormatter

logger = ColorFormatter.get_logger("orchestrator")


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
            group_member: str = est_cfg.get(field)

            if group_member is None:
                continue

            group_member_cfg = cfg_dir / (group_member + ".yaml")

            if group_member_cfg.exists():
                group_cfg = OmegaConf.load(group_member_cfg)  # Case: .yaml file at top-level
            else:
                group_cfg = _load_group(group, group_member)  # Case: .yaml file inside folder with group name

            override_key = f"{field}_override"
            base_cfg = est_cfg.get(override_key)

            if base_cfg:
                if override_key == "dataset_override":
                    est_cfg[override_key] = OmegaConf.merge(base_cfg, group_cfg)
                else:
                    base_dict = OmegaConf.to_container(base_cfg, resolve=False)
                    group_dict = OmegaConf.to_container(group_cfg, resolve=False)
                    est_cfg[override_key] = OmegaConf.create({**base_dict, **group_dict})  # type: ignore
            else:
                est_cfg[override_key] = group_cfg

    return cfg


def hydra_check_if_arg_supported(
    cfg: DictConfig | None,
    arg_name: str,
) -> bool:
    """Check if an argument is valid for the given class

    Args:
        cfg (DictConfig): OmegaConf DictConfig corresponding to the class being instantiated using
            hydra
        arg_name (str): The argument to check. Can be positional or keyword

    Returns:
        bool: Whether the parameter is valid for this class or if the config is None
    """
    if cfg is None:
        # Treat this case separately as this can cause a lot of headache
        caller = inspect.stack()[1]
        logger.debug("Config object is None")
        logger.debug(f"Called from {caller.filename}:{caller.filename} in {caller.function}")
        logger.debug(f"  {caller.code_context[0].strip()}")  # type: ignore
        return False

    cls = hydra.utils.get_class(cfg._target_)
    sig = inspect.signature(cls.__init__).parameters

    return (arg_name in sig) or any(  # check positional parameter
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.values()
    )  # check keyword arguments


def hydra_instantiate(
    cfg: DictConfig,
    **kwargs,
) -> Any:
    """Instantiate a class with hydra using the maximum subset of allowed arguments.

    A target class might not support all the arguments that are provided by the NEEDLE framework,
    so this function instantiates the class with all valid arguments and skips the others.

    Args:
        cfg (DictConfig): The values coming from the config. Must contain the `_target_` key
            in order to be compatible with hydra.
        **kwargs: The values coming from the framework

    Returns:
        Any: An instance of the target class
    """
    if not cfg.__getattr__("_target_"):
        raise ValueError(
            "Module config must include the key `_target_` that points to the location of your module. "
            "See the hydra docs https://hydra.cc/docs/advanced/instantiate_objects/overview/"
        )

    check_for_lightning_import_mismatch(cfg)

    supported_kwargs = {k: v for k, v in kwargs.items() if hydra_check_if_arg_supported(cfg, k)}
    unsupported_kwargs = set(kwargs) - set(supported_kwargs)

    if unsupported_kwargs:
        cls_name = hydra.utils.get_class(cfg._target_).__name__  # type: ignore
        logger.warning(
            f"Class {cls_name} does not support the following arguments: "
            f"{unsupported_kwargs}, which were skipped at instantiation."
        )

    return hydra.utils.instantiate(cfg, **supported_kwargs)


def check_for_lightning_import_mismatch(cfg: DictConfig) -> None:
    """Raise a clear error if the target class inherits from the wrong Lightning package.

    Mixing `pytorch_lightning` and `lightning.pytorch` base classes causes silent
    failures where e.g. a Trainer refuses to accept a LightningModule because they
    come from different class hierarchies.
    """
    try:
        cls = hydra.utils.get_class(cfg._target_)
    except Exception:
        raise ValueError(
            "Module config must include the key `_target_` that points to the location of your module. "
            "See the hydra docs https://hydra.cc/docs/advanced/instantiate_objects/overview/"
        )

    mro_module_paths = [f"{c.__module__}.{c.__qualname__}" for c in inspect.getmro(cls)]
    legacy_bases = [p for p in mro_module_paths if p.startswith("pytorch_lightning.")]

    if not legacy_bases:
        return None

    if issubclass(cls, LegacyModule):
        kind = "LightningModule (model)"
        fix = "from lightning import LightningModule"
        base = "LightningModule"
    elif issubclass(cls, LegacyDataModule):
        kind = "LightningDataModule"
        fix = "from lightning import LightningDataModule"
        base = "LightningDataModule"
    elif issubclass(cls, LegacyTrainer):
        kind = "Trainer"
        fix = "from lightning import Trainer"
        base = "Trainer"
    else:
        kind = "Lightning class"
        fix = "from lightning.pytorch import ..."
        base = "the appropriate Lightning base class"

    raise TypeError(
        f"Your class '{cls.__name__}' inherits from `pytorch_lightning.{base}` (the legacy package), "
        f"but NEEDLE uses the modern `lightning.pytorch` package.\n\n"
        f"Fix: update your {kind} to inherit from the modern package:\n\n"
        f"    # Before (legacy)\n"
        f"    from pytorch_lightning import {base}\n\n"
        f"    # After (modern)\n"
        f"    {fix}\n\n"
    )

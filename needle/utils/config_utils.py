from __future__ import annotations

import difflib
import graphlib
import inspect
from pathlib import Path
from typing import Any, List, Literal, Mapping, Optional, Type, cast

import hydra
import luigi
from hydra.errors import ConfigCompositionException
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import LightningDataModule as LegacyDataModule
from pytorch_lightning import LightningModule as LegacyModule
from pytorch_lightning import Trainer as LegacyTrainer

from needle.utils.config_schema import MainConfig
from needle.utils.logging import ColorFormatter

logger = ColorFormatter.get_logger("config")
OmegaConf.register_new_resolver("if", lambda cond, t, f: t if cond else f)


def validate_graph(self: "MainConfig") -> None:
    """Validate that estimators form a Directed Acyclic Graph and that dependencies exist.

    Args:
        self (MainConfig): The config instance containing the `estimators` mapping.
            Each estimator may declare dependencies via `requires`.

    Raises:
        ValueError: When an estimator depends on an undefined estimator name.

    Returns:
        None: This helper performs only validation and has no side effects.
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
    """Initialize Hydra and compose a `MainConfig` from the given config directory.

    Args:
        config_dir (str): Absolute path to the Hydra config directory.
        config_name (str): Base name of the config file to compose (without `.yaml`).
        overrides (List[str] | None, optional): Hydra override strings. Defaults to None.

    Returns:
        MainConfig: Partially resolved `MainConfig` instance with defaults applied and the
            estimator dependency graph validated.

    Raises:
        ValueError: If default resolution fails or graph validation detects missing dependencies.
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
        OmegaConf.resolve(cfg_as_dict)
        cfg: MainConfig = cast(MainConfig, cfg_as_dict)
        validate_graph(cfg)
        return cfg


def resolve_defaults(
    cfg: DictConfig,
    cfg_dir: Path,
    node: Literal["estimators", "systematics"] = "estimators",
) -> DictConfig:
    """Resolve Hydra defaults for nested estimator or systematic config entries.

    This function resolves fields like `dataset`, `datamodule`, `model`, and `trainer`
    by loading the corresponding group config and populating the matching override field
    (for example `dataset_override`). It mutates `cfg` in place and returns the same
    resolved `DictConfig` object.

    Args:
        cfg (DictConfig): The config object to resolve.
        cfg_dir (Path): The path to the config directory containing group configs.
        node (Literal["estimators", "systematics"], optional): The top-level node to
            iterate. Defaults to "estimators".

    Returns:
        DictConfig: The same `cfg` object with resolved `*_override` entries populated.

    Raises:
        ValueError: When a referenced group member cannot be resolved.
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

    # Mark as resolved to prevent re-resolution
    cfg["_resolved"] = True
    return cfg


def hydra_check_if_arg_supported(
    cfg: DictConfig | None,
    arg_name: str,
) -> bool:
    """Check whether an argument is supported by the target class in `cfg`.

    Args:
        cfg (DictConfig | None): OmegaConf config containing `_target_` for the class.
        arg_name (str): The constructor argument name to validate.

    Returns:
        bool: True when the class accepts the parameter or supports `**kwargs`.
            Returns False when `cfg` is None or the argument is not supported.
    """
    if cfg is None:
        # Treat this case separately as this can cause a lot of headache
        caller = inspect.stack()[1]
        logger.debug("Config object is None")
        logger.debug(f"Called from {caller.filename}:{caller.filename} in {caller.function}")
        logger.debug(f"  {caller.code_context[0].strip()}")  # type: ignore
        return False

    cls = hydra.utils.get_class(cfg._target_)

    if issubclass(cls, luigi.Task):
        is_luigi_parameter = hydra_check_if_luigi_parameter_supported(cls, arg_name=arg_name)
    else:
        is_luigi_parameter = False

    sig = inspect.signature(cls.__init__).parameters
    return (arg_name in sig) or is_luigi_parameter  # check luigi parameters


def hydra_check_if_luigi_parameter_supported(task: Type[luigi.Task], arg_name: str) -> bool:
    """Check if an argument is a luigi.Parameter. These are not regular Args, but instead class
    attributes that are set during the requires() and run() methods.

    Args:
        task (Type[luigi.Task]): Task to check
        arg_name (str): Name of the Parameter to check

    Returns:
        bool: True if the arg is a valid luigi.Parameter attribute of the Task, False otherwise
    """
    for name, var in vars(task).items():
        if isinstance(var, luigi.Parameter) and (name == arg_name):
            return True
    else:
        return False


def hydra_instantiate(
    cfg: DictConfig | Any | None,
    **kwargs,
) -> Any:
    """Instantiate the target class using only supported keyword arguments.

    The function filters `kwargs` to the subset accepted by the target class's
    constructor and skips unsupported parameters, logging a warning for dropped keys.

    Args:
        cfg (DictConfig): Config containing `_target_` and any supported parameters.
        **kwargs: Candidate keyword arguments for instantiation.

    Returns:
        Any: The instantiated object returned by `hydra.utils.instantiate`.

    Raises:
        ValueError: If `_target_` is missing from `cfg`.

    Example:
        >>> instance = hydra_instantiate(cfg, device='cuda', logger=logger)
    """
    if cfg is None:
        raise ValueError("Model config is empty")

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

    Args:
        cfg (DictConfig): Config containing `_target_` for the target class.

    Raises:
        ValueError: If `_target_` is missing or invalid.
        TypeError: If the target class inherits from `pytorch_lightning` instead of
            `lightning.pytorch`.
    """
    cls = hydra.utils.get_class(cfg._target_)

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


def compare_configs(self: MainConfig, other: object) -> Optional[str]:
    """Return None if equal, otherwise a colored unified diff of their YAML representations."""
    import re

    RED, GREEN, CYAN, BOLD, RESET = "\033[31m", "\033[32m", "\033[36m", "\033[1m", "\033[0m"

    if self == other:
        return None

    def _to_lines(cfg: object) -> List[str]:
        if isinstance(cfg, DictConfig):
            return OmegaConf.to_yaml(cfg, resolve=True).splitlines(keepends=True)
        return OmegaConf.to_yaml(OmegaConf.structured(cfg), resolve=True).splitlines(keepends=True)

    self_lines = _to_lines(self)
    other_lines = _to_lines(other)
    raw_diff: List[str] = list(
        difflib.unified_diff(
            self_lines,
            other_lines,
            fromfile=f"{self.__class__.__name__} (self)",
            tofile=f"{other.__class__.__name__} (other)",
            lineterm="",
        )
    )
    if not raw_diff:
        return None

    old_lineno = new_lineno = 0
    out: List[str] = []

    for line in raw_diff:
        if line.startswith("---") or line.startswith("+++"):
            out.append(f"{BOLD}{line}{RESET}")
        elif line.startswith("@@"):
            m = re.match(r"@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@", line)
            if m:
                old_lineno, new_lineno = int(m.group(1)), int(m.group(2))
            out.append(f"{CYAN}{line}{RESET}")
        elif line.startswith("-"):
            out.append(f"{RED}{old_lineno:4d} {line}{RESET}")
            old_lineno += 1
        elif line.startswith("+"):
            out.append(f"{GREEN}{new_lineno:4d} {line}{RESET}")
            new_lineno += 1
        else:
            out.append(f"{old_lineno:4d} {line}")
            old_lineno += 1
            new_lineno += 1
    return "".join(out)

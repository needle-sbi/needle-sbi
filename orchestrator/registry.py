"""
Hydra ConfigStore-based schema validation for Data/Lightning/Model modules.
See https://hydra.cc/docs/tutorials/structured_config/config_store/

Each schema is a ``@dataclass`` registered with the Hydra ``ConfigStore`` under the
appropriate config group (``models``, ``datamodules``, ``trainers``).
"""

from dataclasses import fields
from typing import Any, Mapping

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
from .schemas import (
    MockTransformerSchema,
    SimpleMLPSchema,
    PaddedDataModuleSchema,
    NormFlowSchema,
    LightningTrainerSchema,
)

# ---------------------------------------------------------------------------
# `ConfigStore` registration

# maps _target_ string to schema class for automatic lookup
_SCHEMA_REGISTRY: dict[str, type] = {}


def _register_all_schemas() -> None:
    """Register every schema with the Hydra ConfigStore under the correct group.

    Called once at module import time.  Adding a new model only requires:
    - Define a ``@dataclass`` schema above.
    - Add a ``cs.store(...)`` call here.
    - Create the matching YAML in ``conf/<group>/<name>.yaml``.
    """
    cs = ConfigStore.instance()

    # models
    cs.store(group="models", name="mock_transformer", node=MockTransformerSchema)
    cs.store(group="models", name="simple_mlp", node=SimpleMLPSchema)
    cs.store(group="models", name="norm_flow", node=NormFlowSchema)

    # data-modules
    cs.store(group="datamodules", name="padded", node=PaddedDataModuleSchema)

    # trainers
    cs.store(group="trainers", name="default", node=LightningTrainerSchema)

    # populate the target for schema lookup
    for schema_cls in (
        MockTransformerSchema,
        SimpleMLPSchema,
        NormFlowSchema,
        PaddedDataModuleSchema,
        LightningTrainerSchema,
    ):
        _SCHEMA_REGISTRY[schema_cls._target_] = schema_cls


_register_all_schemas()


def resolve_defaults(cfg: DictConfig) -> DictConfig:
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
        cfg = hydra.compose(overrides=[f"+{group}={name}"])
        return cfg[group]

    estimators: DictConfig = cfg.get("estimators", {})

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


# ---------------------------------------------------------------------------
# Some validation helpers
def _to_dict(config: DictConfig | dict[str, Any]) -> dict[str, Any]:
    if isinstance(config, DictConfig):
        return dict(OmegaConf.to_container(config, resolve=True))  # type: ignore[arg-type]
    return dict(config)


def _validate_against_schema(
    config: DictConfig | dict[str, Any],
    schema_cls: type,
) -> DictConfig:
    """Validate *config* against a ``@dataclass`` schema.

    Raises ``ValueError`` if unknown fields are present, then returns a fully-typed
    ``OmegaConf.structured`` config.
    """
    raw = _to_dict(config)
    allowed_fields = {f.name for f in fields(schema_cls)}
    unknown_fields = set(raw) - allowed_fields
    if unknown_fields:
        raise ValueError(
            f"Unknown config field(s) for {schema_cls.__name__}: {sorted(unknown_fields)}. "
            f"Allowed fields: {sorted(allowed_fields)}"
        )

    schema_instance = schema_cls(**raw)
    return OmegaConf.structured(schema_instance)


def _validate_by_target(config: DictConfig | dict[str, Any]) -> DictConfig | dict[str, Any]:
    """Look up the ``_target_`` in the registry and validate if a schema exists.

    Unknown targets are passed through unchanged, preserving Hydra polymorphism.
    """
    raw = _to_dict(config)
    target = raw.get("_target_")
    schema_cls = _SCHEMA_REGISTRY.get(target) if target else None

    if schema_cls is not None:
        return _validate_against_schema(config, schema_cls)
    return config


# ---------------------------------------------------------------------------
# Public API


def validate_model_config(config: DictConfig | dict[str, Any]) -> DictConfig | dict[str, Any]:
    """Validate known model configs while allowing unknown targets to pass through."""
    return _validate_by_target(config)


def validate_datamodule_config(config: DictConfig | dict[str, Any]) -> DictConfig | dict[str, Any]:
    """Validate known datamodule configs while allowing unknown targets to pass through.

    Includes a custom check for ``multiprocessing_type``.
    """
    raw = _to_dict(config)
    target = raw.get("_target_")

    if target == PaddedDataModuleSchema._target_:
        mp_type = raw.get("multiprocessing_type", "torch")
        if mp_type not in {"torch", "dask"}:
            raise ValueError(
                f"Invalid multiprocessing_type in PaddedDataModule config: "
                f"{mp_type!r}. Allowed values: ['torch', 'dask']"
            )

    return _validate_by_target(config)


def validate_trainer_config(config: DictConfig | dict[str, Any]) -> DictConfig | dict[str, Any]:
    """Validate known trainer configs while allowing unknown targets to pass through."""
    return _validate_by_target(config)

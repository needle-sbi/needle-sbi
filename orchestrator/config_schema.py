"""
Hydra ConfigStore-based schema validation for Data/Lightning/Model modules.
See https://hydra.cc/docs/tutorials/structured_config/config_store/

Each schema is a ``@dataclass`` registered with the Hydra ``ConfigStore`` under the
appropriate config group (``models``, ``datamodules``, ``trainers``).
"""

from dataclasses import dataclass, fields
from typing import Any

from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf


@dataclass
class MockTransformerSchema:
    """Typed schema for ml.lightning.models.mock_transformer.MockTransformerModule."""

    _target_: str = "ml.lightning.models.mock_transformer.MockTransformerModule"
    factor: float = 0.1
    patience: int = 10
    init_lr: float = 1e-3


@dataclass
class SimpleMLPSchema:
    """Typed schema for ml.lightning.models.simple_mlp.SimpleMLPModule."""

    _target_: str = "ml.lightning.models.simple_mlp.SimpleMLPModule"
    hidden_dim: int = 64
    init_lr: float = 1e-3


@dataclass
class PaddedDataModuleSchema:
    """Typed schema for ml.lightning.data.padded_datamodule.PaddedDataModule."""

    _target_: str = "ml.lightning.data.padded_datamodule.PaddedDataModule"
    multiprocessing_type: str = "torch"
    batch_size: int = 1024
    n_workers: int = 0
    shuffle_partitions: bool = True
    shuffle_events: bool = True


@dataclass
class NormFlowSchema:
    """Typed schema for ml.lightning.models.norm_flow.NormFlowModule."""

    _target_: str = "ml.lightning.models.norm_flow.NormFlowModule"
    features_dim: int = 8
    context_dim: int = 0
    flow_type: str = "nsf"
    n_transforms: int = 5
    hidden_dim: int = 128
    init_lr: float = 1e-3


@dataclass
class LightningTrainerSchema:
    """Typed schema for lightning.Trainer."""

    _target_: str = "lightning.Trainer"
    max_epochs: int = 1
    log_every_n_steps: int = 50


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

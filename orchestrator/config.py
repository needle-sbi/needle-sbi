from dataclasses import dataclass, field
from typing import Any, List, Optional

from ml.utils.dataclass import SerializableDataclass


@dataclass
class DatasetConfig(SerializableDataclass):
    paths: str = ""
    features_columns: Optional[List[str]] = field(default_factory=list)
    labels_columns: Optional[List[str]] = field(default_factory=list)
    format: str = "automatic"
    dak_reader_kwargs: dict[str, Any] = field(default_factory=dict)
    max_number_events: int = -1


@dataclass
class SystematicConfig(SerializableDataclass):
    """In contrast to the `EstimatorConfig` dataclass, entries here can be inferred from the Asimov
    dataclass, e.g. by adopting the entries from the parent estimator.
    """

    datamodule: Optional[str] = None
    datamodule_override: Optional[Any] = None
    dataset: Optional[str] = None
    dataset_override: Optional[DatasetConfig] = field(default_factory=DatasetConfig)
    model: Optional[str] = None
    model_override: Optional[Any] = None
    trainer: Optional[str] = None
    trainer_override: Optional[Any] = None


@dataclass
class EnsembleConfig(SerializableDataclass):
    num_ensembles: int = 1
    aggregation_method: str | None = None


@dataclass
class ExpansionConfig(SerializableDataclass):
    ensembles: EnsembleConfig = field(default_factory=EnsembleConfig)
    systematics: dict[str, SystematicConfig] = field(default_factory=dict)
    folds: int = 1

    def __post_init__(self):
        self.systematics.setdefault("nominal", SystematicConfig())


@dataclass
class EstimatorConfig(SerializableDataclass):
    """Config for modules used during training.

    Important:
        The field to be used afterwards in the code is `*_override`, as this is the resolved field,
        the name is just a str and would have to be populated manually afterwards.

    Each field can be defined in two ways:

    1. By name (str): `dataset="fair_universe"`. This string is resolved at runtime using
        `orchestrator.registry.resolve_defaults` to produce the actual DictConfig in the corresponding
        `*_override` field.

    2. By override (DictConfig / dataclass): Usual `dataset_override=DatasetConfig(...)`.
        If provided, the resolver will use this directly and not overwrite it.

    Example usage:

    >>> from omegaconf import OmegaConf
    >>> from ml.utils.dataclass import SerializableDataclass
    >>> from your_project.configs import DatasetConfig, NodeBaseConfig

    # Case 1: Using string reference (to be resolved)
    >>> cfg1 = NodeBaseConfig(
    ...     datamodule="padded",
    ...     datamodule_override=None,
    ...     dataset="fair_universe",
    ...     dataset_override=None,
    ...     model="transformer",
    ...     model_override=None,
    ...     trainer="default",
    ...     trainer_override=None
    ... )
    >>> cfg1.dataset  # string reference
    'fair_universe'
    >>> cfg1.dataset_override  # initially empty
    None

    # After calling `resolve_defaults(cfg1, config_dir)`, the override is populated:
    >>> # orchestrator.registry.resolve_defaults(cfg1, config_dir)
    >>> cfg1.dataset_override.paths
    '/path/to/fair_universe_dataset.yaml'
    >>> cfg1.dataset_override.labels_columns
    ['PRI_n_jets']

    # Case 2: Using inline override directly
    >>> override_ds = DatasetConfig(paths='/custom/path', labels_columns=['PRI_n_jets'])
    >>> cfg2 = NodeBaseConfig(
    ...     datamodule="padded",
    ...     datamodule_override=None,
    ...     dataset_override=override_ds,
    ...     dataset="",  # string can be empty
    ...     model_override=None,
    ...     model="xgboost",
    ...     trainer_override=None,
    ...     trainer="default"
    ... )
    >>> cfg2.dataset_override.paths
    '/custom/path'
    >>> cfg2.dataset_override.labels_columns
    ['PRI_n_jets']
    """

    datamodule: str = ""
    datamodule_override: Optional[Any] = None
    dataset: str = ""
    dataset_override: Optional[DatasetConfig] = field(default_factory=DatasetConfig)
    model: str = ""
    model_override: Optional[Any] = None
    trainer: str = ""
    trainer_override: Optional[Any] = None
    expands: ExpansionConfig = field(default_factory=ExpansionConfig)
    requires: Optional[List[str]] = None


@dataclass
class MainConfig(SerializableDataclass):
    estimators: dict[str, EstimatorConfig] = field(default_factory=dict)

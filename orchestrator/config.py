from dataclasses import dataclass, field
from typing import Any, List, Optional

from ml.utils.dataclass import SerializableDataclass


@dataclass
class DatasetConfig(SerializableDataclass):
    paths: str = ""
    features_columns: Optional[List[str]] = None
    labels_columns: Optional[List[str]] = None
    format: str = "automatic"
    dak_reader_kwargs: Optional[dict[str, Any]] = None
    max_number_events: int = -1


@dataclass
class LightningDataModuleConfig:
    _target_: str  # type[LightningDataModule]
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class LightningModuleConfig:
    _target_: str  # type[LightningModule]
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class LightningTrainerConfig:
    _target_: str  # type[Trainer]
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class MainConfig(SerializableDataclass):
    datamodules: LightningDataModuleConfig
    datasets: DatasetConfig
    models: LightningModuleConfig
    trainers: LightningTrainerConfig

    n_folds: int = 5

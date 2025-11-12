from dataclasses import dataclass, field
from typing import Any, Literal, Type

from lightning import LightningDataModule, LightningModule, Trainer

from ml.utils.dataclass import SerializableDataclass


@dataclass
class DatasetConfig(SerializableDataclass):
    paths: str | list[str]
    features_columns: str | list[str]
    labels_columns: str | list[str]
    format: Literal["parquet", "automatic"] = "automatic"
    dak_reader_kwargs: dict[str, Any] = field(default_factory=dict)
    max_number_events: int = -1


@dataclass
class LightningDataModuleConfig:
    _target_: Type[LightningDataModule]
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class LightningModuleConfig:
    _target_: Type[LightningModule]
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class LightningTrainerConfig:
    _target_: Type[Trainer] = Trainer
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class MainConfig(SerializableDataclass):
    datamodules: LightningDataModuleConfig
    datasets: DatasetConfig
    models: LightningModuleConfig
    trainers: LightningTrainerConfig

    n_folds: int = 5

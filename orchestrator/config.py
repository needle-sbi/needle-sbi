from dataclasses import dataclass, field
from typing import Any, List, Optional

from ml.utils.dataclass import SerializableDataclass


@dataclass
class DatasetConfig(SerializableDataclass):
    paths: str = ""
    features_columns: Optional[List[str]] = None
    labels_columns: Optional[List[str]] = None
    format: str = "automatic"
    dak_reader_kwargs: dict[str, Any] = field(default_factory=dict)
    max_number_events: int = -1


@dataclass
class MainConfig(SerializableDataclass):
    datamodules: Any
    datasets: DatasetConfig
    models: Any
    trainers: Any

    n_folds: int = 5

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class BenchmarkVariables:
    number_events: list[int] = field(default_factory=list)
    number_workers: list[int] = field(default_factory=list)
    batch_sizes: list[int] = field(default_factory=list)
    dataset_types: list[str] = field(default_factory=list)


@dataclass
class DataConfig:
    files_to_load: str | list[str]
    features_columns: str | list[str]
    labels_columns: str | list[str]


@dataclass
class BenchmarkConfig:
    variables: BenchmarkVariables
    data: DataConfig
    verbose: bool
    only_plot: bool
    format: Literal["automatic", "parquet"]

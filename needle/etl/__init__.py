import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from needle.etl.array import NestedArrayIndexer
    from needle.etl.conversion import convert_root_to_parquet
    from needle.etl.dask_ingestor import Ingestor
    from needle.etl.normalization import MinMaxScaler, ScalerProtocol, StandardScaler

__all__ = [
    "Ingestor",
    "NestedArrayIndexer",
    "ScalerProtocol",
    "MinMaxScaler",
    "StandardScaler",
    "convert_root_to_parquet",
]

_MODULE_BY_NAME = {
    "Ingestor": "needle.etl.dask_ingestor",
    "NestedArrayIndexer": "needle.etl.array",
    "ScalerProtocol": "needle.etl.normalization",
    "MinMaxScaler": "needle.etl.normalization",
    "StandardScaler": "needle.etl.normalization",
    "convert_root_to_parquet": "needle.etl.conversion",
}


def __getattr__(name: str) -> object:
    module_name = _MODULE_BY_NAME.get(name)

    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = importlib.import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)

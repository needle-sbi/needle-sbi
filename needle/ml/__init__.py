import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from needle.ml.datasets import (
        PaddedDaskDataset,
        PaddedDataset,
        PaddedDatasetBase,
        PaddedTorchDataset,
        PartitionQueue,
        load_partition,
    )
    from needle.ml.datasets.kfold import KFold
    from needle.ml.lightning.datamodules.padded_datamodule import PaddedDataModule
    from needle.ml.lightning.models.mock_transformer import (
        MockTransformer,
        MockTransformerConfig,
        MockTransformerModule,
    )

__all__ = [
    # datasets
    "PaddedDatasetBase",
    "PaddedDataset",
    "PaddedTorchDataset",
    "PaddedDaskDataset",
    "PartitionQueue",
    "load_partition",
    "KFold",
    # lightning
    "PaddedDataModule",
    "MockTransformer",
    "MockTransformerConfig",
    "MockTransformerModule",
]

_MODULE_BY_NAME = {
    "PaddedDatasetBase": "needle.ml.datasets",
    "PaddedDataset": "needle.ml.datasets",
    "PaddedTorchDataset": "needle.ml.datasets",
    "PaddedDaskDataset": "needle.ml.datasets",
    "PartitionQueue": "needle.ml.datasets",
    "load_partition": "needle.ml.datasets",
    "KFold": "needle.ml.datasets.kfold",
    "PaddedDataModule": "needle.ml.lightning.datamodules.padded_datamodule",
    "MockTransformer": "needle.ml.lightning.models.mock_transformer",
    "MockTransformerConfig": "needle.ml.lightning.models.mock_transformer",
    "MockTransformerModule": "needle.ml.lightning.models.mock_transformer",
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

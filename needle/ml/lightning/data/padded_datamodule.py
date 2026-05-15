from typing import Literal

import lightning as L
from torch.utils.data import DataLoader

from needle.etl.dask_ingestor import Ingestor
from needle.etl.normalization import MinMaxScaler
from needle.ml.datasets import PaddedDaskDataset, PaddedTorchDataset
from needle.ml.datasets.kfold import KFold
from needle.utils.config_schema import DatasetConfig


class PaddedDataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset_config: dict,
        batch_size: int = 1024,
        fold_index: int | None = None,
        n_folds: int | None = None,
        n_workers: int = 0,
        multiprocessing_type: Literal["torch", "dask"] = "torch",
        shuffle_partitions: bool = True,
        shuffle_events: bool = True,
    ) -> None:
        super().__init__()
        self.dataset_config = DatasetConfig(**dataset_config)
        self.batch_size = batch_size
        self.fold_index = fold_index
        self.n_folds = n_folds
        self.n_workers = n_workers
        self.multiprocessing_type = multiprocessing_type
        self.shuffle_partitions = shuffle_partitions
        self.shuffle_events = shuffle_events
        self.scaler = MinMaxScaler()

    def setup(self, stage: str | None = None) -> None:
        features = Ingestor(
            self.dataset_config.paths,
            format=self.dataset_config.format,
            columns=self.dataset_config.features_columns,
            reader_kwargs=self.dataset_config.dak_reader_kwargs,
            max_number_events=self.dataset_config.max_number_events,
        )
        labels = Ingestor(
            self.dataset_config.paths,
            format=self.dataset_config.format,
            columns=self.dataset_config.labels_columns,
            reader_kwargs=self.dataset_config.dak_reader_kwargs,
            max_number_events=self.dataset_config.max_number_events,
        )
        features.array = self.scaler.apply(features.array)
        labels.array = self.scaler.apply(labels.array)

        self.features = features
        self.labels = labels

    @staticmethod
    def get_dataset(name: str):
        match name:
            case "dask":
                return PaddedDaskDataset
            case "torch":
                return PaddedTorchDataset
            case _:
                raise ValueError(name)

    def train_dataloader(self) -> DataLoader:
        if self.fold_index and self.n_folds:
            kfold = KFold(
                fold_index=self.fold_index,
                n_folds=self.n_folds,
                is_training=True,
                divisions=self.features.array.divisions,
            )
        else:
            kfold = None

        dataset = self.get_dataset(self.multiprocessing_type)(
            self.features,
            self.labels,
            shuffle_partitions=self.shuffle_partitions,
            shuffle_events=self.shuffle_events,
            kfold=kfold,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=dataset.SHUFFLE_ALLOWED,
            num_workers=self.n_workers if dataset.TORCH_MULTIPROCESSING_ALLOWED else 0,
        )

    def val_dataloader(self) -> DataLoader:
        if self.fold_index and self.n_folds:
            kfold = KFold(
                fold_index=self.fold_index,
                n_folds=self.n_folds,
                is_training=False,
                divisions=self.features.array.divisions,
            )
        else:
            kfold = None
        dataset = self.get_dataset(self.multiprocessing_type)(
            self.features,
            self.labels,
            shuffle_partitions=self.shuffle_partitions,
            shuffle_events=self.shuffle_events,
            kfold=kfold,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=dataset.SHUFFLE_ALLOWED,
            num_workers=self.n_workers if dataset.TORCH_MULTIPROCESSING_ALLOWED else 0,
        )

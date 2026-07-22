from typing import List, Optional, Annotated

import pandas as pd
import torch
import uproot
from needle.utils.config_schema import DatasetConfig
from needle.etl.array import resolve_paths
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import DataLoader, Dataset
from pydantic import Field

import lightning as L

Percentage = Annotated[float, Field(ge=0.0, le=1.0)]


class PandasDataset(Dataset):
    """Simple dataset wrapper for pandas DataFrames."""

    def __init__(self, df: pd.DataFrame, features: List[str], labels: List[str]):
        self.df = df
        self.features = features
        self.labels = labels

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        features = torch.tensor([row[col] for col in self.features], dtype=torch.float32)
        labels = torch.tensor([row[col] for col in self.labels], dtype=torch.float32)
        return features, labels


class PandasDataModule(L.LightningDataModule):
    """PyTorch Lightning DataModule for pandas DataFrames with kfold support."""

    df: pd.DataFrame
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame

    def __init__(
        self,
        dataset_config: DatasetConfig,
        n_folds: int,
        fold_index: int,
        batch_size: int = 512,
        train_test_split: Percentage = 0.9,
    ):
        super().__init__()
        self.dataset_config = dataset_config
        self.batch_size = batch_size
        self.n_folds = n_folds
        self.fold_index = fold_index
        self.max_number_events = self.dataset_config.max_number_events
        self.train_test_split = train_test_split
        self.features = self.dataset_config.features_columns or []
        self.labels = self.dataset_config.labels_columns or []

        if not self.features:
            raise ValueError(f"Feature column is empty: {self.features}")
        if not self.labels:
            raise ValueError(f"Label column is empty: {self.labels}")

    def setup(self, stage: Optional[str] = None) -> None:
        """Load and split data."""
        self.df = self._load_data()
        self.df = self.df.head(n=self.max_number_events)

        if self.n_folds > 1:
            kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
            folds = list(kfold.split(self.df))
            train_idx, val_idx = folds[self.fold_index]
            self.train_df = self.df.iloc[train_idx].reset_index(drop=True)
            self.val_df = self.df.iloc[val_idx].reset_index(drop=True)
        else:
            self.train_df, self.val_df = train_test_split(self.df, train_size=self.train_test_split)

        self.test_df = self.val_df.copy()

    def _load_data(self) -> pd.DataFrame:
        """Load data from parquet or root files."""
        file_paths = resolve_paths(self.dataset_config.paths)
        file_type = self.dataset_config.format

        for f in ["parquet", "root"]:
            if file_paths[0].endswith(f):
                file_type = f

        columns = []
        if self.dataset_config.features_columns:
            columns.extend(self.dataset_config.features_columns)
        if self.dataset_config.labels_columns:
            columns.extend(self.dataset_config.labels_columns)

        match file_type:
            case "parquet":
                df = pd.concat([pd.read_parquet(file, columns=columns) for file in file_paths], ignore_index=True)
            case "root":
                tree_name = self.dataset_config.dak_reader_kwargs.get("tree_name", "tree")
                df = pd.concat(
                    [file[tree_name].arrays(library="pd") for file in uproot.iterate(file_paths, expressions=columns)],
                    ignore_index=True,
                )
            case _:
                raise ValueError(f"Unsupported file type: {file_type}")

        return df

    @staticmethod
    def get_dataset():
        return PandasDataset

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.get_dataset()(self.train_df, features=self.features, labels=self.labels),
            batch_size=self.batch_size,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.get_dataset()(self.val_df, features=self.features, labels=self.labels),
            batch_size=self.batch_size,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.get_dataset()(self.test_df, features=self.features, labels=self.labels),
            batch_size=self.batch_size,
            shuffle=False,
        )

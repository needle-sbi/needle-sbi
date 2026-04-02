"""
Original author: I. Elsharkawy
Based on https://github.com/ibrahimEls/CNFParameterEstimation
Adapted by K. Schmidt
"""
from itertools import chain
from typing import Annotated, Literal, Optional

import numpy as np
import torch
from lightning import LightningDataModule
from pydantic import Field
from torch.utils.data import DataLoader, Subset, TensorDataset, random_split

from ..utils.selection import createJetData

Percentage = Annotated[float, Field(ge=0.0, le=1.0)]


class NormalizingFlowDatamodule(LightningDataModule):
    """Lightning datamodule for training conditional normalizing flows on jet data.

    This datamodule loads jet features from the FAIR Universe dataset, balances signal and
    background samples, and creates training and validation splits.
    """

    def __init__(
        self,
        train_on_signal: bool,
        num_jets: Literal[0, 1, 2],
        root_dir: str,
        batch_size: int = 1000,
        train_test_split: Percentage = 0.8,
        fold_index: int = 0,
        n_folds: int = 1,
    ) -> None:
        super().__init__()
        self.train_on_signal = train_on_signal  # called 's' in the original code
        self.num_jets = num_jets
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.train_test_split = train_test_split
        self.fold_index = fold_index
        self.n_folds = n_folds

    def prepare_data(self) -> None:
        """Placeholder for Lightning prepare_data hook.

        This datamodule does not perform a separate download or preparation step here.
        """

    def setup(self, stage: Optional[str] = None) -> None:
        """Load jet data, balance signal/background, and split into train/validation sets.

        Args:
            stage (Optional[str]): Stage name, not used by this datamodule.

        Side effects:
            Sets `self.train_dataset`, `self.val_dataset`, `self.X_mean`, and `self.X_std`.

        Raises:
            ValueError: If requested fold_index is out of range or n_folds is invalid.
        """
        j2_data, j2_detlabel, _, _ = createJetData(  # type: ignore
            jet_num=self.num_jets,
            useTestData=False,
            seed=78,
            root_dir=self.root_dir,
        )
        S_tensor = torch.tensor(j2_data[j2_detlabel == 1], dtype=torch.float32)
        BG_tensor = torch.tensor(j2_data[j2_detlabel == 0], dtype=torch.float32)

        # Equalize dataset size between signal and background
        max_size = np.min([len(S_tensor), len(BG_tensor)])

        if self.train_on_signal:
            dataset = TensorDataset(S_tensor[:max_size], BG_tensor[:max_size])
        else:
            dataset = TensorDataset(BG_tensor[:max_size], S_tensor[:max_size])

        if self.n_folds > 1:
            if self.fold_index < 0 or self.fold_index >= self.n_folds:
                raise ValueError(f"fold_index must be in [0, {self.n_folds - 1}], got {self.fold_index}")
            if self.n_folds > len(dataset):
                raise ValueError(
                    f"n_folds must be <= number of samples ({len(dataset)}) when using cross-fold validation"
                )

            total_samples = len(dataset)
            base_fold_size = total_samples // self.n_folds
            remainder = total_samples % self.n_folds
            fold_sizes = [base_fold_size + (1 if i < remainder else 0) for i in range(self.n_folds)]

            start = sum(fold_sizes[: self.fold_index])
            end = start + fold_sizes[self.fold_index]
            val_indices = range(start, end)
            train_indices = chain(range(0, start), range(end, total_samples))

            self.train_dataset = Subset(dataset, train_indices)  # type: ignore
            self.val_dataset = Subset(dataset, val_indices)
        else:
            n_val = int(self.train_test_split * len(dataset))
            n_train = len(dataset) - n_val
            self.train_dataset, self.val_dataset = random_split(dataset, [n_train, n_val])

        self.X_mean = torch.mean(S_tensor, dim=0)
        self.X_std = torch.std(S_tensor, dim=0)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

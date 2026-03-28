"""
Original author: I. Elsharkawy
Based on https://github.com/ibrahimEls/CNFParameterEstimation
Adapted by K. Schmidt
"""

from pathlib import Path
from typing import Annotated, Optional

import numpy as np
import torch
from lightning import LightningDataModule
from pydantic import Field
from torch.utils.data import DataLoader, TensorDataset, random_split
from utils.selection import createJetData

Percentage = Annotated[float, Field(ge=0.0, le=1.0)]


class FairUniverseDatamodule(LightningDataModule):
    def __init__(
        self,
        train_on_signal: bool,
        root_dir: str,
        batch_size: int,
        train_test_split: Percentage,
    ) -> None:
        super().__init__()
        self.train_on_signal = train_on_signal  # called 's' in the original code
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.train_test_split = train_test_split

    def prepare_data(self) -> None:
        ...

    def setup(self, stage: Optional[str] = None) -> None:
        j2_data, j2_detlabel, _, _ = createJetData(  # type: ignore
            jet_num=1,
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

        # Split dataset into training and validation sets
        n_val = int(self.train_test_split * len(dataset))
        n_train = len(dataset) - n_val

        self.train_dataset, self.val_dataset = random_split(dataset, [n_train, n_val])
        self.X_mean = torch.mean(S_tensor, dim=0)
        self.X_std = torch.std(S_tensor, dim=0)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

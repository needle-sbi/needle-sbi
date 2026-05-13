from dataclasses import dataclass
from typing import Any

import lightning as L
import torch

from needle.utils.config_schema import DatasetConfig
from needle.utils.logging import ColorFormatter

logger = ColorFormatter.get_logger("ml")


@dataclass
class MockTransformerConfig:
    factor: float
    patience: float
    init_lr: float


class MockTransformer(torch.nn.Module):
    def __init__(self, num_features: int) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(num_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Mock forward method for testing purposes.
        This method currently does nothing and returns the input tensor.
        """
        return self.linear(x)


class MockTransformerModule(L.LightningModule):
    def __init__(
        self,
        factor: float,
        patience: int,
        init_lr: float,
        dataset_config: dict,
        hidden_dim: int = 512,
        n_hidden: int = 30,
    ) -> None:
        super().__init__()
        self.factor = factor
        self.patience = patience
        self.num_features = len(DatasetConfig(**dataset_config).features_columns)  # type: ignore
        self.init_lr = init_lr
        self.hidden_dim = hidden_dim
        self.n_hidden = n_hidden

    def configure_model(self) -> None:
        self.model = MockTransformer(num_features=self.num_features)

    def configure_optimizers(self):  # type: ignore
        optimizer = torch.optim.Adam(self.parameters(), lr=self.init_lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            factor=self.factor,
            patience=self.patience,
        )
        return {
            "optimizer": optimizer,
            "scheduler": scheduler,
        }

    @property
    def loss_function(self):
        return torch.nn.CrossEntropyLoss()

    def forward(self, input, _target):
        return self.model(input)

    def training_step(self, batch: tuple[torch.Tensor, ...], batch_idx: int) -> Any:
        input, target = batch
        output = self(input, target)
        return self.loss_function(output, target)

    def validation_step(self, batch: tuple[torch.Tensor, ...], batch_idx: int) -> Any:
        input, target = batch
        output = self(input, target)
        val_loss = self.loss_function(output, target)
        self.log("val_loss", val_loss)

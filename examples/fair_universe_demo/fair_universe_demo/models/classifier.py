"""
Original author: I. Elsharkawy
Based on https://github.com/ibrahimEls/CNFParameterEstimation
Adapted by K. Schmidt
"""

from typing import Literal, Mapping

import lightning as L
import torch
import torch.nn.functional as F
from torch import nn


class CombinedClassifier(L.LightningModule):
    """Combined classifier for 1-jet and 2-jet event representations.

    The model uses separate input towers for 1-jet and 2-jet feature tensors,
    then passes both branches through a shared trunk and separate classification heads.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        latent_dim: int = 256,
        lr: float = 1e-3,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.lr = lr
        self.save_hyperparameters()
        self.train_loss_list = []
        self.val_loss_list = []

        self.input_2j = nn.Sequential(
            nn.Linear(31, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.input_1j = nn.Sequential(
            nn.Linear(24, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.shared_trunk = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.GELU(),
        )
        # Classification heads for each category
        self.classifier_2j = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, 1),  # BCE output (logits)
        )
        self.classifier_1j = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, 1),
        )

    def forward(self, x: torch.Tensor, jet_category: Literal[1, 2]) -> torch.Tensor:
        """Compute logits for the requested jet category.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, feature_dim].
            jet_category (Literal[1, 2]): Jet category selector for the input branch.

        Returns:
            torch.Tensor: Output logits tensor for the selected category.
        """
        if jet_category == 2:
            x = self.input_2j(x)
        elif jet_category == 1:
            x = self.input_1j(x)

        # Shared trunk
        x = self.shared_trunk(x)

        # Classification
        if jet_category == 2:
            logits = self.classifier_2j(x)
        elif jet_category == 1:
            logits = self.classifier_1j(x)

        return logits

    def training_step(self, batch: Mapping[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Compute training loss for one batch and log it.

        Args:
            batch (Mapping[str, torch.Tensor]): Batch dictionary containing
                'x_2j', 'x_1j', 'l_2j', and 'l_1j'.
            batch_idx (int): Batch index (not used).

        Returns:
            torch.Tensor: Computed training loss.
        """
        x_2j = batch["x_2j"]
        x_1j = batch["x_1j"]
        l_2j = batch["l_2j"].float()
        l_1j = batch["l_1j"].float()

        logits_2j = self.forward(x_2j, 2).squeeze(1)
        logits_1j = self.forward(x_1j, 1).squeeze(1)

        loss = F.binary_cross_entropy_with_logits(logits_2j, l_2j) + F.binary_cross_entropy_with_logits(logits_1j, l_1j)

        self.train_loss_list.append(loss.item())
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: Mapping[str, torch.Tensor], batch_idx: int):
        """Compute validation loss for one batch and log it.

        Args:
            batch (Mapping[str, torch.Tensor]): Batch dictionary containing
                'x_2j', 'x_1j', 'l_2j', and 'l_1j'.
            batch_idx (int): Batch index (not used).

        Returns:
            torch.Tensor: Computed validation loss.
        """
        x_2j = batch["x_2j"]
        x_1j = batch["x_1j"]
        # x_0j  = batch["x_0j"]
        l_2j = batch["l_2j"].float()
        l_1j = batch["l_1j"].float()
        # l_0j  = batch["l_0j"].float()

        logits_2j = self.forward(x_2j, 2).squeeze(1)
        logits_1j = self.forward(x_1j, 1).squeeze(1)
        # logits_0j = self.forward(x_0j, 0).squeeze(1)

        loss = F.binary_cross_entropy_with_logits(logits_2j, l_2j) + F.binary_cross_entropy_with_logits(logits_1j, l_1j)

        self.val_loss_list.append(loss.item())
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

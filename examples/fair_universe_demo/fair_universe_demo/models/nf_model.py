"""
Original author: I. Elsharkawy
Based on https://github.com/ibrahimEls/CNFParameterEstimation
Adapted by K. Schmidt
"""

from typing import Any, Literal, Tuple

import lightning as L
import torch

from .nf_datamodule import NormalizingFlowDatamodule
from .nf_layers import NormalizingQuadFlow


class ConditionalNormalizingFlowModule(L.LightningModule):
    """Conditional normalizing flow module for jet event densities.

    The model learns a distribution over jet features and can score inputs using
    a flow-based likelihood. It is used for both signal and background jets.
    """

    def __init__(
        self,
        num_jets: Literal[1, 2],
        num_layers: int = 10,
        lr: float = 1e-3,
        c: float = 1,
        clamp_val: float = -10,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.lr = lr
        self.x_mean: torch.Tensor
        self.x_std: torch.Tensor
        self.prior = torch.distributions.normal.Normal(loc=0.0, scale=1.0)
        self.c = c
        self.train_losses = []
        self.val_losses = []
        self.clamp_val = clamp_val
        self.num_jets = num_jets
        self.input_dim = {
            1: 20,  # Case: 1 jet
            2: 27,  # Case: 2 jets
        }[self.num_jets]
        self.flow = NormalizingQuadFlow(self.input_dim, num_layers)
        self.register_buffer("x_mean", torch.zeros(self.input_dim, dtype=torch.float32))
        self.register_buffer("x_std", torch.ones(self.input_dim, dtype=torch.float32))

    def on_train_start(self) -> None:
        datamodule: NormalizingFlowDatamodule = self.trainer.datamodule  # type: ignore
        x_mean_old = self.x_mean.clone()
        x_std_old = self.x_std.clone()

        self.x_mean = self.x_mean.copy_(datamodule.X_mean)
        self.x_std = self.x_std.copy_(datamodule.X_std + 1e-10)

        if torch.all(torch.isclose(self.x_mean, x_mean_old)).float():
            raise RuntimeError(
                f"Normalization for mean failed: New mean={self.x_mean} is same as " f"placeholder={x_mean_old}"
            )
        if torch.all(torch.isclose(self.x_std, x_std_old)).float():
            raise ValueError(
                f"Normalization for std failed: New std={self.x_std} is same as " f"placeholder={x_std_old}"
            )

    def forward(self, x: torch.Tensor, eval: bool = True) -> torch.Tensor:
        """Evaluate the normalizing flow on input data.

        Args:
            x (torch.Tensor): Input tensor with jet features.
            eval (bool): If True, evaluate without tracking gradients.

        Returns:
            torch.Tensor: Log probability scores for each sample.
        """
        if eval:
            with torch.no_grad():
                x = (x - self.x_mean) / self.x_std
                z, log_det = self.flow(x)

                z = torch.nan_to_num(z, nan=0.0, posinf=1e3, neginf=-1e3)

                log_z = self.prior.log_prob(z).sum(dim=1)
                log_prob = log_z + log_det

        else:
            x = (x - self.x_mean) / self.x_std
            z, log_det = self.flow(x)
            log_z = self.prior.log_prob(z).sum(dim=1)
            log_prob = log_z + log_det

        return log_prob

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Compute the training loss for one batch.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): Input batch, typically (x, y).
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: Training loss.
        """
        if len(batch) > 1:
            x, y = batch
            log_prob = self.forward(x, eval=False)
            log_prob_adv = self.forward(y, eval=False)
            log_prob_adv = torch.clamp(log_prob_adv, min=self.clamp_val)
            loss = -self.c * log_prob.mean() + log_prob_adv.mean()
            self.log("train_logprob_adv", log_prob_adv.mean(), prog_bar=True)

        else:
            x = batch[0]
            log_prob = self.forward(x)
            loss = -log_prob.mean()

        self.log("train_logprob", log_prob.mean(), prog_bar=True)
        self.log("train_loss", loss, prog_bar=True)
        self.train_losses.append(loss)

        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Compute the validation loss for one batch.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): Input batch, typically (x, y).
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: Validation loss.
        """
        if len(batch) > 1:
            x, y = batch
            log_prob = self.forward(x, eval=False)
            log_prob_adv = self.forward(y, eval=False)

            log_prob_adv = torch.clamp(log_prob_adv, min=self.clamp_val)
            loss = -self.c * log_prob.mean() + log_prob_adv.mean()

        else:
            x = batch[0]
            log_prob = self.forward(x)
            loss = -log_prob.mean()

        self.log("val_loss", loss, prog_bar=True)
        self.val_losses.append(loss)
        return loss

    def sample(self, num_samples: int, grad: bool = False) -> torch.Tensor:
        """Generate samples from the learned distribution.

        Args:
            num_samples (int): Number of samples to generate.
            grad (bool): If True, retain gradient information during inverse sampling.

        Returns:
            torch.Tensor: Samples in data space.
        """
        z = self.prior.sample((num_samples,))

        if grad:
            x_samples = self.flow.inverse(z)
        else:
            with torch.no_grad():
                x_samples = self.flow.inverse(z)

        return (x_samples * self.x_std) + self.x_mean

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.lr)

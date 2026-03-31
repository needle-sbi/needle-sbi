"""
Original author: I. Elsharkawy
Based on https://github.com/ibrahimEls/CNFParameterEstimation
Adapted by K. Schmidt
"""

import os
from typing import Literal, Tuple

import lightning as L
import torch
from tqdm import tqdm

from ..models.nf_layers import NormalizingQuadFlow


class ConditionalNormalizingFlowModule(L.LightningModule):
    def __init__(
        self,
        num_jets: Literal[0, 1, 2],
        num_layers: int = 10,
        lr: float = 1e-3,
        x_mean: float = 1,
        x_std: float = 0,
        c: float = 1,
        clamp_val: float = -10,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.lr = lr
        self.x_mean = torch.tensor(x_mean, dtype=torch.float32).to(self.device)
        self.x_std = torch.tensor(x_std + 10e-10, dtype=torch.float32).to(self.device)
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

    def forward(self, x: torch.Tensor, eval: bool = True) -> torch.Tensor:
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

    def sample(self, num_samples: int, grad=False) -> torch.Tensor:
        """
        Sample from the learned distribution.
        1. Sample latent variable z from the base distribution.
        2. Apply the inverse flow to obtain samples in data space.
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


def load_nf_models(models_dir: str, device: str):
    """
    Load NormalizingFlowModel models from a directory structure.

    The expected structure is:
        models_dir/
            1_jet/
                *.ckpt   # models for 1 jet (indices 0-3)
            2_jet/
                *.ckpt   # models for 2 jets (indices 4-7)

    Returns:
        A list of loaded models in order (first the 1_jet models, then the 2_jet models).
    """
    models = []

    one_jet_dir = os.path.join(models_dir, "1_jet")
    if not os.path.isdir(one_jet_dir):
        raise FileNotFoundError(f"Directory not found: {one_jet_dir}")

    two_jet_dir = os.path.join(models_dir, "2_jet")
    if not os.path.isdir(two_jet_dir):
        raise FileNotFoundError(f"Directory not found: {two_jet_dir}")

    checkpoints = [os.path.join(one_jet_dir, f) for f in sorted(os.listdir(one_jet_dir)) if f.endswith(".ckpt")] + [
        os.path.join(two_jet_dir, f) for f in sorted(os.listdir(two_jet_dir)) if f.endswith(".ckpt")
    ]

    for ckpt_path in tqdm(checkpoints):
        model = ConditionalNormalizingFlowModule.load_from_checkpoint(ckpt_path).to(device).eval().to(torch.float32)
        models.append(model)

    return models

"""
Original Author: I. Elsharkawy
Based on https://github.com/ibrahimEls/CNFParameterEstimation
Adapted by K. Schmidt
"""
from typing import Tuple

import torch


class AffineCouplingLayerGated(torch.nn.Module):
    """
    A RealNVP-style affine coupling layer with gating.
    For input x ∈ ℝ^d, a binary mask determines which coordinates remain unchanged.
    For the unmasked coordinates, the transformation is:

      y = x_masked + (1 - mask) * [ (1 - g) * x + g * (x * exp(s) + t) ],

    where g = sigmoid(gate) is computed from the network.
    """

    mask: torch.Tensor

    def __init__(self, input_dim: int, mask):
        """
        Args:
            input_dim (int): Dimensionality of the input.
            mask (list or array of length input_dim): Binary mask.
                  A 1 indicates that the coordinate is "masked" (left unchanged).
        """
        super().__init__()
        self.register_buffer("mask", torch.tensor(mask).float())
        self.input_dim = input_dim
        # For the unmasked coordinates, we now output three sets of parameters:
        # scale (s), translation (t), and a gating parameter (gate).
        # Hence the final layer outputs input_dim * 3.
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 64),
            torch.nn.GELU(),
            torch.nn.Linear(64, 64),
            torch.nn.GELU(),
            torch.nn.Linear(64, 64),
            torch.nn.GELU(),
            torch.nn.Linear(64, 64),
            torch.nn.GELU(),
            torch.nn.Linear(64, input_dim * 3),
            torch.nn.Tanh(),  # initial scaling; gating will be passed through a sigmoid later.
        )

    def forward(self, x):
        """Apply the forward transformation of the gated affine coupling layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Transformed tensor and log-determinant.
        """
        # x: (batch_size, input_dim)
        # Preserve the masked coordinates.
        x_masked = x * self.mask
        stg = self.net(x_masked)  # (batch_size, input_dim * 3)
        # Split the network output into three parts: s, t, and gate.
        s, t, gate = stg.chunk(3, dim=1)  # each of shape (batch_size, input_dim)
        # Only update the unmasked coordinates.
        s = s * (1 - self.mask)
        t = t * (1 - self.mask)
        gate = gate * (1 - self.mask)
        # Pass gate parameters through sigmoid to get values in [0,1].
        g = torch.sigmoid(gate)

        # For the unmasked coordinates, define:
        # y = x_masked + (1 - mask) * [ (1-g)*x + g*(x * exp(s) + t) ]
        x_transformed = (1 - g) * x + g * (x * torch.exp(s) + t)
        y = x_masked + (1 - self.mask) * x_transformed

        # The log-determinant for each unmasked coordinate is log((1-g) + g*exp(s))
        log_det = torch.log((1 - g) + g * torch.exp(s) + 1e-8)  # add small epsilon for numerical stability
        log_det = torch.sum(log_det * (1 - self.mask), dim=1)

        return y, log_det

    def inverse(self, y):
        """Apply the inverse transformation of the gated affine coupling layer.

        Args:
            y (torch.Tensor): Output tensor of shape (batch_size, input_dim).

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Inverted tensor and inverse log-determinant.
        """
        # y: (batch_size, input_dim)
        # Preserve masked coordinates.
        y_masked = y * self.mask
        stg = self.net(y_masked)
        s, t, gate = stg.chunk(3, dim=1)
        s = s * (1 - self.mask)
        t = t * (1 - self.mask)
        gate = gate * (1 - self.mask)
        g = torch.sigmoid(gate)

        # For unmasked coordinates, the forward mapping was:
        #   y = (1-g)*x + g*(x * exp(s) + t)
        # which can be rearranged as:
        #   y = x * ((1-g) + g*exp(s)) + g*t
        # So the inverse is:
        #   x = (y - g*t) / ((1-g) + g*exp(s))
        x_unmasked = (y - g * t) / ((1 - g) + g * torch.exp(s) + 1e-8)
        x = y_masked + (1 - self.mask) * x_unmasked

        # The inverse log-determinant is the negative of the forward one.
        log_det = -torch.log((1 - g) + g * torch.exp(s) + 1e-8)
        log_det = torch.sum(log_det * (1 - self.mask), dim=1)
        return x, log_det


class NormalizingFlowGated(torch.nn.Module):
    """
    A normalizing flow as a sequence of affine coupling layers with gating.
    This version works for any input dimension d.
    """

    def __init__(self, input_dim, n_layers):
        super().__init__()
        self.layers = torch.nn.ModuleList()

        for i in range(n_layers):
            # For each layer, create an alternating binary mask.
            # For layer i, the mask is defined such that:
            #   mask[j] = 1 if j % 2 == i % 2, else 0.
            mask = [1 if j % 2 == i % 2 else 0 for j in range(input_dim)]
            self.layers.append(AffineCouplingLayerGated(input_dim, mask))

    def forward(self, x):
        """Run input through the sequence of gated affine coupling layers.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Transformed tensor and total log-determinant.
        """
        log_det_total = 0
        for layer in self.layers:
            x, log_det = layer(x)
            log_det_total += log_det
        return x, log_det_total

    def inverse(self, z):
        """Invert the gated normalizing flow by applying inverse transforms.

        Args:
            z (torch.Tensor): Input tensor in latent space.

        Returns:
            torch.Tensor: Reconstructed tensor in original data space.
        """
        log_det_total = 0
        for layer in reversed(self.layers):
            z, log_det = layer.inverse(z)  # type: ignore
            log_det_total += log_det
        return z


class QuadraticCouplingLayer(torch.nn.Module):
    """
    A RealNVP-style quadratic coupling layer without gating.
    For an input x ∈ ℝ^d, a binary mask determines which coordinates remain unchanged.
    For the unmasked coordinates, the transformation is:

        y = x_masked + (1 - mask) * [ x + s*x^2 + t ],

    where s and t are computed from a neural network applied to the masked coordinates.
    """

    mask: torch.Tensor

    def __init__(self, input_dim, mask):
        """
        Args:
            input_dim (int): Dimensionality of the input.
            mask (list or array of length input_dim): Binary mask.
                A 1 indicates that the coordinate is "masked" (left unchanged).
        """
        super().__init__()
        self.register_buffer("mask", torch.tensor(mask).float())
        self.input_dim = input_dim
        # The network outputs two sets of parameters (s and t) per coordinate.
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 64),
            torch.nn.GELU(),
            torch.nn.Linear(64, 64),
            torch.nn.GELU(),
            torch.nn.Linear(64, 64),
            torch.nn.GELU(),
            torch.nn.Linear(64, 64),
            torch.nn.GELU(),
            torch.nn.Linear(64, input_dim * 2),
            torch.nn.Tanh(),  # initial scaling; gating will be passed through a sigmoid later.
        )

    def forward(self, x):
        """Apply the forward transformation of the quadratic coupling layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Transformed tensor and log-determinant.
        """
        # x: (batch_size, input_dim)
        # Preserve masked coordinates.
        x_masked = x * self.mask
        # Compute s and t from the masked part.
        st = self.net(x_masked)  # (batch_size, input_dim * 2)
        s, t = st.chunk(2, dim=1)  # each of shape (batch_size, input_dim)
        s = s * (1 - self.mask)
        t = t * (1 - self.mask)

        s_max = 0.1  # hyperparameter to control the maximum effect of the quadratic term
        s = s_max * torch.tanh(s)
        # Quadratic transformation for unmasked coordinates:
        #   y = x + s*x^2 + t
        x_transformed = x + s * (x**2) + t
        y = x_masked + (1 - self.mask) * x_transformed

        # The Jacobian for each unmasked coordinate is:
        #   dy/dx = 1 + 2*s*x
        log_det = torch.log(torch.abs(1 + 2 * s * x) + 1e-8)
        log_det = torch.sum(log_det * (1 - self.mask), dim=1)
        return y, log_det

    def inverse(self, y):
        # y: (batch_size, input_dim)
        # Preserve masked coordinates.
        y_masked = y * self.mask
        # Compute s and t from the masked part.
        st = self.net(y_masked)
        s, t = st.chunk(2, dim=1)
        s = s * (1 - self.mask)
        t = t * (1 - self.mask)

        # For unmasked coordinates, we need to invert:
        #   y = x + s*x^2 + t   =>   s*x^2 + x + t - y = 0.
        # Let A = s, B = 1, C = t - y.
        A = s
        B = torch.ones_like(y)
        C = t - y

        # Compute discriminant of quadratic equation.
        discriminant = B**2 - 4 * A * C
        discriminant = torch.clamp(discriminant, min=0.0)
        sqrt_disc = torch.sqrt(discriminant + 1e-8)

        # If A is near zero, fall back to linear solution: x = y - t.
        use_linear = A.abs() < 1e-8
        x_candidate = (-B + sqrt_disc) / (2 * A + 1e-8)
        linear_candidate = y - t

        x_unmasked = torch.where(use_linear, linear_candidate, x_candidate)
        x = y_masked + (1 - self.mask) * x_unmasked

        log_det = -torch.log(torch.abs(1 + 2 * s * x_unmasked) + 1e-8)
        log_det = torch.sum(log_det * (1 - self.mask), dim=1)
        return x, log_det


class NormalizingQuadFlow(torch.nn.Module):
    """
    A normalizing flow composed of a sequence of quadratic coupling layers without gating.
    """

    def __init__(self, input_dim, n_layers):
        super().__init__()
        self.layers = torch.nn.ModuleList()

        for i in range(n_layers):
            mask = [1 if j % 2 == i % 2 else 0 for j in range(input_dim)]

            if i == n_layers - 1:
                self.layers.append(QuadraticCouplingLayer(input_dim, mask))
            else:
                self.layers.append(AffineCouplingLayerGated(input_dim, mask))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run input through the full quadratic flow.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Transformed tensor and total log-determinant.
        """
        log_det_total = 0

        for layer in self.layers:
            x, log_det = layer(x)
            log_det_total += log_det

        return x, log_det_total

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        """Invert the full quadratic flow.

        Args:
            z (torch.Tensor): Latent tensor.

        Returns:
            torch.Tensor: Reconstructed data tensor.
        """
        log_det_total = 0

        for layer in reversed(self.layers):
            z, log_det = layer.inverse(z)  # type: ignore
            log_det_total += log_det

        return z

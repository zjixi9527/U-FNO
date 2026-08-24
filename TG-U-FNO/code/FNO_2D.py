"""Minimal utilities required by the original ``wave3d1.py`` model.

The submitted source snapshot imports these helpers but did not include the
module.  Keeping them here makes the original model importable without changing
its architecture.
"""

from __future__ import annotations

import torch


class GaussianNormalizer:
    def __init__(self, values: torch.Tensor, epsilon: float = 1e-5) -> None:
        self.mean = torch.mean(values)
        self.std = torch.std(values)
        self.epsilon = epsilon

    def encode(self, values: torch.Tensor) -> torch.Tensor:
        return (values - self.mean) / (self.std + self.epsilon)

    def decode(self, values: torch.Tensor) -> torch.Tensor:
        return values * (self.std + self.epsilon) + self.mean


class LpLoss:
    def __init__(self, size_average: bool = True, epsilon: float = 1e-12) -> None:
        self.size_average = size_average
        self.epsilon = epsilon

    def __call__(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        batch_size = prediction.shape[0]
        difference = prediction.reshape(batch_size, -1) - target.reshape(batch_size, -1)
        difference_norm = torch.linalg.vector_norm(difference, dim=1)
        target_norm = torch.linalg.vector_norm(target.reshape(batch_size, -1), dim=1)
        relative_error = difference_norm / torch.clamp(target_norm, min=self.epsilon)
        return relative_error.mean() if self.size_average else relative_error.sum()

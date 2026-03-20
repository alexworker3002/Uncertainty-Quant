from __future__ import annotations

import torch

from .base import UQOutput, predictive_entropy


@torch.no_grad()
def temperature_scaled_predict(model: torch.nn.Module, x: torch.Tensor, temperature: float = 1.0) -> UQOutput:
    model.eval()
    t = max(float(temperature), 1e-6)
    logits = model(x)
    mean_prob = torch.sigmoid(logits / t)
    variance = torch.zeros_like(mean_prob)
    entropy = predictive_entropy(mean_prob)
    return UQOutput(mean_prob=mean_prob, variance=variance, entropy=entropy)

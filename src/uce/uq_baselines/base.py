from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch


@dataclass
class UQOutput:
    mean_prob: torch.Tensor
    variance: torch.Tensor
    entropy: torch.Tensor


def predictive_entropy(prob: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    prob = torch.clamp(prob, eps, 1.0 - eps)
    return -(prob * torch.log(prob) + (1.0 - prob) * torch.log(1.0 - prob))


def summarize_samples(prob_samples: torch.Tensor) -> UQOutput:
    """prob_samples: [T, B, 1, H, W]"""
    mean_prob = prob_samples.mean(dim=0)
    variance = prob_samples.var(dim=0, unbiased=False)
    entropy = predictive_entropy(mean_prob)
    return UQOutput(mean_prob=mean_prob, variance=variance, entropy=entropy)

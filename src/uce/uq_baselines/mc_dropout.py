from __future__ import annotations

import torch

from .base import UQOutput, summarize_samples


def enable_dropout_layers(model: torch.nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, (torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout3d)):
            m.train()


@torch.no_grad()
def mc_dropout_predict(model: torch.nn.Module, x: torch.Tensor, num_samples: int = 20) -> UQOutput:
    model.eval()
    enable_dropout_layers(model)
    samples = []
    for _ in range(num_samples):
        logits = model(x)
        samples.append(torch.sigmoid(logits))
    prob_samples = torch.stack(samples, dim=0)
    return summarize_samples(prob_samples)

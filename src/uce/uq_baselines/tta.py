from __future__ import annotations

import torch

from .base import UQOutput, summarize_samples


@torch.no_grad()
def tta_predict(model: torch.nn.Module, x: torch.Tensor) -> UQOutput:
    model.eval()
    variants = [x, torch.flip(x, dims=[-1]), torch.flip(x, dims=[-2]), torch.flip(x, dims=[-1, -2])]
    preds = []
    for i, xv in enumerate(variants):
        logits = model(xv)
        p = torch.sigmoid(logits)
        if i == 1:
            p = torch.flip(p, dims=[-1])
        elif i == 2:
            p = torch.flip(p, dims=[-2])
        elif i == 3:
            p = torch.flip(p, dims=[-1, -2])
        preds.append(p)
    return summarize_samples(torch.stack(preds, dim=0))

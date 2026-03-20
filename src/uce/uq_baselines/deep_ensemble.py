from __future__ import annotations

from typing import Iterable

import torch

from .base import UQOutput, summarize_samples


@torch.no_grad()
def deep_ensemble_predict(models: Iterable[torch.nn.Module], x: torch.Tensor) -> UQOutput:
    preds = []
    for m in models:
        m.eval()
        logits = m(x)
        preds.append(torch.sigmoid(logits))
    return summarize_samples(torch.stack(preds, dim=0))

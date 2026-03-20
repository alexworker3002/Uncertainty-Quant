from __future__ import annotations

import torch

from .base import UQOutput, summarize_samples


@torch.no_grad()
def deterministic_predict(model: torch.nn.Module, x: torch.Tensor) -> UQOutput:
    model.eval()
    logits = model(x)
    probs = torch.sigmoid(logits)
    # Reuse summarize interface with T=1 for consistency.
    return summarize_samples(probs.unsqueeze(0))

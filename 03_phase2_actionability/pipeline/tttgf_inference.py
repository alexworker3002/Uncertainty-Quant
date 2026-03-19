"""
End-to-end TTTGF test-time inference loop.

Fixes backbone, activates LoRA, iterates THE + Geometric Fidelity for 5-10 steps
to repair structural hallucinations without degrading Dice.
"""

from typing import Callable, Optional

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None

import sys
import os

_project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def tttgf_step(
    model: "nn.Module",
    x: "torch.Tensor",
    the_fn: Callable,
    anchor_logits: "torch.Tensor",
    lr: float = 1e-4,
    the_weight: float = 1.0,
    fidelity_weight: float = 1.0,
) -> tuple["torch.Tensor", float, float]:
    """
    Single TTTGF optimization step.

    Parameters
    ----------
    model : nn.Module
        Model with LoRA (decoder trainable).
    x : torch.Tensor
        Input image.
    the_fn : callable
        Differentiable THE (e.g. the_autograd).
    anchor_logits : torch.Tensor
        Frozen initial prediction for Geometric Fidelity.
    lr, the_weight, fidelity_weight : float
        Step size and loss weights.

    Returns
    -------
    (pred_logits, the_loss, fidelity_loss)
    """
    if torch is None:
        raise ImportError("PyTorch required.")

    model.zero_grad()
    pred = model(x)
    if pred.shape[1] > 1:
        f = torch.softmax(pred, dim=1)[:, 1]  # foreground prob
    else:
        f = torch.sigmoid(pred).squeeze(1)

    the_loss = the_fn(f)
    fidelity_loss = ((pred - anchor_logits) ** 2).mean()

    loss = the_weight * the_loss + fidelity_weight * fidelity_loss
    loss.backward()

    with torch.no_grad():
        for p in model.parameters():
            if p.grad is not None:
                p.data.sub_(p.grad, alpha=lr)
                p.grad.zero_()

    return pred.detach(), float(the_loss.item()), float(fidelity_loss.item())


def tttgf_loop(
    model: "nn.Module",
    x: "torch.Tensor",
    the_fn: Callable,
    n_steps: int = 5,
    lr: float = 1e-4,
    the_weight: float = 1.0,
    fidelity_weight: float = 1.0,
) -> "torch.Tensor":
    """
    Run TTTGF test-time adaptation loop.

    Parameters
    ----------
    model : nn.Module
        Model with frozen backbone and trainable LoRA in decoder.
    x : torch.Tensor
        Input.
    the_fn : callable
        Differentiable THE.
    n_steps : int
        Number of gradient steps.
    lr, the_weight, fidelity_weight : float

    Returns
    -------
    torch.Tensor
        Final adapted prediction.
    """
    if torch is None:
        raise ImportError("PyTorch required.")

    model.eval()
    with torch.no_grad():
        anchor_logits = model(x).detach()

    model.train()
    pred = anchor_logits
    for _ in range(n_steps):
        pred, _, _ = tttgf_step(
            model,
            x,
            the_fn,
            anchor_logits,
            lr=lr,
            the_weight=the_weight,
            fidelity_weight=fidelity_weight,
        )

    return pred

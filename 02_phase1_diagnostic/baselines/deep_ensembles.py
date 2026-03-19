"""
Deep Ensembles baseline for pixel-level uncertainty quantification.

Multiple independently trained models; epistemic uncertainty via predictive
variance across ensemble members.
"""

from typing import List, Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    torch = None
    nn = None
    F = None


def ensemble_forward(
    models: List["nn.Module"],
    x: "torch.Tensor",
) -> "torch.Tensor":
    """
    Run forward pass for each ensemble member.

    Parameters
    ----------
    models : list of nn.Module
        Ensemble of segmentation models.
    x : torch.Tensor
        Input, shape (B, C, ...).

    Returns
    -------
    torch.Tensor
        Stacked logits, shape (n_models, B, num_classes, ...).
    """
    if torch is None:
        raise ImportError("PyTorch required for Deep Ensembles.")

    outputs = []
    for m in models:
        m.eval()
        with torch.no_grad():
            out = m(x)
        outputs.append(out)
    return torch.stack(outputs, dim=0)


def ensemble_uncertainty(
    models: List["nn.Module"],
    x: "torch.Tensor",
    reduction: str = "mean",
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """
    Compute pixel-level epistemic uncertainty via Deep Ensembles.

    Uses predictive variance of class probabilities across ensemble members.

    Parameters
    ----------
    models : list of nn.Module
        Ensemble models (same architecture, different initializations/training).
    x : torch.Tensor
        Input.
    reduction : str
        "mean" or "none".

    Returns
    -------
    (variance_map, mean_variance)
        variance_map: shape (B, 1, ...), per-pixel predictive variance.
        mean_variance: scalar or (B,) for correlation analysis.
    """
    if torch is None:
        raise ImportError("PyTorch required.")

    outputs = ensemble_forward(models, x)
    probs = F.softmax(outputs, dim=2)
    mean_probs = probs.mean(dim=0)
    var_probs = probs.var(dim=0).sum(dim=1, keepdim=True)

    if reduction == "mean":
        spatial_dims = tuple(range(2, var_probs.ndim))
        mean_var = var_probs.mean(dim=spatial_dims)
    else:
        mean_var = var_probs

    return var_probs, mean_var

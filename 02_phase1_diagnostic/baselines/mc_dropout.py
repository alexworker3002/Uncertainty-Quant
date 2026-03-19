"""
MC Dropout baseline for pixel-level uncertainty quantification.

Multiple forward passes with dropout enabled at test time to obtain epistemic
uncertainty via predictive entropy or variance.
"""

from typing import Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    torch = None
    nn = None
    F = None


def mc_dropout_forward(
    model: "nn.Module",
    x: "torch.Tensor",
    n_samples: int = 20,
) -> "torch.Tensor":
    """
    Run N forward passes with dropout enabled.

    Parameters
    ----------
    model : nn.Module
        Segmentation model (must have dropout layers).
    x : torch.Tensor
        Input, shape (B, C, D, H, W) or (B, C, H, W).
    n_samples : int
        Number of MC samples.

    Returns
    -------
    torch.Tensor
        Stacked logits, shape (n_samples, B, num_classes, ...).
    """
    if torch is None:
        raise ImportError("PyTorch required for MC Dropout.")

    model.train()  # Enable dropout
    with torch.no_grad():
        samples = [model(x) for _ in range(n_samples)]
    return torch.stack(samples, dim=0)


def predictive_entropy(probs: "torch.Tensor") -> "torch.Tensor":
    """
    Predictive entropy H = -sum_c p_c log p_c over classes.

    Parameters
    ----------
    probs : torch.Tensor, shape (N, C, ...)
        Class probabilities (e.g. mean over MC samples).

    Returns
    -------
    torch.Tensor
        Per-pixel entropy, shape (N, 1, ...).
    """
    if torch is None:
        raise ImportError("PyTorch required.")
    eps = 1e-8
    ent = -(probs * (probs + eps).log()).sum(dim=1, keepdim=True)
    return ent


def mc_dropout_uncertainty(
    model: "nn.Module",
    x: "torch.Tensor",
    n_samples: int = 20,
    reduction: str = "mean",
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """
    Compute pixel-level epistemic uncertainty via MC Dropout.

    Parameters
    ----------
    model : nn.Module
        Segmentation model.
    x : torch.Tensor
        Input.
    n_samples : int
        Number of MC samples.
    reduction : str
        "mean" (default) or "none". If "mean", returns mean entropy over spatial dims.

    Returns
    -------
    (entropy_map, mean_entropy)
        entropy_map: shape (B, 1, ...), per-pixel entropy.
        mean_entropy: scalar or (B,) mean entropy for correlation analysis.
    """
    if torch is None:
        raise ImportError("PyTorch required.")

    samples = mc_dropout_forward(model, x, n_samples)
    probs = F.softmax(samples, dim=2)
    mean_probs = probs.mean(dim=0)
    entropy_map = predictive_entropy(mean_probs)

    if reduction == "mean":
        spatial_dims = tuple(range(2, entropy_map.ndim))
        mean_entropy = entropy_map.mean(dim=spatial_dims)
    else:
        mean_entropy = entropy_map

    return entropy_map, mean_entropy

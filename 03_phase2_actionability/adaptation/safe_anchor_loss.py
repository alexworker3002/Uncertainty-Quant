"""
Geometric Fidelity (L2) anchor loss for Safe Adaptation.

Prevents Dice degradation during test-time TTTGF updates by anchoring
predictions to the initial (frozen) output.
"""

from typing import Optional

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    torch = None
    nn = None
    F = None


class GeometricFidelityLoss(nn.Module if nn else object):
    """
    L2 anchor: ||f_adapted - f_anchor||^2.

    Ensures test-time updates do not deviate too far from the original
    frozen prediction, providing a Safe Adaptation guarantee.
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        pred: "torch.Tensor",
        anchor: "torch.Tensor",
        mask: Optional["torch.Tensor"] = None,
    ) -> "torch.Tensor":
        """
        Parameters
        ----------
        pred : torch.Tensor
            Adapted prediction (logits or probabilities).
        anchor : torch.Tensor
            Original frozen prediction, same shape as pred.
        mask : torch.Tensor, optional
            Binary mask for valid region (e.g. foreground).

        Returns
        -------
        torch.Tensor
            Scalar loss.
        """
        diff = pred - anchor
        loss = (diff ** 2)
        if mask is not None:
            loss = loss * mask
        if self.reduction == "mean":
            if mask is not None:
                loss = loss.sum() / (mask.sum() + 1e-8)
            else:
                loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss


def geometric_fidelity_loss(
    pred: "torch.Tensor",
    anchor: "torch.Tensor",
    mask: Optional["torch.Tensor"] = None,
) -> "torch.Tensor":
    """
    Convenience function for Geometric Fidelity L2 loss.

    Parameters
    ----------
    pred, anchor : torch.Tensor
        Same shape.
    mask : torch.Tensor, optional

    Returns
    -------
    torch.Tensor
    """
    if torch is None:
        raise ImportError("PyTorch required.")
    return GeometricFidelityLoss(reduction="mean")(pred, anchor, mask)

"""
TTTGF Sparse Gradient Routing via torch.autograd.Function.

Forward: calls Phase 1 persistence + THE.
Backward: maps Sinkhorn gradients w.r.t. (b_i, d_i) to ∂THE/∂f(x) via Pairing Theorem.
Gradient is non-zero only at birth/death voxels.
"""

from typing import Any, Optional, Tuple

import numpy as np

try:
    import torch
    from torch.autograd import Function
except ImportError:
    torch = None
    Function = None

import sys
import os

_project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


class TopologicalHallucinationEnergyFunction(Function):
    """
    Autograd Function for THE with sparse gradient routing.

    Maps gradients from (b_i, d_i) back to f(x) only at birth/death voxels.
    """

    @staticmethod
    def forward(
        ctx: Any,
        f: "torch.Tensor",
        epsilon: float,
        reg_m: float,
        sigma: float,
        min_persistence: float,
    ) -> "torch.Tensor":
        from pathlib import Path
        _core_path = str(Path(_project_root) / "02_phase1_diagnostic" / "core")
        if _core_path not in sys.path:
            sys.path.insert(0, _core_path)
        from persistence_homology import extract_persistence
        from hallucination_energy import (
            compute_the,
            compute_cost_matrix,
            compute_the_gradient_wrt_pairs,
        )

        f_np = f.detach().cpu().numpy().astype(np.float64)
        shape = f_np.shape

        pd = extract_persistence(
            f_np,
            min_persistence=min_persistence,
            homology_dims=(0, 1),
        )

        the_val, P = compute_the(
            pd,
            epsilon=epsilon,
            reg_m=reg_m,
            sigma=sigma,
            return_plan=True,
        )

        C = compute_cost_matrix(pd.pairs, sigma=sigma)
        grad_pairs = compute_the_gradient_wrt_pairs(pd.pairs, P, C, sigma)

        ctx.save_for_backward(
            torch.from_numpy(pd.birth_indices.copy()),
            torch.from_numpy(pd.death_indices.copy()),
            torch.from_numpy(grad_pairs.copy()),
        )
        ctx.shape = shape

        return torch.tensor(the_val, dtype=f.dtype, device=f.device)

    @staticmethod
    def backward(ctx: Any, grad_output: "torch.Tensor") -> Tuple:
        if grad_output is None or grad_output.numel() == 0:
            return (None,) * 5

        birth_indices = ctx.saved_tensors[0].numpy()
        death_indices = ctx.saved_tensors[1].numpy()
        grad_pairs = ctx.saved_tensors[2].numpy()
        shape = ctx.shape

        grad_f = np.zeros(shape, dtype=np.float64)
        g = float(grad_output.item())

        for i in range(len(grad_pairs)):
            bi, di = birth_indices[i], death_indices[i]
            d_the_d_bi = grad_pairs[i, 0]
            d_the_d_di = grad_pairs[i, 1]

            if bi >= 0:
                coord = np.unravel_index(int(bi), shape, order="F")
                grad_f[coord] += g * d_the_d_bi
            if di >= 0 and di != bi:
                coord = np.unravel_index(int(di), shape, order="F")
                grad_f[coord] += g * d_the_d_di

        grad_f_t = torch.from_numpy(grad_f).to(
            dtype=grad_output.dtype,
            device=grad_output.device,
        )
        return grad_f_t, None, None, None, None


def the_autograd(
    f: "torch.Tensor",
    epsilon: float = 0.05,
    reg_m: float = 1.0,
    sigma: float = 0.1,
    min_persistence: float = 0.0,
) -> "torch.Tensor":
    """
    Differentiable THE: forward pass with sparse gradient in backward.

    Parameters
    ----------
    f : torch.Tensor
        Probability field, shape (D, H, W) or (H, W).
    epsilon, reg_m, sigma : float
        THE/Sinkhorn hyperparameters.
    min_persistence : float
        Minimum persistence threshold.

    Returns
    -------
    torch.Tensor
        THE scalar (0-dim).
    """
    if torch is None or Function is None:
        raise ImportError("PyTorch required for THE autograd.")

    return TopologicalHallucinationEnergyFunction.apply(
        f, epsilon, reg_m, sigma, min_persistence
    )

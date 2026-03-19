"""
Topological Hallucination Energy (THE) via Relaxed Sinkhorn Divergence.

Computes differentiable THE(f) as Wasserstein-like distance from persistence
diagram to diagonal. Uses POT for entropy-regularized unbalanced optimal
transport with persistence-aware cost matrix.

Ref: profile.md, 01_theory_and_math/THE.md
"""

from typing import Tuple

import numpy as np

try:
    import ot
except ImportError:
    ot = None

try:
    from .persistence_homology import PersistenceDiagram, extract_persistence
except ImportError:
    from persistence_homology import PersistenceDiagram, extract_persistence


def _diagonal_projection(b: float, d: float) -> tuple:
    """Project (b, d) onto diagonal: pi_Delta(b,d) = ((b+d)/2, (b+d)/2)."""
    mid = (b + d) / 2.0
    return (mid, mid)


def _persistence_weight(pers: float, sigma: float) -> float:
    """Persistence-aware gating: omega(p) = exp(-pers^2 / (2*sigma^2))."""
    return float(np.exp(-(pers**2) / (2.0 * sigma**2)))


def compute_cost_matrix(
    pairs: np.ndarray,
    sigma: float = 0.1,
    p: int = 2,
) -> np.ndarray:
    """
    Compute persistence-aware cost matrix C for THE.

    C_ij = omega(p_i) * ||p_i - pi_Delta(p_j)||_2^p
    where omega(p_i) = exp(-pers(p_i)^2 / (2*sigma^2))

    Parameters
    ----------
    pairs : np.ndarray, shape (M, 2)
        Persistence pairs (birth, death).
    sigma : float
        Decay parameter for persistence gating.
    p : int
        Power for cost (default 2 for squared Euclidean).

    Returns
    -------
    np.ndarray, shape (M, M)
        Cost matrix.
    """
    M = pairs.shape[0]
    C = np.zeros((M, M), dtype=np.float64)

    for i in range(M):
        bi, di = pairs[i, 0], pairs[i, 1]
        pers_i = bi - di
        omega_i = _persistence_weight(pers_i, sigma)

        for j in range(M):
            bj, dj = pairs[j, 0], pairs[j, 1]
            proj_j = _diagonal_projection(bj, dj)
            dist = np.sqrt((bi - proj_j[0]) ** 2 + (di - proj_j[1]) ** 2)
            C[i, j] = omega_i * (dist**p)

    return C


def compute_the(
    pd: PersistenceDiagram,
    epsilon: float = 0.05,
    reg_m: float = 1.0,
    sigma: float = 0.1,
    num_iter_max: int = 1000,
    return_plan: bool = False,
):
    """
    Compute Topological Hallucination Energy via unbalanced Sinkhorn.

    THE = min_P <P,C> - eps*H(P) + reg_m*D_KL(P1||a) + reg_m*D_KL(P^T 1||b)

    Parameters
    ----------
    pd : PersistenceDiagram
        From extract_persistence().
    epsilon : float
        Entropy regularization.
    reg_m : float
        Marginal relaxation (KL) for unbalanced OT.
    sigma : float
        Persistence gating decay.
    num_iter_max : int
        Max Sinkhorn iterations.

    Returns
    -------
    float
        THE scalar (non-negative).
    """
    if ot is None:
        raise ImportError(
            "POT (Python Optimal Transport) is required for THE. "
            "Install with: pip install pot"
        )

    pairs = pd.pairs
    M = pairs.shape[0]

    if M == 0:
        if return_plan:
            return 0.0, np.zeros((0, 0))
        return 0.0

    # Uniform marginals
    a = np.ones(M) / M
    b = np.ones(M) / M

    C = compute_cost_matrix(pairs, sigma=sigma)

    if return_plan:
        P = ot.unbalanced.sinkhorn_unbalanced(
            a,
            b,
            C,
            reg=epsilon,
            reg_m=reg_m,
            method="sinkhorn",
            numItermax=num_iter_max,
        )
        cost = np.sum(P * C)
        return float(np.maximum(cost, 0.0)), P

    # Unbalanced Sinkhorn returns the transport cost
    cost = ot.unbalanced.sinkhorn_unbalanced2(
        a,
        b,
        C,
        reg=epsilon,
        reg_m=reg_m,
        method="sinkhorn",
        numItermax=num_iter_max,
    )

    return float(np.maximum(cost, 0.0))


def compute_the_gradient_wrt_pairs(
    pairs: np.ndarray,
    P: np.ndarray,
    C: np.ndarray,
    sigma: float = 0.1,
) -> np.ndarray:
    """
    Compute dTHE/d(b_i) and dTHE/d(d_i) via envelope theorem.

    dTHE/d(b_i) = sum_j (P_ij + P_ji) * dC_ij/d(b_i) (simplified; full expression
    involves cost derivative). C_ij = omega_i * ||p_i - pi_Delta(p_j)||^2.
    """
    M = pairs.shape[0]
    grad = np.zeros((M, 2))

    for i in range(M):
        bi, di = pairs[i, 0], pairs[i, 1]
        pers_i = bi - di
        omega_i = _persistence_weight(pers_i, sigma)
        d_omega_d_bi = omega_i * (-pers_i / (sigma**2))
        d_omega_d_di = omega_i * (pers_i / (sigma**2))

        d_bi = 0.0
        d_di = 0.0
        for j in range(M):
            bj, dj = pairs[j, 0], pairs[j, 1]
            mj = (bj + dj) / 2.0
            dist_sq = (bi - mj) ** 2 + (di - mj) ** 2
            coeff = (P[i, j] + P[j, i]) if i != j else P[i, j]
            d_dist_sq_d_bi = 2 * (bi - mj)
            d_dist_sq_d_di = 2 * (di - mj)
            d_bi += coeff * (d_omega_d_bi * dist_sq + omega_i * d_dist_sq_d_bi)
            d_di += coeff * (d_omega_d_di * dist_sq + omega_i * d_dist_sq_d_di)

        grad[i, 0] = d_bi
        grad[i, 1] = d_di

    return grad


def the_from_probability(
    f: np.ndarray,
    min_persistence: float = 0.0,
    homology_dims: tuple = (0, 1),
    epsilon: float = 0.05,
    reg_m: float = 1.0,
    sigma: float = 0.1,
) -> Tuple[float, PersistenceDiagram]:
    """
    End-to-end THE computation from probability field.

    Parameters
    ----------
    f : np.ndarray
        Probability field in [0, 1].
    min_persistence : float
        Minimum persistence to keep.
    homology_dims : tuple
        Homology dimensions (default H0, H1).
    epsilon, reg_m, sigma : float
        THE/Sinkhorn hyperparameters.

    Returns
    -------
    (the_value, PersistenceDiagram)
    """
    pd = extract_persistence(f, min_persistence, homology_dims)
    the_val = compute_the(pd, epsilon=epsilon, reg_m=reg_m, sigma=sigma)
    return the_val, pd

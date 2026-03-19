"""
Unified correlation analysis: THE vs pixel-level UQ vs Graph Break Rate.

Compares THE, MC Dropout, Deep Ensembles, and struct-uncertainty baselines
against ground-truth structure quality (e.g., GBR) via Spearman correlation.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy.stats import spearmanr
except ImportError:
    spearmanr = None

import sys
import os

# Add project root for imports
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


@dataclass
class UQMetrics:
    """Container for per-sample UQ metrics from all baselines."""

    the_score: Optional[float] = None
    mc_dropout_entropy: Optional[float] = None
    ensemble_var: Optional[float] = None
    struct_unc_pred_mu: Optional[float] = None  # or per-structure mean
    gbr: Optional[float] = None  # Ground-truth Graph Break Rate


def compute_spearman_correlation(
    x: np.ndarray,
    y: np.ndarray,
) -> Tuple[float, float]:
    """
    Compute Spearman rank correlation and p-value.

    Parameters
    ----------
    x, y : np.ndarray
        Equal-length arrays.

    Returns
    -------
    (rho, p_value)
    """
    if spearmanr is None:
        raise ImportError("scipy required for Spearman correlation.")
    rho, pval = spearmanr(x, y, nan_policy="omit")
    return float(rho), float(pval)


def correlation_analysis(
    metrics_list: List[UQMetrics],
    uq_keys: Optional[List[str]] = None,
    gbr_key: str = "gbr",
) -> Dict[str, Dict[str, float]]:
    """
    Correlate each UQ metric with ground-truth GBR.

    Parameters
    ----------
    metrics_list : list of UQMetrics
        Per-sample metrics.
    uq_keys : list, optional
        Keys to correlate (default: the_score, mc_dropout_entropy, ensemble_var,
        struct_unc_pred_mu).
    gbr_key : str
        Ground-truth key (default "gbr").

    Returns
    -------
    dict
        {uq_key: {"spearman_rho": float, "p_value": float}}
    """
    if uq_keys is None:
        uq_keys = [
            "the_score",
            "mc_dropout_entropy",
            "ensemble_var",
            "struct_unc_pred_mu",
        ]

    gbr_vals = np.array(
        [getattr(m, gbr_key) for m in metrics_list],
        dtype=np.float64,
    )
    valid_gbr = ~np.isnan(gbr_vals)
    gbr_vals = gbr_vals[valid_gbr]

    results = {}

    for key in uq_keys:
        uq_vals = np.array(
            [getattr(m, key) for m in metrics_list],
            dtype=np.float64,
        )
        uq_vals = uq_vals[valid_gbr]
        valid = ~np.isnan(uq_vals) & ~np.isnan(gbr_vals)
        if valid.sum() < 3:
            results[key] = {"spearman_rho": np.nan, "p_value": np.nan}
            continue
        rho, pval = compute_spearman_correlation(uq_vals[valid], gbr_vals[valid])
        results[key] = {"spearman_rho": rho, "p_value": pval}

    return results


def collect_metrics_for_correlation(
    the_fn: Callable[[np.ndarray], float],
    pred_maps: List[np.ndarray],
    mc_dropout_entropies: Optional[List[float]] = None,
    ensemble_vars: Optional[List[float]] = None,
    struct_unc_mus: Optional[List[float]] = None,
    gbr_values: Optional[List[float]] = None,
) -> List[UQMetrics]:
    """
    Build UQMetrics list for correlation analysis.

    Parameters
    ----------
    the_fn : callable
        Function f(pred_map) -> THE scalar.
    pred_maps : list of np.ndarray
        Probability predictions.
    mc_dropout_entropies, ensemble_vars, struct_unc_mus, gbr_values : list, optional
        Per-sample baseline metrics.

    Returns
    -------
    list of UQMetrics
    """
    n = len(pred_maps)
    out = []

    for i in range(n):
        the_val = the_fn(pred_maps[i]) if pred_maps[i].size > 0 else np.nan
        m = UQMetrics(the_score=float(the_val))

        if mc_dropout_entropies is not None and i < len(mc_dropout_entropies):
            m.mc_dropout_entropy = float(mc_dropout_entropies[i])
        if ensemble_vars is not None and i < len(ensemble_vars):
            m.ensemble_var = float(ensemble_vars[i])
        if struct_unc_mus is not None and i < len(struct_unc_mus):
            v = struct_unc_mus[i]
            m.struct_unc_pred_mu = float(v) if v is not None else np.nan
        if gbr_values is not None and i < len(gbr_values):
            m.gbr = float(gbr_values[i])

        out.append(m)

    return out

"""
Bridge to struct-uncertainty (NeurIPS 2023) for structure-wise UQ baseline.

Provides unified interface to run struct-uncertainty inference when the
baseline repo is available and configured. Requires:
- 04_experiments/baselines/struct-uncertainty/ cloned
- Pretrained segmentation + uncertainty model checkpoints
- DIPHA compiled (optional; their pipeline uses it for DMT)
"""

from typing import Optional

import numpy as np

# Lazy import to avoid hard dependency
_struct_uncertainty_available = False
_struct_uncertainty_path: Optional[str] = None


def _get_struct_uncertainty_path() -> Optional[str]:
    """Resolve path to struct-uncertainty baseline."""
    import os

    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    path = os.path.join(base, "04_experiments", "baselines", "struct-uncertainty")
    if os.path.isdir(path):
        return path
    return None


def is_struct_uncertainty_available() -> bool:
    """Check if struct-uncertainty baseline can be used."""
    global _struct_uncertainty_available, _struct_uncertainty_path
    if _struct_uncertainty_path is None:
        _struct_uncertainty_path = _get_struct_uncertainty_path()
    if _struct_uncertainty_path is None:
        return False
    if _struct_uncertainty_available:
        return True
    try:
        import sys

        if _struct_uncertainty_path not in sys.path:
            sys.path.insert(0, _struct_uncertainty_path)
        import unc_model  # noqa: F401

        _struct_uncertainty_available = True
    except Exception:
        pass
    return _struct_uncertainty_available


def compute_struct_uncertainty(
    img: np.ndarray,
    likelihood: np.ndarray,
    unc_model_ckpt: str,
    seg_model_ckpt: Optional[str] = None,
    device: str = "cuda",
) -> Optional[np.ndarray]:
    """
    Run struct-uncertainty inference to get structure-wise (mu, log_var).

    This is a placeholder that documents the expected interface. Full integration
    requires running struct-uncertainty's infer.py with proper config. For
    metric_correlation_analysis, use run_struct_uncertainty_infer_script() to
    invoke their CLI and parse outputs.

    Parameters
    ----------
    img : np.ndarray
        Input image (C, H, W) or (C, D, H, W).
    likelihood : np.ndarray
        Segmentation probability map from base model.
    unc_model_ckpt : str
        Path to UncertaintyModel checkpoint.
    seg_model_ckpt : str, optional
        Path to segmentation model (if needed by their pipeline).
    device : str
        Device for inference.

    Returns
    -------
    np.ndarray or None
        Per-structure uncertainty (mu or combined), or None if unavailable.
    """
    if not is_struct_uncertainty_available():
        return None
    # Full implementation would load models, run DMT, get manifold features,
    # and forward through UncertaintyModel. For now return None as placeholder.
    return None


def run_struct_uncertainty_infer_script(
    params_path: str,
) -> Optional[dict]:
    """
    Run struct-uncertainty infer.py via subprocess and return parsed results.

    Use when struct-uncertainty is configured with datalists and checkpoints.
    Returns dict with keys like 'unc_pred_mu', 'unc_pred_logvar', 'structure_info'.

    Parameters
    ----------
    params_path : str
        Path to infer.json (e.g. datalists/DRIVE/infer.json).

    Returns
    -------
    dict or None
    """
    import subprocess
    import os

    path = _get_struct_uncertainty_path()
    if path is None:
        return None
    infer_py = os.path.join(path, "infer.py")
    if not os.path.isfile(infer_py):
        return None
    try:
        subprocess.run(
            ["python3", infer_py, "--params", params_path],
            cwd=path,
            check=True,
            capture_output=True,
        )
        # Parsing output files would go here; struct-uncertainty writes to
        # output_folder. Caller should specify output path in params.
        return {}
    except Exception:
        return None

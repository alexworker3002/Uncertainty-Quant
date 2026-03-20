"""
Bridge utilities for structure-wise uncertainty baselines and THE integration.

This file now provides:
1) Existing struct-uncertainty compatibility checks.
2) Runnable THE bridge from probability map -> scalar score (+ metadata),
   with stability and runtime controls.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import importlib.util
import numpy as np


# -----------------------------
# Struct-uncertainty baseline
# -----------------------------
_struct_uncertainty_available = False
_struct_uncertainty_path: Optional[str] = None


def _get_struct_uncertainty_path() -> Optional[str]:
    import os

    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    path = os.path.join(base, "04_experiments", "baselines", "struct-uncertainty")
    if os.path.isdir(path):
        return path
    return None


def is_struct_uncertainty_available() -> bool:
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
    if not is_struct_uncertainty_available():
        return None
    # Placeholder for external repo integration.
    return None


def run_struct_uncertainty_infer_script(params_path: str) -> Optional[dict]:
    import os
    import subprocess

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
        return {}
    except Exception:
        return None


# -----------------------------
# THE bridge
# -----------------------------
@dataclass
class THEResult:
    the_score: float
    num_pairs: int
    h0_pairs: int
    h1_pairs: int


def _load_the_module():
    """Load hallucination_energy.py by file path (folder name starts with digit)."""
    import sys

    core_dir = Path(__file__).resolve().parents[1] / "core"
    target = core_dir / "hallucination_energy.py"
    if not target.exists():
        raise FileNotFoundError(f"THE core file not found: {target}")

    if str(core_dir) not in sys.path:
        sys.path.insert(0, str(core_dir))

    spec = importlib.util.spec_from_file_location("hallucination_energy", str(target))
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to create module spec for THE core.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _downsample_for_the(prob: np.ndarray, max_hw: int) -> np.ndarray:
    if max_hw <= 0:
        return prob
    h, w = prob.shape
    max_side = max(h, w)
    if max_side <= max_hw:
        return prob

    step = int(np.ceil(max_side / max_hw))
    return prob[::step, ::step]


def compute_the_for_likelihood(
    likelihood: np.ndarray,
    min_persistence: float = 0.0,
    homology_dims: tuple[int, ...] = (0, 1),
    epsilon: float = 0.05,
    reg_m: float = 1.0,
    sigma: float = 0.1,
    num_iter_max: int = 300,
    max_hw: int = 192,
) -> THEResult:
    """Compute scalar THE from a 2D likelihood map in [0,1]."""
    if likelihood.ndim == 3 and likelihood.shape[0] == 1:
        likelihood = likelihood[0]
    if likelihood.ndim != 2:
        raise ValueError(f"Expected 2D likelihood map, got shape={likelihood.shape}")

    prob = np.asarray(likelihood, dtype=np.float64)
    prob = np.clip(prob, 0.0, 1.0)
    prob = _downsample_for_the(prob, max_hw=max_hw)

    mod = _load_the_module()
    pd = mod.extract_persistence(prob, min_persistence=min_persistence, homology_dims=homology_dims)
    the_val = mod.compute_the(pd, epsilon=epsilon, reg_m=reg_m, sigma=sigma, num_iter_max=num_iter_max)

    dims = pd.dimensions if pd.dimensions.size > 0 else np.array([], dtype=np.int64)
    h0 = int(np.sum(dims == 0))
    h1 = int(np.sum(dims == 1))

    return THEResult(
        the_score=float(the_val),
        num_pairs=int(pd.pairs.shape[0]),
        h0_pairs=h0,
        h1_pairs=h1,
    )


def compute_the_from_uq_npz(
    uq_npz: str,
    min_persistence: float = 0.0,
    homology_dims: tuple[int, ...] = (0, 1),
    epsilon: float = 0.05,
    reg_m: float = 1.0,
    sigma: float = 0.1,
    num_iter_max: int = 300,
    max_samples: int = 0,
    max_hw: int = 192,
    log_every: int = 1,
) -> list[dict]:
    """Run THE over `mean_prob` inside UQ npz and return per-sample results."""
    data = np.load(uq_npz, allow_pickle=True)
    if "mean_prob" not in data:
        raise KeyError(f"mean_prob missing in {uq_npz}")

    mean_prob = data["mean_prob"]
    if mean_prob.ndim != 4:
        raise ValueError(f"Expected mean_prob shape [N,1,H,W], got {mean_prob.shape}")

    names = [f"sample_{i:04d}.png" for i in range(mean_prob.shape[0])]
    if "names" in data:
        names = [str(x) for x in data["names"].tolist()]

    n_total = mean_prob.shape[0]
    n_run = n_total if max_samples <= 0 else min(max_samples, n_total)

    out: list[dict] = []
    for i in range(n_run):
        prob = np.asarray(mean_prob[i, 0], dtype=np.float64)
        r = compute_the_for_likelihood(
            prob,
            min_persistence=min_persistence,
            homology_dims=homology_dims,
            epsilon=epsilon,
            reg_m=reg_m,
            sigma=sigma,
            num_iter_max=num_iter_max,
            max_hw=max_hw,
        )
        out.append(
            {
                "name": names[i],
                "the_score": r.the_score,
                "num_pairs": r.num_pairs,
                "h0_pairs": r.h0_pairs,
                "h1_pairs": r.h1_pairs,
            }
        )
        if log_every > 0 and (i + 1) % log_every == 0:
            print(f"[THE] processed {i+1}/{n_run} samples")

    return out

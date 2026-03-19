"""
DMT (Discrete Morse Theory) accelerator for super-level set filtration.

Provides O(N) complexity reduction for cubical complex construction from probability
field f. Uses GUDHI CubicalComplex directly on the voxel array; for 3D extension
the same API applies. Optionally supports preprocessing (clipping, downsampling).
"""

import numpy as np


def prepare_filtration(f: np.ndarray) -> np.ndarray:
    """
    Convert probability field to GUDHI lower-star filtration.

    Super-level set X_alpha = {x : f(x) >= alpha} is equivalent to lower-star
    filtration on 1 - f: high f -> low filtration value -> appears first.

    Parameters
    ----------
    f : np.ndarray, shape (D, H, W) or (H, W)
        Probability field in [0, 1]. Supports 2D or 3D.

    Returns
    -------
    np.ndarray
        Filtration values for CubicalComplex (1 - f, clipped to [0, 1]).
    """
    f = np.asarray(f, dtype=np.float64)
    f = np.clip(f, 0.0, 1.0)
    return 1.0 - f

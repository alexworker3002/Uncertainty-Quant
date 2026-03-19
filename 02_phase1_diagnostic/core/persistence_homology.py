"""
Persistence homology feature extraction via GUDHI CubicalComplex.

Extracts H0/H1 persistence diagrams from super-level set filtration.
Outputs persistence pairs [(b_i, d_i), ...] and Birth/Death cell indices
for gradient routing (Phase 2).
"""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

try:
    import gudhi
except ImportError:
    gudhi = None


@dataclass
class PersistenceDiagram:
    """Persistence diagram with coordinates in f-space and cell indices."""

    pairs: np.ndarray  # (M, 2) array of (birth, death) in [0, 1]
    dimensions: np.ndarray  # (M,) homological dimension for each pair
    birth_indices: np.ndarray  # (M,) flat indices of birth cells (for gradient)
    death_indices: np.ndarray  # (M,) flat indices of death cells (for gradient)
    shape: Tuple[int, ...]  # shape of the original voxel grid


def extract_persistence(
    f: np.ndarray,
    min_persistence: float = 0.0,
    homology_dims: Tuple[int, ...] = (0, 1),
) -> PersistenceDiagram:
    """
    Extract persistence diagram from probability field using GUDHI.

    Parameters
    ----------
    f : np.ndarray
        Probability field in [0, 1], shape (D, H, W) or (H, W).
    min_persistence : float
        Minimum persistence (birth - death) to keep. Use -1 for all.
    homology_dims : tuple
        Homology dimensions to extract (default H0 and H1).

    Returns
    -------
    PersistenceDiagram
        Pairs (b, d) in f-space, dimensions, and cell indices for routing.
    """
    if gudhi is None:
        raise ImportError(
            "GUDHI is required for persistence homology. "
            "Install with: pip install gudhi"
        )

    # Prepare filtration: 1 - f for super-level set
    try:
        from .dmt_accelerator import prepare_filtration
    except ImportError:
        from dmt_accelerator import prepare_filtration

    filt = prepare_filtration(f)
    shape = filt.shape

    # Build cubical complex
    cc = gudhi.CubicalComplex(top_dimensional_cells=filt)

    # Compute persistence
    cc.compute_persistence(min_persistence=min_persistence)

    pairs_list: List[Tuple[float, float, int, int, int]] = []

    for dim in homology_dims:
        intervals = cc.persistence_intervals_in_dimension(dim)
        if intervals.size == 0:
            continue

        for idx in range(intervals.shape[0]):
            birth_filt, death_filt = intervals[idx, 0], intervals[idx, 1]
            # Convert to f-space: b = 1 - birth_filt, d = 1 - death_filt
            birth_f = 1.0 - float(birth_filt)
            death_f = 1.0 - float(death_filt)
            # Handle infinite death (essential features)
            if np.isinf(death_filt):
                death_f = 0.0
            pairs_list.append((birth_f, death_f, dim, birth_filt, death_filt))

    if not pairs_list:
        return PersistenceDiagram(
            pairs=np.zeros((0, 2)),
            dimensions=np.zeros(0, dtype=np.int64),
            birth_indices=np.zeros(0, dtype=np.int64),
            death_indices=np.zeros(0, dtype=np.int64),
            shape=shape,
        )

    # Get cofaces (cell indices) for gradient routing
    regular_pairs, _ = cc.cofaces_of_persistence_pairs()

    birth_inds: List[int] = []
    death_inds: List[int] = []
    dim_counters: dict = {}

    for birth_f, death_f, dim, _birth_filt, _death_filt in pairs_list:
        if dim not in dim_counters:
            dim_counters[dim] = 0
        i = dim_counters[dim]
        dim_counters[dim] += 1

        if dim < len(regular_pairs) and regular_pairs[dim].size > 0:
            arr = regular_pairs[dim]
            if i < arr.shape[0]:
                pos_idx, neg_idx = int(arr[i, 0]), int(arr[i, 1])
                birth_inds.append(pos_idx)
                death_inds.append(neg_idx)
            else:
                birth_inds.append(-1)
                death_inds.append(-1)
        else:
            birth_inds.append(-1)
            death_inds.append(-1)

    pairs_arr = np.array([[p[0], p[1]] for p in pairs_list], dtype=np.float64)
    dims_arr = np.array([p[2] for p in pairs_list], dtype=np.int64)

    return PersistenceDiagram(
        pairs=pairs_arr,
        dimensions=dims_arr,
        birth_indices=np.array(birth_inds, dtype=np.int64),
        death_indices=np.array(death_inds, dtype=np.int64),
        shape=shape,
    )

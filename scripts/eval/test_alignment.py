from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _require(pkg: str):
    try:
        return __import__(pkg)
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"Missing dependency: {pkg}") from e


def _compute_gudhi_pairs(prob: np.ndarray) -> dict[int, np.ndarray]:
    gudhi = _require("gudhi")
    cc = gudhi.CubicalComplex(top_dimensional_cells=(1.0 - prob.astype(np.float64)))
    cc.compute_persistence(min_persistence=0.0)

    out: dict[int, np.ndarray] = {}
    for dim in (0, 1):
        arr = cc.persistence_intervals_in_dimension(dim)
        if arr.size == 0:
            out[dim] = np.zeros((0, 2), dtype=np.float64)
        else:
            b = 1.0 - arr[:, 0]
            d = np.where(np.isinf(arr[:, 1]), 0.0, 1.0 - arr[:, 1])
            out[dim] = np.stack([b, d], axis=1)
    return out


def _compute_dmt_pairs(prob: np.ndarray) -> dict[int, np.ndarray]:
    _require("dmt_implicit_ext")
    import dmt_implicit_ext as dmt

    ret = dmt.extract_persistence_2d(prob.astype(np.float64), 0.0, [0, 1])
    pairs = np.asarray(ret["pairs"], dtype=np.float64)
    dims = np.asarray(ret["dimensions"], dtype=np.int64)
    out: dict[int, np.ndarray] = {}
    for dim in (0, 1):
        out[dim] = pairs[dims == dim] if pairs.size else np.zeros((0, 2), dtype=np.float64)
    return out


def _sort_pairs(pd: np.ndarray) -> np.ndarray:
    if pd.size == 0:
        return pd.reshape(0, 2)
    order = np.lexsort((pd[:, 1], pd[:, 0]))
    return pd[order]


def _diag_only_distance(pd: np.ndarray) -> float:
    if pd.shape[0] == 0:
        return 0.0
    life = np.maximum(0.0, pd[:, 0] - pd[:, 1])
    return float(np.max(life) * 0.5)


def _safe_bottleneck(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape[0] == 0 and b.shape[0] == 0:
        return 0.0
    if a.shape[0] == 0:
        return _diag_only_distance(b)
    if b.shape[0] == 0:
        return _diag_only_distance(a)

    gudhi = _require("gudhi")
    try:
        return float(gudhi.bottleneck_distance(a, b, e=0.0))
    except Exception:
        return float("inf")


def _build_micro_cases() -> list[tuple[str, np.ndarray]]:
    # Case A: one strong peak at center
    c_peak = np.array(
        [
            [0.1, 0.2, 0.1],
            [0.2, 0.9, 0.2],
            [0.1, 0.2, 0.1],
        ],
        dtype=np.float64,
    )

    # Case B: ring-like pattern (high border, low center)
    c_ring = np.array(
        [
            [0.9, 0.9, 0.9],
            [0.9, 0.1, 0.9],
            [0.9, 0.9, 0.9],
        ],
        dtype=np.float64,
    )

    # Case C: tie-heavy plateaus to stress tie-breaking
    c_tie = np.array(
        [
            [0.7, 0.7, 0.7],
            [0.7, 0.7, 0.7],
            [0.7, 0.7, 0.7],
        ],
        dtype=np.float64,
    )

    return [
        ("micro_peak", c_peak),
        ("micro_ring", c_ring),
        ("micro_tie", c_tie),
    ]


def _print_pair_dump(label: str, dim: int, g_pd: np.ndarray, d_pd: np.ndarray) -> None:
    print(f"  [{label}] dim={dim}")
    print(f"    gudhi pairs ({g_pd.shape[0]}): {_sort_pairs(g_pd)}")
    print(f"    dmt   pairs ({d_pd.shape[0]}): {_sort_pairs(d_pd)}")


def _check_case(prob: np.ndarray, case_name: str, strict: bool, tol: float) -> tuple[float, float]:
    g = _compute_gudhi_pairs(prob)
    d = _compute_dmt_pairs(prob)

    n0_g, n0_d = g[0].shape[0], d[0].shape[0]
    n1_g, n1_d = g[1].shape[0], d[1].shape[0]

    b0 = _safe_bottleneck(g[0], d[0])
    b1 = _safe_bottleneck(g[1], d[1])

    print(
        f"[{case_name}] H0 count gudhi={n0_g}, dmt={n0_d}, bottleneck={b0:.6e} | "
        f"H1 count gudhi={n1_g}, dmt={n1_d}, bottleneck={b1:.6e}"
    )

    _print_pair_dump(case_name, 0, g[0], d[0])
    _print_pair_dump(case_name, 1, g[1], d[1])

    if strict:
        if n0_g != n0_d or n1_g != n1_d:
            raise AssertionError(
                f"Pair count mismatch at case {case_name}: "
                f"H0 gudhi={n0_g} dmt={n0_d}, "
                f"H1 gudhi={n1_g} dmt={n1_d}"
            )
        if b0 >= tol or b1 >= tol:
            raise AssertionError(
                f"Bottleneck threshold failed at case {case_name}: "
                f"H0={b0:.6e}, H1={b1:.6e}, tol={tol:.6e}"
            )

    return b0, b1


def run_alignment(
    num_cases: int,
    seed: int,
    h: int,
    w: int,
    strict: bool,
    tol: float,
    run_micro: bool,
) -> None:
    rng = np.random.default_rng(seed)
    max_bdim0 = 0.0
    max_bdim1 = 0.0

    if run_micro:
        print("[micro-tests] Running handcrafted 3x3 cases for deterministic alignment checks")
        for name, prob in _build_micro_cases():
            b0, b1 = _check_case(prob, name, strict, tol)
            max_bdim0 = max(max_bdim0, b0)
            max_bdim1 = max(max_bdim1, b1)

    print("[random-tests] Running stochastic cases")
    for i in range(num_cases):
        base = rng.random((h, w), dtype=np.float64)
        noise = 0.05 * rng.standard_normal((h, w))
        prob = np.clip(base + noise, 0.0, 1.0)

        b0, b1 = _check_case(prob, f"rand_{i:03d}", strict, tol)
        max_bdim0 = max(max_bdim0, b0)
        max_bdim1 = max(max_bdim1, b1)

    print("\n[summary]")
    print(f"max bottleneck H0: {max_bdim0:.6e}")
    print(f"max bottleneck H1: {max_bdim1:.6e}")

    if strict:
        print(f"Strict assertions PASSED (counts match, bottleneck < {tol:.1e})")
    else:
        print("\nNOTE: strict assertions are intentionally not enforced yet because dmt_cpp is still prototype.")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--num_cases", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--h", type=int, default=64)
    p.add_argument("--w", type=int, default=64)
    p.add_argument("--strict", action="store_true")
    p.add_argument("--tol", type=float, default=1e-5)
    p.add_argument("--write_note", type=str, default="")
    p.add_argument(
        "--run_micro",
        action="store_true",
        help="Run deterministic 3x3 micro-tests for quick GUDHI vs DMT alignment checks",
    )
    args = p.parse_args()

    run_alignment(
        args.num_cases,
        args.seed,
        args.h,
        args.w,
        args.strict,
        args.tol,
        args.run_micro,
    )

    if args.write_note:
        out = Path(args.write_note)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            "Alignment script executed in prototype mode. "
            "Strict equality assertions to be enabled after PH-equivalent DMT implementation.\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()

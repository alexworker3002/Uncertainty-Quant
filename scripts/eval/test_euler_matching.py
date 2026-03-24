from __future__ import annotations

import numpy as np


def main() -> None:
    import dmt_implicit_ext as dmt

    rng = np.random.default_rng(42)
    prob = rng.random((10, 10), dtype=np.float64)

    grad = dmt.debug_build_gradient_2d(prob, 1e-12)
    v2e = np.asarray(grad["v2e"], dtype=np.int64)
    e2v = np.asarray(grad["e2v"], dtype=np.int64)
    e2f = np.asarray(grad["e2f"], dtype=np.int64)
    f2e = np.asarray(grad["f2e"], dtype=np.int64)

    c0 = int(np.sum(v2e < 0))
    c2 = int(np.sum(f2e < 0))
    c1 = int(np.sum((e2v < 0) & (e2f < 0)))

    euler = c0 - c1 + c2

    print(f"critical cells: c0={c0}, c1={c1}, c2={c2}")
    print(f"Euler characteristic: {euler}")

    if euler == 1:
        print("Euler check PASSED")
    else:
        print("Euler check FAILED")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path
import importlib.util
import sys

import numpy as np


def load_bridge_module():
    root = Path(__file__).resolve().parents[2]
    bridge_path = root / "02_phase1_diagnostic" / "baselines" / "struct_uncertainty_bridge.py"
    spec = importlib.util.spec_from_file_location("struct_uncertainty_bridge", str(bridge_path))
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load struct_uncertainty_bridge module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate THE scores from a UQ npz file.")
    p.add_argument("--uq_npz", type=str, required=True)
    p.add_argument("--save_csv", type=str, default="")
    p.add_argument("--epsilon", type=float, default=0.05)
    p.add_argument("--reg_m", type=float, default=1.0)
    p.add_argument("--sigma", type=float, default=0.1)
    p.add_argument("--min_persistence", type=float, default=0.0)
    p.add_argument("--num_iter_max", type=int, default=300)
    p.add_argument("--max_samples", type=int, default=0, help="0 means all samples")
    p.add_argument("--max_hw", type=int, default=192, help="downsample long side to control runtime")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    bridge = load_bridge_module()

    rows = bridge.compute_the_from_uq_npz(
        uq_npz=args.uq_npz,
        min_persistence=args.min_persistence,
        homology_dims=(0, 1),
        epsilon=args.epsilon,
        reg_m=args.reg_m,
        sigma=args.sigma,
        num_iter_max=args.num_iter_max,
        max_samples=args.max_samples,
        max_hw=args.max_hw,
        log_every=1,
    )

    if not rows:
        raise RuntimeError("No THE rows computed")

    the_scores = np.array([r["the_score"] for r in rows], dtype=np.float64)
    num_pairs = np.array([r["num_pairs"] for r in rows], dtype=np.float64)

    print("[THE Metrics]")
    print(f"samples: {len(rows)}")
    print(f"THE mean: {the_scores.mean():.6f}")
    print(f"THE std : {the_scores.std():.6f}")
    print(f"pairs mean: {num_pairs.mean():.2f}")

    if args.save_csv:
        out = Path(args.save_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            f.write("name,the_score,num_pairs,h0_pairs,h1_pairs\n")
            for r in rows:
                f.write(f"{r['name']},{r['the_score']:.8f},{r['num_pairs']},{r['h0_pairs']},{r['h1_pairs']}\n")
        print(f"saved_csv: {out}")


if __name__ == "__main__":
    main()

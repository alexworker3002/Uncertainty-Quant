from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uq_npz", type=str, required=True, help="npz with keys: mean_prob, variance, entropy")
    parser.add_argument("--gt_dir", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    uq_path = Path(args.uq_npz)
    if not uq_path.exists():
        raise FileNotFoundError(uq_path)

    stats = np.load(uq_path)
    for k in ("mean_prob", "variance", "entropy"):
        if k not in stats:
            raise KeyError(f"Missing key '{k}' in {uq_path}")

    mean_prob = stats["mean_prob"]
    variance = stats["variance"]
    entropy = stats["entropy"]

    print("[Scaffold] UQ summary")
    print(f"mean_prob shape: {mean_prob.shape}")
    print(f"variance mean: {variance.mean():.6f}")
    print(f"entropy mean : {entropy.mean():.6f}")
    print("TODO: add ECE/NLL/Brier and risk-coverage metrics once GT loading protocol is fixed.")


if __name__ == "__main__":
    main()

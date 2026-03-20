from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uq_npz", type=str, required=True, help="npz with keys: mean_prob, variance, entropy[, names]")
    parser.add_argument("--gt_dir", type=str, required=True)
    parser.add_argument("--num_bins", type=int, default=15)
    return parser.parse_args()


def load_mask(path: Path) -> np.ndarray:
    arr = np.array(Image.open(path).convert("L"), dtype=np.uint8)
    return (arr > 127).astype(np.uint8)


def center_crop_to_shape(arr: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = arr.shape
    th, tw = shape
    top = max((h - th) // 2, 0)
    left = max((w - tw) // 2, 0)
    return arr[top : top + th, left : left + tw]


def binary_nll(probs: np.ndarray, targets: np.ndarray, eps: float = 1e-8) -> float:
    p = np.clip(probs, eps, 1 - eps)
    nll = -(targets * np.log(p) + (1 - targets) * np.log(1 - p))
    return float(np.mean(nll))


def binary_brier(probs: np.ndarray, targets: np.ndarray) -> float:
    return float(np.mean((probs - targets) ** 2))


def binary_ece(probs: np.ndarray, targets: np.ndarray, num_bins: int = 15) -> float:
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    n = len(probs)
    for i in range(num_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == num_bins - 1:
            mask = (probs >= lo) & (probs <= hi)
        else:
            mask = (probs >= lo) & (probs < hi)
        if not np.any(mask):
            continue
        conf = float(np.mean(probs[mask]))
        acc = float(np.mean(targets[mask]))
        w = float(np.sum(mask) / n)
        ece += abs(conf - acc) * w
    return float(ece)


def aurc_from_entropy(errors: np.ndarray, entropy: np.ndarray) -> tuple[float, float]:
    # coverage from low-uncertainty to high-uncertainty
    order = np.argsort(entropy)
    sorted_err = errors[order].astype(np.float64)
    n = len(sorted_err)

    risks: list[float] = []
    coverages: list[float] = []
    cum_err = 0.0
    for k in range(1, n + 1):
        cum_err += sorted_err[k - 1]
        risk_k = cum_err / k
        cov_k = k / n
        risks.append(risk_k)
        coverages.append(cov_k)

    aurc = float(np.trapz(np.array(risks), np.array(coverages)))
    risk_at_100 = float(risks[-1]) if risks else 0.0
    return aurc, risk_at_100


def main() -> None:
    args = parse_args()
    uq_path = Path(args.uq_npz)
    if not uq_path.exists():
        raise FileNotFoundError(uq_path)

    stats = np.load(uq_path, allow_pickle=True)
    for k in ("mean_prob", "variance", "entropy"):
        if k not in stats:
            raise KeyError(f"Missing key '{k}' in {uq_path}")

    mean_prob = stats["mean_prob"]  # [N,1,H,W]
    variance = stats["variance"]
    entropy = stats["entropy"]

    if "names" in stats:
        names = [str(x) for x in stats["names"].tolist()]
    else:
        names = [f"sample_{i:04d}.png" for i in range(mean_prob.shape[0])]

    gt_dir = Path(args.gt_dir)

    probs_flat_parts: list[np.ndarray] = []
    gts_flat_parts: list[np.ndarray] = []
    entropy_flat_parts: list[np.ndarray] = []

    for i, name in enumerate(names):
        gt_path = gt_dir / name
        if not gt_path.exists():
            continue

        pred_prob = mean_prob[i, 0]
        pred_entropy = entropy[i, 0]
        gt = load_mask(gt_path)

        if gt.shape != pred_prob.shape:
            gt = center_crop_to_shape(gt, pred_prob.shape)

        probs_flat_parts.append(pred_prob.reshape(-1).astype(np.float64))
        gts_flat_parts.append(gt.reshape(-1).astype(np.float64))
        entropy_flat_parts.append(pred_entropy.reshape(-1).astype(np.float64))

    if not probs_flat_parts:
        raise RuntimeError("No matched UQ/GT pairs found for evaluation.")

    probs_flat = np.concatenate(probs_flat_parts, axis=0)
    gts_flat = np.concatenate(gts_flat_parts, axis=0)
    entropy_flat = np.concatenate(entropy_flat_parts, axis=0)

    nll = binary_nll(probs_flat, gts_flat)
    brier = binary_brier(probs_flat, gts_flat)
    ece = binary_ece(probs_flat, gts_flat, num_bins=args.num_bins)

    pred_bin = (probs_flat >= 0.5).astype(np.float64)
    pixel_err = (pred_bin != gts_flat).astype(np.float64)
    aurc, risk100 = aurc_from_entropy(pixel_err, entropy_flat)

    print("[UQ Metrics]")
    print(f"samples matched: {len(probs_flat_parts)}")
    print(f"mean_prob shape: {mean_prob.shape}")
    print(f"variance mean: {variance.mean():.6f}")
    print(f"entropy mean : {entropy.mean():.6f}")
    print(f"NLL  : {nll:.6f}")
    print(f"Brier: {brier:.6f}")
    print(f"ECE  : {ece:.6f}")
    print(f"AURC : {aurc:.6f}")
    print(f"Risk@100% coverage: {risk100:.6f}")


if __name__ == "__main__":
    main()

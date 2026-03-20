from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.stats import spearmanr


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare THE and entropy against per-image segmentation error.")
    p.add_argument("--the_csv", type=str, required=True)
    p.add_argument("--uq_npz", type=str, required=True)
    p.add_argument("--pred_dir", type=str, required=True)
    p.add_argument("--gt_dir", type=str, required=True)
    return p.parse_args()


def load_mask(path: Path) -> np.ndarray:
    arr = np.array(Image.open(path).convert("L"), dtype=np.uint8)
    return (arr > 127).astype(np.uint8)


def center_crop_to_shape(arr: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = arr.shape
    th, tw = shape
    top = max((h - th) // 2, 0)
    left = max((w - tw) // 2, 0)
    return arr[top : top + th, left : left + tw]


def dice_score(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-8) -> float:
    inter = float(np.logical_and(pred > 0, gt > 0).sum())
    union = float((pred > 0).sum() + (gt > 0).sum())
    return (2.0 * inter + eps) / (union + eps)


def load_the_csv(path: Path) -> dict[str, float]:
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    if len(lines) <= 1:
        return {}
    out: dict[str, float] = {}
    for ln in lines[1:]:
        cols = ln.split(",")
        if len(cols) < 2:
            continue
        out[cols[0]] = float(cols[1])
    return out


def main() -> None:
    args = parse_args()

    the_map = load_the_csv(Path(args.the_csv))
    if not the_map:
        raise RuntimeError("Empty THE csv")

    uq = np.load(args.uq_npz, allow_pickle=True)
    names = [str(x) for x in uq["names"].tolist()] if "names" in uq else []
    entropy = uq["entropy"]  # [N,1,H,W]
    if not names:
        names = [f"sample_{i:04d}.png" for i in range(entropy.shape[0])]

    entropy_mean = {names[i]: float(np.mean(entropy[i, 0])) for i in range(min(len(names), entropy.shape[0]))}

    pred_dir = Path(args.pred_dir)
    gt_dir = Path(args.gt_dir)

    rows = []
    for name, the_val in the_map.items():
        pred_path = pred_dir / name
        gt_path = gt_dir / name
        if not pred_path.exists() or not gt_path.exists() or name not in entropy_mean:
            continue

        pred = load_mask(pred_path)
        gt = load_mask(gt_path)
        if gt.shape != pred.shape:
            gt = center_crop_to_shape(gt, pred.shape)

        d = dice_score(pred, gt)
        err = 1.0 - d
        rows.append((name, the_val, entropy_mean[name], err))

    if len(rows) < 3:
        raise RuntimeError("Not enough matched samples for correlation")

    the_vals = np.array([r[1] for r in rows], dtype=np.float64)
    ent_vals = np.array([r[2] for r in rows], dtype=np.float64)
    err_vals = np.array([r[3] for r in rows], dtype=np.float64)

    rho_the, p_the = spearmanr(the_vals, err_vals)
    rho_ent, p_ent = spearmanr(ent_vals, err_vals)

    print("[THE Correlation]")
    print(f"matched_samples: {len(rows)}")
    print(f"Spearman(THE, error): rho={rho_the:.4f}, p={p_the:.4e}")
    print(f"Spearman(Entropy, error): rho={rho_ent:.4f}, p={p_ent:.4e}")


if __name__ == "__main__":
    main()

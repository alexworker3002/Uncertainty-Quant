from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np
from PIL import Image

from uce.metrics.segmentation import dice_score, iou_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir", type=str, required=True)
    parser.add_argument("--gt_dir", type=str, required=True)
    parser.add_argument("--suffix", type=str, default=".png")
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


def main() -> None:
    args = parse_args()
    pred_dir = Path(args.pred_dir)
    gt_dir = Path(args.gt_dir)

    files = sorted([p for p in pred_dir.glob(f"*{args.suffix}")])
    if not files:
        raise FileNotFoundError(f"No prediction files found in {pred_dir}")

    dices, ious = [], []
    for pred_path in files:
        gt_path = gt_dir / pred_path.name
        if not gt_path.exists():
            continue
        pred = load_mask(pred_path)
        gt = load_mask(gt_path)

        if pred.shape != gt.shape:
            gt = center_crop_to_shape(gt, pred.shape)
            if gt.shape != pred.shape:
                continue

        dices.append(dice_score(pred, gt))
        ious.append(iou_score(pred, gt))

    if not dices:
        raise RuntimeError("No matched prediction/ground-truth pairs found.")

    print(f"Dice: {np.mean(dices):.4f}")
    print(f"IoU : {np.mean(ious):.4f}")


if __name__ == "__main__":
    main()

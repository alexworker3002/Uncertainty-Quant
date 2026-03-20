from __future__ import annotations

import argparse
from pathlib import Path
import random


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create deterministic split files for DRIVE-style processed data.")
    p.add_argument("--root", type=str, default="data/processed/drive")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--train_split", type=str, default="train")
    p.add_argument("--test_split", type=str, default="test")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    train_img_dir = root / args.train_split / "images"
    test_img_dir = root / args.test_split / "images"

    if not train_img_dir.exists():
        raise FileNotFoundError(f"Missing {train_img_dir}")

    train_names = sorted([p.name for p in train_img_dir.glob("*.*") if p.is_file()])
    if not train_names:
        raise RuntimeError(f"No files found in {train_img_dir}")

    rnd = random.Random(args.seed)
    rnd.shuffle(train_names)

    n_val = max(1, int(len(train_names) * args.val_ratio)) if len(train_names) > 1 else 0
    val_names = sorted(train_names[:n_val])
    train_names = sorted(train_names[n_val:])

    test_names = []
    if test_img_dir.exists():
        test_names = sorted([p.name for p in test_img_dir.glob("*.*") if p.is_file()])

    split_dir = Path("data/splits")
    split_dir.mkdir(parents=True, exist_ok=True)

    (split_dir / "drive_train.txt").write_text("\n".join(train_names) + ("\n" if train_names else ""), encoding="utf-8")
    (split_dir / "drive_val.txt").write_text("\n".join(val_names) + ("\n" if val_names else ""), encoding="utf-8")
    (split_dir / "drive_test.txt").write_text("\n".join(test_names) + ("\n" if test_names else ""), encoding="utf-8")

    print(f"Wrote splits to {split_dir}")
    print(f"train={len(train_names)}, val={len(val_names)}, test={len(test_names)}")


if __name__ == "__main__":
    main()

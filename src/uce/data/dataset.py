from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

from .transforms import center_crop_pair, default_image_transform, default_mask_transform


class DriveDataset(Dataset):
    """DRIVE-style dataset loader.

    Expected structure:
      root/<split>/images/*
      root/<split>/mask/*

    Optional deterministic split file (one filename per line) can be provided.
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        image_subdir: str = "images",
        mask_subdir: str = "mask",
        input_size: tuple[int, int] = (512, 512),
        split_file: str | Path | None = None,
        image_transform: Callable[[np.ndarray], torch.Tensor] | None = None,
        mask_transform: Callable[[np.ndarray], torch.Tensor] | None = None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.image_dir = self.root / split / image_subdir
        self.mask_dir = self.root / split / mask_subdir
        self.input_size = input_size
        self.split_file = Path(split_file) if split_file else None
        self.image_transform = image_transform or default_image_transform()
        self.mask_transform = mask_transform or default_mask_transform()

        self.samples = self._collect_samples()
        if not self.samples:
            raise FileNotFoundError(f"No image/mask pairs found in {self.image_dir} and {self.mask_dir}")

    def _collect_samples(self) -> list[tuple[Path, Path]]:
        if not self.image_dir.exists():
            return []

        if self.split_file is not None:
            if not self.split_file.exists():
                raise FileNotFoundError(f"split_file not found: {self.split_file}")
            selected = [ln.strip() for ln in self.split_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
            img_paths = [self.image_dir / name for name in selected if (self.image_dir / name).exists()]
        else:
            img_paths = sorted(
                [p for ext in ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp") for p in self.image_dir.glob(ext)]
            )

        pairs: list[tuple[Path, Path]] = []
        for img_path in img_paths:
            mask_path = self.mask_dir / img_path.name
            if mask_path.exists():
                pairs.append((img_path, mask_path))
        return pairs

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        img_path, mask_path = self.samples[idx]
        image = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)

        image, mask = center_crop_pair(image, mask, self.input_size)
        x = self.image_transform(image)
        y = self.mask_transform(mask)

        return {"image": x, "mask": y, "name": img_path.name}


def build_dataloader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    drop_last: bool = False,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=drop_last,
    )


def sample_names(ds: Iterable[dict[str, torch.Tensor | str]], n: int = 3) -> list[str]:
    names: list[str] = []
    for i, item in enumerate(ds):
        if i >= n:
            break
        names.append(str(item["name"]))
    return names

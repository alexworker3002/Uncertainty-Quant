from __future__ import annotations

from typing import Callable, Tuple

import numpy as np
import torch


def _to_chw_tensor(image: np.ndarray) -> torch.Tensor:
    if image.ndim == 2:
        image = image[..., None]
    image = image.astype(np.float32) / 255.0
    chw = np.transpose(image, (2, 0, 1))
    return torch.from_numpy(chw)


def _to_mask_tensor(mask: np.ndarray) -> torch.Tensor:
    if mask.ndim == 3:
        mask = mask[..., 0]
    mask = (mask > 127).astype(np.float32)
    return torch.from_numpy(mask[None, ...])


def default_image_transform() -> Callable[[np.ndarray], torch.Tensor]:
    return _to_chw_tensor


def default_mask_transform() -> Callable[[np.ndarray], torch.Tensor]:
    return _to_mask_tensor


def center_crop_pair(image: np.ndarray, mask: np.ndarray, size: Tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    h, w = image.shape[:2]
    th, tw = size
    top = max((h - th) // 2, 0)
    left = max((w - tw) // 2, 0)
    image_c = image[top : top + th, left : left + tw]
    mask_c = mask[top : top + th, left : left + tw]
    return image_c, mask_c

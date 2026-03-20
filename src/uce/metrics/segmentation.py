from __future__ import annotations

import numpy as np


def dice_score(pred: np.ndarray, target: np.ndarray, eps: float = 1e-8) -> float:
    pred = pred.astype(bool)
    target = target.astype(bool)
    inter = np.logical_and(pred, target).sum()
    return float((2.0 * inter + eps) / (pred.sum() + target.sum() + eps))


def iou_score(pred: np.ndarray, target: np.ndarray, eps: float = 1e-8) -> float:
    pred = pred.astype(bool)
    target = target.astype(bool)
    inter = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    return float((inter + eps) / (union + eps))

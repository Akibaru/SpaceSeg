from __future__ import annotations

import numpy as np


def binary_iou_and_accuracy(
    gt_masks: np.ndarray,
    pred_masks: np.ndarray,
    eps: float = 1e-5,
) -> tuple[float, float]:
    gt = gt_masks.astype(bool)
    pred = pred_masks.astype(bool)
    if gt.shape != pred.shape:
        raise ValueError(f"Metric shape mismatch: gt={gt.shape}, pred={pred.shape}")

    intersection = np.logical_and(gt, pred).sum(axis=(1, 2))
    union = np.logical_or(gt, pred).sum(axis=(1, 2))
    iou = intersection / (union + eps)
    acc = (gt == pred).sum(axis=(1, 2)) / (gt.shape[1] * gt.shape[2] + eps)
    return float(iou.mean()), float(acc.mean())


def count_parameters(model) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


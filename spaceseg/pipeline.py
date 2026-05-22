from __future__ import annotations

from pathlib import Path

import numpy as np

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


def build_spaceseg_predictor(
    model_cfg: str,
    pretrained_checkpoint: str | Path,
    device: str = "cuda",
    finetuned_checkpoint: str | Path | None = None,
    strict_finetuned: bool = True,
) -> SAM2ImagePredictor:
    model = build_sam2(model_cfg, str(pretrained_checkpoint), device=device)
    if finetuned_checkpoint:
        load_finetuned_state(model, finetuned_checkpoint, strict=strict_finetuned)
    return SAM2ImagePredictor(model)


def load_finetuned_state(model, checkpoint_path: str | Path, strict: bool = True):
    import torch

    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu")
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break
    missing, unexpected = model.load_state_dict(checkpoint, strict=strict)
    if missing or unexpected:
        message = f"missing={missing}, unexpected={unexpected}"
        if strict:
            raise RuntimeError(f"Fine-tuned checkpoint load failed: {message}")
        print(f"Loaded fine-tuned checkpoint with non-strict keys: {message}")


def predict_prompt_masks(predictor, image, point_coords, point_labels):
    predictor.set_image(image)
    masks, scores, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=True,
    )
    order = np.argsort(scores[:, 0])[::-1]
    return masks[:, 0].astype(bool), scores[:, 0], order


def stitch_instance_masks(
    masks: np.ndarray,
    scores: np.ndarray | None = None,
    overlap_threshold: float = 0.15,
) -> np.ndarray:
    if masks.size == 0:
        return np.zeros((0, 0), dtype=np.uint8)
    if scores is None:
        order = np.arange(masks.shape[0])
    else:
        order = np.argsort(scores)[::-1]

    seg_map = np.zeros(masks.shape[1:], dtype=np.uint8)
    occupied = np.zeros(masks.shape[1:], dtype=bool)
    instance_id = 1
    for idx in order:
        mask = masks[idx].astype(bool).copy()
        area = mask.sum()
        if area == 0:
            continue
        if np.logical_and(mask, occupied).sum() / area > overlap_threshold:
            continue
        mask[occupied] = False
        if mask.sum() == 0:
            continue
        seg_map[mask] = instance_id
        occupied[mask] = True
        instance_id += 1
    return seg_map


def colorize_segmentation(seg_map: np.ndarray, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rgb = np.zeros((*seg_map.shape, 3), dtype=np.uint8)
    for instance_id in range(1, int(seg_map.max()) + 1):
        rgb[seg_map == instance_id] = rng.integers(32, 256, size=3, dtype=np.uint8)
    return rgb


def overlay_segmentation(image: np.ndarray, color_mask: np.ndarray, alpha: float = 0.5):
    return (image.astype(np.float32) * (1 - alpha) + color_mask.astype(np.float32) * alpha).astype(
        np.uint8
    )


from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class SpaceESPair:
    image_path: Path
    mask_path: Path

    @property
    def name(self) -> str:
        return self.image_path.name

    @property
    def background(self) -> str:
        return self.image_path.name.split("_", 1)[0]


def list_spacees_pairs(root: str | Path) -> list[SpaceESPair]:
    root = Path(root)
    image_dir = root / "image"
    mask_dir = root / "mask"
    if not image_dir.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError(
            f"Expected SpaceES split with image/ and mask/ under {root}"
        )

    pairs: list[SpaceESPair] = []
    for image_path in sorted(image_dir.iterdir()):
        if not image_path.is_file():
            continue
        mask_path = mask_dir / image_path.name
        if mask_path.is_file():
            pairs.append(SpaceESPair(image_path=image_path, mask_path=mask_path))
    if not pairs:
        raise FileNotFoundError(f"No image/mask pairs found under {root}")
    return pairs


def read_image_mask(pair: SpaceESPair, image_size: int = 1024):
    cv2 = _require_cv2()
    image = cv2.imread(str(pair.image_path), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(pair.mask_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image: {pair.image_path}")
    if mask is None:
        raise ValueError(f"Could not read mask: {pair.mask_path}")

    image = image[..., ::-1]
    if image.shape[:2] != (image_size, image_size):
        image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    if mask.shape[:2] != (image_size, image_size):
        mask = cv2.resize(mask, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
    return image, mask


def mask_to_binary(mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 3:
        mask = mask.max(axis=2)
    return (mask > 0).astype(np.uint8)


def connected_component_masks(mask: np.ndarray, min_area: int = 8) -> list[np.ndarray]:
    cv2 = _require_cv2()
    binary = mask_to_binary(mask)
    num_labels, labels = cv2.connectedComponents(binary)
    masks: list[np.ndarray] = []
    for label in range(1, num_labels):
        component = (labels == label).astype(np.uint8)
        if int(component.sum()) >= min_area:
            masks.append(component)
    return masks


def sample_points_from_masks(
    masks: Iterable[np.ndarray],
    rng: np.random.Generator,
    points_per_mask: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    points = []
    for mask in masks:
        coords = np.argwhere(mask > 0)
        if coords.size == 0:
            continue
        indices = rng.choice(len(coords), size=points_per_mask, replace=True)
        for idx in np.atleast_1d(indices):
            y, x = coords[int(idx)]
            points.append([[int(x), int(y)]])
    if not points:
        return np.empty((0, 1, 2), dtype=np.float32), np.empty((0, 1), dtype=np.int64)
    point_coords = np.asarray(points, dtype=np.float32)
    point_labels = np.ones((len(points), 1), dtype=np.int64)
    return point_coords, point_labels


def prepare_prompt_batch(
    mask: np.ndarray,
    rng: np.random.Generator,
    min_area: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    masks = connected_component_masks(mask, min_area=min_area)
    if not masks:
        return (
            np.empty((0, *mask.shape[:2]), dtype=np.uint8),
            np.empty((0, 1, 2), dtype=np.float32),
            np.empty((0, 1), dtype=np.int64),
        )
    point_coords, point_labels = sample_points_from_masks(masks, rng=rng)
    return np.asarray(masks, dtype=np.uint8), point_coords, point_labels


def _require_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "OpenCV is required for SpaceES data loading. Install with "
            "`pip install -e .[spaceseg]` or `pip install opencv-python`."
        ) from exc
    return cv2


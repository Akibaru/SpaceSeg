from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from spaceseg.data import (
    SpaceESPair,
    connected_component_masks,
    read_image_mask,
    sample_points_from_masks,
)
from spaceseg.pipeline import (
    build_spaceseg_predictor,
    colorize_segmentation,
    overlay_segmentation,
    predict_prompt_masks,
    stitch_instance_masks,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run SpaceSeg inference.")
    parser.add_argument("--source", default="examples/spacees_sample/test/image")
    parser.add_argument("--mask-dir", default="examples/spacees_sample/test/mask")
    parser.add_argument("--output-dir", default="outputs/infer")
    parser.add_argument("--pretrained-checkpoint", default="checkpoints/sam2_hiera_small.pt")
    parser.add_argument(
        "--finetuned-checkpoint",
        default="Out/final_best_model_mIoU_8987_mAcc_9998.pth",
    )
    parser.add_argument("--model-cfg", default="sam2_hiera_s.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--min-area", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--point",
        action="append",
        default=[],
        help="Manual prompt point as x,y. Can be repeated when no mask-dir is available.",
    )
    return parser.parse_args()


def iter_sources(source: Path):
    if source.is_file():
        yield source
        return
    for path in sorted(source.iterdir()):
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            yield path


def manual_points(points: list[str]):
    coords = []
    for item in points:
        x_text, y_text = item.split(",", 1)
        coords.append([[float(x_text), float(y_text)]])
    if not coords:
        return np.empty((0, 1, 2), dtype=np.float32), np.empty((0, 1), dtype=np.int64)
    return np.asarray(coords, dtype=np.float32), np.ones((len(coords), 1), dtype=np.int64)


def save_png(path: Path, image):
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    if image.ndim == 3:
        image = image[..., ::-1]
    cv2.imwrite(str(path), image)


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    source = Path(args.source)
    mask_dir = Path(args.mask_dir) if args.mask_dir else None
    output_dir = Path(args.output_dir)

    predictor = build_spaceseg_predictor(
        args.model_cfg,
        args.pretrained_checkpoint,
        device=args.device,
        finetuned_checkpoint=args.finetuned_checkpoint,
        strict_finetuned=True,
    )

    device_type = "cuda" if args.device.startswith("cuda") else "cpu"
    with torch.no_grad(), torch.autocast(device_type=device_type, enabled=args.device.startswith("cuda")):
        for image_path in iter_sources(source):
            if mask_dir:
                mask_path = mask_dir / image_path.name
                pair = SpaceESPair(image_path=image_path, mask_path=mask_path)
                image, mask = read_image_mask(pair, image_size=args.image_size)
                component_masks = connected_component_masks(mask, min_area=args.min_area)
                point_coords, point_labels = sample_points_from_masks(component_masks, rng=rng)
            else:
                import cv2

                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError(f"Could not read image: {image_path}")
                image = cv2.resize(
                    image[..., ::-1],
                    (args.image_size, args.image_size),
                    interpolation=cv2.INTER_LINEAR,
                )
                point_coords, point_labels = manual_points(args.point)

            if len(point_coords) == 0:
                raise ValueError(
                    f"No prompt points for {image_path}. Provide --mask-dir or --point x,y."
                )
            masks, scores, _ = predict_prompt_masks(
                predictor,
                image,
                point_coords,
                point_labels,
            )
            seg_map = stitch_instance_masks(masks, scores=scores)
            color = colorize_segmentation(seg_map, seed=args.seed)
            overlay = overlay_segmentation(image, color)

            stem = image_path.stem
            save_png(output_dir / f"{stem}_instances.png", color)
            save_png(output_dir / f"{stem}_overlay.png", overlay)
            save_png(output_dir / f"{stem}_mask.png", (seg_map > 0).astype(np.uint8) * 255)
            print(f"wrote {stem} to {output_dir}")


if __name__ == "__main__":
    main()


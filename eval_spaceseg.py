from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from spaceseg.data import list_spacees_pairs, prepare_prompt_batch, read_image_mask
from spaceseg.metrics import binary_iou_and_accuracy, count_parameters
from spaceseg.pipeline import build_spaceseg_predictor


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SpaceSeg on SpaceES.")
    parser.add_argument("--data-root", default="data/test")
    parser.add_argument("--pretrained-checkpoint", default="checkpoints/sam2_hiera_small.pt")
    parser.add_argument(
        "--finetuned-checkpoint",
        default="Out/final_best_model_mIoU_8987_mAcc_9998.pth",
    )
    parser.add_argument("--model-cfg", default="sam2_hiera_s.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--min-area", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def predict_masks_for_components(predictor, image, point_coords, point_labels):
    predictor.set_image(image)
    masks, _, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=True,
    )
    return masks[:, 0].astype(bool)


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    pairs = list_spacees_pairs(args.data_root)
    if args.limit > 0:
        pairs = pairs[: args.limit]

    predictor = build_spaceseg_predictor(
        args.model_cfg,
        args.pretrained_checkpoint,
        device=args.device,
        finetuned_checkpoint=args.finetuned_checkpoint,
        strict_finetuned=True,
    )
    total_params, trainable_params = count_parameters(predictor.model)

    total_iou = 0.0
    total_acc = 0.0
    count = 0
    start = time.perf_counter()
    device_type = "cuda" if args.device.startswith("cuda") else "cpu"
    with torch.no_grad(), torch.autocast(device_type=device_type, enabled=args.device.startswith("cuda")):
        for pair in pairs:
            image, mask = read_image_mask(pair, image_size=args.image_size)
            gt_masks, point_coords, point_labels = prepare_prompt_batch(
                mask,
                rng=rng,
                min_area=args.min_area,
            )
            if len(gt_masks) == 0:
                continue
            pred_masks = predict_masks_for_components(
                predictor,
                image,
                point_coords,
                point_labels,
            )
            miou, macc = binary_iou_and_accuracy(gt_masks, pred_masks)
            total_iou += miou
            total_acc += macc
            count += 1
            if count % 100 == 0:
                print(f"processed={count} running_mIoU={total_iou / count:.4f}")

    elapsed = time.perf_counter() - start
    print(f"images={count}")
    print(f"mIoU={total_iou / count if count else 0.0:.4f}")
    print(f"mAcc={total_acc / count if count else 0.0:.4f}")
    print(f"FPS={count / elapsed if elapsed else 0.0:.2f}")
    print(f"params_total={total_params:,}")
    print(f"params_trainable={trainable_params:,}")


if __name__ == "__main__":
    main()


from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from spaceseg.data import list_spacees_pairs, prepare_prompt_batch, read_image_mask
from spaceseg.metrics import binary_iou_and_accuracy
from spaceseg.sdat import build_sdat_transform


def parse_args():
    parser = argparse.ArgumentParser(description="Train SpaceSeg on SpaceES.")
    parser.add_argument("--train-root", default="data/training")
    parser.add_argument("--val-root", default="data/test")
    parser.add_argument("--pretrained-checkpoint", default="checkpoints/sam2_hiera_small.pt")
    parser.add_argument("--model-cfg", default="sam2_hiera_s.yaml")
    parser.add_argument("--output-dir", default="outputs/spaceseg")
    parser.add_argument("--iterations", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=4e-5)
    parser.add_argument("--lambda-iou", type=float, default=0.05)
    parser.add_argument("--validate-every", type=int, default=500)
    parser.add_argument("--val-samples", type=int, default=50)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--min-area", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-sdat", action="store_true")
    parser.add_argument("--train-image-encoder", action="store_true")
    return parser.parse_args()


def set_trainable_modules(model, train_image_encoder: bool):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.sam_prompt_encoder.train(True)
    model.sam_mask_decoder.train(True)
    for parameter in model.sam_prompt_encoder.parameters():
        parameter.requires_grad_(True)
    for parameter in model.sam_mask_decoder.parameters():
        parameter.requires_grad_(True)
    if train_image_encoder:
        model.image_encoder.train(True)
        for parameter in model.image_encoder.parameters():
            parameter.requires_grad_(True)
    else:
        model.image_encoder.eval()


def forward_space_batch(predictor, image, gt_masks, point_coords, point_labels):
    predictor.set_image(image)
    mask_input, unnorm_coords, labels, _ = predictor._prep_prompts(
        point_coords,
        point_labels,
        box=None,
        mask_logits=None,
        normalize_coords=True,
    )
    del mask_input
    sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
        points=(unnorm_coords, labels),
        boxes=None,
        masks=None,
    )

    batched_mode = unnorm_coords.shape[0] > 1
    high_res_features = [
        feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]
    ]
    low_res_masks, pred_scores, _, _ = predictor.model.sam_mask_decoder(
        image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),
        image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=True,
        repeat_image=batched_mode,
        high_res_features=high_res_features,
    )
    pred_masks = predictor._transforms.postprocess_masks(
        low_res_masks,
        predictor._orig_hw[-1],
    )

    gt = torch.tensor(gt_masks.astype(np.float32), device=predictor.model.device)
    pred_prob = torch.sigmoid(pred_masks[:, 0])
    seg_loss = F.binary_cross_entropy(pred_prob, gt)

    pred_binary = pred_prob > 0.5
    intersection = (gt * pred_binary).sum(dim=(1, 2))
    union = gt.sum(dim=(1, 2)) + pred_binary.sum(dim=(1, 2)) - intersection
    iou = intersection / (union + 1e-5)
    score_loss = torch.abs(pred_scores[:, 0] - iou).mean()
    return seg_loss, score_loss, iou, pred_prob.detach()


def read_training_sample(pairs, rng, image_size, min_area, transform):
    pair = pairs[int(rng.integers(len(pairs)))]
    image, mask = read_image_mask(pair, image_size=image_size)
    if transform is not None:
        augmented = transform(image=image, mask=mask)
        image, mask = augmented["image"], augmented["mask"]
    gt_masks, point_coords, point_labels = prepare_prompt_batch(
        mask,
        rng=rng,
        min_area=min_area,
    )
    return pair, image, mask, gt_masks, point_coords, point_labels


def evaluate(predictor, pairs, args, rng):
    predictor.model.eval()
    total_iou = 0.0
    total_acc = 0.0
    count = 0
    device_type = "cuda" if args.device.startswith("cuda") else "cpu"
    with torch.no_grad(), torch.autocast(device_type=device_type, enabled=args.device.startswith("cuda")):
        for _ in range(args.val_samples):
            _, image, _, gt_masks, point_coords, point_labels = read_training_sample(
                pairs,
                rng,
                args.image_size,
                args.min_area,
                transform=None,
            )
            if len(gt_masks) == 0:
                continue
            _, _, _, pred_prob = forward_space_batch(
                predictor,
                image,
                gt_masks,
                point_coords,
                point_labels,
            )
            iou, acc = binary_iou_and_accuracy(
                gt_masks,
                (pred_prob.cpu().numpy() > 0.5),
            )
            total_iou += iou
            total_acc += acc
            count += 1
    predictor.model.sam_prompt_encoder.train(True)
    predictor.model.sam_mask_decoder.train(True)
    if args.train_image_encoder:
        predictor.model.image_encoder.train(True)
    else:
        predictor.model.image_encoder.eval()
    return (
        total_iou / count if count else 0.0,
        total_acc / count if count else 0.0,
    )


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    train_pairs = list_spacees_pairs(args.train_root)
    val_pairs = list_spacees_pairs(args.val_root)
    transform = None if args.no_sdat else build_sdat_transform(args.image_size)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = build_sam2(args.model_cfg, args.pretrained_checkpoint, device=args.device, mode="train")
    set_trainable_modules(model, train_image_encoder=args.train_image_encoder)
    predictor = SAM2ImagePredictor(model)

    optimizer = torch.optim.AdamW(
        [p for p in predictor.model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.device.startswith("cuda"))
    device_type = "cuda" if args.device.startswith("cuda") else "cpu"

    best_miou = 0.0
    best_macc = 0.0
    mean_iou = 0.0
    mean_acc = 0.0

    for itr in range(args.iterations):
        with torch.autocast(device_type=device_type, enabled=args.device.startswith("cuda")):
            _, image, _, gt_masks, point_coords, point_labels = read_training_sample(
                train_pairs,
                rng,
                args.image_size,
                args.min_area,
                transform=transform,
            )
            if len(gt_masks) == 0:
                continue
            seg_loss, score_loss, iou, pred_prob = forward_space_batch(
                predictor,
                image,
                gt_masks,
                point_coords,
                point_labels,
            )
            loss = seg_loss + args.lambda_iou * score_loss

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_iou = float(iou.mean().detach().cpu())
        train_acc = binary_iou_and_accuracy(
            gt_masks,
            pred_prob.cpu().numpy() > 0.5,
        )[1]
        mean_iou = mean_iou * 0.99 + train_iou * 0.01
        mean_acc = mean_acc * 0.99 + train_acc * 0.01

        if itr and itr % args.validate_every == 0:
            val_miou, val_macc = evaluate(predictor, val_pairs, args, rng)
            print(f"val itr={itr} mIoU={val_miou:.4f} mAcc={val_macc:.4f}")
            if val_miou > best_miou:
                best_miou = val_miou
                best_macc = val_macc
                torch.save(predictor.model.state_dict(), output_dir / "best_model.pth")
                print(f"saved best model mIoU={best_miou:.4f} mAcc={best_macc:.4f}")

        if itr % args.log_every == 0:
            print(
                f"itr={itr} train_mIoU={mean_iou:.4f} train_mAcc={mean_acc:.4f} "
                f"loss={float(loss.detach().cpu()):.4f} best_mIoU={best_miou:.4f}"
            )

    torch.save(predictor.model.state_dict(), output_dir / "last_model.pth")
    print(f"training complete best_mIoU={best_miou:.4f} best_mAcc={best_macc:.4f}")


if __name__ == "__main__":
    main()

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from pathlib import Path

import torch
from hydra import compose
from hydra.utils import instantiate
from omegaconf import OmegaConf


def build_sam2(
    config_file,
    ckpt_path=None,
    device="cuda",
    mode="eval",
    hydra_overrides_extra=None,
    apply_postprocessing=True,
    strict_checkpoint=False,
):
    hydra_overrides_extra = list(hydra_overrides_extra or [])

    if apply_postprocessing:
        hydra_overrides_extra += [
            # dynamically fall back to multi-mask if the single mask is not stable
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
        ]
    # Read config and init model
    cfg = compose(config_name=config_file, overrides=hydra_overrides_extra)
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(model, ckpt_path, strict=strict_checkpoint)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model


def build_sam2_video_predictor(
    config_file,
    ckpt_path=None,
    device="cuda",
    mode="eval",
    hydra_overrides_extra=None,
    apply_postprocessing=True,
    strict_checkpoint=False,
):
    hydra_overrides_extra = list(hydra_overrides_extra or [])
    hydra_overrides = [
        "++model._target_=sam2.sam2_video_predictor.SAM2VideoPredictor",
    ]
    if apply_postprocessing:
        hydra_overrides_extra += [
            # dynamically fall back to multi-mask if the single mask is not stable
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
            # the sigmoid mask logits on interacted frames with clicks in the memory encoder so that the encoded masks are exactly as what users see from clicking
            "++model.binarize_mask_from_pts_for_mem_enc=true",
            # fill small holes in the low-res masks up to `fill_hole_area` (before resizing them to the original video resolution)
            "++model.fill_hole_area=8",
        ]
    hydra_overrides.extend(hydra_overrides_extra)

    # Read config and init model
    cfg = compose(config_name=config_file, overrides=hydra_overrides)
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(model, ckpt_path, strict=strict_checkpoint)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model


def _unwrap_checkpoint_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    return checkpoint


def _load_checkpoint(model, ckpt_path, strict=False):
    if ckpt_path is not None:
        checkpoint_path = Path(ckpt_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        sd = _unwrap_checkpoint_state_dict(checkpoint)
        if not isinstance(sd, dict):
            raise RuntimeError(f"Checkpoint {checkpoint_path} does not contain a state dict")

        if strict:
            missing_keys, unexpected_keys = model.load_state_dict(sd, strict=True)
            if missing_keys or unexpected_keys:
                logging.error("Missing keys: %s", missing_keys)
                logging.error("Unexpected keys: %s", unexpected_keys)
                raise RuntimeError(f"Strict checkpoint loading failed for {checkpoint_path}")
            logging.info("Loaded checkpoint strictly from %s", checkpoint_path)
            return

        current_state_dict = model.state_dict()
        filtered_sd = {
            k: v
            for k, v in sd.items()
            if k in current_state_dict and current_state_dict[k].shape == v.shape
        }
        skipped_shape = {
            k: (tuple(v.shape), tuple(current_state_dict[k].shape))
            for k, v in sd.items()
            if k in current_state_dict and current_state_dict[k].shape != v.shape
        }
        current_state_dict.update(filtered_sd)
        model.load_state_dict(current_state_dict, strict=False)

        missing_keys = sorted(set(current_state_dict.keys()) - set(filtered_sd.keys()))
        if missing_keys:
            logging.info("Randomly initialized keys: %s", missing_keys)

        unexpected_keys = sorted(set(sd.keys()) - set(current_state_dict.keys()))
        if unexpected_keys:
            logging.info("Unused checkpoint keys: %s", unexpected_keys)
        if skipped_shape:
            logging.info("Skipped keys with shape mismatch: %s", skipped_shape)

        logging.info("Loaded %d/%d checkpoint tensors from %s", len(filtered_sd), len(sd), checkpoint_path)

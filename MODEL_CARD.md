# SpaceSeg 8987 Checkpoint Model Card

## Model

The release checkpoint corresponds to the SpaceSeg model reported in the paper
**"SpaceSeg: A High-Precision Intelligent Perception Segmentation Method for
Multi-Spacecraft On-Orbit Targets"**:

```text
Out/final_best_model_mIoU_8987_mAcc_9998.pth
```

Paper: https://arxiv.org/abs/2503.11133

It uses SAM2 Hiera-S as the base model and the SpaceSeg MSHARD decoder stored
under the checkpoint key prefix `sam_mask_decoder.unet.*`.

## Intended Use

The checkpoint is intended for research and non-commercial evaluation of
multi-spacecraft segmentation in SpaceES-style imagery.

## Expected Inputs

- RGB image resized to 1024 x 1024.
- Prompt points sampled from coarse foreground masks or provided manually.
- Base checkpoint: `checkpoints/sam2_hiera_small.pt`.

## Reported Metrics

On the full SpaceES test set, the paper reports:

- mIoU: 89.87
- mAcc: 99.98
- FPS: 3.31 under the paper's RTX 4090, 1024 x 1024, 30-prompt protocol.

## Limitations

The model is specialized for SpaceES-like on-orbit spacecraft imagery. It is not
validated for safety-critical autonomous spacecraft operations without additional
testing, calibration, and mission-specific validation.

## Example

```bash
python eval_spaceseg.py \
  --data-root data/test \
  --pretrained-checkpoint checkpoints/sam2_hiera_small.pt \
  --finetuned-checkpoint Out/final_best_model_mIoU_8987_mAcc_9998.pth
```

# SpaceSeg

SpaceSeg is a reproduction repository for **"SpaceSeg: A High-Precision
Intelligent Perception Segmentation Method for Multi-Spacecraft On-Orbit
Targets"**. It adapts SAM 2 to multi-spacecraft segmentation on the SpaceES
dataset.

The original repository started from SAM 2 and a small fine-tuning tutorial.
This version keeps the SAM 2 base code, adds the SpaceSeg reproduction entry
points, and preserves the old experimental scripts as legacy references.

## Method Mapping

- **MSHARD**: implemented in `sam2/modeling/sam/mask_decoder.py` as the
  `sam_mask_decoder.unet` module for compatibility with the published 8987
  checkpoint. It is wired into both the standard and high-resolution SAM2 mask
  decoder paths.
- **SDAT**: implemented in `spaceseg/sdat.py` and enabled by default in
  `train_spaceseg.py`.
- **CCA target organization**: implemented in `spaceseg/data.py` through binary
  mask connected components and point sampling.
- **Task-oriented loss**: implemented in `train_spaceseg.py` as segmentation
  BCE plus IoU-prediction loss with `--lambda-iou 0.05`.

## Installation

Install on a CUDA machine with Python 3.10+:

```bash
pip install -e ".[spaceseg]"
```

Download the SAM2 Hiera-S checkpoint into `checkpoints/`:

```text
checkpoints/sam2_hiera_small.pt
```

The final SpaceSeg checkpoint used in the paper should be placed at:

```text
Out/final_best_model_mIoU_8987_mAcc_9998.pth
```

See `MODEL_CARD.md` and `weights/README.md` for release notes.

## Data Layout

The full SpaceES dataset is intentionally not tracked by git. Put it under:

```text
data/
  training/
    image/
    mask/
  test/
    image/
    mask/
```

Image and mask files must have identical names. A small balanced sample is
included under `examples/spacees_sample/` for smoke tests and demos.

## Train

```bash
python train_spaceseg.py \
  --train-root data/training \
  --val-root data/test \
  --pretrained-checkpoint checkpoints/sam2_hiera_small.pt \
  --output-dir outputs/spaceseg
```

Defaults match the paper protocol where practical: Hiera-S, 1024 x 1024 input,
100000 iterations, AdamW with learning rate `1e-5`, SDAT enabled, and
`lambda_iou=0.05`.

For a quick smoke test:

```bash
python train_spaceseg.py \
  --train-root examples/spacees_sample/train \
  --val-root examples/spacees_sample/test \
  --iterations 1 \
  --validate-every 1
```

## Evaluate

```bash
python eval_spaceseg.py \
  --data-root data/test \
  --pretrained-checkpoint checkpoints/sam2_hiera_small.pt \
  --finetuned-checkpoint Out/final_best_model_mIoU_8987_mAcc_9998.pth
```

With the full SpaceES test set and the 8987 checkpoint, results should be close
to the paper values: **89.87 mIoU** and **99.98 mAcc**.

## Inference

Run on the included sample split:

```bash
python infer_spaceseg.py \
  --source examples/spacees_sample/test/image \
  --mask-dir examples/spacees_sample/test/mask \
  --output-dir outputs/infer
```

When no mask directory is available, provide one or more prompt points:

```bash
python infer_spaceseg.py --source path/to/image.png --mask-dir "" --point 512,512
```

## Legacy Scripts

Files such as `trainnew*.py`, `testn*.py`, `TRAIN.py`, and `TEST_Net.py` are kept
as historical experiment notes. New reproduction work should use
`train_spaceseg.py`, `eval_spaceseg.py`, and `infer_spaceseg.py`.

## License

Code follows the Apache 2.0 license inherited from SAM 2. The SpaceES sample
data and SpaceSeg released weights are provided for research and non-commercial
use only; see `DATASET.md` and `MODEL_CARD.md`.


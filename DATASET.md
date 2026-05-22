# SpaceES Dataset Notes

SpaceES is the multi-scale on-orbit multi-spacecraft segmentation dataset used
by SpaceSeg. The full dataset is not committed to this repository.

## Expected Layout

```text
data/
  training/
    image/
    mask/
  test/
    image/
    mask/
```

Each image must have a mask with the same file name. Masks are interpreted as
foreground when pixel value is greater than zero. Instance prompts are generated
by connected component analysis during training and evaluation.

## Included Sample

`examples/spacees_sample/` contains a small balanced subset for smoke tests:

- 32 training pairs: 8 each from earth, mars, moon, and stars backgrounds.
- 8 test pairs: 2 each from earth, mars, moon, and stars backgrounds.

This sample is useful for verifying installation and I/O only. Metrics on this
subset are not comparable to the paper results.

## Use Terms

The included SpaceES sample data is provided for research and non-commercial use
only. Do not redistribute it as a standalone dataset without permission from the
dataset owner.


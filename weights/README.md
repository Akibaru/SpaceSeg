# Weights

Large model files are not tracked directly by git.

The final paper checkpoint is intended to be distributed separately through a
GitHub Release asset or Git LFS. It is intentionally not committed to the git
repository.

Place the released checkpoint here or in `Out/`:

```text
Out/final_best_model_mIoU_8987_mAcc_9998.pth
```

Recommended public release options:

- GitHub Release asset named `final_best_model_mIoU_8987_mAcc_9998.pth`.
- Git LFS, if the hosting repository has LFS quota available.

Use terms: SpaceSeg weights are provided for research and non-commercial use.

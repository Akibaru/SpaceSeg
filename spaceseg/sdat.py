from __future__ import annotations


def build_sdat_transform(image_size: int = 1024):
    """Build the Spatial Domain Adaptation Transform used in SpaceSeg training."""
    try:
        import albumentations as A
        import cv2
    except ImportError as exc:
        raise ImportError(
            "SDAT requires albumentations and opencv-python. Install with "
            "`pip install -e .[spaceseg]`."
        ) from exc

    return A.Compose(
        [
            A.RandomRotate90(p=0.5),
            A.PadIfNeeded(
                min_height=image_size,
                min_width=image_size,
                border_mode=cv2.BORDER_CONSTANT,
                p=1.0,
            ),
            A.RandomCrop(height=image_size, width=image_size, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ElasticTransform(alpha=1, sigma=50, p=0.3),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=30,
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.5,
            ),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
                p=0.5,
            ),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
            A.RandomGamma(gamma_limit=(80, 120), p=0.5),
        ]
    )


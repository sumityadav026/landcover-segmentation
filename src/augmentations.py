"""
Augmentation pipelines for Urban Land Cover Mapping semantic segmentation.
Implements Albumentations training and evaluation transforms ensuring identical spatial
transformations on images and masks, photometric augmentations, and ImageNet normalization.
"""

from typing import Optional, Callable, Dict, Any, Tuple
import numpy as np

# Optional Albumentations import with graceful fallback
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False

# ImageNet normalization statistics
IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)


def get_training_augmentation(
    p_hflip: float = 0.5,
    p_vflip: float = 0.5,
    p_rot90: float = 0.5,
    brightness_limit: float = 0.2,
    contrast_limit: float = 0.2,
    p_brightness_contrast: float = 0.5,
) -> Any:
    """
    Construct Albumentations augmentation pipeline for training.

    Spatial transformations (applied identically to image and mask):
      - HorizontalFlip
      - VerticalFlip
      - RandomRotate90

    Photometric transformations (applied strictly to aerial image only):
      - RandomBrightnessContrast

    Normalization:
      - ImageNet mean and standard deviation
      - Conversion to PyTorch tensors (Image: [3, H, W] Float, Mask: [H, W] Long)

    Returns:
        Albumentations Compose object.
    """
    if not ALBUMENTATIONS_AVAILABLE:
        raise ImportError(
            "albumentations is required for get_training_augmentation(). "
            "Please run: pip install albumentations"
        )

    return A.Compose([
        # Spatial transforms (identical on image and mask)
        A.HorizontalFlip(p=p_hflip),
        A.VerticalFlip(p=p_vflip),
        A.RandomRotate90(p=p_rot90),
        
        # Photometric transforms (applied to image only)
        A.RandomBrightnessContrast(
            brightness_limit=brightness_limit,
            contrast_limit=contrast_limit,
            p=p_brightness_contrast,
        ),
        
        # ImageNet normalization & tensor conversion
        A.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
            max_pixel_value=255.0,
        ),
        ToTensorV2(transpose_mask=False),
    ])


def get_validation_augmentation() -> Any:
    """
    Construct Albumentations pipeline for validation and testing.
    Deterministic: ImageNet normalization and tensor conversion only.
    No stochastic/random augmentations applied.

    Returns:
        Albumentations Compose object.
    """
    if not ALBUMENTATIONS_AVAILABLE:
        raise ImportError(
            "albumentations is required for get_validation_augmentation(). "
            "Please run: pip install albumentations"
        )

    return A.Compose([
        A.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
            max_pixel_value=255.0,
        ),
        ToTensorV2(transpose_mask=False),
    ])


# Pure NumPy spatial transformation helpers for testing/verification without albumentations
def apply_spatial_transforms_numpy(
    image: np.ndarray,
    mask: np.ndarray,
    hflip: bool = False,
    vflip: bool = False,
    k_rot: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply identical geometric spatial transformations to both image [H, W, 3] and mask [H, W].

    Args:
        image: 3D uint8 array [H, W, 3].
        mask: 2D int64 array [H, W].
        hflip: Whether to flip horizontally.
        vflip: Whether to flip vertically.
        k_rot: Number of 90-degree counter-clockwise rotations (0, 1, 2, 3).

    Returns:
        Tuple of (transformed_image, transformed_mask).
    """
    img_out = image.copy()
    mask_out = mask.copy()

    if hflip:
        img_out = np.fliplr(img_out)
        mask_out = np.fliplr(mask_out)

    if vflip:
        img_out = np.flipud(img_out)
        mask_out = np.flipud(mask_out)

    if k_rot % 4 != 0:
        img_out = np.rot90(img_out, k=k_rot)
        mask_out = np.rot90(mask_out, k=k_rot)

    return img_out, mask_out


def denormalize_image(
    tensor_or_arr: np.ndarray,
    mean: Tuple[float, float, float] = IMAGENET_MEAN,
    std: Tuple[float, float, float] = IMAGENET_STD,
) -> np.ndarray:
    """
    Reverse ImageNet normalization to recover displayable RGB uint8 image [H, W, 3].

    Args:
        tensor_or_arr: Normalized image array [3, H, W] or [H, W, 3].
        mean: Normalization means.
        std: Normalization stds.

    Returns:
        3D numpy array [H, W, 3] of dtype uint8 in [0, 255].
    """
    arr = np.array(tensor_or_arr, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[0] == 3:
        # Transpose [3, H, W] -> [H, W, 3]
        arr = arr.transpose(1, 2, 0)

    mean_arr = np.array(mean, dtype=np.float32)
    std_arr = np.array(std, dtype=np.float32)

    denorm = (arr * std_arr + mean_arr) * 255.0
    denorm = np.clip(denorm, 0.0, 255.0).astype(np.uint8)
    return denorm


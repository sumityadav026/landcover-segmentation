"""
PyTorch Dataset and DataLoader implementation for LandCover.ai semantic segmentation.
Handles 512x512 tile loading, Albumentations transforms, integer mask validation, and batching.
"""

import os
import glob
import logging
from typing import Optional, Callable, Dict, Any, Tuple, List, Union

import numpy as np
from PIL import Image

logger = logging.getLogger("LandCover.Dataset")

# Optional PyTorch imports with graceful fallback for inspection/testing
try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    class Dataset:  # type: ignore
        pass
    class DataLoader:  # type: ignore
        pass

# Optional Albumentations imports
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


def load_image(file_path: str) -> np.ndarray:
    """
    Load an aerial image from disk as an RGB uint8 numpy array [H, W, 3].

    Args:
        file_path: Path to the image file (.png, .jpg, .tif).

    Returns:
        3D numpy array [H, W, 3] of dtype uint8 in RGB order.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Image file not found: {file_path}")

    if CV2_AVAILABLE:
        bgr = cv2.imread(file_path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise IOError(f"Failed to load image using OpenCV: {file_path}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    else:
        with Image.open(file_path) as pil_img:
            return np.array(pil_img.convert("RGB"), dtype=np.uint8)


def load_mask(file_path: str) -> np.ndarray:
    """
    Load a segmentation mask from disk as an integer class ID numpy array [H, W].
    Preserves exact integer values in {0, 1, 2, 3, 4} without normalization or interpolation.

    Args:
        file_path: Path to mask file (.png, .tif).

    Returns:
        2D numpy array [H, W] of dtype int64 with discrete class IDs.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Mask file not found: {file_path}")

    if CV2_AVAILABLE:
        mask = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise IOError(f"Failed to load mask using OpenCV: {file_path}")
        # If OpenCV loaded an extra channel (e.g. RGBA or 3-channel grayscale), collapse to 2D
        if mask.ndim == 3:
            mask = mask[:, :, 0]
    else:
        with Image.open(file_path) as pil_img:
            mask = np.array(pil_img)
            if mask.ndim == 3:
                mask = mask[:, :, 0]

    # Validate mask values
    unique_vals = np.unique(mask)
    if np.any(unique_vals < 0) or np.any(unique_vals > 4):
        raise ValueError(
            f"Invalid class ID found in mask {file_path}: {unique_vals}. "
            f"Valid classes are 0 (Background), 1 (Building), 2 (Woodland), 3 (Water), 4 (Road)."
        )

    return mask.astype(np.int64)


def get_default_transform(split: str = "train") -> Any:
    """
    Create default Albumentations transform for Stage 2.
    For train: HorizontalFlip, VerticalFlip, RandomRotate90, Brightness/Contrast, ImageNet Normalization.
    For val/test: ImageNet Normalization only.

    Args:
        split: 'train', 'val', or 'test'.

    Returns:
        Albumentations Compose transform or None if albumentations is unavailable.
    """
    try:
        from src.augmentations import get_training_augmentation, get_validation_augmentation
        if split == "train":
            return get_training_augmentation()
        else:
            return get_validation_augmentation()
    except (ImportError, Exception):
        if not ALBUMENTATIONS_AVAILABLE:
            return None

        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        return A.Compose([
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
            ToTensorV2(transpose_mask=False),
        ])


class LandCoverDataset(Dataset):
    """
    PyTorch Dataset for LandCover.ai 512x512 tiles.
    Organized as:
      root_dir/
        images/
          {patch_id}.png / .jpg
        masks/
          {patch_id}.png
    """

    def __init__(
        self,
        root_dir: str,
        split: Optional[str] = None,
        transform: Optional[Any] = None,
        return_metadata: bool = False,
    ):
        """
        Initialize the dataset.

        Args:
            root_dir: Directory containing 'images/' and 'masks/' folders,
                      or path to parent folder with split ('data', split='train').
            split: Optional split subdirectory name ('train', 'val', or 'test').
            transform: Albumentations or callable transform applied to image and mask.
            return_metadata: If True, returns (image, mask, metadata_dict).
        """
        if split is not None:
            self.base_dir = os.path.join(root_dir, split)
        else:
            self.base_dir = root_dir

        self.images_dir = os.path.join(self.base_dir, "images")
        self.masks_dir = os.path.join(self.base_dir, "masks")
        self.transform = transform
        self.return_metadata = return_metadata

        if not os.path.exists(self.images_dir):
            raise FileNotFoundError(f"Images directory not found at: {self.images_dir}")
        if not os.path.exists(self.masks_dir):
            raise FileNotFoundError(f"Masks directory not found at: {self.masks_dir}")

        # Index paired samples
        self.samples = self._find_pairs()
        logger.info(
            f"Initialized LandCoverDataset for '{split or self.base_dir}' with {len(self.samples)} pairs."
        )

    def _find_pairs(self) -> List[Tuple[str, str, str]]:
        """
        Search for paired image and mask files.

        Returns:
            List of tuples: (sample_id, image_path, mask_path).
        """
        supported_exts = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff")
        image_files = []
        for ext in supported_exts:
            image_files.extend(glob.glob(os.path.join(self.images_dir, ext)))

        image_files.sort()
        if not image_files:
            logger.warning(f"No image files found in {self.images_dir}")
            return []

        pairs: List[Tuple[str, str, str]] = []
        missing_masks = 0

        for img_path in image_files:
            base_name = os.path.splitext(os.path.basename(img_path))[0]

            # Candidate mask file patterns:
            # 1. Exact stem: {base_name}.png / {base_name}.tif
            # 2. Suffix stem: {base_name}_m.png (official split.py output)
            candidates = [
                os.path.join(self.masks_dir, f"{base_name}.png"),
                os.path.join(self.masks_dir, f"{base_name}.tif"),
                os.path.join(self.masks_dir, f"{base_name}_m.png"),
                os.path.join(self.masks_dir, f"{base_name}_m.tif"),
            ]

            mask_path = next((c for c in candidates if os.path.exists(c)), None)
            if mask_path is not None:
                pairs.append((base_name, img_path, mask_path))
            else:
                missing_masks += 1

        if missing_masks > 0:
            logger.warning(
                f"Found {missing_masks} images in {self.images_dir} with missing masks."
            )

        return pairs

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Union[Tuple[Any, Any], Tuple[Any, Any, Dict[str, Any]]]:
        """
        Get sample by index.

        Args:
            idx: Integer sample index.

        Returns:
            Tuple of (image, mask) where:
              - image: FloatTensor [3, 512, 512] (or [H, W, 3] numpy array if no transform)
              - mask: LongTensor [512, 512] (or [H, W] int64 numpy array if no transform)
            If return_metadata is True, returns (image, mask, metadata).
        """
        sample_id, img_path, mask_path = self.samples[idx]

        # Load raw arrays
        image = load_image(img_path)
        mask = load_mask(mask_path)

        # Validate spatial alignment
        if image.shape[:2] != mask.shape[:2]:
            raise ValueError(
                f"Dimension mismatch for sample '{sample_id}': "
                f"Image shape {image.shape} vs Mask shape {mask.shape}"
            )

        # Apply transforms
        if self.transform is not None:
            # Albumentations expects (image=..., mask=...)
            if ALBUMENTATIONS_AVAILABLE and isinstance(self.transform, (A.Compose, A.BasicTransform)):
                augmented = self.transform(image=image, mask=mask)
                image = augmented["image"]
                mask = augmented["mask"]
            else:
                # Custom callable
                image, mask = self.transform(image, mask)
        else:
            # Default fallback when no transform is provided
            if TORCH_AVAILABLE:
                # Convert to PyTorch tensors manually:
                # Image: [H, W, 3] uint8 -> [3, H, W] float32 normalized to [0, 1]
                image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
                # Mask: [H, W] int64
                mask = torch.from_numpy(mask).long()

        if self.return_metadata:
            meta = {
                "id": sample_id,
                "image_path": img_path,
                "mask_path": mask_path,
                "height": image.shape[-2] if hasattr(image, "shape") else 512,
                "width": image.shape[-1] if hasattr(image, "shape") else 512,
            }
            return image, mask, meta

        return image, mask

    def get_raw_sample(self, idx: int) -> Tuple[np.ndarray, np.ndarray, str]:
        """
        Fetch unaugmented, raw uint8 RGB image and int64 mask for analysis or plotting.

        Args:
            idx: Integer sample index.

        Returns:
            Tuple of (raw_image_np, raw_mask_np, sample_id).
        """
        sample_id, img_path, mask_path = self.samples[idx]
        image = load_image(img_path)
        mask = load_mask(mask_path)
        return image, mask, sample_id


def get_dataloaders(
    data_dir: str = "data_rebuilt",
    batch_size: int = 8,
    num_workers: int = 2,
    pin_memory: bool = True,
    train_transform: Optional[Any] = None,
    val_transform: Optional[Any] = None,
    test_transform: Optional[Any] = None,
) -> Tuple[Any, Any, Any]:
    """
    Construct PyTorch DataLoaders for train, val, and test splits.
    Defaults to 'data_rebuilt' and resolves to verified Drive/local paths.

    Args:
        data_dir: Root dataset directory containing 'train', 'val', 'test' folders.
        batch_size: Number of samples per batch (default: 8).
        num_workers: Subprocesses for data loading (default: 2).
        pin_memory: Pin memory for faster CPU to GPU transfer (default: True).
        train_transform: Transform for train set (defaults to get_default_transform('train')).
        val_transform: Transform for val set (defaults to get_default_transform('val')).
        test_transform: Transform for test set (defaults to get_default_transform('test')).

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required to create DataLoaders.")

    # Resolve data_dir strictly prioritizing data_rebuilt
    candidates = [
        data_dir,
        "/content/drive/MyDrive/landcover-segmentation/data_rebuilt",
        "/content/data_rebuilt",
        "data_rebuilt",
    ]
    resolved_dir = data_dir
    for cand in candidates:
        if cand and os.path.exists(cand) and os.path.exists(os.path.join(cand, "train")):
            resolved_dir = cand
            break

    train_tf = train_transform if train_transform is not None else get_default_transform("train")
    val_tf = val_transform if val_transform is not None else get_default_transform("val")
    test_tf = test_transform if test_transform is not None else get_default_transform("test")

    train_dataset = LandCoverDataset(root_dir=resolved_dir, split="train", transform=train_tf)
    val_dataset = LandCoverDataset(root_dir=resolved_dir, split="val", transform=val_tf)
    test_dataset = LandCoverDataset(root_dir=resolved_dir, split="test", transform=test_tf)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=False,
    )

    logger.info(
        f"DataLoaders created: Train batches={len(train_loader)}, "
        f"Val batches={len(val_loader)}, Test batches={len(test_loader)}"
    )

    return train_loader, val_loader, test_loader


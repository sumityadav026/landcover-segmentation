"""
Utility functions for Urban Land Cover Mapping semantic segmentation project.
Provides reproducibility tools, configuration loading, color mapping, and visualization helpers.
"""

import os
import random
import json
import logging
from typing import Dict, Optional, Tuple, Any

import numpy as np

# Configure standard logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("LandCover")

# Default official class mapping and visualization colors
DEFAULT_CLASSES: Dict[int, str] = {
    0: "Background",
    1: "Building",
    2: "Woodland",
    3: "Water",
    4: "Road",
}

DEFAULT_CLASS_COLORS: Dict[int, Tuple[int, int, int]] = {
    0: (0, 0, 0),        # Background: Black
    1: (255, 0, 0),      # Building: Red
    2: (0, 255, 0),      # Woodland: Green
    3: (0, 0, 255),      # Water: Blue
    4: (255, 255, 0),    # Road: Yellow
}


def set_seed(seed: int = 42) -> None:
    """
    Set random seed across all libraries to ensure full reproducibility.

    Args:
        seed: Integer seed value (default: 42).
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        logger.info(f"PyTorch seed set to {seed} (deterministic mode enabled)")
    except ImportError:
        logger.debug("PyTorch not installed; set seed for random and numpy only.")


def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """
    Load project configuration YAML file.

    Args:
        config_path: Relative or absolute path to configuration file.

    Returns:
        Nested dictionary with parsed configuration parameters.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    try:
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return config
    except ImportError:
        # Fallback minimal YAML parser for basic key-value / nested structures if pyyaml is missing
        logger.warning("PyYAML not found. Falling back to built-in JSON / default config loader.")
        # If a json mirror exists or simple parsing
        raise ImportError("pyyaml is required to load configs/config.yaml. Please install pyyaml.")


def mask_to_rgb(
    mask: np.ndarray,
    color_map: Optional[Dict[int, Tuple[int, int, int]]] = None,
) -> np.ndarray:
    """
    Convert a 2D segmentation mask with integer class IDs into an RGB visualization image.

    Args:
        mask: 2D numpy array [H, W] with integer class labels in {0, 1, 2, 3, 4}.
        color_map: Optional dictionary mapping class ID -> (R, G, B) tuple.

    Returns:
        3D numpy array [H, W, 3] of dtype uint8 with RGB color mapping.
    """
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask array [H, W], got shape {mask.shape}")

    palette = color_map or DEFAULT_CLASS_COLORS
    h, w = mask.shape
    rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id, color in palette.items():
        # Match mask pixels equal to class_id
        match = (mask == class_id)
        if np.any(match):
            rgb_mask[match] = color

    return rgb_mask


def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.5,
    color_map: Optional[Dict[int, Tuple[int, int, int]]] = None,
) -> np.ndarray:
    """
    Overlay a colored segmentation mask on top of an aerial RGB image.

    Args:
        image: 3D numpy array [H, W, 3] of dtype uint8 (RGB).
        mask: 2D numpy array [H, W] of integer class IDs.
        alpha: Blend factor between 0.0 (only image) and 1.0 (only mask).
        color_map: Optional dictionary mapping class ID -> (R, G, B) tuple.

    Returns:
        3D numpy array [H, W, 3] blended image.
    """
    if image.shape[:2] != mask.shape[:2]:
        raise ValueError(
            f"Image spatial shape {image.shape[:2]} does not match mask spatial shape {mask.shape[:2]}"
        )

    colored_mask = mask_to_rgb(mask, color_map=color_map)
    # Blend only where mask is not background (class 0)
    blended = image.copy()
    non_bg = (mask > 0)
    blended[non_bg] = (
        (1.0 - alpha) * image[non_bg] + alpha * colored_mask[non_bg]
    ).astype(np.uint8)

    return blended


def save_json(data: Dict[str, Any], file_path: str) -> None:
    """Save dictionary to a formatted JSON file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved JSON data to {file_path}")


def load_json(file_path: str) -> Dict[str, Any]:
    """Load dictionary from JSON file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"JSON file not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


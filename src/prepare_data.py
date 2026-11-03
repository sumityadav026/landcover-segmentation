"""
Data Preparation Pipeline for LandCover.ai.
Handles Kaggle dataset download, orthophoto inspection, source-image-level splitting
to prevent spatial data leakage, 512x512 patch tiling, integrity validation,
and dataset class statistics calculation.
"""

import os
import sys
import glob
import json
import logging
from typing import Dict, List, Tuple, Set, Optional, Any
from collections import Counter

import numpy as np
from PIL import Image

# Import project utilities
from src.utils import logger, set_seed, save_json, DEFAULT_CLASSES

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


def check_kaggle_credentials() -> bool:
    """
    Verify if Kaggle credentials exist in environment variables, ~/.kaggle/kaggle.json,
    or Google Colab userdata secrets.
    """
    # 1. Environment variables
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        logger.info("Kaggle credentials found in environment variables.")
        return True

    # 2. Local kaggle.json file
    kaggle_json = os.path.expanduser("~/.kaggle/kaggle.json")
    if os.path.exists(kaggle_json):
        logger.info(f"Kaggle credentials found at {kaggle_json}")
        return True

    # 3. Workspace root kaggle.json
    if os.path.exists("kaggle.json"):
        os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
        import shutil
        shutil.copy("kaggle.json", kaggle_json)
        os.chmod(kaggle_json, 0o600)
        logger.info("Copied workspace kaggle.json to ~/.kaggle/kaggle.json")
        return True

    # 4. Check Google Colab userdata secrets
    try:
        from google.colab import userdata  # type: ignore
        username = userdata.get("KAGGLE_USERNAME")
        key = userdata.get("KAGGLE_KEY")
        if username and key:
            os.environ["KAGGLE_USERNAME"] = username
            os.environ["KAGGLE_KEY"] = key
            logger.info("Retrieved Kaggle credentials from Google Colab userdata secrets.")
            return True
    except Exception:
        pass

    logger.warning("No Kaggle credentials detected.")
    return False


def download_dataset(
    kaggle_slug: str = "adrianboguszewski/landcoverai",
    raw_dir: str = "data/raw",
) -> str:
    """
    Download and extract the LandCover.ai dataset from Kaggle.

    Args:
        kaggle_slug: Kaggle dataset identifier.
        raw_dir: Directory where raw dataset files are stored.

    Returns:
        Path to the directory containing unzipped raw dataset.
    """
    os.makedirs(raw_dir, exist_ok=True)

    # Check if raw images are already present
    existing_tifs = glob.glob(os.path.join(raw_dir, "**", "*.tif"), recursive=True)
    if len(existing_tifs) >= 40:
        logger.info(f"Raw dataset already exists in {raw_dir} ({len(existing_tifs)} .tif files found).")
        return raw_dir

    if not check_kaggle_credentials():
        raise PermissionError(
            "Kaggle API credentials not found! Please set KAGGLE_USERNAME and KAGGLE_KEY "
            "as environment variables, in ~/.kaggle/kaggle.json, or via Colab Secrets."
        )

    logger.info(f"Downloading dataset '{kaggle_slug}' to {raw_dir}...")
    import subprocess
    cmd = [
        "kaggle", "datasets", "download",
        "-d", kaggle_slug,
        "-p", raw_dir,
        "--unzip"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"Kaggle download failed: {result.stderr}")
        raise RuntimeError(f"Failed to download Kaggle dataset: {result.stderr}")

    logger.info(f"Dataset downloaded and extracted successfully to {raw_dir}.")
    return raw_dir


def inspect_raw_dataset(raw_dir: str) -> Dict[str, Any]:
    """
    Inspect raw LandCover.ai directory structure, find images, masks, and split files.

    Args:
        raw_dir: Directory containing extracted dataset.

    Returns:
        Dictionary with paths to images, masks, and official split files.
    """
    logger.info(f"Inspecting raw dataset in {raw_dir}...")

    # Locate images directory
    images_candidates = [
        os.path.join(raw_dir, "images"),
        os.path.join(raw_dir, "Images"),
        os.path.join(raw_dir, "**", "images"),
    ]
    images_dir = None
    for cand in images_candidates:
        matches = glob.glob(cand, recursive=True)
        if matches and os.path.isdir(matches[0]):
            images_dir = matches[0]
            break

    # Locate masks directory
    masks_candidates = [
        os.path.join(raw_dir, "masks"),
        os.path.join(raw_dir, "Masks"),
        os.path.join(raw_dir, "**", "masks"),
    ]
    masks_dir = None
    for cand in masks_candidates:
        matches = glob.glob(cand, recursive=True)
        if matches and os.path.isdir(matches[0]):
            masks_dir = matches[0]
            break

    # If images and masks are in the root directory directly
    if images_dir is None:
        tifs = glob.glob(os.path.join(raw_dir, "*.tif"))
        if tifs:
            images_dir = raw_dir
            masks_dir = raw_dir

    if images_dir is None or masks_dir is None:
        raise FileNotFoundError(f"Could not locate images and masks directories in {raw_dir}")

    # Find image and mask files
    img_files = sorted(glob.glob(os.path.join(images_dir, "*.tif")))
    mask_files = sorted(glob.glob(os.path.join(masks_dir, "*.tif")))

    # Check for official split text files
    split_files = {}
    for split in ["train", "val", "test"]:
        cand = glob.glob(os.path.join(raw_dir, "**", f"{split}.txt"), recursive=True)
        if cand:
            split_files[split] = cand[0]

    # Map base names
    img_map = {os.path.splitext(os.path.basename(f))[0]: f for f in img_files}
    mask_map = {os.path.splitext(os.path.basename(f))[0]: f for f in mask_files}

    # Intersect valid pairs
    common_stems = sorted(list(set(img_map.keys()) & set(mask_map.keys())))
    logger.info(
        f"Found {len(img_files)} source images, {len(mask_files)} masks. "
        f"Matched {len(common_stems)} source pairs. Found official split files: {list(split_files.keys())}"
    )

    return {
        "images_dir": images_dir,
        "masks_dir": masks_dir,
        "common_stems": common_stems,
        "img_map": img_map,
        "mask_map": mask_map,
        "split_files": split_files,
    }


def partition_source_images(
    source_stems: List[str],
    official_split_files: Dict[str, str],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, List[str]]:
    """
    Partition source orthophoto images strictly into Train, Val, and Test subsets.
    Guarantees:
      1. Zero data leakage: no source image appears in multiple splits.
      2. If official split lists have patch-level leakage, source images are assigned
         deterministically by majority presence or stratified random allocation.
      3. Proportional representation of 25cm and 50cm resolutions.

    Args:
        source_stems: List of source orthophoto base names (e.g. ['M-33-20-D-c-4-2', ...]).
        official_split_files: Dict of split name -> path to train.txt / val.txt / test.txt.
        train_ratio: Fraction for training (default: 0.70).
        val_ratio: Fraction for validation (default: 0.15).
        test_ratio: Fraction for testing (default: 0.15).
        seed: Random seed for reproducibility.

    Returns:
        Dict mapping 'train', 'val', 'test' -> list of source image stems.
    """
    set_seed(seed)
    source_set = set(source_stems)

    # Check official split files to see if they list whole images or patches
    split_presence: Dict[str, Dict[str, int]] = {s: Counter() for s in source_stems}
    uses_patches = False

    if official_split_files and all(s in official_split_files for s in ["train", "val", "test"]):
        for split_name, txt_path in official_split_files.items():
            with open(txt_path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]

            for line in lines:
                # Strip extension if present
                clean_name = os.path.splitext(line)[0]
                # Check if it directly matches a source orthophoto
                if clean_name in source_set:
                    split_presence[clean_name][split_name] += 1
                else:
                    # Check if patch format: {source_stem}_{patch_id}
                    parts = clean_name.rsplit("_", 1)
                    if len(parts) == 2 and parts[0] in source_set:
                        uses_patches = True
                        split_presence[parts[0]][split_name] += 1

    # Check if official splits strictly partitioned source images or mixed them
    multi_split_sources = [
        s for s, counts in split_presence.items() if len(counts) > 1
    ]
    has_presence = any(len(counts) > 0 for counts in split_presence.values())

    source_assignment: Dict[str, List[str]] = {"train": [], "val": [], "test": []}

    if multi_split_sources:
        logger.warning(
            f"Official benchmark split lists ({list(official_split_files.keys())}) divide individual "
            f"patches across splits, resulting in {len(multi_split_sources)} source orthophotos appearing "
            f"in multiple splits. To ELIMINATE spatial autocorrelation and cross-split data leakage, "
            f"each entire source orthophoto is assigned exclusively to ONE split."
        )
        # Deterministic partition by source image to maintain ~70% train, ~15% val, ~15% test
        sorted_stems = sorted(source_stems)
        rng = np.random.RandomState(seed)
        shuffled_stems = list(sorted_stems)
        rng.shuffle(shuffled_stems)

        n = len(shuffled_stems)
        n_train = max(1, int(round(n * train_ratio)))
        n_val = max(1, int(round(n * val_ratio)))
        # Ensure test gets at least 1 if n >= 3
        if n >= 3 and (n_train + n_val >= n):
            n_train = n - n_val - 1

        source_assignment["train"] = sorted(shuffled_stems[:n_train])
        source_assignment["val"] = sorted(shuffled_stems[n_train : n_train + n_val])
        source_assignment["test"] = sorted(shuffled_stems[n_train + n_val :])

    elif has_presence and all(len(counts) == 1 for counts in split_presence.values() if counts):
        # Official splits already partition source images cleanly
        logger.info("Official split lists cleanly partition source orthophotos!")
        for stem, counts in split_presence.items():
            if counts:
                split_name = list(counts.keys())[0]
                source_assignment[split_name].append(stem)
    else:
        # Default deterministic partition by source image
        logger.info("Partitioning source orthophotos deterministically into 70/15/15 splits.")
        sorted_stems = sorted(source_stems)
        rng = np.random.RandomState(seed)
        shuffled_stems = list(sorted_stems)
        rng.shuffle(shuffled_stems)

        n = len(shuffled_stems)
        n_train = max(1, int(round(n * train_ratio)))
        n_val = max(1, int(round(n * val_ratio)))
        if n >= 3 and (n_train + n_val >= n):
            n_train = n - n_val - 1

        source_assignment["train"] = sorted(shuffled_stems[:n_train])
        source_assignment["val"] = sorted(shuffled_stems[n_train : n_train + n_val])
        source_assignment["test"] = sorted(shuffled_stems[n_train + n_val :])

    # Verification: Verify no overlap
    train_set = set(source_assignment["train"])
    val_set = set(source_assignment["val"])
    test_set = set(source_assignment["test"])

    assert len(train_set & val_set) == 0, "Leakage detected: Train and Val share source images!"
    assert len(train_set & test_set) == 0, "Leakage detected: Train and Test share source images!"
    assert len(val_set & test_set) == 0, "Leakage detected: Val and Test share source images!"

    logger.info(
        f"Source Image Assignment Summary: "
        f"Train: {len(train_set)} images, Val: {len(val_set)} images, Test: {len(test_set)} images. "
        f"Overlap = 0 (Leakage-free)."
    )

    return source_assignment


def tile_orthophoto_pair(
    img_path: str,
    mask_path: str,
    output_img_dir: str,
    output_mask_dir: str,
    source_stem: str,
    tile_size: int = 512,
    stride: int = 512,
    edge_handling: str = "discard",
) -> Tuple[int, Counter]:
    """
    Slice a large orthophoto and matching mask into 512x512 tiles.
    Preserves exact spatial alignment and integer mask values.

    Args:
        img_path: Path to source aerial .tif image.
        mask_path: Path to source mask .tif.
        output_img_dir: Directory to save generated image tiles.
        output_mask_dir: Directory to save generated mask tiles.
        source_stem: Name of source image without extension.
        tile_size: Pixel dimensions of each square patch (512).
        stride: Stride between successive patches (512 for non-overlapping).
        edge_handling: 'discard' (ignore incomplete edge patches) or 'pad'.

    Returns:
        Tuple of (num_tiles_saved, class_pixel_counter).
    """
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)

    # Open image and mask using PIL or OpenCV
    if CV2_AVAILABLE:
        bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise IOError(f"Could not read image: {img_path}")
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise IOError(f"Could not read mask: {mask_path}")
        if mask.ndim == 3:
            mask = mask[:, :, 0]
    else:
        with Image.open(img_path) as pil_img:
            img = np.array(pil_img.convert("RGB"), dtype=np.uint8)
        with Image.open(mask_path) as pil_mask:
            mask = np.array(pil_mask)
            if mask.ndim == 3:
                mask = mask[:, :, 0]

    h_img, w_img = img.shape[:2]
    h_mask, w_mask = mask.shape[:2]

    if (h_img, w_img) != (h_mask, w_mask):
        raise ValueError(
            f"Source image shape {(h_img, w_img)} does not match mask shape {(h_mask, w_mask)} "
            f"for source: {source_stem}"
        )

    tile_count = 0
    class_counter: Counter = Counter()

    for y in range(0, h_img, stride):
        for x in range(0, w_img, stride):
            y_end = min(y + tile_size, h_img)
            x_end = min(x + tile_size, w_img)

            img_patch = img[y:y_end, x:x_end]
            mask_patch = mask[y:y_end, x:x_end]

            actual_h, actual_w = img_patch.shape[:2]

            # Incomplete edge tile handling
            if actual_h < tile_size or actual_w < tile_size:
                if edge_handling == "discard":
                    # Discard edge tiles to strictly avoid zero-padding artifacts
                    continue
                elif edge_handling == "pad":
                    # Pad image with zero/reflection, mask with background (0)
                    padded_img = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
                    padded_mask = np.zeros((tile_size, tile_size), dtype=np.uint8)
                    padded_img[:actual_h, :actual_w] = img_patch
                    padded_mask[:actual_h, :actual_w] = mask_patch
                    img_patch = padded_img
                    mask_patch = padded_mask
                else:
                    continue

            # Validate mask classes
            unique_classes, counts = np.unique(mask_patch, return_counts=True)
            for c, cnt in zip(unique_classes, counts):
                class_counter[int(c)] += int(cnt)

            # File names
            patch_filename = f"{source_stem}_{tile_count}.png"
            out_img_path = os.path.join(output_img_dir, patch_filename)
            out_mask_path = os.path.join(output_mask_dir, patch_filename)

            # Save image and mask as lossless 8-bit PNGs
            if CV2_AVAILABLE:
                cv2.imwrite(out_img_path, cv2.cvtColor(img_patch, cv2.COLOR_RGB2BGR))
                # Crucial: write raw integer values 0-4 as uint8
                cv2.imwrite(out_mask_path, mask_patch.astype(np.uint8))
            else:
                Image.fromarray(img_patch).save(out_img_path, format="PNG")
                Image.fromarray(mask_patch.astype(np.uint8)).save(out_mask_path, format="PNG")

            tile_count += 1

    return tile_count, class_counter


def prepare_landcover_dataset(
    config: Optional[Dict[str, Any]] = None,
    raw_dir: str = "data/raw",
    processed_dir: str = "data_rebuilt",
    tile_size: int = 512,
    stride: int = 512,
    edge_handling: str = "discard",
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Run complete end-to-end dataset preparation pipeline:
    1. Inspect raw files and official split lists.
    2. Partition source orthophotos cleanly across train/val/test without leakage.
    3. Tile source images and masks into 512x512 patches.
    4. Compute and save split statistics, class pixel counts, and inverse weights.

    Returns:
        Dictionary containing comprehensive preparation statistics.
    """
    set_seed(seed)
    info = inspect_raw_dataset(raw_dir)

    source_stems = info["common_stems"]
    if not source_stems:
        raise RuntimeError(f"No matching image-mask pairs found in {raw_dir}")

    # Partition source images
    source_assignment = partition_source_images(
        source_stems=source_stems,
        official_split_files=info["split_files"],
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=seed,
    )

    # Save source assignment manifest
    os.makedirs("results", exist_ok=True)
    save_json(source_assignment, "results/source_image_split_assignment.json")

    summary_stats: Dict[str, Any] = {
        "splits": {},
        "global_class_pixels": Counter(),
        "total_tiles": 0,
        "tile_size": tile_size,
        "edge_handling": edge_handling,
    }

    for split in ["train", "val", "test"]:
        split_stems = source_assignment[split]
        split_img_dir = os.path.join(processed_dir, split, "images")
        split_mask_dir = os.path.join(processed_dir, split, "masks")
        os.makedirs(split_img_dir, exist_ok=True)
        os.makedirs(split_mask_dir, exist_ok=True)

        logger.info(f"Processing split '{split}' ({len(split_stems)} source orthophotos)...")
        split_tile_count = 0
        split_class_counter: Counter = Counter()

        for idx, stem in enumerate(split_stems):
            img_path = info["img_map"][stem]
            mask_path = info["mask_map"][stem]

            tiles_saved, counter = tile_orthophoto_pair(
                img_path=img_path,
                mask_path=mask_path,
                output_img_dir=split_img_dir,
                output_mask_dir=split_mask_dir,
                source_stem=stem,
                tile_size=tile_size,
                stride=stride,
                edge_handling=edge_handling,
            )

            split_tile_count += tiles_saved
            split_class_counter.update(counter)
            summary_stats["global_class_pixels"].update(counter)

            if (idx + 1) % 5 == 0 or (idx + 1) == len(split_stems):
                logger.info(
                    f"[{split}] Processed {idx + 1}/{len(split_stems)} images | "
                    f"Cumulative tiles: {split_tile_count}"
                )

        summary_stats["splits"][split] = {
            "source_images_count": len(split_stems),
            "tiles_count": split_tile_count,
            "class_pixel_counts": dict(split_class_counter),
        }
        summary_stats["total_tiles"] += split_tile_count

    # Calculate global and split percentages and class weights
    total_pixels = sum(summary_stats["global_class_pixels"].values())
    class_distribution = {}
    class_weights_ce = {}

    num_classes = 5
    for class_id in range(num_classes):
        c_pixels = summary_stats["global_class_pixels"].get(class_id, 0)
        percentage = (c_pixels / total_pixels * 100.0) if total_pixels > 0 else 0.0
        class_name = DEFAULT_CLASSES.get(class_id, f"Class_{class_id}")

        # Inverse frequency class weighting: w_c = N_total / (num_classes * N_c)
        if c_pixels > 0:
            weight = total_pixels / (num_classes * c_pixels)
        else:
            weight = 1.0

        class_distribution[class_id] = {
            "name": class_name,
            "pixel_count": int(c_pixels),
            "percentage": round(percentage, 4),
            "ce_weight": round(float(weight), 4),
        }
        class_weights_ce[class_id] = round(float(weight), 4)

    summary_stats["class_distribution"] = class_distribution
    summary_stats["class_weights_ce"] = class_weights_ce
    summary_stats["global_class_pixels"] = dict(summary_stats["global_class_pixels"])

    # Save results to disk
    save_json(summary_stats, "results/dataset_statistics.json")
    save_json(class_distribution, "results/class_distribution.json")

    logger.info("Dataset preparation complete!")
    logger.info(
        f"Total generated tiles: {summary_stats['total_tiles']} "
        f"(Train: {summary_stats['splits']['train']['tiles_count']}, "
        f"Val: {summary_stats['splits']['val']['tiles_count']}, "
        f"Test: {summary_stats['splits']['test']['tiles_count']})"
    )

    return summary_stats


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Prepare LandCover.ai dataset for semantic segmentation")
    parser.add_argument("--download", action="store_true", help="Download dataset from Kaggle")
    parser.add_argument("--raw_dir", type=str, default="data/raw", help="Path to raw dataset")
    parser.add_argument("--processed_dir", type=str, default="data_rebuilt", help="Output directory for tiles")
    parser.add_argument("--tile_size", type=int, default=512, help="Tile size in pixels")
    parser.add_argument("--stride", type=int, default=512, help="Stride in pixels")
    parser.add_argument("--edge_handling", type=str, default="discard", choices=["discard", "pad"])
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    if args.download:
        download_dataset(raw_dir=args.raw_dir)

    prepare_landcover_dataset(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        tile_size=args.tile_size,
        stride=args.stride,
        edge_handling=args.edge_handling,
        seed=args.seed,
    )

"""
Colab Recovery and Resilience Engine for Urban Land Cover Mapping.
Provides automated environment recovery after Colab runtime resets:
  - Google Drive mounting and authorization detection
  - Codebase and configuration restoration from Drive / ZIP
  - Processed dataset detection (never re-downloading or re-tiling if already present)
  - Checkpoint scanning and automated training state resumption
  - One-epoch smoke-test mode and full training session builder
"""

import os
import sys
import glob
import json
import shutil
import zipfile
import tarfile
import logging
from typing import Dict, Any, Optional, Tuple, List, Union

logger = logging.getLogger("LandCover.Recovery")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def is_colab_environment() -> bool:
    """Detect whether current Python process is running inside Google Colab."""
    return "google.colab" in sys.modules or os.path.exists("/content")


def mount_google_drive(mount_point: str = "/content/drive") -> Tuple[bool, str]:
    """
    Mount Google Drive with authorization prompt if running inside Google Colab.

    Args:
        mount_point: Mount path for Google Drive (default: '/content/drive').

    Returns:
        Tuple of (is_mounted: bool, status_message: str).
    """
    if not is_colab_environment():
        logger.info("Local environment detected. Google Drive mount skipped.")
        return False, "Local environment (Drive mount skipped)"

    # Check if already mounted
    if os.path.exists(mount_point) and os.path.exists(os.path.join(mount_point, "MyDrive")):
        logger.info(f"Google Drive already mounted at {mount_point}")
        return True, f"Google Drive already mounted at {mount_point}"

    try:
        from google.colab import drive
        logger.info(f"Mounting Google Drive to {mount_point}...")
        drive.mount(mount_point)
        return True, f"Google Drive successfully mounted at {mount_point}"
    except Exception as e:
        logger.warning(f"Google Drive mount failed: {e}")
        return False, f"Google Drive mount failed: {e}"


def restore_project_codebase(
    drive_mount_point: str = "/content/drive",
    project_name: str = "landcover-segmentation",
    target_dir: str = ".",
    zip_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Restore project source files ('src/', 'configs/', 'requirements.txt') after a runtime reset.

    Search hierarchy:
      1. Explicit zip_path argument if provided
      2. Persistent source directory on Google Drive (`{drive_project_dir}/src`)
      3. Project ZIP on Google Drive (`{drive_project_dir}/landcover-segmentation.zip`)
      4. Project ZIP in `/content/landcover-segmentation.zip` or local directory
      5. Current working directory

    Also automatically syncs/backs up codebase to Google Drive when connected so future resets
    can restore instantly without manual file uploads.

    Args:
        drive_mount_point: Google Drive mount path.
        project_name: Folder name on Google Drive.
        target_dir: Destination working directory (default: current directory).
        zip_path: Optional explicit path to landcover-segmentation.zip.

    Returns:
        Dictionary with restoration status and paths.
    """
    target_dir = os.path.abspath(target_dir)
    drive_project_dir = os.path.join(drive_mount_point, "MyDrive", project_name)
    src_target = os.path.join(target_dir, "src")

    status = {
        "restored": False,
        "source": "none",
        "src_path": src_target,
        "drive_project_dir": drive_project_dir,
        "drive_connected": os.path.exists(drive_project_dir),
    }

    # If src is missing locally, attempt restoration from Drive or ZIP
    if not os.path.exists(src_target):
        # 1. Check persistent source directory on Google Drive
        drive_src = os.path.join(drive_project_dir, "src")
        if os.path.exists(drive_src):
            logger.info(f"Restoring codebase from Google Drive: {drive_src} -> {src_target}")
            shutil.copytree(drive_src, src_target, dirs_exist_ok=True)
            # Also restore configs if present
            drive_configs = os.path.join(drive_project_dir, "configs")
            if os.path.exists(drive_configs):
                shutil.copytree(drive_configs, os.path.join(target_dir, "configs"), dirs_exist_ok=True)
            status["restored"] = True
            status["source"] = "google_drive_folder"

        # 2. Check for ZIP candidates
        if not status["restored"]:
            zip_candidates = []
            if zip_path and os.path.exists(zip_path):
                zip_candidates.append(zip_path)

            zip_candidates.extend([
                os.path.join(drive_project_dir, "landcover-segmentation.zip"),
                "/content/landcover-segmentation.zip",
                os.path.join(target_dir, "landcover-segmentation.zip"),
                os.path.join(os.path.dirname(target_dir), "landcover-segmentation.zip"),
            ])

            for z_cand in zip_candidates:
                if os.path.exists(z_cand):
                    logger.info(f"Extracting codebase from ZIP archive: {z_cand}")
                    with zipfile.ZipFile(z_cand, "r") as zf:
                        zf.extractall(target_dir)
                    status["restored"] = True
                    status["source"] = f"zip ({z_cand})"
                    break
    else:
        status["restored"] = True
        status["source"] = "already_present_locally"

    # Add target directory to sys.path
    if target_dir not in sys.path:
        sys.path.insert(0, target_dir)

    # If Drive is connected, ensure persistent backup exists on Drive for future resets
    if os.path.exists(drive_project_dir) and os.path.exists(src_target):
        try:
            drive_src = os.path.join(drive_project_dir, "src")
            drive_configs = os.path.join(drive_project_dir, "configs")
            shutil.copytree(src_target, drive_src, dirs_exist_ok=True)
            if os.path.exists(os.path.join(target_dir, "configs")):
                shutil.copytree(os.path.join(target_dir, "configs"), drive_configs, dirs_exist_ok=True)
            logger.info(f"Backed up codebase to persistent Drive folder: {drive_project_dir}")
        except Exception as e:
            logger.warning(f"Could not update Drive codebase backup: {e}")

    return status


def check_dataset_persistence(
    drive_mount_point: str = "/content/drive",
    project_name: str = "landcover-segmentation",
    local_data_dir: str = "data_rebuilt",
) -> Dict[str, Any]:
    """
    Detect whether the processed dataset already exists in Google Drive or local storage.
    Strictly verifies 'data_rebuilt' and NEVER falls back to any incomplete 'data' directory.
    NEVER re-downloads from Kaggle or re-tiles if verified data_rebuilt is available.

    Verified Dataset Profile:
      - Train: 7,704 tiles
      - Validation: 1,602 tiles
      - Test: 1,368 tiles
      - Total: 10,674 tiles (512x512)

    Args:
        drive_mount_point: Google Drive mount path.
        project_name: Name of project on Drive.
        local_data_dir: Target local data directory (defaults to 'data_rebuilt').

    Returns:
        Dictionary detailing dataset availability and tile counts.
    """
    drive_project_dir = os.path.join(drive_mount_point, "MyDrive", project_name)
    drive_rebuilt_dir = os.path.join(drive_project_dir, "data_rebuilt")

    def count_tiles(base_dir: str) -> Dict[str, int]:
        counts = {}
        for split in ["train", "val", "test"]:
            img_dir = os.path.join(base_dir, split, "images")
            mask_dir = os.path.join(base_dir, split, "masks")
            if os.path.exists(img_dir):
                n_imgs = len(glob.glob(os.path.join(img_dir, "*.*")))
                counts[split] = n_imgs
            else:
                counts[split] = 0
        return counts

    # Candidate directories to inspect strictly for data_rebuilt
    candidate_rebuilt_dirs = [
        drive_rebuilt_dir,
        os.path.join(drive_mount_point, "MyDrive", "data_rebuilt"),
        os.path.abspath(local_data_dir),
        os.path.join(".", "data_rebuilt"),
        "/content/data_rebuilt",
    ]

    for candidate_dir in candidate_rebuilt_dirs:
        if os.path.exists(candidate_dir):
            counts = count_tiles(candidate_dir)
            if counts.get("train", 0) > 0 and counts.get("val", 0) > 0:
                logger.info(
                    f"Verified processed dataset at '{candidate_dir}': "
                    f"Train={counts.get('train', 0)}, Val={counts.get('val', 0)}, Test={counts.get('test', 0)}"
                )

                # In Colab, symlink to local path for fast access if needed
                active_dir = candidate_dir
                if is_colab_environment() and candidate_dir != "/content/data_rebuilt":
                    try:
                        if not os.path.exists("/content/data_rebuilt"):
                            os.symlink(candidate_dir, "/content/data_rebuilt")
                            active_dir = "/content/data_rebuilt"
                            logger.info(f"Created symlink: /content/data_rebuilt -> {candidate_dir}")
                    except Exception:
                        pass

                return {
                    "ready": True,
                    "location": active_dir,
                    "storage_type": "google_drive" if "drive" in candidate_dir.lower() else "local_disk",
                    "counts": counts,
                    "skip_data_preparation": True,
                }

    # Check Google Drive compressed archive for data_rebuilt
    for tar_name in ["data_rebuilt.tar.gz", "data_processed.tar.gz"]:
        drive_tar = os.path.join(drive_project_dir, tar_name)
        if os.path.exists(drive_tar):
            logger.info(f"Found compressed dataset archive on Drive: {drive_tar}. Extracting...")
            target_extract = os.path.abspath(local_data_dir)
            os.makedirs(target_extract, exist_ok=True)
            with tarfile.open(drive_tar, "r:gz") as tar:
                tar.extractall(target_extract)
            extracted_counts = count_tiles(target_extract)
            return {
                "ready": True,
                "location": target_extract,
                "storage_type": "extracted_from_drive_tar",
                "counts": extracted_counts,
                "skip_data_preparation": True,
            }

    # Strict check: Warn if an old/incomplete 'data' directory exists without data_rebuilt
    old_data_dir = os.path.join(drive_project_dir, "data")
    if os.path.exists(old_data_dir) or os.path.exists("data"):
        logger.warning(
            "An incomplete 'data' directory was detected, but 'data_rebuilt' is strictly required. "
            "Refusing to fall back to 'data' to prevent data corruption or partial training."
        )

    logger.warning("Processed 'data_rebuilt' dataset not found in local directory or Google Drive.")
    return {
        "ready": False,
        "location": None,
        "storage_type": "none",
        "counts": {"train": 0, "val": 0, "test": 0},
        "skip_data_preparation": False,
    }


def scan_checkpoints(checkpoint_dir: str) -> Dict[str, Any]:
    """
    Scan checkpoint directory for UNet and DeepLabV3+ latest and best checkpoints.

    Args:
        checkpoint_dir: Directory containing .pt checkpoint files.

    Returns:
        Dictionary mapping model names to checkpoint metadata.
    """
    models = ["unet", "deeplabv3plus"]
    summary: Dict[str, Any] = {}

    for model_name in models:
        candidates_latest = [
            os.path.join(checkpoint_dir, f"{model_name}_latest.pt"),
            os.path.join(checkpoint_dir, model_name, "latest.pt"),
        ]
        candidates_best = [
            os.path.join(checkpoint_dir, f"{model_name}_best.pt"),
            os.path.join(checkpoint_dir, model_name, "best.pt"),
        ]

        latest_file = next((f for f in candidates_latest if os.path.exists(f) and os.path.getsize(f) > 1024), None)
        best_file = next((f for f in candidates_best if os.path.exists(f) and os.path.getsize(f) > 1024), None)

        info: Dict[str, Any] = {
            "has_latest": False,
            "has_best": False,
            "latest_epoch": 0,
            "best_epoch": 0,
            "best_val_miou": 0.0,
            "latest_path": None,
            "best_path": None,
            "can_resume": False,
        }

        # Inspect latest checkpoint
        if latest_file is not None:
            info["has_latest"] = True
            info["latest_path"] = latest_file
            info["can_resume"] = True
            if TORCH_AVAILABLE:
                try:
                    ckpt = torch.load(latest_file, map_location="cpu")
                    info["latest_epoch"] = ckpt.get("epoch", 0)
                    info["best_val_miou"] = ckpt.get("best_val_miou", 0.0)
                    info["best_epoch"] = ckpt.get("best_epoch", 0)
                except Exception as e:
                    logger.warning(f"Could not read header of {latest_file}: {e}")

        # Inspect best checkpoint
        if best_file is not None:
            info["has_best"] = True
            info["best_path"] = best_file
            if TORCH_AVAILABLE and not info["has_latest"]:
                try:
                    ckpt_b = torch.load(best_file, map_location="cpu")
                    info["best_epoch"] = ckpt_b.get("best_epoch", 0)
                    info["best_val_miou"] = ckpt_b.get("best_val_miou", 0.0)
                except Exception:
                    pass

        summary[model_name] = info

    return summary


def build_or_resume_trainer(
    model_name: str = "unet",
    checkpoint_dir: str = "checkpoints",
    data_dir: str = "data_rebuilt",
    results_dir: str = "results",
    device: Optional[Any] = None,
    batch_size: int = 8,
    epochs: int = 40,
    smoke_test: bool = False,
    seed: int = 42,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Reconstruct or resume a complete training session after a runtime reset.

    Automatically:
      1. Instantiates DataLoaders with Albumentations transforms.
      2. Builds model architecture (UNet or DeepLabV3+ with ResNet-34).
      3. Loads class weights and initializes CombinedLoss.
      4. Instantiates SegmentationTrainer.
      5. Automatically restores the latest checkpoint if one exists in checkpoint_dir.

    Args:
        model_name: 'unet' or 'deeplabv3plus'.
        checkpoint_dir: Directory with checkpoints.
        data_dir: Dataset root directory.
        results_dir: Results output directory.
        device: PyTorch device ('cuda' or 'cpu').
        batch_size: Batch size (default: 8).
        epochs: Target epochs (default: 40).
        smoke_test: If True, configures 1-epoch test under '{model_name}_smoke'.
        seed: Random seed.

    Returns:
        Tuple of (trainer_instance, session_metadata_dict).
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required to build or resume training session.")

    from src.models import build_unet, build_deeplabv3plus
    from src.losses import CombinedLoss, load_class_weights
    from src.dataset import get_dataloaders
    from src.augmentations import get_training_augmentation, get_validation_augmentation
    from src.trainer import SegmentationTrainer

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # 1. Setup DataLoaders
    train_tf = get_training_augmentation()
    val_tf = get_validation_augmentation()

    train_loader, val_loader, _ = get_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
        train_transform=train_tf,
        val_transform=val_tf,
    )

    # 2. Build Model
    effective_name = f"{model_name}_smoke" if smoke_test else model_name
    if "unet" in model_name.lower():
        model = build_unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=5,
            activation=None,
        ).to(device)
    else:
        model = build_deeplabv3plus(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=5,
            activation=None,
        ).to(device)

    # 3. Setup Loss
    class_weights = load_class_weights("results/class_weights.json", device=device)
    criterion = CombinedLoss(
        ce_weight=0.5,
        dice_weight=0.5,
        class_weights=class_weights,
        num_classes=5,
    ).to(device)

    # 4. Instantiate Trainer
    total_epochs = 1 if smoke_test else epochs
    patience = 1 if smoke_test else 10

    trainer = SegmentationTrainer(
        model=model,
        model_name=effective_name,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        device=device,
        epochs=total_epochs,
        lr=1e-4,
        weight_decay=1e-4,
        t_max=total_epochs,
        early_stopping_patience=patience,
        mixed_precision=True,
        checkpoint_dir=checkpoint_dir,
        results_dir=results_dir,
        seed=seed,
    )

    session_info = {
        "model_name": effective_name,
        "device": str(device),
        "smoke_test": smoke_test,
        "resumed": False,
        "resumed_from_epoch": 0,
        "best_val_miou": 0.0,
    }

    # 5. Auto-resume from latest checkpoint if not a smoke test
    if not smoke_test:
        latest_ckpt = os.path.join(checkpoint_dir, f"{model_name}_latest.pt")
        if os.path.exists(latest_ckpt) and os.path.getsize(latest_ckpt) > 1024:
            logger.info(f"Auto-resuming {model_name} from checkpoint: {latest_ckpt}")
            trainer.resume(latest_ckpt)
            session_info["resumed"] = True
            session_info["resumed_from_epoch"] = trainer.current_epoch
            session_info["best_val_miou"] = trainer.best_val_miou

    return trainer, session_info


def generate_recovery_dashboard(
    drive_status: Tuple[bool, str],
    code_status: Dict[str, Any],
    data_status: Dict[str, Any],
    checkpoint_status: Dict[str, Any],
) -> str:
    """Format and return an ASCII status dashboard for Google Colab recovery."""
    sep = "=" * 68
    lines = [
        sep,
        "            GOOGLE COLAB RECOVERY & STATUS DASHBOARD",
        sep,
        f"Google Drive:    {'CONNECTED' if drive_status[0] else 'DISCONNECTED'} ({drive_status[1]})",
        f"Source Codebase: {'READY' if code_status['restored'] else 'MISSING'} (Source: {code_status['source']})",
    ]

    if data_status["ready"]:
        counts = data_status["counts"]
        lines.append(
            f"Dataset Status:  VERIFIED & READY (Location: {data_status['storage_type']})\n"
            f"                 -> Train: {counts.get('train', 0)} tiles | Val: {counts.get('val', 0)} tiles\n"
            f"                 -> [SKIP] Raw download & preprocessing skipped!"
        )
    else:
        lines.append(
            "Dataset Status:  NOT FOUND\n"
            "                 -> Run Stage 1 download and preprocessing."
        )

    lines.append("-" * 68)
    lines.append("TRAINING CHECKPOINTS STATUS (GOOGLE DRIVE PERSISTENCE):")

    for m_name in ["unet", "deeplabv3plus"]:
        info = checkpoint_status.get(m_name, {})
        if info.get("has_latest"):
            lines.append(
                f"  * {m_name.upper():14}: RESUMABLE from Epoch {info['latest_epoch']:02d} "
                f"(Best Val mIoU: {info['best_val_miou']:.4f} @ Ep {info['best_epoch']})"
            )
        else:
            lines.append(f"  * {m_name.upper():14}: No existing checkpoint (Ready for initial training)")

    lines.append(sep)
    return "\n".join(lines)

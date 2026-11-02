"""
Command-line training script for Urban Land Cover Mapping semantic segmentation.
Usage:
    python src/train.py --model unet --epochs 40 --batch_size 8
    python src/train.py --model deeplabv3plus --epochs 40 --batch_size 8
    python src/train.py --model both --epochs 40 --batch_size 8
    python src/train.py --resume checkpoints/unet_latest.pt
"""

import os
import argparse
import logging

from src.utils import logger, set_seed
from src.dataset import get_dataloaders
from src.augmentations import get_training_augmentation, get_validation_augmentation
from src.models import build_unet, build_deeplabv3plus
from src.losses import CombinedLoss, load_class_weights
from src.trainer import SegmentationTrainer

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def train_single_model(
    model_name: str,
    train_loader: Any,
    val_loader: Any,
    device: Any,
    class_weights: Any,
    epochs: int = 40,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    patience: int = 10,
    checkpoints_dir: str = "checkpoints",
    results_dir: str = "results",
    resume_path: str = None,
    seed: int = 42,
) -> dict:
    """Train a single segmentation architecture."""
    logger.info(f"\n{'='*30} TRAINING {model_name.upper()} {'='*30}")

    if model_name.lower() == "unet":
        model = build_unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=5,
            activation=None,
        )
    elif model_name.lower() in ("deeplabv3plus", "deeplabv3+"):
        model = build_deeplabv3plus(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=5,
            activation=None,
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    criterion = CombinedLoss(
        ce_weight=0.5,
        dice_weight=0.5,
        class_weights=class_weights,
        num_classes=5,
    )

    trainer = SegmentationTrainer(
        model=model,
        model_name=model_name.lower(),
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        device=device,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        t_max=epochs,
        early_stopping_patience=patience,
        mixed_precision=True,
        checkpoint_dir=checkpoints_dir,
        results_dir=results_dir,
        seed=seed,
    )

    if resume_path and os.path.exists(resume_path):
        trainer.resume(resume_path)

    results = trainer.fit()
    return results


def main():
    parser = argparse.ArgumentParser(description="Train LandCover Semantic Segmentation Models")
    parser.add_argument("--model", type=str, default="both", choices=["unet", "deeplabv3plus", "both"])
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--checkpoints_dir", type=str, default="checkpoints")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required to run model training.")

    set_seed(args.seed)

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Target execution device: {device}")

    # Build augmentations & DataLoaders
    train_tf = get_training_augmentation()
    val_tf = get_validation_augmentation()

    train_loader, val_loader, _ = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
        train_transform=train_tf,
        val_transform=val_tf,
    )

    class_weights = load_class_weights("results/class_distribution.json", device=device)

    models_to_train = ["unet", "deeplabv3plus"] if args.model == "both" else [args.model]

    for model_name in models_to_train:
        train_single_model(
            model_name=model_name,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            class_weights=class_weights,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
            checkpoints_dir=args.checkpoints_dir,
            results_dir=args.results_dir,
            resume_path=args.resume,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()


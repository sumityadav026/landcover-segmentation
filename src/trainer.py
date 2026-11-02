"""
Training engine for Urban Land Cover Mapping semantic segmentation.
Implements the reusable SegmentationTrainer supporting:
  - AdamW optimizer & CosineAnnealingLR scheduler
  - Combined Loss: 0.5 * Weighted CE + 0.5 * Multiclass Dice
  - Automatic Mixed Precision (AMP) & GradScaler
  - Epoch-wise validation mIoU evaluation
  - Resumable checkpointing (latest vs best)
  - Early stopping with configurable patience (10 epochs)
  - Wall-clock epoch timing & training history logging
  - Graceful CUDA out-of-memory handling
"""

import os
import time
import json
import logging
from typing import Dict, Any, Optional, Tuple, List, Union

import numpy as np

# Import project components
from src.utils import logger, set_seed, save_json
from src.metrics import MetricTracker
from src.losses import CombinedLoss, load_class_weights

try:
    import torch
    import torch.nn as nn
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from torch.cuda.amp import autocast, GradScaler
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class SegmentationTrainer:
    """
    Standardized trainer for segmentation models (UNet and DeepLabV3+).
    Guarantees 100% identical training conditions and transparent tracking.
    """

    def __init__(
        self,
        model: Any,
        model_name: str,
        train_loader: Any,
        val_loader: Any,
        criterion: Optional[Any] = None,
        optimizer: Optional[Any] = None,
        scheduler: Optional[Any] = None,
        device: Optional[Any] = None,
        epochs: int = 40,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        t_max: int = 40,
        eta_min: float = 1e-6,
        early_stopping_patience: int = 10,
        mixed_precision: bool = True,
        max_grad_norm: float = 1.0,
        checkpoint_dir: str = "checkpoints",
        results_dir: str = "results",
        seed: int = 42,
    ):
        """
        Initialize the trainer.
        """
        set_seed(seed)
        self.model_name = model_name
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.max_grad_norm = max_grad_norm
        self.checkpoint_dir = checkpoint_dir
        self.results_dir = results_dir

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

        # Device configuration
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Mixed precision setup
        self.use_amp = mixed_precision and (self.device.type == "cuda")
        self.scaler = GradScaler(enabled=self.use_amp)

        # Loss function
        if criterion is not None:
            self.criterion = criterion.to(self.device)
        else:
            class_weights = load_class_weights("results/class_distribution.json", device=self.device)
            self.criterion = CombinedLoss(
                ce_weight=0.5,
                dice_weight=0.5,
                class_weights=class_weights,
                num_classes=5,
            ).to(self.device)

        # Optimizer: AdamW
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )

        # Learning Rate Scheduler: CosineAnnealingLR
        if scheduler is not None:
            self.scheduler = scheduler
        else:
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=t_max,
                eta_min=eta_min,
            )

        # Tracking state
        self.current_epoch = 0
        self.best_val_miou = 0.0
        self.best_epoch = 0
        self.patience_counter = 0

        self.history: Dict[str, List[Any]] = {
            "epoch": [],
            "train_loss": [],
            "train_ce_loss": [],
            "train_dice_loss": [],
            "val_loss": [],
            "val_ce_loss": [],
            "val_dice_loss": [],
            "val_miou": [],
            "val_miou_no_bg": [],
            "val_pixel_accuracy": [],
            "lr": [],
            "epoch_duration_sec": [],
        }

        logger.info(
            f"Initialized SegmentationTrainer for [{model_name}] on [{self.device}] | "
            f"AMP: {self.use_amp} | Epochs: {epochs} | Patience: {early_stopping_patience}"
        )

    def train_one_epoch(self) -> Dict[str, float]:
        """
        Execute one training epoch over train_loader with mixed precision.

        Returns:
            Dictionary with average train_loss, ce_loss, and dice_loss.
        """
        self.model.train()
        total_loss = 0.0
        total_ce = 0.0
        total_dice = 0.0
        num_batches = len(self.train_loader)

        for batch_idx, (images, masks) in enumerate(self.train_loader):
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            try:
                # Forward pass with Automatic Mixed Precision
                with autocast(enabled=self.use_amp):
                    logits = self.model(images)
                    loss, metrics = self.criterion(logits, masks, return_dict=True)

                # Backward pass with GradScaler
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                    if self.max_grad_norm > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    if self.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()

                total_loss += metrics["loss"]
                total_ce += metrics["ce_loss"]
                total_dice += metrics["dice_loss"]

            except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                if "CUDA out of memory" in str(e) or isinstance(e, torch.cuda.OutOfMemoryError):
                    logger.error(
                        f"CUDA Out-Of-Memory encountered during training epoch {self.current_epoch + 1}, "
                        f"batch {batch_idx + 1}! Cleaning cache..."
                    )
                    torch.cuda.empty_cache()
                    raise RuntimeError(
                        "CUDA OOM detected! Please reduce batch size from 8 to 4 in configs/config.yaml."
                    ) from e
                else:
                    raise e

        avg_loss = total_loss / max(1, num_batches)
        avg_ce = total_ce / max(1, num_batches)
        avg_dice = total_dice / max(1, num_batches)

        return {
            "loss": avg_loss,
            "ce_loss": avg_ce,
            "dice_loss": avg_dice,
        }

    @torch.no_grad()
    def validate(self) -> Dict[str, Any]:
        """
        Execute evaluation over val_loader, calculating loss and global confusion matrix mIoU.

        Returns:
            Dictionary with validation loss and mIoU metrics.
        """
        self.model.eval()
        total_loss = 0.0
        total_ce = 0.0
        total_dice = 0.0
        num_batches = len(self.val_loader)

        tracker = MetricTracker(num_classes=5)

        for images, masks in self.val_loader:
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)

            with autocast(enabled=self.use_amp):
                logits = self.model(images)
                loss, metrics = self.criterion(logits, masks, return_dict=True)

            total_loss += metrics["loss"]
            total_ce += metrics["ce_loss"]
            total_dice += metrics["dice_loss"]

            # Accumulate global confusion matrix
            preds = torch.argmax(logits, dim=1)
            tracker.update(preds, masks)

        eval_metrics = tracker.compute()
        eval_metrics["loss"] = total_loss / max(1, num_batches)
        eval_metrics["ce_loss"] = total_ce / max(1, num_batches)
        eval_metrics["dice_loss"] = total_dice / max(1, num_batches)

        return eval_metrics

    def save_checkpoint(
        self,
        is_best: bool = False,
        extra_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Save training checkpoint to disk / Google Drive.

        Saves:
          1. 'latest_checkpoint.pt': saved after every single epoch for interruption resumption.
          2. 'best_checkpoint.pt': saved ONLY when validation mIoU improves over prior best.
        """
        checkpoint_state = {
            "epoch": self.current_epoch,
            "model_name": self.model_name,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "scaler_state_dict": self.scaler.state_dict() if self.use_amp else None,
            "best_val_miou": self.best_val_miou,
            "best_epoch": self.best_epoch,
            "patience_counter": self.patience_counter,
            "history": self.history,
            "extra_info": extra_info or {},
        }

        # 1. Always save latest checkpoint
        latest_path = os.path.join(self.checkpoint_dir, f"{self.model_name}_latest.pt")
        torch.save(checkpoint_state, latest_path)

        # 2. Save best checkpoint strictly when improved
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, f"{self.model_name}_best.pt")
            torch.save(checkpoint_state, best_path)
            logger.info(
                f"[{self.model_name}] New best checkpoint saved to {best_path} "
                f"(Val mIoU: {self.best_val_miou:.5f} at Epoch {self.current_epoch})"
            )

    def resume(self, checkpoint_path: str) -> None:
        """
        Resume training from a saved checkpoint file.
        Restores model weights, optimizer, scheduler, scaler, epoch index, and history.

        Args:
            checkpoint_path: Path to checkpoint .pt file.
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

        logger.info(f"Loading checkpoint for [{self.model_name}] from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.scheduler and checkpoint.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if self.use_amp and checkpoint.get("scaler_state_dict"):
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        self.current_epoch = checkpoint["epoch"]
        self.best_val_miou = checkpoint.get("best_val_miou", 0.0)
        self.best_epoch = checkpoint.get("best_epoch", 0)
        self.patience_counter = checkpoint.get("patience_counter", 0)
        self.history = checkpoint.get("history", self.history)

        logger.info(
            f"Successfully resumed [{self.model_name}] from Epoch {self.current_epoch} | "
            f"Best Val mIoU: {self.best_val_miou:.5f} (Epoch {self.best_epoch})"
        )

    def fit(self, target_epochs: Optional[int] = None) -> Dict[str, Any]:
        """
        Execute full training loop over epochs with early stopping.

        Args:
            target_epochs: Optional override for total epochs (defaults to self.epochs).

        Returns:
            Dictionary with training history and best performance metrics.
        """
        total_epochs = target_epochs or self.epochs
        logger.info(
            f"Starting training for [{self.model_name}] from Epoch {self.current_epoch + 1} "
            f"to {total_epochs}..."
        )

        overall_start_time = time.time()

        while self.current_epoch < total_epochs:
            self.current_epoch += 1
            epoch_start_time = time.time()
            current_lr = self.optimizer.param_groups[0]["lr"]

            # 1. Training Phase
            train_results = self.train_one_epoch()

            # 2. Validation Phase
            val_results = self.validate()
            val_miou = val_results["miou"]

            # 3. Learning Rate Scheduler Step
            if self.scheduler is not None:
                self.scheduler.step()

            epoch_duration = time.time() - epoch_start_time

            # 4. Check for Improvement and Best Checkpoint
            is_best = val_miou > self.best_val_miou
            if is_best:
                self.best_val_miou = val_miou
                self.best_epoch = self.current_epoch
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            # 5. Log History
            self.history["epoch"].append(self.current_epoch)
            self.history["train_loss"].append(round(train_results["loss"], 5))
            self.history["train_ce_loss"].append(round(train_results["ce_loss"], 5))
            self.history["train_dice_loss"].append(round(train_results["dice_loss"], 5))
            self.history["val_loss"].append(round(val_results["loss"], 5))
            self.history["val_ce_loss"].append(round(val_results["ce_loss"], 5))
            self.history["val_dice_loss"].append(round(val_results["dice_loss"], 5))
            self.history["val_miou"].append(round(val_miou, 5))
            self.history["val_miou_no_bg"].append(round(val_results["miou_no_bg"], 5))
            self.history["val_pixel_accuracy"].append(round(val_results["pixel_accuracy"], 5))
            self.history["lr"].append(current_lr)
            self.history["epoch_duration_sec"].append(round(epoch_duration, 2))

            # 6. Save Checkpoint after every epoch
            self.save_checkpoint(is_best=is_best, extra_info={"val_metrics": val_results})

            # 7. Print Progress
            logger.info(
                f"Epoch [{self.current_epoch:02d}/{total_epochs:02d}] "
                f"Train Loss: {train_results['loss']:.4f} | "
                f"Val Loss: {val_results['loss']:.4f} | "
                f"Val mIoU: {val_miou:.4f} (Best: {self.best_val_miou:.4f} @ Ep {self.best_epoch}) | "
                f"LR: {current_lr:.2e} | Time: {epoch_duration:.1f}s"
            )

            # 8. Early Stopping Check
            if self.patience_counter >= self.early_stopping_patience:
                logger.warning(
                    f"[{self.model_name}] Early stopping triggered at Epoch {self.current_epoch}! "
                    f"No improvement for {self.early_stopping_patience} consecutive epochs. "
                    f"Best Val mIoU: {self.best_val_miou:.5f} (Epoch {self.best_epoch})."
                )
                break

        total_time = time.time() - overall_start_time
        logger.info(
            f"Training completed for [{self.model_name}] in {total_time / 60:.2f} minutes | "
            f"Best Val mIoU: {self.best_val_miou:.5f} at Epoch {self.best_epoch}"
        )

        # Save history JSON
        history_path = os.path.join(self.results_dir, f"{self.model_name}_training_history.json")
        save_json(self.history, history_path)

        return {
            "model_name": self.model_name,
            "best_epoch": self.best_epoch,
            "best_val_miou": self.best_val_miou,
            "total_time_seconds": round(total_time, 2),
            "history": self.history,
        }

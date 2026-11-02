"""
Evaluation metrics for Urban Land Cover Mapping semantic segmentation.
Implements:
  1. Confusion Matrix Accumulator
  2. Per-Class Intersection over Union (IoU)
  3. Mean Intersection over Union (mIoU)
  4. Mean IoU excluding Background
  5. Pixel Accuracy
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional, List, Union

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class MetricTracker:
    """
    Streaming metric tracker that accumulates a confusion matrix across batches
    and computes global IoU, mIoU, and Pixel Accuracy at the end of an evaluation epoch.
    """

    def __init__(
        self,
        num_classes: int = 5,
        class_names: Optional[Dict[int, str]] = None,
        ignore_index: Optional[int] = None,
    ):
        """
        Args:
            num_classes: Number of discrete categories (5).
            class_names: Dictionary mapping class ID -> human-readable string.
            ignore_index: Optional class ID to exclude from accumulation.
        """
        self.num_classes = num_classes
        self.class_names = class_names or {
            0: "Background",
            1: "Building",
            2: "Woodland",
            3: "Water",
            4: "Road",
        }
        self.ignore_index = ignore_index
        self.reset()

    def reset(self) -> None:
        """Clear the accumulated confusion matrix."""
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        self.total_samples = 0

    def update(
        self,
        preds: Union[np.ndarray, "torch.Tensor"],
        targets: Union[np.ndarray, "torch.Tensor"],
    ) -> None:
        """
        Accumulate confusion matrix from predictions and ground-truth targets.

        Args:
            preds: Predicted class IDs [B, H, W] or logits [B, C, H, W].
            targets: Ground-truth class IDs [B, H, W].
        """
        if TORCH_AVAILABLE and isinstance(preds, torch.Tensor):
            if preds.ndim == 4:
                # If logits [B, C, H, W], take argmax over class dimension
                preds = torch.argmax(preds, dim=1)
            preds_np = preds.detach().cpu().numpy().astype(np.int64)
        else:
            preds_np = np.array(preds, dtype=np.int64)
            if preds_np.ndim == 4:
                preds_np = np.argmax(preds_np, axis=1)

        if TORCH_AVAILABLE and isinstance(targets, torch.Tensor):
            targets_np = targets.detach().cpu().numpy().astype(np.int64)
        else:
            targets_np = np.array(targets, dtype=np.int64)

        if preds_np.shape != targets_np.shape:
            raise ValueError(
                f"Prediction shape {preds_np.shape} does not match target shape {targets_np.shape}"
            )

        # Flatten arrays
        p_flat = preds_np.flatten()
        t_flat = targets_np.flatten()

        # Filter ignore_index or invalid labels if specified
        valid_mask = (t_flat >= 0) & (t_flat < self.num_classes)
        if self.ignore_index is not None:
            valid_mask &= (t_flat != self.ignore_index)

        p_valid = p_flat[valid_mask]
        t_valid = t_flat[valid_mask]

        # Fast bincount confusion matrix accumulation
        # index = target * num_classes + pred
        indices = t_valid * self.num_classes + p_valid
        counts = np.bincount(indices, minlength=self.num_classes ** 2)
        self.confusion_matrix += counts.reshape((self.num_classes, self.num_classes))
        self.total_samples += len(preds_np)

    def compute(self) -> Dict[str, Any]:
        """
        Compute global metrics accumulated over all batches.

        Returns:
            Dictionary containing:
              - 'confusion_matrix': [num_classes, num_classes] ndarray
              - 'per_class_iou': dict mapping class_name -> float IoU
              - 'miou': mean IoU across all classes
              - 'miou_no_bg': mean IoU excluding background (class 0)
              - 'pixel_accuracy': overall fraction of correctly classified pixels
        """
        cm = self.confusion_matrix
        total_pixels = cm.sum()

        if total_pixels == 0:
            return {
                "confusion_matrix": cm,
                "per_class_iou": {name: 0.0 for name in self.class_names.values()},
                "miou": 0.0,
                "miou_no_bg": 0.0,
                "pixel_accuracy": 0.0,
            }

        # True Positives: diagonal elements M[c, c]
        tp = np.diag(cm).astype(np.float64)
        # False Positives: column sum - diagonal
        fp = cm.sum(axis=0).astype(np.float64) - tp
        # False Negatives: row sum - diagonal
        fn = cm.sum(axis=1).astype(np.float64) - tp

        denominator = tp + fp + fn
        per_class_iou = {}
        valid_ious = []
        valid_ious_no_bg = []

        for c in range(self.num_classes):
            name = self.class_names.get(c, f"Class_{c}")
            if denominator[c] > 0:
                iou_c = float(tp[c] / denominator[c])
            else:
                # Class not present in ground truth or predictions
                iou_c = 1.0 if (tp[c] == 0 and fp[c] == 0 and fn[c] == 0) else 0.0

            per_class_iou[name] = round(iou_c, 5)
            valid_ious.append(iou_c)
            if c > 0:  # Exclude background (class 0)
                valid_ious_no_bg.append(iou_c)

        miou = float(np.mean(valid_ious)) if valid_ious else 0.0
        miou_no_bg = float(np.mean(valid_ious_no_bg)) if valid_ious_no_bg else 0.0
        pixel_acc = float(tp.sum() / total_pixels)

        return {
            "confusion_matrix": cm,
            "per_class_iou": per_class_iou,
            "miou": round(miou, 5),
            "miou_no_bg": round(miou_no_bg, 5),
            "pixel_accuracy": round(pixel_acc, 5),
        }

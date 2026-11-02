"""
Loss functions for Urban Land Cover Mapping semantic segmentation.
Implements:
  1. Class-Weighted Cross Entropy Loss
  2. Multiclass Dice Loss
  3. Combined Loss: 0.5 * Weighted CE + 0.5 * Multiclass Dice
"""

import json
import os
import logging
from typing import Optional, Dict, Any, Union, Tuple, List
import numpy as np

logger = logging.getLogger("LandCover.Losses")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    nn = object  # type: ignore


def load_class_weights(
    weights_path: str = "results/class_distribution.json",
    device: Optional[Any] = None,
) -> Any:
    """
    Load class weights from Stage 1 dataset statistics.

    Args:
        weights_path: Path to class_distribution.json or dataset_statistics.json.
        device: PyTorch device ('cuda' or 'cpu').

    Returns:
        PyTorch FloatTensor of shape [5] or list of float weights.
    """
    if not os.path.exists(weights_path):
        logger.warning(f"Class weights file not found at {weights_path}. Using uniform weights.")
        weights = [1.0] * 5
    else:
        with open(weights_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Handle class_distribution.json structure or dataset_statistics.json structure
        if "class_weights_ce" in data:
            weight_dict = data["class_weights_ce"]
        elif "class_distribution" in data:
            weight_dict = {k: v["ce_weight"] for k, v in data["class_distribution"].items()}
        elif "0" in data and isinstance(data["0"], dict) and "ce_weight" in data["0"]:
            weight_dict = {k: v["ce_weight"] for k, v in data.items()}
        else:
            weight_dict = data

        weights = [float(weight_dict[str(i)]) for i in range(5)]

    logger.info(f"Loaded class weights: {weights}")

    if TORCH_AVAILABLE:
        tensor_weights = torch.tensor(weights, dtype=torch.float32)
        if device is not None:
            tensor_weights = tensor_weights.to(device)
        return tensor_weights

    return weights


if TORCH_AVAILABLE:
    class MulticlassDiceLoss(nn.Module):
        """
        Multiclass Dice Loss for semantic segmentation.
        Computes soft Dice coefficient across all classes from unnormalized logits.
        """

        def __init__(
            self,
            num_classes: int = 5,
            smooth: float = 1e-5,
            ignore_index: Optional[int] = None,
        ):
            """
            Args:
                num_classes: Number of segmentation categories (5).
                smooth: Smoothing factor to avoid division by zero.
                ignore_index: Class ID to ignore in loss calculation, if any.
            """
            super().__init__()
            self.num_classes = num_classes
            self.smooth = smooth
            self.ignore_index = ignore_index

        def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            """
            Compute multiclass Dice loss.

            Args:
                logits: Unnormalized network output [B, C, H, W].
                targets: Ground truth integer class IDs [B, H, W].

            Returns:
                Scalar loss tensor.
            """
            if logits.ndim != 4:
                raise ValueError(f"Expected 4D logits [B, C, H, W], got shape {logits.shape}")
            if targets.ndim != 3:
                raise ValueError(f"Expected 3D targets [B, H, W], got shape {targets.shape}")

            # Apply softmax along channel dimension
            probs = F.softmax(logits, dim=1)

            # Convert targets to one-hot encoding [B, C, H, W]
            targets_one_hot = F.one_hot(targets, num_classes=self.num_classes)
            # Permute from [B, H, W, C] to [B, C, H, W]
            targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()

            # Flatten spatial dimensions: [B, C, H*W]
            probs_flat = probs.view(probs.size(0), self.num_classes, -1)
            targets_flat = targets_one_hot.view(targets_one_hot.size(0), self.num_classes, -1)

            # Intersection and Cardinality per class
            intersection = (probs_flat * targets_flat).sum(dim=2)
            cardinality = probs_flat.sum(dim=2) + targets_flat.sum(dim=2)

            dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)

            if self.ignore_index is not None and 0 <= self.ignore_index < self.num_classes:
                # Mask out ignored class
                valid_mask = [i for i in range(self.num_classes) if i != self.ignore_index]
                dice_score = dice_score[:, valid_mask]

            # Mean Dice loss over classes and batch
            dice_loss = 1.0 - dice_score.mean()
            return dice_loss


    class CombinedLoss(nn.Module):
        """
        Combined loss function for urban semantic segmentation:
          L_total = ce_weight * L_CE + dice_weight * L_Dice
        Default:
          0.5 * Class-Weighted CrossEntropy + 0.5 * Multiclass Dice
        """

        def __init__(
            self,
            ce_weight: float = 0.5,
            dice_weight: float = 0.5,
            class_weights: Optional[Union[torch.Tensor, List[float]]] = None,
            num_classes: int = 5,
            smooth: float = 1e-5,
            ignore_index: Optional[int] = None,
        ):
            """
            Args:
                ce_weight: Weight multiplier for CrossEntropy loss (0.5).
                dice_weight: Weight multiplier for Dice loss (0.5).
                class_weights: Tensor or list of 5 class weights for CrossEntropy.
                num_classes: Number of semantic classes (5).
                smooth: Smoothing epsilon for Dice denominator.
                ignore_index: Optional class ID to ignore in loss calculation.
            """
            super().__init__()
            self.ce_weight = ce_weight
            self.dice_weight = dice_weight
            self.num_classes = num_classes

            # Register class weights for CrossEntropy
            if class_weights is not None:
                if not isinstance(class_weights, torch.Tensor):
                    class_weights = torch.tensor(class_weights, dtype=torch.float32)
                self.register_buffer("class_weights", class_weights)
            else:
                self.class_weights = None

            # CrossEntropy loss expects [B, C, H, W] logits and [B, H, W] target
            self.ce_loss_fn = nn.CrossEntropyLoss(
                weight=self.class_weights,
                ignore_index=-100 if ignore_index is None else ignore_index,
            )
            # Multiclass Dice loss
            self.dice_loss_fn = MulticlassDiceLoss(
                num_classes=num_classes,
                smooth=smooth,
                ignore_index=ignore_index,
            )

        def forward(
            self,
            logits: torch.Tensor,
            targets: torch.Tensor,
            return_dict: bool = False,
        ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, float]]]:
            """
            Calculate combined weighted loss.

            Args:
                logits: Unnormalized model predictions [B, C, H, W].
                targets: Integer ground truth masks [B, H, W].
                return_dict: If True, returns (total_loss, metrics_dict).

            Returns:
                total_loss scalar tensor, or tuple with loss breakdown.
            """
            # Ensure target is long dtype
            if targets.dtype != torch.long:
                targets = targets.long()

            ce_loss = self.ce_loss_fn(logits, targets)
            dice_loss = self.dice_loss_fn(logits, targets)

            total_loss = (self.ce_weight * ce_loss) + (self.dice_weight * dice_loss)

            if return_dict:
                metrics = {
                    "loss": total_loss.item(),
                    "ce_loss": ce_loss.item(),
                    "dice_loss": dice_loss.item(),
                }
                return total_loss, metrics

            return total_loss

else:
    # Minimal fallbacks if PyTorch is not yet installed
    class MulticlassDiceLoss:  # type: ignore
        pass

    class CombinedLoss:  # type: ignore
        pass


# Pure NumPy implementation for offline verification and testing
def numpy_combined_loss(
    logits: np.ndarray,
    targets: np.ndarray,
    class_weights: Optional[List[float]] = None,
    ce_weight: float = 0.5,
    dice_weight: float = 0.5,
    num_classes: int = 5,
    smooth: float = 1e-5,
) -> Dict[str, float]:
    """
    Calculate combined loss in pure NumPy for environment validation and unit testing.

    Args:
        logits: Numpy array [B, C, H, W] of raw logits.
        targets: Numpy array [B, H, W] of class IDs in {0, 1, 2, 3, 4}.
        class_weights: List of 5 class weights.
        ce_weight: CE multiplier (0.5).
        dice_weight: Dice multiplier (0.5).
        num_classes: Number of classes (5).
        smooth: Epsilon smoothing.

    Returns:
        Dictionary with 'total_loss', 'ce_loss', and 'dice_loss'.
    """
    b, c, h, w = logits.shape
    weights = np.array(class_weights if class_weights is not None else [1.0] * c, dtype=np.float32)

    # 1. Softmax calculation with numerical stability
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    # 2. Cross Entropy Loss with class weighting
    ce_losses = []
    for bi in range(b):
        for hi in range(h):
            for wi in range(w):
                target_cls = int(targets[bi, hi, wi])
                p = max(probs[bi, target_cls, hi, wi], 1e-12)
                w_cls = weights[target_cls]
                ce_losses.append(-w_cls * np.log(p))
    ce_loss = float(np.mean(ce_losses))

    # 3. Multiclass Dice Loss
    dice_scores = []
    for cls_idx in range(num_classes):
        p_c = probs[:, cls_idx, :, :].flatten()
        y_c = (targets == cls_idx).astype(np.float32).flatten()
        intersection = np.sum(p_c * y_c)
        cardinality = np.sum(p_c) + np.sum(y_c)
        dice_c = (2.0 * intersection + smooth) / (cardinality + smooth)
        dice_scores.append(dice_c)
    dice_loss = float(1.0 - np.mean(dice_scores))

    total_loss = (ce_weight * ce_loss) + (dice_weight * dice_loss)
    return {
        "total_loss": float(total_loss),
        "ce_loss": float(ce_loss),
        "dice_loss": float(dice_loss),
    }


"""
Model architectures for Urban Land Cover Mapping semantic segmentation.
Implements UNet and DeepLabV3+ with pretrained ResNet-34 encoders using segmentation_models_pytorch,
along with parameter counting, architecture summaries, and VRAM estimation utilities.
"""

import logging
from typing import Dict, Any, Tuple, Optional

logger = logging.getLogger("LandCover.Models")

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    nn = object  # type: ignore

try:
    import segmentation_models_pytorch as smp
    SMP_AVAILABLE = True
except ImportError:
    SMP_AVAILABLE = False


def build_unet(
    encoder_name: str = "resnet34",
    encoder_weights: Optional[str] = "imagenet",
    in_channels: int = 3,
    classes: int = 5,
    activation: Optional[str] = None,
) -> Any:
    """
    Construct UNet model with ResNet-34 encoder and skip connections.

    Args:
        encoder_name: Name of encoder backbone (default: 'resnet34').
        encoder_weights: Pretrained weights (default: 'imagenet').
        in_channels: Input image channels (default: 3 for RGB).
        classes: Number of output segmentation classes (default: 5).
        activation: Output activation ('softmax', 'sigmoid', or None for raw logits).

    Returns:
        smp.Unet model instance.
    """
    if not SMP_AVAILABLE:
        raise ImportError(
            "segmentation_models_pytorch is required to build UNet. "
            "Please run: pip install segmentation-models-pytorch"
        )

    logger.info(
        f"Building UNet (Encoder: {encoder_name}, Weights: {encoder_weights}, "
        f"In channels: {in_channels}, Classes: {classes})"
    )
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        activation=activation,
    )
    return model


def build_deeplabv3plus(
    encoder_name: str = "resnet34",
    encoder_weights: Optional[str] = "imagenet",
    in_channels: int = 3,
    classes: int = 5,
    activation: Optional[str] = None,
) -> Any:
    """
    Construct DeepLabV3+ model with ResNet-34 encoder and ASPP decoder.

    Args:
        encoder_name: Name of encoder backbone (default: 'resnet34').
        encoder_weights: Pretrained weights (default: 'imagenet').
        in_channels: Input image channels (default: 3 for RGB).
        classes: Number of output segmentation classes (default: 5).
        activation: Output activation ('softmax', 'sigmoid', or None for raw logits).

    Returns:
        smp.DeepLabV3Plus model instance.
    """
    if not SMP_AVAILABLE:
        raise ImportError(
            "segmentation_models_pytorch is required to build DeepLabV3Plus. "
            "Please run: pip install segmentation-models-pytorch"
        )

    logger.info(
        f"Building DeepLabV3+ (Encoder: {encoder_name}, Weights: {encoder_weights}, "
        f"In channels: {in_channels}, Classes: {classes})"
    )
    model = smp.DeepLabV3Plus(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        activation=activation,
    )
    return model


def get_model(
    model_name: str,
    encoder_name: str = "resnet34",
    encoder_weights: Optional[str] = "imagenet",
    in_channels: int = 3,
    classes: int = 5,
    activation: Optional[str] = None,
) -> Any:
    """
    Factory function to instantiate segmentation models by name.

    Args:
        model_name: Architecture name ('unet' or 'deeplabv3plus' / 'deeplabv3+').
        encoder_name: Backbone encoder name.
        encoder_weights: Pretrained weights name.
        in_channels: Input image channels (3).
        classes: Output segmentation classes (5).
        activation: Activation function for output logits.

    Returns:
        Instantiated PyTorch segmentation model.
    """
    name_clean = model_name.lower().replace("+", "plus").replace("-", "").replace("_", "")
    if "unet" in name_clean:
        return build_unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=activation,
        )
    elif "deeplab" in name_clean:
        return build_deeplabv3plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=activation,
        )
    else:
        raise ValueError(
            f"Unsupported model architecture: '{model_name}'. "
            f"Choose from: 'unet', 'deeplabv3plus'."
        )


def count_parameters(model: Any) -> Dict[str, int]:
    """
    Compute parameter statistics for a PyTorch model.

    Args:
        model: PyTorch nn.Module.

    Returns:
        Dictionary with 'total', 'trainable', and 'non_trainable' parameter counts.
    """
    if not TORCH_AVAILABLE or not hasattr(model, "parameters"):
        return {"total": 0, "trainable": 0, "non_trainable": 0}

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    return {
        "total": total_params,
        "trainable": trainable_params,
        "non_trainable": non_trainable_params,
    }


def estimate_model_memory(
    model: Any,
    batch_size: int = 8,
    input_resolution: Tuple[int, int] = (512, 512),
    in_channels: int = 3,
    precision_bytes: int = 4,  # 4 bytes for float32, 2 bytes for float16
) -> Dict[str, Any]:
    """
    Estimate theoretical VRAM requirement for training a model on GPU.

    Components estimated:
      1. Model Parameters (Weights & Biases)
      2. Gradients (Identical in size to trainable parameters)
      3. Optimizer States (AdamW requires 2 FP32 states per parameter: 8 bytes per param)
      4. Forward Activations (Empirical multiplier for 512x512 feature maps)

    Args:
        model: PyTorch nn.Module.
        batch_size: Training batch size (default: 8).
        input_resolution: (H, W) spatial dimensions (default: 512, 512).
        in_channels: Input channels (default: 3).
        precision_bytes: Bytes per parameter (4 for FP32, 2 for FP16).

    Returns:
        Dictionary with detailed memory estimates in Megabytes (MB) and Gigabytes (GB).
    """
    params_info = count_parameters(model)
    total_params = params_info["total"]
    trainable_params = params_info["trainable"]

    # 1. Weights Memory (MB)
    weights_mb = (total_params * precision_bytes) / (1024 ** 2)

    # 2. Gradients Memory (MB)
    gradients_mb = (trainable_params * precision_bytes) / (1024 ** 2)

    # 3. Optimizer State Memory (AdamW stores 1st and 2nd moments in FP32 = 8 bytes per param)
    optimizer_mb = (trainable_params * 8) / (1024 ** 2)

    # 4. Activation Memory (Rough empirical estimation for 512x512 ResNet-34 decoder)
    # High-resolution feature maps across UNet/ASPP decoder layers
    # Approximately 180-250 MB per sample at 512x512 in FP32, halved in FP16
    h, w = input_resolution
    scale_factor = (h * w) / (512 * 512)
    activation_per_sample_mb = 220.0 * scale_factor * (precision_bytes / 4.0)
    activations_mb = activation_per_sample_mb * batch_size

    # Total estimated training memory
    total_training_mb = weights_mb + gradients_mb + optimizer_mb + activations_mb
    total_training_gb = total_training_mb / 1024.0

    # Total estimated inference memory (weights + 1 forward activation)
    total_inference_mb = weights_mb + (activation_per_sample_mb * 1)
    total_inference_gb = total_inference_mb / 1024.0

    return {
        "batch_size": batch_size,
        "precision": "FP32" if precision_bytes == 4 else "FP16 (Mixed Precision)",
        "weights_mb": round(weights_mb, 2),
        "gradients_mb": round(gradients_mb, 2),
        "optimizer_mb": round(optimizer_mb, 2),
        "activations_mb": round(activations_mb, 2),
        "total_training_mb": round(total_training_mb, 2),
        "total_training_gb": round(total_training_gb, 2),
        "total_inference_mb": round(total_inference_mb, 2),
        "total_inference_gb": round(total_inference_gb, 2),
        "fits_colab_t4": total_training_gb < 14.5,  # Colab T4 has ~15 GB VRAM
    }

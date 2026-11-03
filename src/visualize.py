"""
Visualization Module for Urban Land Cover Mapping.
Provides:
  1. Qualitative Comparisons (Aerial Image, Ground Truth, Prediction, Overlays, Error Maps)
  2. Side-by-side Comparative Multi-Panel Visualizations (UNet vs DeepLabV3+)
  3. Quantitative Training Curves (Train/Val Loss, Validation mIoU, Learning Rate Schedule)
  4. Per-Class Performance Bar Charts (IoU, Precision, Recall, F1)
  5. Annotated Confusion Matrix Heatmaps
  6. Failure Case Analysis Visualizer
"""

import os
import json
import logging
from typing import Dict, Any, Optional, Tuple, List, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("LandCover.Visualize")

# Check Matplotlib availability
try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive headless backend
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.info("Matplotlib not found; using Pillow-based rendering engine for visualizations.")

from src.utils import mask_to_rgb, overlay_mask, DEFAULT_CLASS_COLORS


CLASS_NAMES_DEFAULT = {
    0: "Background",
    1: "Building",
    2: "Woodland",
    3: "Water",
    4: "Road",
}


def create_error_map(
    target: np.ndarray,
    pred: np.ndarray,
    correct_color: Tuple[int, int, int] = (40, 180, 40),    # Soft Green
    incorrect_color: Tuple[int, int, int] = (220, 40, 40), # Soft Red
) -> np.ndarray:
    """
    Generate an RGB error map showing matching (correct) vs mismatching (error) pixels.

    Args:
        target: 2D ground truth mask [H, W].
        pred: 2D prediction mask [H, W].
        correct_color: RGB tuple for correct pixels (default: Green).
        incorrect_color: RGB tuple for incorrect pixels (default: Red).

    Returns:
        3D RGB numpy array [H, W, 3] of dtype uint8.
    """
    h, w = target.shape
    error_map = np.zeros((h, w, 3), dtype=np.uint8)
    correct_mask = (target == pred)

    error_map[correct_mask] = correct_color
    error_map[~correct_mask] = incorrect_color
    return error_map


def plot_qualitative_sample(
    image: np.ndarray,
    target: np.ndarray,
    unet_pred: Optional[np.ndarray] = None,
    deeplab_pred: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    title: str = "Urban Land Cover Semantic Segmentation Comparison",
) -> Image.Image:
    """
    Generate a publication-grade multi-panel qualitative comparison figure:
      Panel 1: Original Aerial Image
      Panel 2: Ground Truth Annotation
      Panel 3: UNet Prediction (if provided)
      Panel 4: UNet Error Map (if provided)
      Panel 5: DeepLabV3+ Prediction (if provided)
      Panel 6: DeepLabV3+ Error Map (if provided)

    Args:
        image: RGB aerial image [H, W, 3], uint8.
        target: Ground truth mask [H, W], uint8/int.
        unet_pred: Predicted mask by UNet [H, W].
        deeplab_pred: Predicted mask by DeepLabV3+ [H, W].
        save_path: Optional file path to save output image.
        title: Title of visualization.

    Returns:
        Pillow Image of the rendered multi-panel canvas.
    """
    panels = []
    labels = []

    # 1. Aerial Image
    panels.append(image if image.ndim == 3 else np.stack([image]*3, axis=-1))
    labels.append("Aerial Image")

    # 2. Ground Truth Mask
    gt_rgb = mask_to_rgb(target)
    panels.append(gt_rgb)
    labels.append("Ground Truth")

    # 3. UNet Prediction & Error Map
    if unet_pred is not None:
        unet_rgb = mask_to_rgb(unet_pred)
        unet_err = create_error_map(target, unet_pred)
        panels.append(unet_rgb)
        labels.append("U-Net Prediction")
        panels.append(unet_err)
        labels.append("U-Net Error Map")

    # 4. DeepLabV3+ Prediction & Error Map
    if deeplab_pred is not None:
        deep_rgb = mask_to_rgb(deeplab_pred)
        deep_err = create_error_map(target, deeplab_pred)
        panels.append(deep_rgb)
        labels.append("DeepLabV3+ Prediction")
        panels.append(deep_err)
        labels.append("DeepLabV3+ Error Map")

    n_cols = len(panels)
    tile_w, tile_h = panels[0].shape[1], panels[0].shape[0]

    # If Matplotlib is available, generate via Matplotlib
    if MATPLOTLIB_AVAILABLE:
        fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4.5), dpi=300)
        if n_cols == 1:
            axes = [axes]
        fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)

        for ax, panel, lbl in zip(axes, panels, labels):
            ax.imshow(panel)
            ax.set_title(lbl, fontsize=11, pad=6)
            ax.axis("off")

        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            plt.savefig(save_path, bbox_inches="tight", dpi=300)
            logger.info(f"Saved qualitative comparison to {save_path}")
        plt.close(fig)

    # Always generate Pillow canvas for standalone return & fallback
    pad = 10
    header_h = 50
    footer_h = 40
    canvas_w = n_cols * tile_w + (n_cols + 1) * pad
    canvas_h = tile_h + header_h + footer_h + pad * 2

    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(250, 250, 252))
    draw = ImageDraw.Draw(canvas)

    # Draw Title
    draw.text((pad + 10, 15), title, fill=(20, 25, 35))

    for idx, (panel, lbl) in enumerate(zip(panels, labels)):
        x_pos = pad + idx * (tile_w + pad)
        y_pos = header_h + pad

        panel_img = Image.fromarray(panel.astype(np.uint8))
        canvas.paste(panel_img, (x_pos, y_pos))

        # Panel label
        draw.text((x_pos + 5, y_pos + tile_h + 8), lbl, fill=(40, 45, 55))

    if save_path and not MATPLOTLIB_AVAILABLE:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        canvas.save(save_path)
        logger.info(f"Saved qualitative comparison (Pillow) to {save_path}")

    return canvas


def plot_training_curves(
    history: Dict[str, List[float]],
    model_name: str = "Model",
    save_path: Optional[str] = "results/figures/training_curves.png",
) -> None:
    """
    Plot training loss, validation loss, and validation mIoU progression over epochs.

    Args:
        history: Dictionary containing 'train_loss', 'val_loss', 'val_miou', and optionally 'lr'.
        model_name: Name of architecture ('UNet' or 'DeepLabV3+').
        save_path: Destination path for figure file.
    """
    epochs = list(range(1, len(history.get("train_loss", [])) + 1))
    if not epochs:
        logger.warning("Empty history provided; skipping training curves plot.")
        return

    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])
    val_miou = history.get("val_miou", [])
    lrs = history.get("lr", [])

    if MATPLOTLIB_AVAILABLE:
        fig, axes = plt.subplots(1, 3 if lrs else 2, figsize=(15 if lrs else 11, 4.5), dpi=300)
        fig.suptitle(f"{model_name.upper()} Training Dynamics & Convergence", fontsize=14, fontweight="bold")

        # Loss Curve
        ax_loss = axes[0]
        ax_loss.plot(epochs, train_loss, label="Train Loss", color="#1f77b4", linewidth=2)
        if val_loss:
            ax_loss.plot(epochs, val_loss, label="Val Loss", color="#ff7f0e", linewidth=2, linestyle="--")
        ax_loss.set_xlabel("Epoch", fontsize=11)
        ax_loss.set_ylabel("Combined Loss", fontsize=11)
        ax_loss.set_title("Loss Progression", fontsize=12)
        ax_loss.grid(True, linestyle=":", alpha=0.6)
        ax_loss.legend(fontsize=10)

        # mIoU Curve
        ax_iou = axes[1]
        ax_iou.plot(epochs, val_miou, label="Validation mIoU", color="#2ca02c", linewidth=2)
        ax_iou.set_xlabel("Epoch", fontsize=11)
        ax_iou.set_ylabel("mIoU", fontsize=11)
        ax_iou.set_title("Validation mIoU Metric", fontsize=12)
        ax_iou.grid(True, linestyle=":", alpha=0.6)
        ax_iou.legend(fontsize=10)

        # Learning Rate Schedule (if tracked)
        if lrs:
            ax_lr = axes[2]
            ax_lr.plot(epochs, lrs, label="Learning Rate", color="#9467bd", linewidth=2)
            ax_lr.set_xlabel("Epoch", fontsize=11)
            ax_lr.set_ylabel("LR", fontsize=11)
            ax_lr.set_title("Cosine LR Schedule", fontsize=12)
            ax_lr.grid(True, linestyle=":", alpha=0.6)
            ax_lr.legend(fontsize=10)

        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            plt.savefig(save_path, bbox_inches="tight", dpi=300)
            logger.info(f"Saved training curves to {save_path}")
        plt.close(fig)
    else:
        # Standalone Pillow chart fallback
        w, h = 900, 400
        canvas = Image.new("RGB", (w, h), color=(255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        draw.text((20, 15), f"{model_name.upper()} Training Convergence History ({len(epochs)} Epochs)", fill=(30, 30, 30))

        # Draw simple line plots for loss and mIoU
        # Panel 1: Loss (x: 50..420, y: 70..350)
        # Panel 2: mIoU (x: 480..850, y: 70..350)
        draw.rectangle([50, 70, 420, 350], outline=(200, 200, 200), width=1)
        draw.text((60, 75), "Combined Loss", fill=(50, 50, 50))
        draw.rectangle([480, 70, 850, 350], outline=(200, 200, 200), width=1)
        draw.text((490, 75), "Val mIoU", fill=(50, 50, 50))

        if len(train_loss) > 1:
            max_l = max(max(train_loss), 1.0)
            min_l = min(min(train_loss), 0.0)
            rng_l = max_l - min_l if max_l != min_l else 1.0

            for i in range(len(train_loss) - 1):
                x1 = 60 + int((i / (len(train_loss) - 1)) * 340)
                y1 = 330 - int(((train_loss[i] - min_l) / rng_l) * 240)
                x2 = 60 + int(((i + 1) / (len(train_loss) - 1)) * 340)
                y2 = 330 - int(((train_loss[i + 1] - min_l) / rng_l) * 240)
                draw.line([(x1, y1), (x2, y2)], fill=(31, 119, 180), width=2)

        if len(val_miou) > 1:
            for i in range(len(val_miou) - 1):
                x1 = 490 + int((i / (len(val_miou) - 1)) * 340)
                y1 = 330 - int((val_miou[i]) * 240)
                x2 = 490 + int(((i + 1) / (len(val_miou) - 1)) * 340)
                y2 = 330 - int((val_miou[i + 1]) * 240)
                draw.line([(x1, y1), (x2, y2)], fill=(44, 160, 44), width=2)

        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            canvas.save(save_path)
            logger.info(f"Saved training curves (Pillow fallback) to {save_path}")


def plot_per_class_comparison(
    unet_metrics: Dict[str, Any],
    deeplab_metrics: Dict[str, Any],
    save_path: Optional[str] = "results/figures/per_class_iou_comparison.png",
) -> None:
    """
    Generate grouped bar chart comparing per-class IoU for UNet vs DeepLabV3+.

    Args:
        unet_metrics: Metrics dict from evaluation for UNet.
        deeplab_metrics: Metrics dict from evaluation for DeepLabV3+.
        save_path: Destination path for figure.
    """
    classes = ["Background", "Building", "Woodland", "Water", "Road"]
    unet_ious = [unet_metrics.get("per_class_iou", {}).get(c, 0.0) * 100.0 for c in classes]
    deep_ious = [deeplab_metrics.get("per_class_iou", {}).get(c, 0.0) * 100.0 for c in classes]

    if MATPLOTLIB_AVAILABLE:
        fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
        x = np.arange(len(classes))
        bar_width = 0.35

        rects1 = ax.bar(x - bar_width / 2, unet_ious, bar_width, label="U-Net (ResNet-34)", color="#3470a3")
        rects2 = ax.bar(x + bar_width / 2, deep_ious, bar_width, label="DeepLabV3+ (ResNet-34)", color="#e07a5f")

        ax.set_ylabel("Intersection over Union (IoU %)", fontsize=11)
        ax.set_title("Per-Class IoU Benchmark: U-Net vs DeepLabV3+", fontsize=13, fontweight="bold", pad=12)
        ax.set_xticks(x)
        ax.set_xticklabels(classes, fontsize=10)
        ax.set_ylim(0, 105)
        ax.grid(axis="y", linestyle=":", alpha=0.7)
        ax.legend(fontsize=10)

        # Label bars with values
        for rect in rects1:
            h = rect.get_height()
            ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)
        for rect in rects2:
            h = rect.get_height()
            ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            plt.savefig(save_path, bbox_inches="tight", dpi=300)
            logger.info(f"Saved per-class comparison chart to {save_path}")
        plt.close(fig)
    else:
        # Pillow bar chart fallback
        w, h = 800, 450
        canvas = Image.new("RGB", (w, h), color=(255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        draw.text((20, 15), "Per-Class IoU Benchmark: U-Net vs DeepLabV3+", fill=(20, 20, 20))

        # Base axis: y=380, x from 80 to 740
        draw.line([(80, 380), (740, 380)], fill=(100, 100, 100), width=2)
        step = (740 - 80) // len(classes)

        for i, c in enumerate(classes):
            bx = 80 + i * step + 20
            # UNet bar
            u_h = int((unet_ious[i] / 100.0) * 300)
            draw.rectangle([bx, 380 - u_h, bx + 35, 380], fill=(52, 112, 163))
            # DeepLab bar
            d_h = int((deep_ious[i] / 100.0) * 300)
            draw.rectangle([bx + 40, 380 - d_h, bx + 75, 380], fill=(224, 122, 95))
            # Label
            draw.text((bx + 15, 390), c, fill=(30, 30, 30))

        # Legend
        draw.rectangle([500, 25, 520, 40], fill=(52, 112, 163))
        draw.text((530, 25), "U-Net", fill=(20, 20, 20))
        draw.rectangle([620, 25, 640, 40], fill=(224, 122, 95))
        draw.text((650, 25), "DeepLabV3+", fill=(20, 20, 20))

        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            canvas.save(save_path)
            logger.info(f"Saved per-class comparison chart (Pillow fallback) to {save_path}")


def plot_confusion_matrix(
    confusion_matrix: Union[np.ndarray, List[List[float]]],
    class_names: Optional[Dict[int, str]] = None,
    model_name: str = "Model",
    normalize: bool = True,
    save_path: Optional[str] = "results/figures/confusion_matrix.png",
) -> None:
    """
    Plot annotated confusion matrix heatmap.

    Args:
        confusion_matrix: 5x5 array of counts or percentages.
        class_names: Mapping class ID -> string.
        model_name: Model title.
        normalize: Whether input matrix is normalized percentage.
        save_path: Destination path for figure.
    """
    cm = np.array(confusion_matrix, dtype=np.float64)
    names = [class_names.get(i, f"C_{i}") for i in range(cm.shape[0])] if class_names else ["Background", "Building", "Woodland", "Water", "Road"]

    if normalize and cm.max() > 100.0:
        row_sums = cm.sum(axis=1, keepdims=True)
        with np.errstate(divide='ignore', invalid='ignore'):
            cm = np.where(row_sums > 0, (cm / row_sums) * 100.0, 0.0)

    if MATPLOTLIB_AVAILABLE:
        fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.set_ylabel("Recall %" if normalize else "Pixel Count", rotation=-90, va="bottom", fontsize=10)

        ax.set(
            xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            xticklabels=names,
            yticklabels=names,
            title=f"{model_name.upper()} Normalized Confusion Matrix (%)",
            ylabel="Ground Truth",
            xlabel="Predicted Class",
        )
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")

        # Loop over data dimensions and create text annotations
        fmt = ".1f" if normalize else ".0f"
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j, i, format(cm[i, j], fmt) + ("%" if normalize else ""),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=9,
                )

        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            plt.savefig(save_path, bbox_inches="tight", dpi=300)
            logger.info(f"Saved confusion matrix heatmap to {save_path}")
        plt.close(fig)
    else:
        # Pillow confusion matrix grid fallback
        cell_size = 70
        n = cm.shape[0]
        w = n * cell_size + 200
        h = n * cell_size + 150

        canvas = Image.new("RGB", (w, h), color=(255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        draw.text((20, 15), f"{model_name.upper()} Confusion Matrix (%)", fill=(20, 20, 20))

        start_x, start_y = 120, 80
        max_val = max(float(cm.max()), 1.0)

        for i in range(n):
            draw.text((20, start_y + i * cell_size + 25), names[i], fill=(30, 30, 30))
            for j in range(n):
                cx = start_x + j * cell_size
                cy = start_y + i * cell_size
                val = cm[i, j]
                # Blue intensity
                intensity = int(255 - (val / max_val) * 200)
                cell_color = (intensity, intensity, 255)
                draw.rectangle([cx, cy, cx + cell_size, cy + cell_size], fill=cell_color, outline=(200, 200, 200))
                text_col = (255, 255, 255) if val > max_val / 2 else (0, 0, 0)
                draw.text((cx + 15, cy + 25), f"{val:.1f}%", fill=text_col)

        for j in range(n):
            draw.text((start_x + j * cell_size + 10, start_y + n * cell_size + 15), names[j], fill=(30, 30, 30))

        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            canvas.save(save_path)
            logger.info(f"Saved confusion matrix (Pillow fallback) to {save_path}")


def generate_all_visualizations(
    results_dir: str = "results",
    figures_dir: str = "results/figures",
) -> List[str]:
    """
    Generate all primary visualization artifacts from available results.

    Returns:
        List of generated figure file paths.
    """
    os.makedirs(figures_dir, exist_ok=True)
    generated_files = []

    # 1. Load evaluation summary if exists
    eval_summary_path = os.path.join(results_dir, "evaluation_summary.json")
    unet_eval = {}
    deeplab_eval = {}
    if os.path.exists(eval_summary_path):
        with open(eval_summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
            unet_eval = summary.get("unet", {}).get("metrics", {})
            deeplab_eval = summary.get("deeplabv3plus", {}).get("metrics", {})

    # If metrics not found in summary, check individual JSONs
    if not unet_eval and os.path.exists(os.path.join(results_dir, "evaluation_metrics_unet.json")):
        with open(os.path.join(results_dir, "evaluation_metrics_unet.json"), "r", encoding="utf-8") as f:
            unet_eval = json.load(f)
    if not deeplab_eval and os.path.exists(os.path.join(results_dir, "evaluation_metrics_deeplabv3plus.json")):
        with open(os.path.join(results_dir, "evaluation_metrics_deeplabv3plus.json"), "r", encoding="utf-8") as f:
            deeplab_eval = json.load(f)

    # 2. Per-class IoU comparison bar chart
    if unet_eval or deeplab_eval:
        bar_path = os.path.join(figures_dir, "per_class_iou_comparison.png")
        plot_per_class_comparison(unet_eval, deeplab_eval, save_path=bar_path)
        generated_files.append(bar_path)

    # 3. Confusion matrices
    if "normalized_confusion_matrix" in unet_eval:
        cm_path = os.path.join(figures_dir, "confusion_matrix_unet.png")
        plot_confusion_matrix(
            unet_eval["normalized_confusion_matrix"],
            model_name="U-Net",
            save_path=cm_path,
        )
        generated_files.append(cm_path)

    if "normalized_confusion_matrix" in deeplab_eval:
        cm_deep_path = os.path.join(figures_dir, "confusion_matrix_deeplabv3plus.png")
        plot_confusion_matrix(
            deeplab_eval["normalized_confusion_matrix"],
            model_name="DeepLabV3+",
            save_path=cm_deep_path,
        )
        generated_files.append(cm_deep_path)

    # 4. Training Convergence curves (if checkpoint/history exists)
    for m in ["unet", "deeplabv3plus"]:
        hist_candidates = [
            f"checkpoints/{m}/history.json",
            f"/content/drive/MyDrive/landcover-segmentation/checkpoints/{m}/history.json",
        ]
        for hc in hist_candidates:
            if os.path.exists(hc):
                with open(hc, "r", encoding="utf-8") as f:
                    hist_data = json.load(f)
                curve_path = os.path.join(figures_dir, f"training_curves_{m}.png")
                plot_training_curves(hist_data, model_name=m, save_path=curve_path)
                generated_files.append(curve_path)
                break

    # 5. Synthetic demo qualitative sample if data exists or create synthetic sample
    dummy_img = np.zeros((512, 512, 3), dtype=np.uint8)
    dummy_img[100:300, 100:300] = [180, 175, 170]  # building texture
    dummy_img[350:500, :] = [30, 100, 40]          # woodland
    dummy_img[:, 240:270] = [80, 80, 85]           # road

    dummy_target = np.zeros((512, 512), dtype=np.uint8)
    dummy_target[100:300, 100:300] = 1             # Building
    dummy_target[350:500, :] = 2                   # Woodland
    dummy_target[:, 240:270] = 4                   # Road

    dummy_pred_unet = dummy_target.copy()
    dummy_pred_unet[100:130, 100:130] = 0          # small error
    dummy_pred_deep = dummy_target.copy()
    dummy_pred_deep[350:370, 0:50] = 0             # boundary error

    sample_qual_path = os.path.join(figures_dir, "qualitative_comparison_sample.png")
    plot_qualitative_sample(
        image=dummy_img,
        target=dummy_target,
        unet_pred=dummy_pred_unet,
        deeplab_pred=dummy_pred_deep,
        save_path=sample_qual_path,
        title="Urban Land Cover Multi-Model Qualitative Comparison",
    )
    generated_files.append(sample_qual_path)

    logger.info(f"Generated {len(generated_files)} visualization figures in {figures_dir}")
    return generated_files


if __name__ == "__main__":
    generate_all_visualizations()

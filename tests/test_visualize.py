"""
Unit tests for Phase 4: Visualization Pipeline (Qualitative Overlays, Error Maps, Training Curves, Confusion Matrices).
"""

import os
import shutil
import unittest
import numpy as np
from PIL import Image

from src.visualize import (
    create_error_map,
    plot_qualitative_sample,
    plot_training_curves,
    plot_per_class_comparison,
    plot_confusion_matrix,
    generate_all_visualizations,
)


class TestVisualizationPipeline(unittest.TestCase):
    def setUp(self):
        """Set up temporary test directory."""
        self.test_dir = "tests/temp_test_vis"
        os.makedirs(self.test_dir, exist_ok=True)

    def tearDown(self):
        """Clean up temporary test directory."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_create_error_map(self):
        """Verify error map produces green for matches and red for errors."""
        target = np.array([
            [1, 1, 2],
            [3, 4, 0],
        ], dtype=np.uint8)
        pred = np.array([
            [1, 0, 2],  # (0, 1) is error
            [3, 4, 1],  # (1, 2) is error
        ], dtype=np.uint8)

        err_map = create_error_map(target, pred, correct_color=(0, 255, 0), incorrect_color=(255, 0, 0))

        self.assertEqual(err_map.shape, (2, 3, 3))
        # (0, 0) should be green
        np.testing.assert_array_equal(err_map[0, 0], [0, 255, 0])
        # (0, 1) should be red
        np.testing.assert_array_equal(err_map[0, 1], [255, 0, 0])
        # (1, 1) should be green
        np.testing.assert_array_equal(err_map[1, 1], [0, 255, 0])
        # (1, 2) should be red
        np.testing.assert_array_equal(err_map[1, 2], [255, 0, 0])

    def test_plot_qualitative_sample(self):
        """Verify qualitative comparison image composition and saving."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        gt = np.zeros((100, 100), dtype=np.uint8)
        unet_pred = np.zeros((100, 100), dtype=np.uint8)
        deeplab_pred = np.zeros((100, 100), dtype=np.uint8)

        save_path = os.path.join(self.test_dir, "test_qual.png")
        canvas = plot_qualitative_sample(
            image=img,
            target=gt,
            unet_pred=unet_pred,
            deeplab_pred=deeplab_pred,
            save_path=save_path,
        )

        self.assertIsInstance(canvas, Image.Image)
        self.assertTrue(os.path.exists(save_path))
        with Image.open(save_path) as saved_img:
            self.assertGreater(saved_img.width, 100)
            self.assertGreater(saved_img.height, 100)

    def test_plot_training_curves(self):
        """Verify training curves can be rendered and saved."""
        history = {
            "train_loss": [0.85, 0.62, 0.45, 0.35, 0.28],
            "val_loss": [0.90, 0.68, 0.50, 0.39, 0.32],
            "val_miou": [0.42, 0.58, 0.69, 0.73, 0.76],
        }
        save_path = os.path.join(self.test_dir, "test_curves.png")
        plot_training_curves(history, model_name="UNet", save_path=save_path)
        self.assertTrue(os.path.exists(save_path))

    def test_plot_per_class_comparison(self):
        """Verify per-class comparison chart rendering."""
        unet_m = {
            "per_class_iou": {"Background": 0.88, "Building": 0.72, "Woodland": 0.84, "Water": 0.78, "Road": 0.55}
        }
        deep_m = {
            "per_class_iou": {"Background": 0.89, "Building": 0.74, "Woodland": 0.85, "Water": 0.80, "Road": 0.56}
        }
        save_path = os.path.join(self.test_dir, "test_bar.png")
        plot_per_class_comparison(unet_m, deep_m, save_path=save_path)
        self.assertTrue(os.path.exists(save_path))

    def test_plot_confusion_matrix(self):
        """Verify confusion matrix heatmap rendering."""
        cm = [
            [85.2, 5.1, 4.3, 2.1, 3.3],
            [4.2, 78.4, 6.1, 1.2, 10.1],
            [3.1, 2.5, 88.2, 4.1, 2.1],
            [1.5, 0.8, 3.2, 92.5, 2.0],
            [5.0, 12.1, 3.4, 1.5, 78.0],
        ]
        save_path = os.path.join(self.test_dir, "test_cm.png")
        plot_confusion_matrix(cm, model_name="U-Net", save_path=save_path)
        self.assertTrue(os.path.exists(save_path))

    def test_generate_all_visualizations(self):
        """Verify batch generation of visualization artifacts."""
        fig_dir = os.path.join(self.test_dir, "figures")
        generated = generate_all_visualizations(results_dir=self.test_dir, figures_dir=fig_dir)
        self.assertGreaterEqual(len(generated), 1)
        for p in generated:
            self.assertTrue(os.path.exists(p))


if __name__ == "__main__":
    unittest.main()

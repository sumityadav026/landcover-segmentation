"""
Unit tests for Stage 2: Augmentations, Model Architecture Configurations,
Combined Loss Function, and Memory Estimation.
"""

import os
import unittest
import numpy as np

from src.augmentations import (
    apply_spatial_transforms_numpy,
    denormalize_image,
    IMAGENET_MEAN,
    IMAGENET_STD,
)
from src.losses import load_class_weights, numpy_combined_loss
from src.models import estimate_model_memory, count_parameters


class TestStage2ModelDevelopment(unittest.TestCase):
    def setUp(self):
        """Set up synthetic test data for augmentation and loss testing."""
        np.random.seed(42)
        self.h, self.w = 64, 64
        # Synthetic aerial image [H, W, 3] uint8
        self.image = np.random.randint(50, 200, (self.h, self.w, 3), dtype=np.uint8)
        # Synthetic mask [H, W] int64 with classes 0 to 4
        self.mask = np.random.randint(0, 5, (self.h, self.w), dtype=np.int64)

        # Place a unique asymmetric marker at (10, 20) in both image and mask
        self.image[10, 20] = [255, 0, 0]  # Red pixel
        self.mask[10, 20] = 1            # Building class

    def test_spatial_transforms_identical_on_image_and_mask(self):
        """Verify that horizontal flip, vertical flip, and 90-degree rotations apply identically."""
        # 1. Horizontal Flip
        img_h, mask_h = apply_spatial_transforms_numpy(self.image, self.mask, hflip=True)
        # Expected new location of marker: (10, w - 1 - 20) = (10, 43)
        expected_x = self.w - 1 - 20
        np.testing.assert_array_equal(img_h[10, expected_x], [255, 0, 0])
        self.assertEqual(mask_h[10, expected_x], 1)

        # 2. Vertical Flip
        img_v, mask_v = apply_spatial_transforms_numpy(self.image, self.mask, vflip=True)
        # Expected new location of marker: (h - 1 - 10, 20) = (53, 20)
        expected_y = self.h - 1 - 10
        np.testing.assert_array_equal(img_v[expected_y, 20], [255, 0, 0])
        self.assertEqual(mask_v[expected_y, 20], 1)

        # 3. 90-degree Rotation (k=1)
        img_rot, mask_rot = apply_spatial_transforms_numpy(self.image, self.mask, k_rot=1)
        # Verify image and mask shapes remain identical
        self.assertEqual(img_rot.shape[:2], mask_rot.shape)
        # Verify mask integer class range is preserved
        self.assertTrue(np.all((mask_rot >= 0) & (mask_rot <= 4)))

    def test_imagenet_normalization_and_denormalization(self):
        """Verify ImageNet normalization reversibility within rounding tolerance."""
        img_float = self.image.astype(np.float32) / 255.0
        # Normalize
        mean_arr = np.array(IMAGENET_MEAN, dtype=np.float32)
        std_arr = np.array(IMAGENET_STD, dtype=np.float32)
        normalized = (img_float - mean_arr) / std_arr

        # Denormalize
        denorm = denormalize_image(normalized)
        self.assertEqual(denorm.shape, self.image.shape)
        self.assertEqual(denorm.dtype, np.uint8)
        # Maximum difference due to float-to-uint8 quantization should be <= 1
        max_diff = np.max(np.abs(denorm.astype(int) - self.image.astype(int)))
        self.assertLessEqual(max_diff, 1)

    def test_load_class_weights(self):
        """Verify that class weights from Stage 1 dataset statistics or class_weights.json load properly."""
        weights = load_class_weights("results/class_distribution.json")
        if hasattr(weights, "tolist"):
            weights = weights.tolist()

        self.assertEqual(len(weights), 5)
        for w in weights:
            self.assertGreater(w, 0.0, "Class weight must be strictly positive")

        # Test loading from custom class_weights.json format
        custom_weights_path = "tests/temp_test_weights.json"
        import json
        with open(custom_weights_path, "w") as f:
            json.dump({"0": 0.5, "1": 2.5, "2": 1.2, "3": 1.8, "4": 3.1}, f)

        try:
            loaded_custom = load_class_weights(custom_weights_path)
            if hasattr(loaded_custom, "tolist"):
                loaded_custom = loaded_custom.tolist()
            self.assertEqual(len(loaded_custom), 5)
            self.assertAlmostEqual(loaded_custom[1], 2.5, places=4)
        finally:
            if os.path.exists(custom_weights_path):
                os.remove(custom_weights_path)

    def test_numpy_combined_loss_perfect_vs_random_predictions(self):
        """Verify that combined loss (0.5 CE + 0.5 Dice) behaves correctly."""
        b, c, h, w = 2, 5, 16, 16
        targets = np.random.randint(0, c, (b, h, w), dtype=np.int64)

        # Perfect prediction: high positive logits for true class, large negative for others
        perfect_logits = np.ones((b, c, h, w), dtype=np.float32) * -10.0
        for bi in range(b):
            for hi in range(h):
                for wi in range(w):
                    cls_id = targets[bi, hi, wi]
                    perfect_logits[bi, cls_id, hi, wi] = 10.0

        loss_perfect = numpy_combined_loss(perfect_logits, targets, num_classes=5)
        # Perfect predictions should have near-zero Dice loss and near-zero CE loss
        self.assertLess(loss_perfect["dice_loss"], 0.05)
        self.assertLess(loss_perfect["ce_loss"], 0.05)
        self.assertLess(loss_perfect["total_loss"], 0.05)

        # Uniform random predictions
        random_logits = np.random.randn(b, c, h, w).astype(np.float32)
        loss_random = numpy_combined_loss(random_logits, targets, num_classes=5)
        # Random predictions should have substantially higher loss
        self.assertGreater(loss_random["total_loss"], loss_perfect["total_loss"])

        # Check combined weighting: total == 0.5 * CE + 0.5 * Dice
        expected_total = 0.5 * loss_random["ce_loss"] + 0.5 * loss_random["dice_loss"]
        self.assertAlmostEqual(loss_random["total_loss"], expected_total, places=5)

    def test_memory_estimation(self):
        """Verify VRAM estimation logic for Google Colab GPU compatibility."""
        # Simulated model wrapper with dummy parameter count (~26M parameters, similar to ResNet34 UNet)
        class MockModel:
            def parameters(self):
                # Return dummy parameter with 26 million elements
                import collections
                Param = collections.namedtuple("Param", ["numel", "requires_grad"])
                return [Param(numel=26_000_000, requires_grad=True)]

        mock_model = MockModel()
        mem_info = estimate_model_memory(mock_model, batch_size=8, input_resolution=(512, 512))

        self.assertEqual(mem_info["batch_size"], 8)
        self.assertIn("total_training_gb", mem_info)
        self.assertIn("total_inference_gb", mem_info)
        # Check that it calculates values properly
        self.assertGreater(mem_info["total_training_gb"], 1.0)
        # Must fit within 15 GB of Colab T4
        self.assertTrue(mem_info["fits_colab_t4"])


if __name__ == "__main__":
    unittest.main()


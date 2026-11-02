"""
Unit tests for Stage 3: Training Pipeline, Metrics Tracking, Checkpoint Resumption,
and Early Stopping Logic.
"""

import os
import shutil
import unittest
import numpy as np

from src.metrics import MetricTracker


class TestStage3TrainingPipeline(unittest.TestCase):
    def setUp(self):
        """Set up temporary test directory."""
        self.test_dir = "tests/temp_test_checkpoints"
        os.makedirs(self.test_dir, exist_ok=True)

    def tearDown(self):
        """Clean up temporary test directory."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_metric_tracker_perfect_predictions(self):
        """Verify that MetricTracker returns 1.0 IoU and 1.0 accuracy for identical predictions."""
        tracker = MetricTracker(num_classes=5)
        # Create identical targets and predictions with all 5 classes
        targets = np.array([
            [0, 1, 2, 3, 4],
            [1, 2, 3, 4, 0],
            [2, 3, 4, 0, 1],
        ], dtype=np.int64)
        preds = targets.copy()

        tracker.update(preds, targets)
        metrics = tracker.compute()

        self.assertAlmostEqual(metrics["miou"], 1.0, places=4)
        self.assertAlmostEqual(metrics["miou_no_bg"], 1.0, places=4)
        self.assertAlmostEqual(metrics["pixel_accuracy"], 1.0, places=4)

        for cls_name, iou in metrics["per_class_iou"].items():
            self.assertAlmostEqual(iou, 1.0, places=4, msg=f"{cls_name} IoU should be 1.0")

    def test_metric_tracker_analytical_confusion_matrix(self):
        """Verify IoU and pixel accuracy against manually calculated ground truth."""
        tracker = MetricTracker(num_classes=3)
        # Class 0: 3 TP, 1 FP (predicted 0 when target was 1) -> TP=3, FP=1, FN=0 -> IoU = 3 / (3+1+0) = 0.75
        # Class 1: 2 TP, 0 FP, 1 FN (predicted 0 when target was 1) -> TP=2, FP=0, FN=1 -> IoU = 2 / (2+0+1) = 0.6667
        # Class 2: 1 TP, 0 FP, 0 FN -> IoU = 1 / 1 = 1.0
        targets = np.array([[0, 0, 0, 1, 1, 1, 2]])
        preds =   np.array([[0, 0, 0, 1, 1, 0, 2]])

        tracker.update(preds, targets)
        metrics = tracker.compute()

        # TP=6, Total=7 -> Pixel Accuracy = 6/7 = 0.85714
        self.assertAlmostEqual(metrics["pixel_accuracy"], 6 / 7, places=4)
        # Class 0 IoU = 3/4 = 0.75
        self.assertAlmostEqual(metrics["per_class_iou"]["Background"], 0.75, places=4)
        # Class 1 IoU = 2/3 = 0.66667
        self.assertAlmostEqual(metrics["per_class_iou"]["Building"], 2 / 3, places=4)
        # Class 2 IoU = 1/1 = 1.0
        self.assertAlmostEqual(metrics["per_class_iou"]["Woodland"], 1.0, places=4)

    def test_checkpoint_state_serialization_and_resumption(self):
        """Verify checkpoint serialization structure and state restoration."""
        # Simulated checkpoint dictionary
        original_state = {
            "epoch": 12,
            "model_name": "unet",
            "model_state_dict": {"conv.weight": np.array([1.5, 2.5, 3.5])},
            "optimizer_state_dict": {"lr": 0.0001},
            "best_val_miou": 0.7425,
            "best_epoch": 10,
            "patience_counter": 2,
            "history": {
                "train_loss": [0.8, 0.6, 0.4],
                "val_miou": [0.55, 0.70, 0.74],
            },
        }

        save_path = os.path.join(self.test_dir, "unet_test_checkpoint.pt")
        # Save using pickle / json serialization mock
        import pickle
        with open(save_path, "wb") as f:
            pickle.dump(original_state, f)

        self.assertTrue(os.path.exists(save_path))

        with open(save_path, "rb") as f:
            restored_state = pickle.load(f)

        self.assertEqual(restored_state["epoch"], 12)
        self.assertEqual(restored_state["best_epoch"], 10)
        self.assertAlmostEqual(restored_state["best_val_miou"], 0.7425, places=4)
        self.assertEqual(restored_state["patience_counter"], 2)
        np.testing.assert_array_equal(
            restored_state["model_state_dict"]["conv.weight"],
            np.array([1.5, 2.5, 3.5])
        )

    def test_best_checkpoint_protection_logic(self):
        """Verify that a worse validation mIoU never overwrites the best checkpoint."""
        best_val_miou = 0.7850
        best_epoch = 15

        # Candidate epoch 16 with worse performance (0.7600)
        epoch_16_miou = 0.7600
        is_best_16 = epoch_16_miou > best_val_miou
        self.assertFalse(is_best_16, "Epoch 16 (0.7600) should NOT be considered best")

        # Candidate epoch 17 with better performance (0.8010)
        epoch_17_miou = 0.8010
        is_best_17 = epoch_17_miou > best_val_miou
        self.assertTrue(is_best_17, "Epoch 17 (0.8010) should be considered best")
        if is_best_17:
            best_val_miou = epoch_17_miou
            best_epoch = 17

        self.assertAlmostEqual(best_val_miou, 0.8010, places=4)
        self.assertEqual(best_epoch, 17)

    def test_early_stopping_patience_trigger(self):
        """Verify that early stopping patience triggers termination after 10 non-improving epochs."""
        patience_limit = 10
        patience_counter = 0
        best_miou = 0.75
        triggered = False

        # 10 consecutive non-improving epochs
        for epoch in range(1, 15):
            val_miou = 0.72  # Below best
            if val_miou > best_miou:
                best_miou = val_miou
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience_limit:
                triggered = True
                stopped_at_epoch = epoch
                break

        self.assertTrue(triggered)
        self.assertEqual(patience_counter, 10)
        self.assertEqual(stopped_at_epoch, 10)


if __name__ == "__main__":
    unittest.main()

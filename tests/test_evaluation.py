"""
Unit tests for Phase 3: Model Evaluation, Latency Benchmarking, and Comparative Metrics.
"""

import os
import shutil
import unittest
import numpy as np

from src.evaluate import evaluate_model, benchmark_latency, compare_models, run_test_evaluation


class TestEvaluationPipeline(unittest.TestCase):
    def setUp(self):
        """Set up temporary test directory."""
        self.test_dir = "tests/temp_test_eval"
        os.makedirs(self.test_dir, exist_ok=True)

    def tearDown(self):
        """Clean up temporary test directory."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_evaluate_model_offline(self):
        """Verify metric accumulation and serialization on synthetic batches."""
        # 2 batches of synthetic predictions and targets
        # Batch 1: perfect predictions
        b1_targets = np.array([
            [0, 1, 2, 3, 4],
            [1, 2, 3, 4, 0],
        ], dtype=np.int64)
        b1_preds = b1_targets.copy()

        # Batch 2: partial errors
        b2_targets = np.array([[1, 1, 2, 2, 3]], dtype=np.int64)
        b2_preds =   np.array([[1, 0, 2, 3, 3]], dtype=np.int64)

        mock_dataloader = [
            (b1_preds, b1_targets),
            (b2_preds, b2_targets),
        ]

        def mock_model(x):
            return x

        report = evaluate_model(mock_model, mock_dataloader, device="cpu", num_classes=5)

        self.assertIn("pixel_accuracy", report)
        self.assertIn("miou", report)
        self.assertIn("miou_no_bg", report)
        self.assertIn("per_class_iou", report)
        self.assertIn("per_class_precision", report)
        self.assertIn("per_class_recall", report)
        self.assertIn("per_class_f1", report)
        self.assertIn("confusion_matrix", report)
        self.assertIn("normalized_confusion_matrix", report)

        # Total pixels = 10 + 5 = 15. Correct = 10 (from b1) + 3 (from b2: targets[0,0]=1, targets[0,2]=2, targets[0,4]=3) = 13
        # Pixel accuracy = 13 / 15 = 0.86667
        self.assertAlmostEqual(report["pixel_accuracy"], 13 / 15, places=4)
        self.assertEqual(len(report["confusion_matrix"]), 5)
        self.assertEqual(len(report["confusion_matrix"][0]), 5)

    def test_benchmark_latency_structure(self):
        """Verify benchmark_latency returns valid statistical fields."""
        latency_info = benchmark_latency(model=None, warmup=2, iterations=5)
        self.assertIn("mean_latency_ms", latency_info)
        self.assertIn("std_latency_ms", latency_info)
        self.assertIn("throughput_fps", latency_info)
        self.assertIn("p50_latency_ms", latency_info)
        self.assertIn("p95_latency_ms", latency_info)
        self.assertIn("p99_latency_ms", latency_info)
        self.assertGreater(latency_info["throughput_fps"], 0)

    def test_compare_models_csv_and_table(self):
        """Verify comparison table generation and CSV output."""
        sample_eval = {
            "unet": {
                "metrics": {
                    "miou": 0.7250,
                    "miou_no_bg": 0.6980,
                    "pixel_accuracy": 0.8920,
                    "mean_precision": 0.8120,
                    "mean_recall": 0.7850,
                    "mean_f1": 0.7980,
                    "per_class_iou": {
                        "Background": 0.85,
                        "Building": 0.70,
                        "Woodland": 0.82,
                        "Water": 0.75,
                        "Road": 0.52,
                    },
                },
                "latency": {"mean_latency_ms": 11.2, "throughput_fps": 89.3},
                "params": {"total_millions": 24.43},
            },
            "deeplabv3plus": {
                "metrics": {
                    "miou": 0.7410,
                    "miou_no_bg": 0.7150,
                    "pixel_accuracy": 0.9010,
                    "mean_precision": 0.8350,
                    "mean_recall": 0.8010,
                    "mean_f1": 0.8170,
                    "per_class_iou": {
                        "Background": 0.87,
                        "Building": 0.73,
                        "Woodland": 0.83,
                        "Water": 0.77,
                        "Road": 0.53,
                    },
                },
                "latency": {"mean_latency_ms": 15.8, "throughput_fps": 63.3},
                "params": {"total_millions": 26.74},
            },
        }

        csv_path = os.path.join(self.test_dir, "test_comparison.csv")
        table_str = compare_models(sample_eval, output_csv_path=csv_path)

        self.assertTrue(os.path.exists(csv_path))
        with open(csv_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("UNET", content)
        self.assertIn("DEEPLABV3PLUS", content)
        self.assertIn("mIoU (%)", content)
        self.assertIn("Building IoU (%)", content)

        self.assertIn("UNET", table_str)
        self.assertIn("DEEPLABV3PLUS", table_str)

    def test_run_test_evaluation_end_to_end(self):
        """Verify orchestrator runs cleanly and creates all output artifacts."""
        output_dir = os.path.join(self.test_dir, "eval_results")
        results = run_test_evaluation(
            config_path="configs/config.yaml",
            model_name="both",
            output_dir=output_dir,
        )

        self.assertIn("unet", results)
        self.assertIn("deeplabv3plus", results)
        self.assertTrue(os.path.exists(os.path.join(output_dir, "evaluation_metrics_unet.json")))
        self.assertTrue(os.path.exists(os.path.join(output_dir, "evaluation_metrics_deeplabv3plus.json")))
        self.assertTrue(os.path.exists(os.path.join(output_dir, "model_comparison.csv")))
        self.assertTrue(os.path.exists(os.path.join(output_dir, "evaluation_summary.json")))


if __name__ == "__main__":
    unittest.main()


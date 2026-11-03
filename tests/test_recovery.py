"""
Unit tests for Colab Recovery and Resilience Engine (src/recovery.py).
Tests dataset detection, checkpoint scanning, codebase restoration, and dashboard generation.
"""

import os
import shutil
import unittest
import zipfile
import pickle

from src.recovery import (
    check_dataset_persistence,
    scan_checkpoints,
    restore_project_codebase,
    generate_recovery_dashboard,
)


class TestColabRecovery(unittest.TestCase):
    def setUp(self):
        """Set up temporary test environment."""
        self.test_dir = "tests/temp_test_recovery"
        os.makedirs(self.test_dir, exist_ok=True)
        self.mock_drive = os.path.join(self.test_dir, "mock_drive")
        self.mock_local = os.path.join(self.test_dir, "mock_local")
        os.makedirs(self.mock_drive, exist_ok=True)
        os.makedirs(self.mock_local, exist_ok=True)

    def tearDown(self):
        """Clean up temporary test files."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_check_dataset_persistence_local(self):
        """Verify that existing processed dataset is detected and skips re-downloading."""
        # Create mock train and val folders with sample tile files
        train_img_dir = os.path.join(self.mock_local, "data", "train", "images")
        val_img_dir = os.path.join(self.mock_local, "data", "val", "images")
        os.makedirs(train_img_dir, exist_ok=True)
        os.makedirs(val_img_dir, exist_ok=True)

        for i in range(5):
            with open(os.path.join(train_img_dir, f"tile_{i}.png"), "w") as f:
                f.write("mock")
        for i in range(2):
            with open(os.path.join(val_img_dir, f"tile_{i}.png"), "w") as f:
                f.write("mock")

        res = check_dataset_persistence(
            drive_mount_point=self.mock_drive,
            project_name="landcover-segmentation",
            local_data_dir=os.path.join(self.mock_local, "data"),
        )

        self.assertTrue(res["ready"])
        self.assertTrue(res["skip_data_preparation"])
        self.assertEqual(res["counts"]["train"], 5)
        self.assertEqual(res["counts"]["val"], 2)

    def test_check_dataset_persistence_missing(self):
        """Verify that missing dataset returns ready=False."""
        res = check_dataset_persistence(
            drive_mount_point=self.mock_drive,
            project_name="landcover-segmentation",
            local_data_dir=os.path.join(self.mock_local, "non_existent_data"),
        )

        self.assertFalse(res["ready"])
        self.assertFalse(res["skip_data_preparation"])

    def test_scan_checkpoints_detection(self):
        """Verify checkpoint scanning and metadata extraction."""
        ckpt_dir = os.path.join(self.mock_local, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)

        # Create mock checkpoint files with pickle serialization (matching torch.save structure)
        mock_unet = {
            "epoch": 14,
            "best_epoch": 12,
            "best_val_miou": 0.7654,
            "model_name": "unet",
        }
        mock_deeplab = {
            "epoch": 8,
            "best_epoch": 8,
            "best_val_miou": 0.7412,
            "model_name": "deeplabv3plus",
        }

        # Write dummy files larger than 1KB
        for name, data in [("unet", mock_unet), ("deeplabv3plus", mock_deeplab)]:
            file_path = os.path.join(ckpt_dir, f"{name}_latest.pt")
            with open(file_path, "wb") as f:
                pickle.dump(data, f)
                # Pad with extra bytes to exceed 1024 bytes
                f.write(b"0" * 2000)

        summary = scan_checkpoints(ckpt_dir)

        self.assertTrue(summary["unet"]["has_latest"])
        self.assertTrue(summary["deeplabv3plus"]["has_latest"])

    def test_restore_project_codebase_from_zip(self):
        """Verify codebase restoration from zip archive."""
        zip_path = os.path.join(self.mock_local, "landcover-segmentation.zip")
        dest_dir = os.path.join(self.mock_local, "target_ws")
        os.makedirs(dest_dir, exist_ok=True)

        # Create a mock zip with src/__init__.py
        with zipfile.ZipFile(zip_path, "w") as z:
            z.writestr("src/__init__.py", "# mock init")
            z.writestr("configs/config.yaml", "project: mock")

        status = restore_project_codebase(
            drive_mount_point=self.mock_drive,
            project_name="landcover-segmentation",
            target_dir=dest_dir,
        )

        self.assertTrue(status["restored"])
        self.assertTrue(os.path.exists(os.path.join(dest_dir, "src", "__init__.py")))

    def test_recovery_dashboard_formatting(self):
        """Verify dashboard text output format."""
        dashboard = generate_recovery_dashboard(
            drive_status=(True, "Mounted at /content/drive"),
            code_status={"restored": True, "source": "Google Drive"},
            data_status={
                "ready": True,
                "storage_type": "local_ssd",
                "counts": {"train": 120, "val": 25},
            },
            checkpoint_status={
                "unet": {"has_latest": True, "latest_epoch": 10, "best_val_miou": 0.75, "best_epoch": 9},
                "deeplabv3plus": {"has_latest": False},
            },
        )

        self.assertIn("GOOGLE COLAB RECOVERY", dashboard)
        self.assertIn("CONNECTED", dashboard)
        self.assertIn("VERIFIED & READY", dashboard)
        self.assertIn("UNET", dashboard)
        self.assertIn("Epoch 10", dashboard)


if __name__ == "__main__":
    unittest.main()


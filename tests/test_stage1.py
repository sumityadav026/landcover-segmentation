"""
Unit tests for Stage 1: Data Preparation, Tiling, Splitting, and Dataset Loading.
Executes locally in standard environment using PIL and NumPy.
"""

import os
import shutil
import unittest
import numpy as np
from PIL import Image

from src.utils import set_seed, mask_to_rgb, overlay_mask, DEFAULT_CLASS_COLORS
from src.dataset import LandCoverDataset, load_image, load_mask
from src.prepare_data import (
    partition_source_images,
    tile_orthophoto_pair,
    prepare_landcover_dataset,
)


class TestStage1DataPipeline(unittest.TestCase):
    def setUp(self):
        """Set up temporary test directories and synthetic test images."""
        set_seed(42)
        self.test_dir = "tests/temp_test_data"
        self.raw_dir = os.path.join(self.test_dir, "raw")
        self.processed_dir = os.path.join(self.test_dir, "processed")

        os.makedirs(os.path.join(self.raw_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(self.raw_dir, "masks"), exist_ok=True)

        # Generate 4 synthetic large orthophotos and masks
        self.source_stems = [
            "M-33-20-D-c-4-2",
            "M-33-20-D-c-4-3",
            "N-34-66-C-c-4-1",
            "N-34-66-C-c-4-2",
        ]

        # Dimension: 1100 x 1200 (contains 2x2 = 4 full 512x512 tiles, plus edge leftovers)
        self.h, self.w = 1100, 1200
        for stem in self.source_stems:
            # Synthetic 3-channel RGB image
            img_arr = np.random.randint(0, 256, (self.h, self.w, 3), dtype=np.uint8)
            img_path = os.path.join(self.raw_dir, "images", f"{stem}.tif")
            Image.fromarray(img_arr).save(img_path)

            # Synthetic mask with classes 0 to 4
            mask_arr = np.random.randint(0, 5, (self.h, self.w), dtype=np.uint8)
            mask_path = os.path.join(self.raw_dir, "masks", f"{stem}.tif")
            Image.fromarray(mask_arr).save(mask_path)

    def tearDown(self):
        """Clean up temporary test files."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_mask_to_rgb_and_overlay(self):
        """Test color palette conversion and visual mask overlay."""
        # Create a small 5x5 test mask with each class
        mask = np.array([
            [0, 1, 2, 3, 4],
            [4, 3, 2, 1, 0],
            [1, 1, 2, 2, 3],
            [0, 0, 4, 4, 4],
            [2, 3, 1, 0, 4],
        ], dtype=np.int64)

        rgb = mask_to_rgb(mask)
        self.assertEqual(rgb.shape, (5, 5, 3))
        self.assertEqual(rgb.dtype, np.uint8)

        # Check background (0,0) is Black (0,0,0)
        np.testing.assert_array_equal(rgb[0, 0], [0, 0, 0])
        # Check building (0,1) is Red (255,0,0)
        np.testing.assert_array_equal(rgb[0, 1], [255, 0, 0])
        # Check woodland (0,2) is Green (0,255,0)
        np.testing.assert_array_equal(rgb[0, 2], [0, 255, 0])
        # Check water (0,3) is Blue (0,0,255)
        np.testing.assert_array_equal(rgb[0, 3], [0, 0, 255])
        # Check road (0,4) is Yellow (255,255,0)
        np.testing.assert_array_equal(rgb[0, 4], [255, 255, 0])

        # Test overlay
        img = np.ones((5, 5, 3), dtype=np.uint8) * 100
        blended = overlay_mask(img, mask, alpha=0.5)
        self.assertEqual(blended.shape, (5, 5, 3))

    def test_partition_source_images_no_leakage(self):
        """Test that source-image partitioning strictly prevents data leakage."""
        stems = [f"ortho_{i:02d}" for i in range(41)]
        assignment = partition_source_images(
            source_stems=stems,
            official_split_files={},
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            seed=42,
        )

        train_set = set(assignment["train"])
        val_set = set(assignment["val"])
        test_set = set(assignment["test"])

        # Check mutual exclusivity
        self.assertEqual(len(train_set & val_set), 0, "Train and Val share source images!")
        self.assertEqual(len(train_set & test_set), 0, "Train and Test share source images!")
        self.assertEqual(len(val_set & test_set), 0, "Val and Test share source images!")

        # Check total count preserved
        self.assertEqual(len(train_set) + len(val_set) + len(test_set), 41)

    def test_tiling_edge_handling(self):
        """Test tiling of 1100x1200 image into 512x512 tiles with edge discarding."""
        stem = self.source_stems[0]
        img_path = os.path.join(self.raw_dir, "images", f"{stem}.tif")
        mask_path = os.path.join(self.raw_dir, "masks", f"{stem}.tif")

        out_img_dir = os.path.join(self.processed_dir, "test_tiles", "images")
        out_mask_dir = os.path.join(self.processed_dir, "test_tiles", "masks")

        # 1100 // 512 = 2 rows, 1200 // 512 = 2 cols -> 4 complete tiles
        num_tiles, counter = tile_orthophoto_pair(
            img_path=img_path,
            mask_path=mask_path,
            output_img_dir=out_img_dir,
            output_mask_dir=out_mask_dir,
            source_stem=stem,
            tile_size=512,
            stride=512,
            edge_handling="discard",
        )

        self.assertEqual(num_tiles, 4)
        saved_imgs = os.listdir(out_img_dir)
        saved_masks = os.listdir(out_mask_dir)
        self.assertEqual(len(saved_imgs), 4)
        self.assertEqual(len(saved_masks), 4)

        # Inspect first saved tile
        first_img = load_image(os.path.join(out_img_dir, saved_imgs[0]))
        first_mask = load_mask(os.path.join(out_mask_dir, saved_masks[0]))

        self.assertEqual(first_img.shape, (512, 512, 3))
        self.assertEqual(first_mask.shape, (512, 512))
        self.assertTrue(np.all((first_mask >= 0) & (first_mask <= 4)))

    def test_dataset_and_class_distribution(self):
        """Test end-to-end preparation pipeline on synthetic data and verify Dataset class."""
        stats = prepare_landcover_dataset(
            raw_dir=self.raw_dir,
            processed_dir=self.processed_dir,
            tile_size=512,
            stride=512,
            edge_handling="discard",
            seed=42,
        )

        # 4 source images * 4 tiles each = 16 tiles total
        self.assertEqual(stats["total_tiles"], 16)
        self.assertIn("train", stats["splits"])
        self.assertIn("val", stats["splits"])
        self.assertIn("test", stats["splits"])

        # Check class distribution calculations
        class_dist = stats["class_distribution"]
        self.assertEqual(len(class_dist), 5)
        total_pct = sum(class_dist[c]["percentage"] for c in range(5))
        self.assertAlmostEqual(total_pct, 100.0, places=2)

        # Test LandCoverDataset instantiation and item loading
        # Find which split has tiles
        for split in ["train", "val", "test"]:
            if stats["splits"][split]["tiles_count"] > 0:
                ds = LandCoverDataset(root_dir=self.processed_dir, split=split)
                self.assertGreater(len(ds), 0)
                raw_img, raw_mask, sample_id = ds.get_raw_sample(0)
                self.assertEqual(raw_img.shape, (512, 512, 3))
                self.assertEqual(raw_mask.shape, (512, 512))
                self.assertTrue(np.all(raw_mask >= 0))
                self.assertTrue(np.all(raw_mask <= 4))
                break


if __name__ == "__main__":
    unittest.main()


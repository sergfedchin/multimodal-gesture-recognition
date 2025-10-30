"""Data saving utilities for processed images and annotations across all splits."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
from PIL import Image
import torch

logger = logging.getLogger("gesture_processing")


class DataSaver:
    """Handles saving of processed data in required structure for all splits."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize data saver.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.output_folder: Path = config["paths"]["output_folder"]
        self.gesture_name: str = config["paths"]["gesture_folder"].name
        self.image_format: str = config["processing"]["image_format"]
        self.save_torch: bool = config["processing"]["save_torch"]

        # Store annotations in memory (small footprint)
        self.annotations = {}

        # Create output directory structure
        self._create_directory_structure()

    def _create_directory_structure(self) -> None:
        """Create required output directory structure."""
        gesture_folder = self.output_folder / self.gesture_name

        folders = [
            gesture_folder,
            gesture_folder / "rgb" / "body" / "png",
            gesture_folder / "rgb" / "hands" / "png",
            gesture_folder / "depth" / "body" / "png",
            gesture_folder / "depth" / "hands" / "png",
        ] + (
            [
                gesture_folder / "rgb" / "body" / "torch",
                gesture_folder / "rgb" / "hands" / "torch",
                gesture_folder / "depth" / "body" / "torch",
                gesture_folder / "depth" / "hands" / "torch",
            ]
            if self.save_torch
            else []
        )

        for folder in folders:
            folder.mkdir(parents=True, exist_ok=True)

        logger.info(f"Created output directory structure at: {gesture_folder}")

    def save_body_data(
        self, img_id: str, rgb_image: Image.Image, depth_map: np.ndarray
    ) -> None:
        """
        Save body RGB image and depth map.

        Args:
            img_id: Image identifier (with split prefix)
            rgb_image: RGB image (PIL)
            depth_map: Depth map (numpy array)
        """
        gesture_folder = self.output_folder / self.gesture_name

        # Save RGB
        self._save_rgb_image(gesture_folder / "rgb" / "body", img_id, rgb_image)

        # Save depth
        self._save_depth_map(gesture_folder / "depth" / "body", img_id, depth_map)

    def save_hand_data(
        self, img_id: str, hand_idx: int, rgb_image: Image.Image, depth_map: np.ndarray
    ) -> None:
        """
        Save hand RGB image and depth map.

        Args:
            img_id: Image identifier (with split prefix)
            hand_idx: Hand index (0, 1, ...)
            rgb_image: RGB image (PIL)
            depth_map: Depth map (numpy array)
        """
        gesture_folder = self.output_folder / self.gesture_name
        hand_id = f"{img_id}_{hand_idx}"

        # Save RGB
        self._save_rgb_image(gesture_folder / "rgb" / "hands", hand_id, rgb_image)

        # Save depth
        self._save_depth_map(gesture_folder / "depth" / "hands", hand_id, depth_map)

    def _save_rgb_image(self, base_path: Path, img_id: str, image: Image.Image) -> None:
        """Save RGB image in both PNG and torch formats."""
        # Save PNG
        png_path = base_path / "png" / f"{img_id}.png"
        image.save(png_path, "PNG")

        if self.save_torch:
            # Save torch tensor
            torch_path = base_path / "torch" / f"{img_id}.pth"
            img_array = np.array(image).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_array).to(torch.float16)
            torch.save(img_tensor, torch_path)

    def _save_depth_map(
        self, base_path: Path, img_id: str, depth_map: np.ndarray
    ) -> None:
        """Save depth map in both PNG and torch formats."""
        # Normalize depth for visualization
        depth_min, depth_max = depth_map.min(), depth_map.max()
        if depth_max - depth_min > 0:
            depth_normalized = (
                (depth_map - depth_min) / (depth_max - depth_min) * 255
            ).astype(np.uint8)
        else:
            depth_normalized = np.zeros_like(depth_map, dtype=np.uint8)

        # Save PNG
        png_path = base_path / "png" / f"{img_id}.png"
        depth_img = Image.fromarray(depth_normalized, mode="L")
        depth_img.save(png_path, "PNG")

        if self.save_torch:
            # Save torch tensor (raw depth values)
            torch_path = base_path / "torch" / f"{img_id}.pth"
            depth_tensor = torch.from_numpy(depth_map).to(torch.float16)
            torch.save(depth_tensor, torch_path)

    def add_annotation_entry(
        self,
        img_id: str,
        split_name: str,
        original_id: str,
        body_bbox: List[float],
        adjusted_hands_bboxes: List[List[float]],
    ) -> None:
        """
        Add an annotation entry (stored in memory, saved at the end).

        Args:
            img_id: Full image ID with split prefix
            split_name: Split name
            original_id: Original image ID
            body_bbox: Body bounding box
            adjusted_hands_bboxes: Adjusted hand bounding boxes
        """
        self.annotations[img_id] = {
            "split": split_name,
            "original_id": original_id,
            "body_bbox": body_bbox,
            "adjusted_hands_bboxes": adjusted_hands_bboxes,
        }

    def save_annotations(self) -> None:
        """Save all accumulated annotations to JSON file."""
        gesture_folder = self.output_folder / self.gesture_name
        annotations_path = gesture_folder / "annotations.json"

        with open(annotations_path, "w") as f:
            json.dump(self.annotations, f, indent=4)

        logger.info(f"Saved annotations to: {annotations_path}")

        # Log statistics by split
        split_counts = {}
        for img_id, ann in self.annotations.items():
            split = ann["split"]
            split_counts[split] = split_counts.get(split, 0) + 1

        logger.info("=" * 60)
        logger.info("PROCESSING SUMMARY:")
        for split, count in sorted(split_counts.items()):
            logger.info(f"  {split}: {count} images")
        logger.info(f"  TOTAL: {len(self.annotations)} images")
        logger.info("=" * 60)

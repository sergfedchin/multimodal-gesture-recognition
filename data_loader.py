"""
Data loading utilities for multi-modal hand gesture classification
Handles RGB images, depth maps, and efficient data streaming
"""

import logging
from collections import Counter
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

logger = logging.getLogger(__name__)


class HandGestureDataset(Dataset):
    """
    Multi-modal dataset for hand gesture classification.
    Supports RGB, depth, or both modalities.

    Args:
        metadata_df: DataFrame with image paths and labels
        data_root: Root directory containing image files
        modality: "rgb", "depth", or "fusion"
        image_size: Target image size (assumed square)
        augmentation_cfg: Configuration for data augmentation
        device: Device to load tensors onto ("cpu" or "cuda")
    """

    def __init__(
        self,
        metadata_df: pd.DataFrame,
        data_root: str,
        modality: str = "fusion",
        image_size: int = 224,
        augmentation_cfg: Optional[Dict] = None,
        device: str = "cuda",
    ):
        self.metadata_df = metadata_df.reset_index(drop=True)
        self.data_root = Path(data_root)
        self.modality = modality
        self.image_size = image_size
        self.device = device

        # Create gesture label mapping
        self.gesture_classes = sorted(metadata_df["gesture_class"].unique().tolist())
        self.gesture_to_idx = {g: idx for idx, g in enumerate(self.gesture_classes)}
        self.idx_to_gesture = {idx: g for g, idx in self.gesture_to_idx.items()}

        logger.info(
            f"Dataset with {len(self.gesture_classes)} classes: {self.gesture_classes}"
        )

        # Initialize augmentation
        self.augmentation_cfg = augmentation_cfg or {}
        self.rgb_transform = self._build_rgb_transform(
            is_train=augmentation_cfg.get("enable_augmentation", False)
            if augmentation_cfg
            else False
        )
        self.depth_transform = self._build_depth_transform()

    def _build_rgb_transform(self, is_train: bool = False):
        """Build transformation pipeline for RGB images"""
        base_transforms = [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet statistics
                std=[0.229, 0.224, 0.225],
            ),
        ]

        augmentation = []
        if is_train and self.augmentation_cfg:
            if self.augmentation_cfg.get("random_flip", True):
                augmentation.append(transforms.RandomHorizontalFlip(p=0.5))
            if self.augmentation_cfg.get("random_rotation", True):
                degrees = self.augmentation_cfg.get("rotation_degrees", 10)
                augmentation.append(transforms.RandomRotation(degrees))
            if self.augmentation_cfg.get("color_jitter", True):
                augmentation.append(
                    transforms.ColorJitter(
                        brightness=self.augmentation_cfg.get(
                            "color_jitter_brightness", 0.2
                        ),
                        contrast=self.augmentation_cfg.get(
                            "color_jitter_contrast", 0.2
                        ),
                        saturation=self.augmentation_cfg.get(
                            "color_jitter_saturation", 0.2
                        ),
                        hue=self.augmentation_cfg.get("color_jitter_hue", 0.1),
                    )
                )

        augmentation.extend(
            [transforms.Resize((self.image_size, self.image_size)), *base_transforms]
        )

        if is_train and self.augmentation_cfg.get("random_erasing", False):
            augmentation.append(
                transforms.RandomErasing(
                    p=self.augmentation_cfg.get("erasing_probability", 0.1)
                )
            )

        return transforms.Compose(augmentation)

    def _build_depth_transform(self):
        """Build transformation pipeline for depth maps"""
        return transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                # Normalize to [0, 1] for depth (will be further normalized in model if needed)
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )

    def __len__(self) -> int:
        return len(self.metadata_df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.

        Returns:
            Dictionary with keys:
            - "rgb": [3, H, W] if modality in ["rgb", "fusion"]
            - "depth": [1, H, W] if modality in ["depth", "fusion"]
            - "label": scalar tensor with class index
            - "metadata": dict with gesture_class, original_id, hand_index, etc.
        """
        row = self.metadata_df.iloc[idx]

        output = {
            "label": torch.tensor(
                self.gesture_to_idx[row["gesture_class"]], dtype=torch.long
            )
        }

        # Store metadata - ALWAYS include all keys with default values
        # This ensures all samples have the same metadata structure for batching
        output["metadata"] = {
            "gesture_class": row["gesture_class"],
            "original_id": str(row["original_id"]),
            "hand_index": int(row["hand_index"]),
            "user_id": str(row["user_id"]) if pd.notna(row["user_id"]) else "",
            "age": float(row["age"]) if pd.notna(row["age"]) else -1.0,
            "gender": str(row["gender"]) if pd.notna(row["gender"]) else "",
            "race": str(row["race"]) if pd.notna(row["race"]) else "",
        }

        # Load RGB image if needed
        if self.modality in ["rgb", "fusion"]:
            rgb_path = self.data_root / row["rgb_image_path"]
            try:
                from PIL import Image

                rgb_image = Image.open(rgb_path).convert("RGB")
                output["rgb"] = self.rgb_transform(rgb_image)
            except Exception as e:
                logger.warning(f"Failed to load RGB image {rgb_path}: {e}")
                output["rgb"] = torch.zeros((3, self.image_size, self.image_size))

        # Load depth map if needed
        if self.modality in ["depth", "fusion"]:
            depth_path = self.data_root / row["depth_image_path"]
            try:
                from PIL import Image

                depth_image = Image.open(depth_path).convert(
                    "L"
                )  # Convert to grayscale
                output["depth"] = self.depth_transform(depth_image)
            except Exception as e:
                logger.warning(f"Failed to load depth image {depth_path}: {e}")
                output["depth"] = torch.zeros((1, self.image_size, self.image_size))

        return output


def create_dataloaders(
    config: Dict,
    device: str = "cuda",
    drop_last_train: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader, Dict]:
    """
    Create train, validation, validation metrics, and test dataloaders with STRATIFIED subsampling.
    
    Args:
        config: Configuration dictionary
        device: Device for tensors
        drop_last_train: Whether to drop incomplete batches in training
        
    Returns:
        (train_loader, val_loader, val_metrics_loader, test_loader, dataset_info)
    """
    
    # Load parquet files
    train_df = pd.read_parquet(config["dataset"]["train_parquet"])
    val_df = pd.read_parquet(config["dataset"]["val_parquet"])
    test_df = pd.read_parquet(config["dataset"]["test_parquet"])
    
    logger.info(f"Original dataset sizes: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    
    # Apply STRATIFIED subsampling to preserve class distribution
    seed = config.get("seed", 42)
    
    if config["dataset"]["train_subsample"] < 1.0:
        train_size = int(len(train_df) * config["dataset"]["train_subsample"])
        train_df, _ = train_test_split(
            train_df,
            train_size=train_size,
            stratify=train_df["label"],
            random_state=seed
        )
        logger.info(f"Train set stratified subsample: {len(train_df)} samples")
    
    if config["dataset"]["val_subsample"] < 1.0:
        val_size = int(len(val_df) * config["dataset"]["val_subsample"])
        val_df, _ = train_test_split(
            val_df,
            train_size=val_size,
            stratify=val_df["label"],
            random_state=seed
        )
        logger.info(f"Val set stratified subsample: {len(val_df)} samples")
    
    if config["dataset"]["test_subsample"] < 1.0:
        test_size = int(len(test_df) * config["dataset"]["test_subsample"])
        test_df, _ = train_test_split(
            test_df,
            train_size=test_size,
            stratify=test_df["label"],
            random_state=seed
        )
        logger.info(f"Test set stratified subsample: {len(test_df)} samples")
    
    logger.info(f"After stratified subsampling: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    
    # Create datasets
    train_dataset = HandGestureDataset(
        metadata_df=train_df,
        data_root=config["dataset"]["data_root"],
        modality=config["model"]["modality"],
        image_size=config["model"]["image_size"],
        augmentation_cfg=config.get("augmentation", {}),
        device=device,
    )
    
    val_dataset = HandGestureDataset(
        metadata_df=val_df,
        data_root=config["dataset"]["data_root"],
        modality=config["model"]["modality"],
        image_size=config["model"]["image_size"],
        augmentation_cfg=None,  # No augmentation for validation
        device=device,
    )
    
    test_dataset = HandGestureDataset(
        metadata_df=test_df,
        data_root=config["dataset"]["data_root"],
        modality=config["model"]["modality"],
        image_size=config["model"]["image_size"],
        augmentation_cfg=None,  # No augmentation for testing
        device=device,
    )
    
    # Create smaller validation metrics subset with STRATIFIED sampling for fast metrics during training
    val_metrics_subset_size = config["evaluation"].get("val_metrics_subset_size", 0)
    
    if val_metrics_subset_size > 0 and val_metrics_subset_size < len(val_dataset):
        logger.info(f"Creating stratified validation metrics subset of size {val_metrics_subset_size} for periodic detailed metrics")
        
        # Get labels for stratification
        val_labels = val_df["label"].tolist()
        
        # Stratified sampling
        all_indices = list(range(len(val_dataset)))
        val_metrics_indices, _ = train_test_split(
            all_indices,
            train_size=val_metrics_subset_size,
            stratify=val_labels,
            random_state=seed
        )
        
        val_metrics_dataset = Subset(val_dataset, val_metrics_indices)
        
        # Log class distribution verification
        subset_labels = [val_labels[i] for i in val_metrics_indices]
        original_dist = Counter(val_labels)
        subset_dist = Counter(subset_labels)
        
        logger.info("Validation metrics subset class distribution:")
        logger.info(f"  Full validation: {len(val_labels)} samples, {len(original_dist)} classes")
        logger.info(f"  Metrics subset: {len(subset_labels)} samples, {len(subset_dist)} classes")
        
    else:
        val_metrics_dataset = val_dataset
        logger.info("Using full validation set for metrics computation (no separate subset)")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["dataset"]["num_workers"],
        pin_memory=config["hardware"]["pin_memory"],
        drop_last=drop_last_train,
        persistent_workers=True,
        prefetch_factor=4,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["dataset"]["num_workers"],
        pin_memory=config["hardware"]["pin_memory"],
        drop_last=False,
    )
    
    val_metrics_loader = DataLoader(
        val_metrics_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["dataset"]["num_workers"],
        pin_memory=config["hardware"]["pin_memory"],
        drop_last=False,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["dataset"]["num_workers"],
        pin_memory=config["hardware"]["pin_memory"],
        drop_last=False,
    )
    
    dataset_info = {
        "num_classes": len(train_dataset.gesture_classes),
        "gesture_classes": train_dataset.gesture_classes,
        "gesture_to_idx": train_dataset.gesture_to_idx,
        "idx_to_gesture": train_dataset.idx_to_gesture,
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "val_metrics_size": len(val_metrics_dataset),
        "test_size": len(test_dataset),
    }
    
    logger.info("Final dataloader sizes:")
    logger.info(f"  Train: {dataset_info['train_size']} samples")
    logger.info(f"  Validation (full): {dataset_info['val_size']} samples - for loss computation")
    logger.info(f"  Validation (metrics): {dataset_info['val_metrics_size']} samples - for detailed metrics")
    logger.info(f"  Test: {dataset_info['test_size']} samples")
    
    return train_loader, val_loader, val_metrics_loader, test_loader, dataset_info

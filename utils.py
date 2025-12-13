"""
Utility functions for inference and model analysis
"""

import os
import random
import numpy as np
import toml
import torch
import torch.nn as nn
from typing import Dict, List, Tuple
import logging
import json

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility across all libraries.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

    # Make cudnn deterministic (may reduce performance slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set Python hash seed for additional reproducibility
    os.environ["PYTHONHASHSEED"] = str(seed)

    logger.info(f"Random seed set to {seed}")


def load_config(config_path: str) -> dict:
    """Load configuration from TOML file"""
    with open(config_path, "r") as f:
        config = toml.load(f)
    logger.info(f"Loaded config from {config_path}")
    return config


def setup_device(config: dict) -> torch.device:
    """Setup GPU/CPU device"""
    if config["hardware"]["device"] == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA device. GPUs available: {torch.cuda.device_count()}")
        if config["hardware"]["num_gpus"] > 1:
            logger.warning("Multi-GPU training not yet implemented. Using single GPU.")
    else:
        device = torch.device("cpu")
        logger.warning("Using CPU. Training will be slow.")

    return device


class ModelInference:
    """
    Inference utility for making predictions on new images
    """

    def __init__(
        self,
        model: nn.Module,
        checkpoint_path: str,
        config: Dict,
        device: torch.device,
    ):
        self.model = model
        self.device = device
        self.config = config

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        logger.info(f"Loaded model from {checkpoint_path}")

    @torch.no_grad()
    def predict_batch(
        self,
        rgb: torch.Tensor = None,
        depth: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions on a batch.

        Args:
            rgb: [B, 3, 224, 224] RGB images or None
            depth: [B, 1, 224, 224] Depth maps or None

        Returns:
            (predictions, confidences): Class indices and confidence scores
        """
        modality = self.config["model"]["modality"]

        # Move to device
        if rgb is not None:
            rgb = rgb.to(self.device)
        if depth is not None:
            depth = depth.to(self.device)

        # Forward pass
        if modality == "rgb":
            logits = self.model(rgb)
        elif modality == "depth":
            logits = self.model(depth)
        elif modality == "fusion":
            logits = self.model(rgb, depth)
        else:
            raise ValueError(f"Unknown modality: {modality}")

        # Get predictions and confidences
        probs = torch.softmax(logits, dim=1)
        confidences, predictions = torch.max(probs, dim=1)

        return predictions.cpu(), confidences.cpu()

    @torch.no_grad()
    def predict_single(
        self,
        rgb: torch.Tensor = None,
        depth: torch.Tensor = None,
        gesture_classes: List[str] = None,
    ) -> Dict:
        """
        Make prediction on single sample with interpretation.

        Args:
            rgb: [1, 3, 224, 224] or [3, 224, 224]
            depth: [1, 1, 224, 224] or [1, 224, 224]
            gesture_classes: List of gesture class names

        Returns:
            Dictionary with prediction, confidence, and top-5 results
        """
        # Add batch dimension if needed
        if rgb is not None and rgb.ndim == 3:
            rgb = rgb.unsqueeze(0)
        if depth is not None and depth.ndim == 3:
            depth = depth.unsqueeze(0)

        # Get predictions
        pred_idx, confidence = self.predict_batch(rgb, depth)
        pred_idx = pred_idx[0].item()
        confidence = confidence[0].item()

        result = {
            "predicted_class_idx": pred_idx,
            "confidence": float(confidence),
        }

        if gesture_classes:
            result["predicted_class"] = gesture_classes[pred_idx]

        # Get top-5 predictions
        if rgb is not None:
            logits = self.model(rgb.to(self.device))
        else:
            logits = self.model(depth.to(self.device))

        probs = torch.softmax(logits, dim=1)[0]
        top5_probs, top5_indices = torch.topk(probs, k=5)

        top5 = []
        for prob, idx in zip(top5_probs.cpu(), top5_indices.cpu()):
            top5_item = {
                "class_idx": int(idx),
                "probability": float(prob),
            }
            if gesture_classes:
                top5_item["class"] = gesture_classes[int(idx)]
            top5.append(top5_item)

        result["top5"] = top5

        return result


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Count total and trainable parameters.

    Returns:
        (total_params, trainable_params)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def freeze_backbone(model: nn.Module):
    """Freeze VSSD backbone parameters (for transfer learning)"""
    if hasattr(model, "backbone"):
        for param in model.backbone.parameters():
            param.requires_grad = False
    elif hasattr(model, "rgb_backbone"):
        for param in model.rgb_backbone.parameters():
            param.requires_grad = False
    elif hasattr(model, "depth_backbone"):
        for param in model.depth_backbone.parameters():
            param.requires_grad = False

    logger.info("Froze backbone parameters")


def unfreeze_backbone(model: nn.Module):
    """Unfreeze VSSD backbone parameters"""
    if hasattr(model, "backbone"):
        for param in model.backbone.parameters():
            param.requires_grad = True
    elif hasattr(model, "rgb_backbone"):
        for param in model.rgb_backbone.parameters():
            param.requires_grad = True
    elif hasattr(model, "depth_backbone"):
        for param in model.depth_backbone.parameters():
            param.requires_grad = True

    logger.info("Unfroze backbone parameters")


def get_learning_rates(model: nn.Module) -> Dict:
    """
    Get current learning rates for different parameter groups.
    Useful for debugging learning rate scheduler.
    """
    lr_dict = {}
    for i, param_group in enumerate(model.parameters()):
        if hasattr(param_group, "lr"):
            lr_dict[f"group_{i}"] = param_group.lr
    return lr_dict


def export_config(config: Dict, output_path: str):
    """Export configuration to JSON for reproducibility"""
    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)
    logger.info(f"Exported config to {output_path}")


def analyze_model_architecture(model: nn.Module, input_shapes: Dict) -> Dict:
    """
    Analyze model architecture and create detailed summary.

    Args:
        model: PyTorch model
        input_shapes: Dict with input shapes, e.g., {"rgb": (1, 3, 224, 224)}

    Returns:
        Dictionary with architecture analysis
    """
    analysis = {
        "num_layers": len(list(model.modules())),
        "num_parameters": count_parameters(model),
        "trainable_parameters": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
    }

    # Estimate FLOPs (rough approximation)
    try:
        from fvcore.nn import FlopCounterMode

        if "rgb" in input_shapes and "depth" in input_shapes:
            # Fusion model
            rgb = torch.randn(input_shapes["rgb"])
            depth = torch.randn(input_shapes["depth"])
            with FlopCounterMode(model) as fcm:
                _ = model(rgb, depth)
            analysis["estimated_flops"] = fcm.flop_counts
        elif "rgb" in input_shapes:
            # RGB model
            rgb = torch.randn(input_shapes["rgb"])
            with FlopCounterMode(model) as fcm:
                _ = model(rgb)
            analysis["estimated_flops"] = fcm.flop_counts
        elif "depth" in input_shapes:
            # Depth model
            depth = torch.randn(input_shapes["depth"])
            with FlopCounterMode(model) as fcm:
                _ = model(depth)
            analysis["estimated_flops"] = fcm.flop_counts
    except ImportError:
        logger.warning("fvcore not installed, skipping FLOPs calculation")

    return analysis


def compare_configs(config1: Dict, config2: Dict) -> Dict:
    """
    Compare two configurations and return differences.
    Useful for ablation studies.
    """
    diffs = {}

    all_keys = set(config1.keys()) | set(config2.keys())

    for key in all_keys:
        if key not in config1:
            diffs[key] = f"Only in config2: {config2[key]}"
        elif key not in config2:
            diffs[key] = f"Only in config1: {config1[key]}"
        elif isinstance(config1[key], dict) and isinstance(config2[key], dict):
            # Recursively compare nested dicts
            nested_diffs = compare_configs(config1[key], config2[key])
            if nested_diffs:
                diffs[key] = nested_diffs
        elif config1[key] != config2[key]:
            diffs[key] = {
                "config1": config1[key],
                "config2": config2[key],
            }

    return diffs

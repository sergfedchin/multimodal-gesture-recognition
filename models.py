"""
Multi-modal model architectures using VSSD as backbone
Supports RGB-only, depth-only, and fusion approaches
"""

import torch
import torch.nn as nn
from typing import Dict
import logging

from vssd_integration import build_vssd_backbone

logger = logging.getLogger(__name__)


def _make_classification_head(
    in_dim: int, num_classes: int, dropout: float
) -> nn.Module:
    """
    Create classification head.

    Args:
        in_dim: Input feature dimension
        num_classes: Number of output classes
        dropout: Dropout rate

    Returns:
        Classification head module
    """
    return nn.Sequential(
        nn.Linear(in_dim, in_dim // 2),
        nn.BatchNorm1d(in_dim // 2),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(in_dim // 2, num_classes),
    )


class GatedFusionVec(nn.Module):
    """
    Gated fusion module for combining RGB and depth feature vectors.
    Uses learned gating mechanism to adaptively weight each modality.

    Input shapes:
    - rgb_vec: [B, C]
    - depth_vec: [B, C]

    Output: [B, C]
    """

    def __init__(self, feature_dim: int):
        super().__init__()
        self.feature_dim = feature_dim

        # Gate network: learns to weight RGB vs Depth
        self.gate = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, feature_dim),
            nn.Sigmoid(),
        )

    def forward(self, rgb_vec: torch.Tensor, depth_vec: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rgb_vec: [B, C]
            depth_vec: [B, C]

        Returns:
            fused: [B, C]
        """
        # Convert from tTensor to regular tensor if needed
        # Use as_subclass to properly convert tensor subclasses
        if type(rgb_vec).__name__ == "tTensor":
            rgb_vec = rgb_vec.as_subclass(torch.Tensor)
        if type(depth_vec).__name__ == "tTensor":
            depth_vec = depth_vec.as_subclass(torch.Tensor)

        # Concatenate
        combined = torch.cat([rgb_vec, depth_vec], dim=1)  # [B, 2C]

        # Learn gating weights
        gate_weights = self.gate(combined)  # [B, C]

        # Apply gating
        fused = rgb_vec * gate_weights + depth_vec * (1 - gate_weights)

        return fused


class AdaptiveFusionVec(nn.Module):
    """
    Adaptive fusion that learns per-channel weights for RGB and depth vectors.
    """

    def __init__(self, feature_dim: int):
        super().__init__()
        self.feature_dim = feature_dim

        # Channel-wise attention for RGB and Depth
        self.rgb_attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 16),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim // 16, feature_dim),
            nn.Sigmoid(),
        )

        self.depth_attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 16),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim // 16, feature_dim),
            nn.Sigmoid(),
        )

    def forward(self, rgb_vec: torch.Tensor, depth_vec: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rgb_vec: [B, C]
            depth_vec: [B, C]

        Returns:
            fused: [B, C]
        """
        # Convert from tTensor to regular tensor if needed
        if type(rgb_vec).__name__ == "tTensor":
            rgb_vec = rgb_vec.as_subclass(torch.Tensor)
        if type(depth_vec).__name__ == "tTensor":
            depth_vec = depth_vec.as_subclass(torch.Tensor)

        # Compute channel-wise attention for each modality
        rgb_weight = self.rgb_attention(rgb_vec)  # [B, C]
        depth_weight = self.depth_attention(depth_vec)  # [B, C]

        # Normalize weights to sum to 1
        total_weight = rgb_weight + depth_weight + 1e-8
        rgb_weight = rgb_weight / total_weight
        depth_weight = depth_weight / total_weight

        # Fuse with learned weights
        fused = rgb_vec * rgb_weight + depth_vec * depth_weight

        return fused


class ConcatFusionVec(nn.Module):
    """Simple concatenation + MLP fusion for vectors"""

    def __init__(self, feature_dim: int):
        super().__init__()
        self.fusion_mlp = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, rgb_vec: torch.Tensor, depth_vec: torch.Tensor) -> torch.Tensor:
        # Convert from tTensor to regular tensor if needed
        if type(rgb_vec).__name__ == "tTensor":
            rgb_vec = rgb_vec.as_subclass(torch.Tensor)
        if type(depth_vec).__name__ == "tTensor":
            depth_vec = depth_vec.as_subclass(torch.Tensor)

        combined = torch.cat([rgb_vec, depth_vec], dim=1)  # [B, 2C]
        fused = self.fusion_mlp(combined)  # [B, C]
        return fused


class RGBOnlyClassifier(nn.Module):
    """
    Single-modality RGB classifier using VSSD backbone.

    Architecture:
    Input RGB: [B, 3, 224, 224]
        ↓
    VSSD Backbone (pretrained on ImageNet)
        ↓
    Pooled features: [B, feat_dim]
        ↓
    Classification Head (FC layers + dropout)
        ↓
    Output logits: [B, num_classes]
    """

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config

        # Build VSSD backbone
        self.backbone, self.feat_dim = build_vssd_backbone(
            variant=config["model"]["vssd_variant"],
            image_size=config["model"]["image_size"],
            in_chans=3,
            vssd_repo_root=config["model"]["vssd_repo_path"],
            pretrained_ckpt=config["model"].get("pretrained_ckpt", ""),
        )

        # Classification head
        self.head = _make_classification_head(
            self.feat_dim, config["model"]["num_classes"], config["model"]["dropout"]
        )

        logger.info(
            f"Built RGB-only classifier with VSSD-{config['model']['vssd_variant'].upper()}"
        )
        logger.info(f"  Feature dimension: {self.feat_dim}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, 224, 224] - RGB image

        Returns:
            logits: [B, num_classes]
        """
        # VSSD backbone outputs pooled features
        features = self.backbone(x)  # [B, feat_dim]

        # Convert from VSSD's tTensor to regular torch.Tensor
        if type(features).__name__ == 'tTensor':
            features = features.as_subclass(torch.Tensor)

        # Classification
        logits = self.head(features)  # [B, num_classes]

        return logits


class DepthOnlyClassifier(nn.Module):
    """
    Single-modality depth classifier using VSSD backbone.

    Architecture:
    Input Depth: [B, 1, 224, 224]
        ↓
    Channel expansion to 3 channels
        ↓
    VSSD Backbone
        ↓
    Pooled features: [B, feat_dim]
        ↓
    Classification Head
        ↓
    Output logits: [B, num_classes]
    """

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config

        # Convert single-channel depth to 3-channel for pretrained weights
        self.depth_to_rgb = nn.Conv2d(1, 3, kernel_size=1)

        # Build VSSD backbone
        self.backbone, self.feat_dim = build_vssd_backbone(
            variant=config["model"]["vssd_variant"],
            image_size=config["model"]["image_size"],
            in_chans=3,
            vssd_repo_root=config["model"]["vssd_repo_path"],
            pretrained_ckpt=config["model"].get("pretrained_ckpt", ""),
        )

        # Classification head
        self.head = _make_classification_head(
            self.feat_dim, config["model"]["num_classes"], config["model"]["dropout"]
        )

        logger.info(
            f"Built depth-only classifier with VSSD-{config['model']['vssd_variant'].upper()}"
        )
        logger.info(f"  Feature dimension: {self.feat_dim}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 1, 224, 224] - Depth map

        Returns:
            logits: [B, num_classes]
        """
        # Expand depth to 3 channels
        x = self.depth_to_rgb(x)  # [B, 3, 224, 224]

        # VSSD backbone
        features = self.backbone(x)  # [B, feat_dim]

        # Convert from VSSD's tTensor to regular torch.Tensor
        if type(features).__name__ == 'tTensor':
            features = features.as_subclass(torch.Tensor)

        # Classification
        logits = self.head(features)  # [B, num_classes]

        return logits


class DualBranchFusionClassifier(nn.Module):
    """
    Dual-branch fusion classifier using two VSSD backbones.

    Architecture:
    RGB Input: [B, 3, 224, 224]        Depth Input: [B, 1, 224, 224]
         ↓                                    ↓
    VSSD RGB Branch                     Channel Expansion + VSSD Depth Branch
         ↓                                    ↓
    RGB Features: [B, feat_dim]         Depth Features: [B, feat_dim]
         ↓                                    ↓
              Fusion Module (gated/adaptive/concat)
                        ↓
              Fused Features: [B, feat_dim]
                        ↓
              Classification Head
                        ↓
              Output logits: [B, num_classes]
    """

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.fusion_method = config["model"]["fusion_method"]

        # RGB branch
        self.rgb_backbone, self.feat_dim = build_vssd_backbone(
            variant=config["model"]["vssd_variant"],
            image_size=config["model"]["image_size"],
            in_chans=3,
            vssd_repo_root=config["model"]["vssd_repo_path"],
            pretrained_ckpt=config["model"].get("pretrained_ckpt", ""),
        )

        # Depth branch with channel expansion
        self.depth_to_rgb = nn.Conv2d(1, 3, kernel_size=1)
        self.depth_backbone, _ = build_vssd_backbone(
            variant=config["model"]["vssd_variant"],
            image_size=config["model"]["image_size"],
            in_chans=3,
            vssd_repo_root=config["model"]["vssd_repo_path"],
            pretrained_ckpt=config["model"].get("pretrained_ckpt", ""),
        )

        # Fusion module
        if self.fusion_method == "gated_fusion":
            self.fusion = GatedFusionVec(self.feat_dim)
        elif self.fusion_method == "adaptive_fusion":
            self.fusion = AdaptiveFusionVec(self.feat_dim)
        elif self.fusion_method == "concat":
            self.fusion = ConcatFusionVec(self.feat_dim)
        elif self.fusion_method == "addition":
            # Create a simple wrapper for addition that handles tTensor
            def addition_fusion(rgb, depth):
                if type(rgb).__name__ == 'tTensor':
                    rgb = rgb.as_subclass(torch.Tensor)
                if type(depth).__name__ == 'tTensor':
                    depth = depth.as_subclass(torch.Tensor)
                return rgb + depth
            self.fusion = addition_fusion
        elif self.fusion_method == "weighted_sum":
            self.alpha = nn.Parameter(torch.tensor(0.5))
            def weighted_fusion(rgb, depth):
                if type(rgb).__name__ == 'tTensor':
                    rgb = rgb.as_subclass(torch.Tensor)
                if type(depth).__name__ == 'tTensor':
                    depth = depth.as_subclass(torch.Tensor)
                return self.alpha * rgb + (1 - self.alpha) * depth
            self.fusion = weighted_fusion

        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")

        # Classification head
        self.head = _make_classification_head(
            self.feat_dim, config["model"]["num_classes"], config["model"]["dropout"]
        )

        logger.info(
            f"Built dual-branch fusion classifier with VSSD-{config['model']['vssd_variant'].upper()}"
        )
        logger.info(f"  Fusion method: {self.fusion_method}")
        logger.info(f"  Feature dimension: {self.feat_dim}")

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rgb: [B, 3, 224, 224] - RGB image
            depth: [B, 1, 224, 224] - Depth map

        Returns:
            logits: [B, num_classes]
        """
        # RGB branch
        rgb_features = self.rgb_backbone(rgb)  # [B, feat_dim]

        # Depth branch
        depth_expanded = self.depth_to_rgb(depth)  # [B, 3, 224, 224]
        depth_features = self.depth_backbone(depth_expanded)  # [B, feat_dim]

        # Fusion (conversion handled inside fusion modules now)
        fused_features = self.fusion(rgb_features, depth_features)  # [B, feat_dim]

        # Classification
        logits = self.head(fused_features)  # [B, num_classes]

        return logits


def build_model(config: Dict) -> nn.Module:
    """
    Factory function to build the appropriate model based on configuration.

    Args:
        config: Configuration dictionary with model settings

    Returns:
        model: PyTorch model
    """
    modality = config["model"]["modality"]

    if modality == "rgb":
        model = RGBOnlyClassifier(config)
    elif modality == "depth":
        model = DepthOnlyClassifier(config)
    elif modality == "fusion":
        model = DualBranchFusionClassifier(config)
    else:
        raise ValueError(f"Unknown modality: {modality}")

    return model

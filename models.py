"""
Multi-modal model architectures with configurable backbones
Supports RGB-only, depth-only, and fusion approaches with multiple architectures
"""

import logging
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

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


# ============================================================================
# FEATURE EXTRACTORS
# ============================================================================


def build_feature_extractor(
    architecture: str,
    in_channels: int = 3,
    pretrained: bool = True,
    vssd_config: Dict = None,
    freeze_backbone: bool = False,
) -> Tuple[nn.Module, int]:
    """
    Build feature extractor based on architecture type.
    Args:
        architecture: Architecture name ("vssd", "resnet18", "resnet50", "vit")
        in_channels: Number of input channels
        pretrained: Whether to use pretrained weights
        vssd_config: Configuration for VSSD (if architecture="vssd")
    Returns:
        (extractor, feature_dim): Feature extractor and output dimension
    """
    architecture = architecture.lower()

    if architecture == "vssd":
        if vssd_config is None:
            raise ValueError("vssd_config required for VSSD architecture")
        from vssd_integration import build_vssd_backbone

        return build_vssd_backbone(
            variant=vssd_config["variant"],
            image_size=vssd_config["image_size"],
            in_chans=in_channels,
            vssd_repo_root=vssd_config["vssd_repo_path"],
            pretrained_ckpt=vssd_config.get("pretrained_ckpt", ""),
        )

    elif architecture == "resnet18":
        from torchvision.models import ResNet18_Weights, resnet18

        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = resnet18(weights=weights)

        if in_channels != 3:
            model.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

        # NEW: Freeze early layers if requested
        if freeze_backbone and pretrained:
            # Freeze conv1, bn1, layer1, layer2
            for name, param in model.named_parameters():
                if any(x in name for x in ["conv1", "bn1", "layer1", "layer2"]):
                    param.requires_grad = False
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in model.parameters())
            logger.info(
                f"Froze early ResNet18 layers: {trainable:,}/{total:,} trainable ({trainable / total * 100:.1f}%)"
            )

        # Remove classifier
        model = nn.Sequential(*list(model.children())[:-2])
        extractor = nn.Sequential(model, nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())
        feature_dim = 512
        logger.info(
            f"Built ResNet18 extractor (pretrained={pretrained}, frozen={freeze_backbone}), feature_dim={feature_dim}"
        )
        return extractor, feature_dim

    elif architecture == "resnet50":
        from torchvision.models import ResNet50_Weights, resnet50

        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        model = resnet50(weights=weights)

        # Modify first conv if not 3 channels
        if in_channels != 3:
            model.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

        # Remove classifier
        model = nn.Sequential(*list(model.children())[:-2])  # Remove avgpool and fc
        extractor = nn.Sequential(model, nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())
        feature_dim = 2048
        logger.info(
            f"Built ResNet50 extractor (pretrained={pretrained}), feature_dim={feature_dim}"
        )
        return extractor, feature_dim

    elif architecture == "vit":
        from torchvision.models import ViT_B_16_Weights, vit_b_16

        weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        model = vit_b_16(weights=weights)

        # Modify patch embedding if not 3 channels
        if in_channels != 3:
            old_conv = model.conv_proj
            model.conv_proj = nn.Conv2d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
            )

        # Remove classification head
        model.heads = nn.Identity()
        feature_dim = 768
        logger.info(
            f"Built ViT-B/16 extractor (pretrained={pretrained}), feature_dim={feature_dim}"
        )
        return model, feature_dim

    else:
        raise ValueError(f"Unknown architecture: {architecture}")


# ============================================================================
# FUSION MODULES
# ============================================================================


class GatedFusionVec(nn.Module):
    """
    Gated fusion module for combining RGB and depth feature vectors.
    Uses learned gating mechanism to adaptively weight each modality.
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


class CrossModalAttentionFusion(nn.Module):
    """
    Cross-modal attention fusion: RGB attends to depth and vice versa.
    This allows each modality to focus on relevant information from the other.
    """

    def __init__(self, feature_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads

        # Multi-head cross-attention: RGB queries depth
        self.rgb_to_depth_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Multi-head cross-attention: Depth queries RGB
        self.depth_to_rgb_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Layer normalization
        self.norm_rgb = nn.LayerNorm(feature_dim)
        self.norm_depth = nn.LayerNorm(feature_dim)

        # Fusion MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        logger.info(f"Built CrossModalAttentionFusion with {num_heads} heads")

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

        # Add sequence dimension for attention: [B, C] -> [B, 1, C]
        rgb_seq = rgb_vec.unsqueeze(1)  # [B, 1, C]
        depth_seq = depth_vec.unsqueeze(1)  # [B, 1, C]

        # Cross-attention: RGB attends to depth
        rgb_attended, _ = self.rgb_to_depth_attn(
            query=rgb_seq, key=depth_seq, value=depth_seq
        )  # [B, 1, C]
        rgb_attended = rgb_attended.squeeze(1)  # [B, C]
        rgb_enhanced = self.norm_rgb(rgb_vec + rgb_attended)  # Residual connection

        # Cross-attention: Depth attends to RGB
        depth_attended, _ = self.depth_to_rgb_attn(
            query=depth_seq, key=rgb_seq, value=rgb_seq
        )  # [B, 1, C]
        depth_attended = depth_attended.squeeze(1)  # [B, C]
        depth_enhanced = self.norm_depth(
            depth_vec + depth_attended
        )  # Residual connection

        # Fuse enhanced features
        combined = torch.cat([rgb_enhanced, depth_enhanced], dim=1)  # [B, 2C]
        fused = self.fusion_mlp(combined)  # [B, C]

        return fused


class ProbabilityFusion(nn.Module):
    """
    Probability-based fusion with multi-head attention mechanism.

    This module fuses logits from multiple modalities by:
    1. Converting logits to probability distributions via softmax
    2. Learning per-class, per-modality weights using multiple attention heads
    3. Combining head outputs with learned weights

    Benefits:
    - Interpretable: shows which modality contributes to each class prediction
    - Flexible: multiple heads allow different fusion strategies
    - Efficient: operates on probability distributions (post-feature extraction)

    Args:
        num_modalities: Number of input modalities (e.g., 2 for RGB + Depth)
        num_classes: Number of output classes
        num_heads: Number of attention heads (default=1 for simplicity)
    """

    def __init__(
        self, num_modalities: int = 2, num_classes: int = 34, num_heads: int = 1
    ):
        super().__init__()
        self.num_modalities = num_modalities
        self.num_classes = num_classes
        self.num_heads = num_heads

        # Learnable weights for each head: [num_heads, num_modalities, num_classes]
        # Each head learns different modality importance per class
        self.head_weights = nn.Parameter(
            torch.randn(num_heads, num_modalities, num_classes) * 0.01 + 1.0
        )

        # Learnable weights to combine outputs from different heads: [num_heads]
        self.head_combine = nn.Parameter(torch.ones(num_heads))

        logger.info(
            f"Built ProbabilityFusion with {num_modalities} modalities, "
            f"{num_classes} classes, {num_heads} head(s)"
        )

    def forward(
        self, logits_list: list
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Fuse logits from multiple modalities.

        Args:
            logits_list: List of logit tensors, one per modality
                        Each tensor has shape [B, num_classes]
                        Example: [rgb_logits, depth_logits]

        Returns:
            fused_probs: Fused probability distribution [B, num_classes]
            weights: Tuple of (head_normalized, combine_weights) for interpretability
                     head_normalized: [num_heads, num_modalities, num_classes]
                     combine_weights: [num_heads]
        """
        # Convert logits to probability distributions
        probs_list = [F.softmax(logits, dim=-1) for logits in logits_list]
        # probs_list: List[[B, num_classes], ...]

        # Normalize head weights across modalities (softmax over modality dimension)
        head_normalized = F.softmax(self.head_weights, dim=1)
        # head_normalized: [num_heads, num_modalities, num_classes]

        # Compute weighted probability distributions for each head
        head_outputs = []
        for h in range(self.num_heads):
            # For each head, compute weighted sum of modality probabilities
            weighted_probs = sum(
                probs * head_normalized[h, i, :].unsqueeze(0)
                for i, probs in enumerate(probs_list)
            )
            # weighted_probs: [B, num_classes]
            head_outputs.append(weighted_probs)

        # Normalize weights for combining heads (softmax over heads)
        combine_weights = F.softmax(self.head_combine, dim=0)
        # combine_weights: [num_heads]

        # Combine head outputs with learned weights
        fused_probs = sum(
            h_out * combine_weights[h] for h, h_out in enumerate(head_outputs)
        )
        # fused_probs: [B, num_classes]

        # Return fused probabilities and weights for interpretability
        return fused_probs, (head_normalized, combine_weights)


# ============================================================================
# CLASSIFIER MODELS
# ============================================================================


class RGBOnlyClassifier(nn.Module):
    """
    Single-modality RGB classifier with configurable backbone.
    """

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config

        # Build feature extractor
        architecture = config["model"]["architecture"]
        freeze_backbone = config["model"].get("freeze_backbone", False)

        vssd_config = None
        if architecture == "vssd":
            vssd_config = {
                "variant": config["model"]["vssd_variant"],
                "image_size": config["model"]["image_size"],
                "vssd_repo_path": config["model"]["vssd_repo_path"],
                "pretrained_ckpt": config["model"].get("pretrained_ckpt", ""),
            }

        self.backbone, self.feat_dim = build_feature_extractor(
            architecture=architecture,
            in_channels=3,
            pretrained=config["model"]["pretrained"],
            vssd_config=vssd_config,
            freeze_backbone=freeze_backbone,
        )

        # Classification head
        self.head = _make_classification_head(
            self.feat_dim, config["model"]["num_classes"], config["model"]["dropout"]
        )

        logger.info(f"Built RGB-only classifier with {architecture.upper()}")
        logger.info(f"  Feature dimension: {self.feat_dim}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, 224, 224] - RGB image
        Returns:
            logits: [B, num_classes]
        """
        # Extract features
        features = self.backbone(x)  # [B, feat_dim]

        # Convert from VSSD's tTensor to regular torch.Tensor if needed
        if type(features).__name__ == "tTensor":
            features = features.as_subclass(torch.Tensor)

        # Classification
        logits = self.head(features)  # [B, num_classes]

        return logits


class DepthOnlyClassifier(nn.Module):
    """
    Single-modality depth classifier with configurable backbone.
    """

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config

        # Convert single-channel depth to 3-channel for pretrained weights
        self.depth_to_rgb = nn.Conv2d(1, 3, kernel_size=1)

        # Build feature extractor
        architecture = config["model"]["architecture"]
        freeze_backbone = config["model"].get("freeze_backbone", False)

        vssd_config = None
        if architecture == "vssd":
            vssd_config = {
                "variant": config["model"]["vssd_variant"],
                "image_size": config["model"]["image_size"],
                "vssd_repo_path": config["model"]["vssd_repo_path"],
                "pretrained_ckpt": config["model"].get("pretrained_ckpt", ""),
            }

        self.backbone, self.feat_dim = build_feature_extractor(
            architecture=architecture,
            in_channels=3,
            pretrained=config["model"]["pretrained"],
            vssd_config=vssd_config,
            freeze_backbone=freeze_backbone,
        )

        # Classification head
        self.head = _make_classification_head(
            self.feat_dim, config["model"]["num_classes"], config["model"]["dropout"]
        )

        logger.info(f"Built depth-only classifier with {architecture.upper()}")
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

        # Extract features
        features = self.backbone(x)  # [B, feat_dim]

        # Convert from VSSD's tTensor to regular torch.Tensor if needed
        if type(features).__name__ == "tTensor":
            features = features.as_subclass(torch.Tensor)

        # Classification
        logits = self.head(features)  # [B, num_classes]

        return logits


class DualBranchFusionClassifier(nn.Module):
    """
    Dual-branch fusion classifier with configurable backbones and fusion strategies.
    """

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.fusion_method = config["model"]["fusion_method"]

        # Build RGB feature extractor
        architecture = config["model"]["architecture"]
        freeze_backbone = config["model"].get("freeze_backbone", False)

        vssd_config = None
        if architecture == "vssd":
            vssd_config = {
                "variant": config["model"]["vssd_variant"],
                "image_size": config["model"]["image_size"],
                "vssd_repo_path": config["model"]["vssd_repo_path"],
                "pretrained_ckpt": config["model"].get("pretrained_ckpt", ""),
            }

        self.rgb_backbone, self.feat_dim = build_feature_extractor(
            architecture=architecture,
            in_channels=3,
            pretrained=config["model"]["pretrained"],
            vssd_config=vssd_config,
            freeze_backbone=freeze_backbone,
        )

        # Depth branch with channel expansion
        self.depth_to_rgb = nn.Conv2d(1, 3, kernel_size=1)
        self.depth_backbone, _ = build_feature_extractor(
            architecture=architecture,
            in_channels=3,
            pretrained=config["model"]["pretrained"],
            vssd_config=vssd_config,
            freeze_backbone=freeze_backbone,
        )

        # RGB and Depth classification heads (for probability_fusion)
        if self.fusion_method == "probability_fusion":
            num_classes = config["model"]["num_classes"]

            self.rgb_head = _make_classification_head(
                self.feat_dim, num_classes, config["model"]["dropout"]
            )
            self.depth_head = _make_classification_head(
                self.feat_dim, num_classes, config["model"]["dropout"]
            )

            # Probability fusion module
            num_heads = config["model"].get("probability_fusion_heads", 1)
            self.fusion = ProbabilityFusion(
                num_modalities=2, num_classes=num_classes, num_heads=num_heads
            )

            # Store fusion weights for analysis
            self.last_fusion_weights = None

            logger.info(f"Built probability_fusion with {num_heads} head(s)")

        else:
            # Feature-level fusion modules
            if self.fusion_method == "gated_fusion":
                self.fusion = GatedFusionVec(self.feat_dim)
            elif self.fusion_method == "adaptive_fusion":
                self.fusion = AdaptiveFusionVec(self.feat_dim)
            elif self.fusion_method == "concat":
                self.fusion = ConcatFusionVec(self.feat_dim)
            elif self.fusion_method == "cross_attention":
                self.fusion = CrossModalAttentionFusion(
                    self.feat_dim, num_heads=8, dropout=config["model"]["dropout"]
                )
            elif self.fusion_method == "addition":
                # Simple addition wrapper
                def addition_fusion(rgb, depth):
                    if type(rgb).__name__ == "tTensor":
                        rgb = rgb.as_subclass(torch.Tensor)
                    if type(depth).__name__ == "tTensor":
                        depth = depth.as_subclass(torch.Tensor)
                    return rgb + depth

                self.fusion = addition_fusion
            elif self.fusion_method == "weighted_sum":
                self.alpha = nn.Parameter(torch.tensor(0.5))

                def weighted_fusion(rgb, depth):
                    if type(rgb).__name__ == "tTensor":
                        rgb = rgb.as_subclass(torch.Tensor)
                    if type(depth).__name__ == "tTensor":
                        depth = depth.as_subclass(torch.Tensor)
                    return self.alpha * rgb + (1 - self.alpha) * depth

                self.fusion = weighted_fusion
            else:
                raise ValueError(f"Unknown fusion method: {self.fusion_method}")

            # Classification head (for feature-level fusion)
            self.head = _make_classification_head(
                self.feat_dim,
                config["model"]["num_classes"],
                config["model"]["dropout"],
            )

        logger.info(f"Built dual-branch fusion classifier with {architecture.upper()}")
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

        if self.fusion_method == "probability_fusion":
            # Probability-level fusion
            # Convert features to tensors if needed
            if type(rgb_features).__name__ == "tTensor":
                rgb_features = rgb_features.as_subclass(torch.Tensor)
            if type(depth_features).__name__ == "tTensor":
                depth_features = depth_features.as_subclass(torch.Tensor)

            # Get logits from each modality
            rgb_logits = self.rgb_head(rgb_features)  # [B, num_classes]
            depth_logits = self.depth_head(depth_features)  # [B, num_classes]

            # Fuse probability distributions
            fused_probs, fusion_weights = self.fusion([rgb_logits, depth_logits])

            # Store weights for analysis (optional)
            self.last_fusion_weights = fusion_weights

            # Convert back to logits for loss computation
            # Add small epsilon to avoid log(0)
            logits = torch.log(fused_probs + 1e-8)

        else:
            # Feature-level fusion
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

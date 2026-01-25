"""
Model architectures for multi-modal gesture recognition
Supports RGB, depth, and fusion approaches with various backbones

Key features:
- Automatic projection layers for dimension alignment
- ALL fusion methods support different feature dimensions
- Flexible architecture combinations (e.g., VSSD small + VSSD base)
"""

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from multimodal_gesture_recognition.vssd_integration import build_vssd_backbone

logger = logging.getLogger(__name__)


# ======================================================================================
# FEATURE EXTRACTORS (BACKBONES)
# ======================================================================================

def build_feature_extractor(
    architecture: str,
    in_channels: int = 3,
    pretrained: bool = True,
    vssd_config: Optional[Dict] = None,
    freeze_backbone: bool = False,
) -> Tuple[nn.Module, int]:
    """
    Build a feature extraction backbone.

    Args:
        architecture: "resnet18/34/50", "vit", or "vssd"
        in_channels: Number of input channels (3 for RGB, 1 for depth)
        pretrained: Use ImageNet pretrained weights
        vssd_config: Configuration for VSSD (required if architecture="vssd")
        freeze_backbone: Freeze backbone parameters

    Returns:
        (backbone, feature_dim)
    """

    if architecture == "resnet18":
        backbone = models.resnet18(pretrained=pretrained)
        backbone.fc = nn.Identity()  # Remove classification head
        feat_dim = 512

        # Modify first conv layer for depth (1 channel)
        if in_channels != 3:
            backbone.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

    elif architecture == "resnet34":
        backbone = models.resnet34(pretrained=pretrained)
        backbone.fc = nn.Identity()  # Remove classification head
        feat_dim = 512
        # Modify first conv layer for depth (1 channel)
        if in_channels != 3:
            backbone.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

    elif architecture == "resnet50":
        backbone = models.resnet50(pretrained=pretrained)
        backbone.fc = nn.Identity()
        feat_dim = 2048

        if in_channels != 3:
            backbone.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

    elif architecture == "vit":
        from torchvision.models import vit_b_16, ViT_B_16_Weights
        if pretrained:
            backbone = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        else:
            backbone = vit_b_16()
        backbone.heads = nn.Identity()
        feat_dim = 768

    elif architecture == "vssd":
        if vssd_config is None:
            raise ValueError("vssd_config is required for VSSD architecture")

        backbone, feat_dim = build_vssd_backbone(
            variant=vssd_config["variant"],
            image_size=vssd_config["image_size"],
            vssd_repo_root=vssd_config["vssd_repo_path"],
            pretrained_ckpt=vssd_config.get("pretrained_ckpt", ""),
        )

    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    # Freeze backbone if requested
    if freeze_backbone:
        for param in backbone.parameters():
            param.requires_grad = False
        logger.info(f"✓ Frozen {architecture} backbone")

    return backbone, feat_dim


def make_classification_head(
    feat_dim: int, num_classes: int, dropout: float = 0.3
) -> nn.Module:
    """Create a classification head with dropout"""
    return nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(feat_dim, num_classes),
    )


# ======================================================================================
# FUSION MODULES (UPGRADED: support different feature dimensions)
# ======================================================================================

class GatedFusionVec(nn.Module):
    """
    Gated fusion for feature vectors.
    NOW SUPPORTS DIFFERENT DIMENSIONS via concatenation-based gating.
    """
    def __init__(self, rgb_dim: int, depth_dim: int, output_dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(rgb_dim + depth_dim, output_dim),
            nn.Sigmoid(),
        )
        # Project to output dimension if needed
        self.rgb_proj = nn.Linear(rgb_dim, output_dim) if rgb_dim != output_dim else nn.Identity()
        self.depth_proj = nn.Linear(depth_dim, output_dim) if depth_dim != output_dim else nn.Identity()

    def forward(self, rgb_vec: torch.Tensor, depth_vec: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rgb_vec: [B, rgb_dim]
            depth_vec: [B, depth_dim]
        Returns:
            fused: [B, output_dim]
        """
        # Convert from tTensor to regular tensor if needed
        if type(rgb_vec).__name__ == "tTensor":
            rgb_vec = rgb_vec.as_subclass(torch.Tensor)
        if type(depth_vec).__name__ == "tTensor":
            depth_vec = depth_vec.as_subclass(torch.Tensor)

        # Project to unified dimension
        rgb_proj = self.rgb_proj(rgb_vec)  # [B, output_dim]
        depth_proj = self.depth_proj(depth_vec)  # [B, output_dim]

        # Compute gate from concatenated features
        combined = torch.cat([rgb_vec, depth_vec], dim=1)  # [B, rgb_dim + depth_dim]
        gate = self.gate(combined)  # [B, output_dim]

        # Gated fusion
        fused = rgb_proj * gate + depth_proj * (1 - gate)
        return fused


class AdaptiveFusionVec(nn.Module):
    """
    Adaptive fusion with learned weights.
    NOW SUPPORTS DIFFERENT DIMENSIONS.
    """
    def __init__(self, rgb_dim: int, depth_dim: int, output_dim: int):
        super().__init__()
        self.weight_network = nn.Sequential(
            nn.Linear(rgb_dim + depth_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.Linear(output_dim, 2),
            nn.Softmax(dim=1),
        )
        # Project to output dimension
        self.rgb_proj = nn.Linear(rgb_dim, output_dim) if rgb_dim != output_dim else nn.Identity()
        self.depth_proj = nn.Linear(depth_dim, output_dim) if depth_dim != output_dim else nn.Identity()

    def forward(self, rgb_vec: torch.Tensor, depth_vec: torch.Tensor) -> torch.Tensor:
        """Fuse with learned weights"""
        # Convert from tTensor to regular tensor if needed
        if type(rgb_vec).__name__ == "tTensor":
            rgb_vec = rgb_vec.as_subclass(torch.Tensor)
        if type(depth_vec).__name__ == "tTensor":
            depth_vec = depth_vec.as_subclass(torch.Tensor)

        # Project to unified dimension
        rgb_proj = self.rgb_proj(rgb_vec)  # [B, output_dim]
        depth_proj = self.depth_proj(depth_vec)  # [B, output_dim]

        # Compute adaptive weights from concatenated features
        combined = torch.cat([rgb_vec, depth_vec], dim=1)
        weights = self.weight_network(combined)  # [B, 2]

        rgb_weight = weights[:, 0:1]  # [B, 1]
        depth_weight = weights[:, 1:2]  # [B, 1]

        fused = rgb_proj * rgb_weight + depth_proj * depth_weight
        return fused


class ConcatFusionVec(nn.Module):
    """
    Concatenation + MLP fusion.
    NATURALLY SUPPORTS DIFFERENT DIMENSIONS (no modification needed).
    """
    def __init__(self, rgb_dim: int, depth_dim: int, output_dim: int):
        super().__init__()
        self.fusion_mlp = nn.Sequential(
            nn.Linear(rgb_dim + depth_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, rgb_vec: torch.Tensor, depth_vec: torch.Tensor) -> torch.Tensor:
        """Concatenate and transform"""
        # Convert from tTensor to regular tensor if needed
        if type(rgb_vec).__name__ == "tTensor":
            rgb_vec = rgb_vec.as_subclass(torch.Tensor)
        if type(depth_vec).__name__ == "tTensor":
            depth_vec = depth_vec.as_subclass(torch.Tensor)

        combined = torch.cat([rgb_vec, depth_vec], dim=1)  # [B, rgb_dim + depth_dim]
        fused = self.fusion_mlp(combined)  # [B, output_dim]
        return fused


class CrossModalAttentionFusion(nn.Module):
    """
    Cross-modal attention fusion with projection layers.
    NOW SUPPORTS DIFFERENT DIMENSIONS via projection to unified space.
    """
    def __init__(
        self, 
        rgb_dim: int, 
        depth_dim: int, 
        unified_dim: int,
        num_heads: int = 8, 
        dropout: float = 0.1
    ):
        super().__init__()
        self.unified_dim = unified_dim
        self.num_heads = num_heads

        # Project to unified dimension for attention
        self.rgb_proj = nn.Linear(rgb_dim, unified_dim) if rgb_dim != unified_dim else nn.Identity()
        self.depth_proj = nn.Linear(depth_dim, unified_dim) if depth_dim != unified_dim else nn.Identity()

        # Multi-head cross-attention: RGB queries depth
        self.rgb_to_depth_attn = nn.MultiheadAttention(
            embed_dim=unified_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Multi-head cross-attention: Depth queries RGB
        self.depth_to_rgb_attn = nn.MultiheadAttention(
            embed_dim=unified_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Layer normalization
        self.norm_rgb = nn.LayerNorm(unified_dim)
        self.norm_depth = nn.LayerNorm(unified_dim)

        # Fusion MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(unified_dim * 2, unified_dim),
            nn.LayerNorm(unified_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(unified_dim, unified_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        logger.info(f"Built CrossModalAttentionFusion: RGB({rgb_dim})→{unified_dim}, Depth({depth_dim})→{unified_dim}, {num_heads} heads")

    def forward(self, rgb_vec: torch.Tensor, depth_vec: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rgb_vec: [B, rgb_dim]
            depth_vec: [B, depth_dim]
        Returns:
            fused: [B, unified_dim]
        """
        # Convert from tTensor to regular tensor if needed
        if type(rgb_vec).__name__ == "tTensor":
            rgb_vec = rgb_vec.as_subclass(torch.Tensor)
        if type(depth_vec).__name__ == "tTensor":
            depth_vec = depth_vec.as_subclass(torch.Tensor)

        # Project to unified dimension
        rgb_unified = self.rgb_proj(rgb_vec)  # [B, unified_dim]
        depth_unified = self.depth_proj(depth_vec)  # [B, unified_dim]

        # Add sequence dimension for attention: [B, unified_dim] -> [B, 1, unified_dim]
        rgb_seq = rgb_unified.unsqueeze(1)
        depth_seq = depth_unified.unsqueeze(1)

        # Cross-attention: RGB attends to depth
        rgb_attended, _ = self.rgb_to_depth_attn(
            query=rgb_seq, key=depth_seq, value=depth_seq
        )  # [B, 1, unified_dim]
        rgb_attended = rgb_attended.squeeze(1)  # [B, unified_dim]
        rgb_enhanced = self.norm_rgb(rgb_unified + rgb_attended)  # Residual

        # Cross-attention: Depth attends to RGB
        depth_attended, _ = self.depth_to_rgb_attn(
            query=depth_seq, key=rgb_seq, value=rgb_seq
        )  # [B, 1, unified_dim]
        depth_attended = depth_attended.squeeze(1)  # [B, unified_dim]
        depth_enhanced = self.norm_depth(depth_unified + depth_attended)  # Residual

        # Fuse enhanced features
        combined = torch.cat([rgb_enhanced, depth_enhanced], dim=1)  # [B, 2*unified_dim]
        fused = self.fusion_mlp(combined)  # [B, unified_dim]

        return fused


class ProbabilityFusion(nn.Module):
    """
    Probability-based fusion with multi-head attention.
    NATURALLY SUPPORTS DIFFERENT DIMENSIONS (works on logits, not features).
    """

    def __init__(
        self,
        num_modalities: int = 2,
        num_classes: int = 34,
        num_heads: int = 1,
    ):
        super().__init__()
        self.num_modalities = num_modalities
        self.num_classes = num_classes
        self.num_heads = num_heads

        # Each head learns different modality importance per class
        self.head_weights = nn.Parameter(
            torch.randn(num_heads, num_modalities, num_classes) * 0.01 + 1.0
        )

        # Learnable weights to combine outputs from different heads
        self.head_combine = nn.Parameter(torch.ones(num_heads))

        logger.info(
            f"Built ProbabilityFusion with {num_modalities} modalities, "
            f"{num_classes} classes, {num_heads} heads"
        )

    def forward(
        self, logits_list: list
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Fuse logits from multiple modalities.

        Args:
            logits_list: List of logit tensors [rgb_logits, depth_logits]
                         Each: [B, num_classes]

        Returns:
            fused_probs: [B, num_classes]
            weights: (head_normalized, combine_weights) for interpretability
        """
        # Convert logits to probabilities
        probs_list = [F.softmax(logits, dim=-1) for logits in logits_list]

        # Normalize head weights across modalities
        head_normalized = F.softmax(self.head_weights, dim=1)  # [num_heads, num_modalities, num_classes]

        # Compute weighted probabilities for each head
        head_outputs = []
        for h in range(self.num_heads):
            weighted_probs = sum(
                probs * head_normalized[h, i, :].unsqueeze(0)
                for i, probs in enumerate(probs_list)
            )
            head_outputs.append(weighted_probs)

        # Normalize weights for combining heads
        combine_weights = F.softmax(self.head_combine, dim=0)  # [num_heads]

        # Combine head outputs
        fused_probs = sum(
            h_out * combine_weights[h]
            for h, h_out in enumerate(head_outputs)
        )

        return fused_probs, (head_normalized, combine_weights)


# ======================================================================================
# CLASSIFIER MODELS
# ======================================================================================

class RGBOnlyClassifier(nn.Module):
    """Single-modality RGB classifier with configurable backbone."""

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config

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

        # Build feature extractor
        self.backbone, self.feat_dim = build_feature_extractor(
            architecture=architecture,
            in_channels=3,
            pretrained=config["model"]["pretrained"],
            vssd_config=vssd_config,
            freeze_backbone=freeze_backbone,
        )

        # Classification head
        self.head = make_classification_head(
            self.feat_dim,
            config["model"]["num_classes"],
            config["model"]["dropout"],
        )

        logger.info(f"Built RGB-only classifier with {architecture.upper()}")
        logger.info(f"  Feature dimension: {self.feat_dim}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, H, W] - RGB image
        Returns:
            logits: [B, num_classes]
        """
        features = self.backbone(x)

        # Convert from VSSD's tTensor if needed
        if type(features).__name__ == "tTensor":
            features = features.as_subclass(torch.Tensor)

        logits = self.head(features)
        return logits


class DepthOnlyClassifier(nn.Module):
    """Single-modality depth classifier with configurable backbone."""

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config

        # Convert 1-channel depth to 3-channel
        self.depth_to_rgb = nn.Conv2d(1, 3, kernel_size=1)

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

        # Build feature extractor
        self.backbone, self.feat_dim = build_feature_extractor(
            architecture=architecture,
            in_channels=3,
            pretrained=config["model"]["pretrained"],
            vssd_config=vssd_config,
            freeze_backbone=freeze_backbone,
        )

        # Classification head
        self.head = make_classification_head(
            self.feat_dim,
            config["model"]["num_classes"],
            config["model"]["dropout"],
        )

        logger.info(f"Built depth-only classifier with {architecture.upper()}")
        logger.info(f"  Feature dimension: {self.feat_dim}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 1, H, W] - Depth map
        Returns:
            logits: [B, num_classes]
        """
        x = self.depth_to_rgb(x)  # [B, 3, H, W]
        features = self.backbone(x)

        # Convert from VSSD's tTensor if needed
        if type(features).__name__ == "tTensor":
            features = features.as_subclass(torch.Tensor)

        logits = self.head(features)
        return logits


class DualBranchFusionClassifier(nn.Module):
    """
    Dual-branch fusion classifier with DIFFERENT backbones for RGB and Depth.

    KEY FEATURES:
    - Supports ANY combination of architectures (e.g., VSSD small + VSSD base)
    - Automatic projection layers for dimension alignment
    - ALL fusion methods work with different feature dimensions
    - Optimal memory usage (projections only when needed)
    """

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.fusion_method = config["model"]["fusion_method"]

        # =====================================================================
        # RGB BRANCH CONFIGURATION
        # =====================================================================
        rgb_architecture = config["model"].get("rgb_architecture", config["model"]["architecture"])
        rgb_pretrained = config["model"].get("rgb_pretrained", config["model"]["pretrained"])
        rgb_freeze_backbone = config["model"].get("rgb_freeze_backbone", config["model"].get("freeze_backbone", False))

        rgb_vssd_config = None
        if rgb_architecture == "vssd":
            rgb_vssd_variant = config["model"].get("rgb_vssd_variant", config["model"]["vssd_variant"])
            rgb_vssd_config = {
                "variant": rgb_vssd_variant,
                "image_size": config["model"]["image_size"],
                "vssd_repo_path": config["model"]["vssd_repo_path"],
                "pretrained_ckpt": config["model"].get("pretrained_ckpt", ""),
            }
            logger.info(f"RGB branch: VSSD-{rgb_vssd_variant.upper()}, pretrained={rgb_pretrained}, freeze={rgb_freeze_backbone}")
        else:
            logger.info(f"RGB branch: {rgb_architecture.upper()}, pretrained={rgb_pretrained}, freeze={rgb_freeze_backbone}")

        # Build RGB backbone
        self.rgb_backbone, self.rgb_feat_dim = build_feature_extractor(
            architecture=rgb_architecture,
            in_channels=3,
            pretrained=rgb_pretrained,
            vssd_config=rgb_vssd_config,
            freeze_backbone=rgb_freeze_backbone,
        )

        # =====================================================================
        # DEPTH BRANCH CONFIGURATION
        # =====================================================================
        depth_architecture = config["model"].get("depth_architecture", config["model"]["architecture"])
        depth_pretrained = config["model"].get("depth_pretrained", config["model"]["pretrained"])
        depth_freeze_backbone = config["model"].get("depth_freeze_backbone", config["model"].get("freeze_backbone", False))

        self.depth_to_rgb = nn.Conv2d(1, 3, kernel_size=1)

        depth_vssd_config = None
        if depth_architecture == "vssd":
            depth_vssd_variant = config["model"].get("depth_vssd_variant", config["model"]["vssd_variant"])
            depth_vssd_config = {
                "variant": depth_vssd_variant,
                "image_size": config["model"]["image_size"],
                "vssd_repo_path": config["model"]["vssd_repo_path"],
                "pretrained_ckpt": config["model"].get("pretrained_ckpt", ""),
            }
            logger.info(f"Depth branch: VSSD-{depth_vssd_variant.upper()}, pretrained={depth_pretrained}, freeze={depth_freeze_backbone}")
        else:
            logger.info(f"Depth branch: {depth_architecture.upper()}, pretrained={depth_pretrained}, freeze={depth_freeze_backbone}")

        # Build Depth backbone
        self.depth_backbone, self.depth_feat_dim = build_feature_extractor(
            architecture=depth_architecture,
            in_channels=3,
            pretrained=depth_pretrained,
            vssd_config=depth_vssd_config,
            freeze_backbone=depth_freeze_backbone,
        )

        # =====================================================================
        # UNIFIED DIMENSION FOR FUSION
        # =====================================================================
        # Use max dimension or custom unified_feature_dim
        self.unified_dim = config["model"].get(
            "unified_feature_dim",
            max(self.rgb_feat_dim, self.depth_feat_dim)
        )

        logger.info(f"Feature dimensions: RGB={self.rgb_feat_dim}, Depth={self.depth_feat_dim}, Unified={self.unified_dim}")

        # =====================================================================
        # FUSION MODULE SELECTION
        # =====================================================================
        if self.fusion_method == "probability_fusion":
            # Probability fusion: no dimension constraints
            num_classes = config["model"]["num_classes"]

            self.rgb_head = make_classification_head(
                self.rgb_feat_dim, num_classes, config["model"]["dropout"]
            )
            self.depth_head = make_classification_head(
                self.depth_feat_dim, num_classes, config["model"]["dropout"]
            )

            num_heads = config["model"].get("probability_fusion_heads", 1)
            self.fusion = ProbabilityFusion(
                num_modalities=2,
                num_classes=num_classes,
                num_heads=num_heads,
            )
            self.last_fusion_weights = None
            logger.info(f"✓ Probability fusion with {num_heads} heads (no projection needed)")

        else:
            # Feature-level fusion: use projection layers
            if self.fusion_method == "gated_fusion":
                self.fusion = GatedFusionVec(
                    self.rgb_feat_dim, self.depth_feat_dim, self.unified_dim
                )
            elif self.fusion_method == "adaptive_fusion":
                self.fusion = AdaptiveFusionVec(
                    self.rgb_feat_dim, self.depth_feat_dim, self.unified_dim
                )
            elif self.fusion_method == "concat":
                self.fusion = ConcatFusionVec(
                    self.rgb_feat_dim, self.depth_feat_dim, self.unified_dim
                )
            elif self.fusion_method == "cross_attention":
                self.fusion = CrossModalAttentionFusion(
                    self.rgb_feat_dim, self.depth_feat_dim, self.unified_dim,
                    num_heads=8,
                    dropout=config["model"]["dropout"],
                )
            elif self.fusion_method == "addition":
                # Addition requires same dimension: use projections
                self.rgb_proj = nn.Linear(self.rgb_feat_dim, self.unified_dim)                     if self.rgb_feat_dim != self.unified_dim else nn.Identity()
                self.depth_proj = nn.Linear(self.depth_feat_dim, self.unified_dim)                     if self.depth_feat_dim != self.unified_dim else nn.Identity()

                def addition_fusion(rgb, depth):
                    if type(rgb).__name__ == "tTensor":
                        rgb = rgb.as_subclass(torch.Tensor)
                    if type(depth).__name__ == "tTensor":
                        depth = depth.as_subclass(torch.Tensor)
                    rgb_proj = self.rgb_proj(rgb)
                    depth_proj = self.depth_proj(depth)
                    return rgb_proj + depth_proj

                self.fusion = addition_fusion
                logger.info(f"✓ Addition fusion with projection: {self.rgb_feat_dim}→{self.unified_dim}, {self.depth_feat_dim}→{self.unified_dim}")

            elif self.fusion_method == "weighted_sum":
                # Weighted sum requires same dimension: use projections
                self.alpha = nn.Parameter(torch.tensor(0.5))
                self.rgb_proj = nn.Linear(self.rgb_feat_dim, self.unified_dim)                     if self.rgb_feat_dim != self.unified_dim else nn.Identity()
                self.depth_proj = nn.Linear(self.depth_feat_dim, self.unified_dim)                     if self.depth_feat_dim != self.unified_dim else nn.Identity()

                def weighted_fusion(rgb, depth):
                    if type(rgb).__name__ == "tTensor":
                        rgb = rgb.as_subclass(torch.Tensor)
                    if type(depth).__name__ == "tTensor":
                        depth = depth.as_subclass(torch.Tensor)
                    rgb_proj = self.rgb_proj(rgb)
                    depth_proj = self.depth_proj(depth)
                    return self.alpha * rgb_proj + (1 - self.alpha) * depth_proj

                self.fusion = weighted_fusion
                logger.info(f"✓ Weighted sum fusion with projection: {self.rgb_feat_dim}→{self.unified_dim}, {self.depth_feat_dim}→{self.unified_dim}")

            else:
                raise ValueError(f"Unknown fusion method: {self.fusion_method}")

            # Classification head for feature-level fusion
            self.head = make_classification_head(
                self.unified_dim,
                config["model"]["num_classes"],
                config["model"]["dropout"],
            )

        logger.info("✓ Built dual-branch fusion classifier")
        logger.info(f"  RGB: {rgb_architecture.upper()} (dim={self.rgb_feat_dim})")
        logger.info(f"  Depth: {depth_architecture.upper()} (dim={self.depth_feat_dim})")
        logger.info(f"  Fusion: {self.fusion_method} (unified_dim={self.unified_dim})")

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rgb: [B, 3, H, W]
            depth: [B, 1, H, W]
        Returns:
            logits or probs: [B, num_classes]
        """
        # Extract features
        rgb_features = self.rgb_backbone(rgb)
        depth_expanded = self.depth_to_rgb(depth)
        depth_features = self.depth_backbone(depth_expanded)

        if self.fusion_method == "probability_fusion":
            # Convert to tensors
            if type(rgb_features).__name__ == "tTensor":
                rgb_features = rgb_features.as_subclass(torch.Tensor)
            if type(depth_features).__name__ == "tTensor":
                depth_features = depth_features.as_subclass(torch.Tensor)

            # Get logits
            rgb_logits = self.rgb_head(rgb_features)
            depth_logits = self.depth_head(depth_features)

            # Fuse probabilities
            fused_probs, weights = self.fusion([rgb_logits, depth_logits])
            self.last_fusion_weights = weights

            return fused_probs
        else:
            # Convert to tensors
            if type(rgb_features).__name__ == "tTensor":
                rgb_features = rgb_features.as_subclass(torch.Tensor)
            if type(depth_features).__name__ == "tTensor":
                depth_features = depth_features.as_subclass(torch.Tensor)

            # Feature-level fusion
            fused_features = self.fusion(rgb_features, depth_features)

            # Classification
            logits = self.head(fused_features)
            return logits


# ======================================================================================
# MODEL BUILDER
# ======================================================================================

def build_model(config: Dict) -> nn.Module:
    """
    Factory function to build the appropriate model.

    Args:
        config: Configuration dictionary

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
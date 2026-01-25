"""
Integration layer for VSSD (VMAMBA2) backbone from the official repository.
Dynamically imports and builds VSSD models with proper configuration.
"""

import importlib.util
import sys
from pathlib import Path
from typing import Tuple, Optional
import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


def _import_vssd_vmamba2(vssd_repo_root: str):
    """
    Dynamically import VMAMBA2 class from the VSSD repository.
    Sets up proper package structure to handle relative imports.
    
    Args:
        vssd_repo_root: Absolute path to VSSD repository root
        
    Returns:
        VMAMBA2 class
    """
    # Paths to required files
    models_dir = Path(vssd_repo_root) / "classification" / "models"
    mamba2_path = models_dir / "mamba2.py"
    mamba_util_path = models_dir / "mamba_util.py"
    
    assert mamba2_path.is_file(), f"VSSD mamba2.py not found: {mamba2_path}"
    assert mamba_util_path.is_file(), f"VSSD mamba_util.py not found: {mamba_util_path}"
    
    # Add classification directory to path temporarily
    cls_dir = Path(vssd_repo_root) / "classification"
    if str(cls_dir) not in sys.path:
        sys.path.insert(0, str(cls_dir))
    
    try:
        # Create a fake package structure for proper relative imports
        # First, create the parent "models" package
        if "vssd_models" not in sys.modules:
            import types
            vssd_models_pkg = types.ModuleType("vssd_models")
            vssd_models_pkg.__package__ = "vssd_models"
            vssd_models_pkg.__path__ = [str(models_dir)]
            sys.modules["vssd_models"] = vssd_models_pkg
        
        # Import mamba_util as part of the package
        spec_util = importlib.util.spec_from_file_location(
            "vssd_models.mamba_util", 
            mamba_util_path,
            submodule_search_locations=[str(models_dir)]
        )
        mamba_util_module = importlib.util.module_from_spec(spec_util)
        mamba_util_module.__package__ = "vssd_models"
        sys.modules["vssd_models.mamba_util"] = mamba_util_module
        # Also add as absolute import for the try block in mamba2.py
        sys.modules["mamba_util"] = mamba_util_module
        spec_util.loader.exec_module(mamba_util_module)
        
        # Import mamba2 as part of the package
        spec_mamba2 = importlib.util.spec_from_file_location(
            "vssd_models.mamba2",
            mamba2_path,
            submodule_search_locations=[str(models_dir)]
        )
        mamba2_module = importlib.util.module_from_spec(spec_mamba2)
        mamba2_module.__package__ = "vssd_models"
        sys.modules["vssd_models.mamba2"] = mamba2_module
        spec_mamba2.loader.exec_module(mamba2_module)
        
        logger.info("Successfully imported VSSD VMAMBA2 module")
        return mamba2_module.VMAMBA2
        
    except Exception as e:
        logger.error(f"Failed to import VSSD VMAMBA2: {e}")
        raise
    finally:
        # Clean up sys.path
        if str(cls_dir) in sys.path:
            sys.path.remove(str(cls_dir))


def build_vssd_backbone(
    variant: str = "tiny",
    image_size: int = 224,
    in_chans: int = 3,
    vssd_repo_root: str = "",
    pretrained_ckpt: Optional[str] = None,
) -> Tuple[nn.Module, int]:
    """
    Build VSSD backbone from official repository.
    
    Args:
        variant: VSSD model variant (micro/tiny/small/base)
        image_size: Input image size
        in_chans: Number of input channels
        vssd_repo_root: Path to VSSD repository
        pretrained_ckpt: Optional path to pretrained checkpoint
        
    Returns:
        (model, feature_dim): VSSD backbone and output feature dimension
    """
    
    # Import VMAMBA2 from official repo
    VMAMBA2 = _import_vssd_vmamba2(vssd_repo_root)
    
    # Configuration mapping from official VSSD YAML configs
    # Source: classification/configs/vssd/*.yaml
    configs = {
        "micro": {
            "embed_dim": 48,
            "depths": [2, 2, 8, 4],
            "num_heads": [2, 4, 8, 16],
            "d_state": 48,
            "drop_path_rate": 0.1,
        },
        "tiny": {
            "embed_dim": 64,
            "depths": [2, 4, 8, 4],
            "num_heads": [2, 4, 8, 16],
            "d_state": 64,
            "drop_path_rate": 0.2,
        },
        "small": {
            "embed_dim": 64,
            "depths": [3, 4, 21, 5],
            "num_heads": [2, 4, 8, 16],
            "d_state": 64,
            "drop_path_rate": 0.4,
        },
        "base": {
            "embed_dim": 96,
            "depths": [3, 4, 21, 5],
            "num_heads": [3, 6, 12, 24],
            "d_state": 64,
            "drop_path_rate": 0.6,
        },
    }
    
    if variant not in configs:
        raise ValueError(f"Unknown variant: {variant}. Choose from {list(configs.keys())}")
    
    cfg = configs[variant]
    
    logger.info(f"Building VSSD-{variant.upper()} backbone")
    logger.info(f"  embed_dim: {cfg['embed_dim']}, depths: {cfg['depths']}")
    
    # Build model with configuration from official repo
    model = VMAMBA2(
        img_size=image_size,
        in_chans=in_chans,
        patch_size=4,
        embed_dim=cfg["embed_dim"],
        depths=cfg["depths"],
        num_heads=cfg["num_heads"],
        mlp_ratio=4.0,
        drop_rate=0.0,
        drop_path_rate=cfg["drop_path_rate"],
        ape=False,
        simple_downsample=True,
        simple_patch_embed=True,
        ssd_expansion=2,
        ssd_ngroups=1,
        ssd_chunk_size=256,
        linear_attn_duality=True,
        lepe=False,
        attn_types=['mamba2', 'mamba2', 'mamba2', 'mamba2'],
        bidirection=False,
        d_state=cfg["d_state"],
        partial_win_size=-1,
        ssd_aexp=False,
        ssd_positve_dA=True,
        win_only=[False] * 4,
        res_scale=[False] * 4,
        ssd_norm_da=False,
        zact=False,
        rope=False,
        ssd_linear_norm=False,
        win_norm=False,
        async_state=[None] * 4,
        ab_bias=False,
        rmt_downsample=False,
        rmt_patch_embed=False,
        multi_branch=True,
        use_cpe=True,
        num_classes=0,  # No classification head - we'll add our own
    )
    
    # Load pretrained weights if provided
    if pretrained_ckpt and Path(pretrained_ckpt).exists():
        logger.info(f"Loading pretrained weights from {pretrained_ckpt}")
        try:
            checkpoint = torch.load(pretrained_ckpt, map_location="cpu")
            
            # Handle different checkpoint formats
            if "model" in checkpoint:
                state_dict = checkpoint["model"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            elif "model_ema" in checkpoint:
                state_dict = checkpoint["model_ema"]
            else:
                state_dict = checkpoint
            
            # Remove 'head' weights if present (we use our own)
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith("head")}
            
            # Load with strict=False to ignore missing head weights
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            logger.info(f"Loaded pretrained weights. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
            
        except Exception as e:
            logger.warning(f"Failed to load pretrained weights: {e}")
    elif pretrained_ckpt:
        logger.warning(f"Pretrained checkpoint not found: {pretrained_ckpt}")
    
    # Calculate feature dimension (from VMAMBA2 architecture)
    # Feature dim = embed_dim * 2^(len(depths)-1)
    feat_dim = cfg["embed_dim"] * (2 ** (len(cfg["depths"]) - 1))
    
    logger.info(f"VSSD backbone built. Feature dimension: {feat_dim}")
    
    return model, feat_dim

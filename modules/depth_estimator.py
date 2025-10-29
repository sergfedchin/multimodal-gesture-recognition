"""Pixel-Perfect Depth estimation module."""

import logging
from typing import Any, Dict

import torch

from modules.ppd.models.ppd import PixelPerfectDepth
from modules.ppd.utils.set_seed import set_seed

logger = logging.getLogger('gesture_processing')


class DepthEstimator:
    """Handles depth estimation using Pixel-Perfect Depth model."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Pixel-Perfect Depth estimator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config['pixel_perfect_depth']
        self.device_str = self.config['device']
        self.model: PixelPerfectDepth = None
        self._setup_model()
    
    def _setup_model(self) -> None:
        """Setup Pixel-Perfect Depth model following their official structure."""
        logger.info("Setting up Pixel-Perfect Depth model...")
        
        try:
            # Set random seed for reproducibility
            set_seed(666)
            
            # Initialize model with semantics checkpoint
            semantics_pth = self.config['semantics_checkpoint']
            sampling_steps = self.config['sampling_steps']
            
            logger.info(f"Loading Pixel-Perfect Depth with semantics from: {semantics_pth}")
            logger.info(f"Sampling steps: {sampling_steps}")
            
            self.model = PixelPerfectDepth(
                semantics_pth=str(semantics_pth),
                sampling_steps=sampling_steps
            )
            
            # Load main PPD checkpoint
            ppd_checkpoint = self.config['ppd_checkpoint']
            logger.info(f"Loading PPD checkpoint from: {ppd_checkpoint}")
            
            state_dict = torch.load(ppd_checkpoint, map_location='cpu')
            self.model.load_state_dict(state_dict, strict=False)
            
            # Move to device and set to eval mode
            self.model = self.model.to(self.device_str).eval()
            
            # Enable fp16 if requested
            if self.config.get('use_fp16', True) and self.device_str.startswith('cuda'):
                logger.info("Enabling mixed precision (fp16)")
                self.model = self.model.half()
            
            logger.info("Pixel-Perfect Depth model loaded successfully")
            
        except ImportError as e:
            logger.error(f"Failed to import Pixel-Perfect Depth modules: {str(e)}")
            logger.error("Make sure the 'ppd' module is placed in the modules/ directory")
            logger.error("Expected structure: modules/ppd/models/ppd.py and modules/ppd/utils/set_seed.py")
            raise
        except Exception as e:
            logger.error(f"Failed to load Pixel-Perfect Depth model: {str(e)}")
            raise
    
    def get_model(self):
        """
        Get the underlying model for direct inference.
        
        Returns:
            The PixelPerfectDepth model
        """
        return self.model

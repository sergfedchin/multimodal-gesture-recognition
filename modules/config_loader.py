"""Configuration loader for the gesture processing pipeline."""

from pathlib import Path
from typing import Dict, Any
import toml


class ConfigLoader:
    """Handles loading and validation of configuration files."""
    
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """
        Load and validate configuration from TOML file.
        
        Args:
            config_path: Path to the config.toml file
            
        Returns:
            Dictionary containing configuration parameters
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If configuration is invalid
        """
        config_file = Path(config_path)
        
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            config = toml.load(config_file)
        except Exception as e:
            raise ValueError(f"Failed to parse configuration file: {str(e)}")
        
        # Validate configuration
        ConfigLoader._validate_config(config)
        
        # Convert paths to Path objects
        config = ConfigLoader._normalize_paths(config)
        
        return config
    
    @staticmethod
    def _validate_config(config: Dict[str, Any]) -> None:
        """
        Validate configuration parameters.
        
        Args:
            config: Configuration dictionary
            
        Raises:
            ValueError: If configuration is invalid
        """
        # Check required sections
        required_sections = ['paths', 'yolov13', 'pixel_perfect_depth', 'processing']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        # Validate paths section
        paths = config['paths']
        required_paths = ['annotations_folder', 'gesture_folder', 'output_folder']
        for path_key in required_paths:
            if path_key not in paths:
                raise ValueError(f"Missing required path: {path_key}")
        
        # Validate YOLOv13 settings
        yolo_config = config['yolov13']
        if yolo_config['model_variant'] not in ['n', 's', 'l', 'x']:
            raise ValueError("YOLOv13 model_variant must be one of: n, s, l, x")
        
        if not 0 <= yolo_config['confidence_threshold'] <= 1:
            raise ValueError("YOLOv13 confidence_threshold must be between 0 and 1")
        
        # Validate Pixel-Perfect Depth settings
        depth_config = config['pixel_perfect_depth']
        if depth_config['inference_height'] % 64 != 0 or depth_config['inference_width'] % 64 != 0:
            raise ValueError("Depth inference dimensions must be multiples of 64")
    
    @staticmethod
    def _normalize_paths(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert string paths to Path objects.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Configuration with normalized paths
        """
        path_keys = ['annotations_folder', 'gesture_folder', 'output_folder', 'temp_folder']
        for key in path_keys:
            if key in config['paths']:
                config['paths'][key] = Path(config['paths'][key])
        
        if 'model_weights' in config['yolov13']:
            config['yolov13']['model_weights'] = Path(config['yolov13']['model_weights'])
        
        if 'checkpoint_path' in config['pixel_perfect_depth']:
            config['pixel_perfect_depth']['checkpoint_path'] = Path(config['pixel_perfect_depth']['checkpoint_path'])
        
        if 'log_file' in config.get('logging', {}) and config['logging']['log_file']:
            config['logging']['log_file'] = Path(config['logging']['log_file'])
        
        return config

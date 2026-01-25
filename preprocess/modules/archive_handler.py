"""Handles image folder loading and annotation loading for all splits."""

import json
from pathlib import Path
from typing import Dict, List, Any
import logging

logger = logging.getLogger('gesture_processing')


class ArchiveHandler:
    """Manages image folder loading and annotation loading across all splits."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize archive handler.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.paths = config['paths']
        self.gesture_folder = Path(self.paths['gesture_folder'])
        self.annotations_folder = self.paths['annotations_folder']
        self.gesture_name = self.gesture_folder.name
    
    def load_images(self) -> List[Path]:
        """
        Load image paths from gesture folder (no extraction needed).
        
        Returns:
            List of paths to image files
            
        Raises:
            FileNotFoundError: If folder doesn't exist
        """
        if not self.gesture_folder.exists():
            raise FileNotFoundError(f"Gesture folder not found: {self.gesture_folder}")
        
        logger.info(f"Loading images from folder: {self.gesture_folder}")
        
        # Get list of image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_files = [
            f for f in self.gesture_folder.rglob('*')
            if f.suffix.lower() in image_extensions and f.is_file()
        ]
        
        if not image_files:
            raise FileNotFoundError(f"No images found in {self.gesture_folder}")
        
        logger.info(f"Found {len(image_files)} images")
        
        return sorted(image_files)
    
    def load_annotations(self) -> Dict[str, Dict[str, Any]]:
        """
        Load annotations for the current gesture from ALL splits.
        Searches train/, test/, and val/ subfolders automatically.
        
        Returns:
            Dictionary with structure: {split: {image_id: annotation_data}}
            
        Raises:
            FileNotFoundError: If no annotation files are found
        """
        all_annotations = {}
        splits_found = []
        
        # Try to find annotation files in different split folders
        possible_splits = ['train', 'test', 'val', 'validation']
        
        logger.info(f"Searching for {self.gesture_name}.json in annotation splits...")
        
        for split in possible_splits:
            annotation_file = self.annotations_folder / split / f"{self.gesture_name}.json"
            
            if annotation_file.exists():
                logger.info(f"Found annotation file: {annotation_file}")
                
                with open(annotation_file, 'r') as f:
                    split_annotations = json.load(f)
                
                # Normalize split name (validation -> val)
                normalized_split = 'val' if split == 'validation' else split
                all_annotations[normalized_split] = split_annotations
                splits_found.append(normalized_split)
                
                logger.info(f"Loaded {len(split_annotations)} annotations for {normalized_split} split")
        
        if not all_annotations:
            # Try loading from root annotations folder (no split structure)
            annotation_file = self.annotations_folder / f"{self.gesture_name}.json"
            if annotation_file.exists():
                logger.info(f"Found annotation file in root: {annotation_file}")
                with open(annotation_file, 'r') as f:
                    annotations = json.load(f)
                # If no split structure, treat as single split
                all_annotations['all'] = annotations
                splits_found.append('all')
                logger.info(f"Loaded {len(annotations)} annotations (no split structure)")
            else:
                raise FileNotFoundError(
                    f"No annotation file found for gesture '{self.gesture_name}' "
                    f"in {self.annotations_folder} or its subfolders (train/test/val)"
                )
        
        logger.info(f"Total splits found: {splits_found}")
        total_images = sum(len(anns) for anns in all_annotations.values())
        logger.info(f"Total annotations loaded: {total_images}")
        
        return all_annotations
    
    def match_images_to_splits(
        self,
        image_paths: List[Path],
        annotations: Dict[str, Dict[str, Any]]
    ) -> Dict[str, List[Path]]:
        """
        Match images to their corresponding splits based on annotations.
        
        Args:
            image_paths: List of image file paths
            annotations: Dictionary of annotations by split
            
        Returns:
            Dictionary mapping splits to lists of image paths
        """
        split_images = {split: [] for split in annotations.keys()}
        unmatched_images = []
        
        logger.info(f"Matching {len(image_paths)} images to splits...")
        
        for img_path in image_paths:
            img_stem = img_path.stem
            matched = False
            
            # Check which split contains this image
            for split, split_anns in annotations.items():
                if img_stem in split_anns:
                    split_images[split].append(img_path)
                    matched = True
                    break
            
            if not matched:
                unmatched_images.append(img_stem)
        
        # Log statistics
        for split, imgs in split_images.items():
            logger.info(f"Split '{split}': {len(imgs)} images")
        
        if unmatched_images:
            logger.warning(f"Found {len(unmatched_images)} unmatched images (not in annotations)")
            if len(unmatched_images) <= 10:
                logger.warning(f"Unmatched images: {unmatched_images}")
        
        return split_images

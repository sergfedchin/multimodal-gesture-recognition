"""Unified image processor that handles one image at a time to save memory."""

import gc
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
from PIL import Image
import cv2
import torch
import torch.nn.functional as F
from tqdm import tqdm

from modules.data_saver import DataSaver
from modules.depth_estimator import DepthEstimator

logger = logging.getLogger('gesture_processing')


class UnifiedProcessor:
    """Processes images one-by-one: crop, depth estimation, and save."""
    
    def __init__(self, config: Dict[str, Any], depth_model: DepthEstimator, data_saver: DataSaver):
        """
        Initialize unified processor.
        
        Args:
            config: Configuration dictionary
            depth_model: Initialized DepthEstimator model
            data_saver: DataSaver instance for saving results
        """
        self.config = config
        self.depth_model = depth_model
        self.data_saver = data_saver
        self.show_progress = config['processing'].get('show_progress', True)
        self.use_fp16 = config['pixel_perfect_depth'].get('use_fp16', True)
        self.device = torch.device(config['pixel_perfect_depth']['device'])
    
    def process_images(
        self,
        image_paths: List[Path],
        person_bboxes: Dict[str, List[float]],
        annotations: Dict[str, Any],
        split_name: str
    ) -> Dict[str, Any]:
        """
        Process images one-by-one through the complete pipeline.
        
        Args:
            image_paths: List of image file paths
            person_bboxes: Dictionary of person bounding boxes
            annotations: Annotations for this split
            split_name: Name of the current split
            
        Returns:
            Dictionary with statistics about processed images
        """
        stats = {
            'total_images': len(image_paths),
            'processed_images': 0,
            'total_hands': 0,
            'skipped_images': []
        }
        
        logger.info(f"Processing {len(image_paths)} images one-by-one for split '{split_name}'")
        
        # Create progress bar
        if self.show_progress:
            pbar = tqdm(
                image_paths,
                desc=f"Processing {split_name}",
                unit="img",
                dynamic_ncols=True,
                smoothing=0
            )
        else:
            pbar = image_paths
        
        for i, img_path in enumerate(pbar):
            img_stem = img_path.stem
            
            # Check if we have person bbox and annotations
            if img_stem not in person_bboxes:
                logger.warning(f"No person bbox for {img_stem}, skipping")
                stats['skipped_images'].append(img_stem)
                continue
            
            if img_stem not in annotations or 'bboxes' not in annotations[img_stem]:
                logger.warning(f"No hand annotations for {img_stem}, skipping")
                stats['skipped_images'].append(img_stem)
                continue
            
            # Process single image through complete pipeline
            try:
                num_hands = self._process_single_image(
                    img_path,
                    img_stem,
                    person_bboxes[img_stem],
                    annotations[img_stem],
                    split_name
                )
                
                stats['processed_images'] += 1
                stats['total_hands'] += num_hands
                
            except Exception as e:
                logger.error(f"Error processing {img_stem}: {str(e)}")
                stats['skipped_images'].append(img_stem)
                continue

            if i % 100 == 0:
                gc.collect()
        
        if self.show_progress and isinstance(pbar, tqdm):
            pbar.close()
        
        logger.info(f"Processed {stats['processed_images']}/{stats['total_images']} images")
        logger.info(f"Total hands cropped: {stats['total_hands']}")
        if stats['skipped_images']:
            logger.warning(f"Skipped {len(stats['skipped_images'])} images")
        
        return stats
    
    def _process_single_image(
        self,
        img_path: Path,
        img_stem: str,
        person_bbox: List[float],
        annotation: Dict[str, Any],
        split_name: str
    ) -> int:
        """
        Process a single image through the complete pipeline.
        
        Steps:
        1. Load image from disk
        2. Crop to person bbox (expanded to include hands)
        3. Estimate depth map
        4. Save body RGB and depth
        5. Crop and save hands RGB and depth
        
        Args:
            img_path: Path to image file
            img_stem: Image filename stem
            person_bbox: Person bounding box [x, y, w, h] (normalized)
            annotation: Annotation data for this image
            split_name: Current split name
            
        Returns:
            Number of hands processed
        """
        # Step 1: Load image
        original_img = Image.open(img_path).convert('RGB')
        img_width, img_height = original_img.size
        
        # Step 2: Expand person bbox to include hands and crop
        hand_bboxes = annotation.get('bboxes', [])
        expanded_bbox = self._expand_bbox_for_hands(person_bbox, hand_bboxes)
        
        cropped_body_rgb = self._crop_image(original_img, expanded_bbox)
        
        # Free original image memory
        del original_img
        
        # Step 3: Estimate depth map
        body_depth_map = self._estimate_depth(cropped_body_rgb)
        
        # Step 4: Save body RGB and depth
        self.data_saver.save_body_data(img_stem, cropped_body_rgb, body_depth_map)
        
        # Step 5: Crop and save hands
        adjusted_hand_bboxes = self._adjust_hand_bboxes(
            hand_bboxes,
            expanded_bbox,
            (img_width, img_height),
            cropped_body_rgb.size
        )
        
        num_hands = len(adjusted_hand_bboxes)
        
        for hand_idx, hand_bbox in enumerate(adjusted_hand_bboxes):
            # Crop hand from RGB
            hand_rgb = self._crop_image(cropped_body_rgb, hand_bbox)
            
            # Crop hand from depth
            depth_pil = Image.fromarray(body_depth_map)
            hand_depth = self._crop_image(depth_pil, hand_bbox)
            hand_depth_array = np.array(hand_depth)
            
            # Save hand crops
            self.data_saver.save_hand_data(
                img_stem,
                hand_idx,
                hand_rgb,
                hand_depth_array
            )
        
        # Store annotation data
        self.data_saver.add_annotation_entry(
            img_stem,
            split_name,
            img_stem,
            expanded_bbox,
            adjusted_hand_bboxes
        )
        
        # Free memory
        del cropped_body_rgb
        del body_depth_map
        
        return num_hands
    
    def _estimate_depth(self, image: Image.Image) -> np.ndarray:
        """
        Estimate depth for a single image.
        
        Args:
            image: PIL Image
            
        Returns:
            Depth map as numpy array
        """
        # Convert PIL Image to OpenCV format (BGR)
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        H, W = image_cv.shape[:2]
        
        # Run inference using model's infer_image method
        with torch.no_grad():
            if self.use_fp16 and self.device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    depth, _ = self.depth_model.infer_image(image_cv)
            else:
                depth, _ = self.depth_model.infer_image(image_cv)
        
        # Resize depth to original image size
        depth = F.interpolate(
            depth, 
            size=(H, W), 
            mode='bilinear', 
            align_corners=False
        )[0, 0]
        
        # Convert to numpy
        depth_np = depth.squeeze().cpu().numpy()
        
        return depth_np
    
    @staticmethod
    def _expand_bbox_for_hands(
        person_bbox: List[float],
        hand_bboxes: List[List[float]]
    ) -> List[float]:
        """
        Expand person bbox to include all hand bboxes.
        
        Args:
            person_bbox: Person bbox [x, y, w, h] (normalized)
            hand_bboxes: List of hand bboxes [x, y, w, h] (normalized)
            
        Returns:
            Expanded bbox [x, y, w, h] (normalized)
        """
        px, py, pw, ph = person_bbox
        px2, py2 = px + pw, py + ph
        
        # Find extent of all hand bboxes
        min_x, min_y = px, py
        max_x, max_y = px2, py2
        
        for hand_bbox in hand_bboxes:
            hx, hy, hw, hh = hand_bbox
            hx2, hy2 = hx + hw, hy + hh
            
            min_x = min(min_x, hx)
            min_y = min(min_y, hy)
            max_x = max(max_x, hx2)
            max_y = max(max_y, hy2)
        
        # Clamp to image bounds
        min_x = max(0.0, min_x)
        min_y = max(0.0, min_y)
        max_x = min(1.0, max_x)
        max_y = min(1.0, max_y)
        
        return [min_x, min_y, max_x - min_x, max_y - min_y]
    
    @staticmethod
    def _crop_image(image: Image.Image, bbox: List[float]) -> Image.Image:
        """
        Crop image using normalized bbox.
        
        Args:
            image: PIL Image
            bbox: Normalized bbox [x, y, w, h]
            
        Returns:
            Cropped image
        """
        img_width, img_height = image.size
        x, y, w, h = bbox
        
        # Convert to pixel coordinates
        x1 = int(x * img_width)
        y1 = int(y * img_height)
        x2 = int((x + w) * img_width)
        y2 = int((y + h) * img_height)
        
        # Crop
        cropped = image.crop((x1, y1, x2, y2))
        
        return cropped
    
    @staticmethod
    def _adjust_hand_bboxes(
        hand_bboxes: List[List[float]],
        person_bbox: List[float],
        original_size: Tuple[int, int],
        cropped_size: Tuple[int, int]
    ) -> List[List[float]]:
        """
        Adjust hand bboxes from original image to cropped image coordinates.
        
        Args:
            hand_bboxes: Hand bboxes in original image (normalized)
            person_bbox: Person bbox used for cropping (normalized)
            original_size: Original image size (width, height)
            cropped_size: Cropped image size (width, height)
            
        Returns:
            Adjusted hand bboxes (normalized to cropped image)
        """
        px, py, pw, ph = person_bbox
        orig_width, orig_height = original_size
        crop_width, crop_height = cropped_size
        
        adjusted_bboxes = []
        
        for hand_bbox in hand_bboxes:
            hx, hy, hw, hh = hand_bbox
            
            # Convert to pixel coordinates in original image
            hx_px = hx * orig_width
            hy_px = hy * orig_height
            hw_px = hw * orig_width
            hh_px = hh * orig_height
            
            # Convert person bbox to pixels
            px_px = px * orig_width
            py_px = py * orig_height
            
            # Adjust to cropped image coordinates
            hx_crop = hx_px - px_px
            hy_crop = hy_px - py_px
            
            # Normalize to cropped image size
            hx_norm = hx_crop / crop_width
            hy_norm = hy_crop / crop_height
            hw_norm = hw_px / crop_width
            hh_norm = hh_px / crop_height
            
            adjusted_bboxes.append([hx_norm, hy_norm, hw_norm, hh_norm])
        
        return adjusted_bboxes

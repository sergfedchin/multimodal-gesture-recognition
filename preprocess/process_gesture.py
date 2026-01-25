"""
Main processing script for HaGRIDv2.1 gesture dataset preprocessing.
This script processes one gesture at a time across ALL splits (train/test/val)
with YOLOv13 person detection and Pixel-Perfect Depth estimation.

Memory-efficient: processes images one-by-one without storing all data in RAM.
"""

import gc
import sys

from modules.config_loader import ConfigLoader
from modules.archive_handler import ArchiveHandler
from modules.person_detector import PersonDetector
from modules.depth_estimator import DepthEstimator
from modules.unified_processor import UnifiedProcessor
from modules.data_saver import DataSaver
from modules.logger import setup_logger


def main():
    """Main processing pipeline."""
    # Load configuration first (before logger setup)
    config = ConfigLoader.load_config("config.toml")
    
    # Setup logging with config
    logger = setup_logger(config)
    logger.info("=" * 80)
    logger.info("Starting HaGRID Gesture Dataset Processing Pipeline")
    logger.info("Memory-Efficient: One-by-One Image Processing")
    logger.info("Processing ALL splits (train/test/val) together")
    logger.info("=" * 80)
    
    try:
        logger.info("Configuration loaded successfully")
        
        # Initialize handlers
        logger.info("Initializing processing modules...")
        archive_handler = ArchiveHandler(config)
        logger.info("All modules initialized successfully")
        
        # Step 1: Load images from folder (no extraction needed)
        logger.info("-" * 80)
        logger.info("STEP 1: Loading images from gesture folder")
        logger.info("-" * 80)
        image_paths = archive_handler.load_images()
        logger.info(f"Loaded {len(image_paths)} images")
        
        # Step 2: Load annotations from ALL splits
        logger.info("-" * 80)
        logger.info("STEP 2: Loading annotations from all splits")
        logger.info("-" * 80)
        all_annotations = archive_handler.load_annotations()
        
        # Step 3: Match images to splits
        logger.info("-" * 80)
        logger.info("STEP 3: Matching images to splits")
        logger.info("-" * 80)
        split_images = archive_handler.match_images_to_splits(image_paths, all_annotations)
        
        # Process each split
        total_stats = {
            'total_processed': 0,
            'total_hands': 0,
            'splits': {}
        }
        
        depth_estimator = DepthEstimator(config)
        data_saver = DataSaver(config)
        unified_processor = UnifiedProcessor(
            config,
            depth_estimator.get_model(),
            data_saver
        )
        for split_name, split_image_paths in split_images.items():
            if not split_image_paths:
                logger.warning(f"No images found for split '{split_name}', skipping")
                continue
            
            gc.collect()
            logger.info("=" * 80)
            logger.info(f"PROCESSING SPLIT: {split_name.upper()}")
            logger.info("=" * 80)
            
            split_annotations = all_annotations[split_name]
            
            # Step 4: Detect persons with YOLOv13
            logger.info("-" * 80)
            logger.info(f"STEP 4: Detecting persons with YOLOv13 ({split_name})")
            logger.info("-" * 80)
            person_detector = PersonDetector(config)
            person_bboxes = person_detector.detect_persons_batch(
                split_image_paths, 
                split_annotations
            )
            del person_detector
            gc.collect()
            logger.info(f"Person detection completed for {len(person_bboxes)} images")
            
            # Steps 5-7 combined: Process images one-by-one
            logger.info("-" * 80)
            logger.info(f"STEPS 5-7: Processing images ({split_name})")
            logger.info("  - Crop to person bbox")
            logger.info("  - Estimate depth")
            logger.info("  - Save body and hands data")
            logger.info("-" * 80)
            
            # Initialize unified processor
            split_stats = unified_processor.process_images(
                split_image_paths,
                person_bboxes,
                split_annotations,
                split_name
            )
            gc.collect()
            
            # Update total stats
            total_stats['total_processed'] += split_stats['processed_images']
            total_stats['total_hands'] += split_stats['total_hands']
            total_stats['splits'][split_name] = split_stats
            
            logger.info(f"Split '{split_name}' complete:")
            logger.info(f"  Processed: {split_stats['processed_images']} images")
            logger.info(f"  Hands cropped: {split_stats['total_hands']}")
            logger.info(f"  Skipped: {len(split_stats['skipped_images'])}")
        
        # Step 8: Save annotations (all splits together)
        logger.info("=" * 80)
        logger.info("STEP 8: Saving consolidated annotations")
        logger.info("=" * 80)
        data_saver.save_annotations()
        
        # Final summary
        logger.info("=" * 80)
        logger.info("FINAL SUMMARY:")
        logger.info(f"  Total images processed: {total_stats['total_processed']}")
        logger.info(f"  Total hands cropped: {total_stats['total_hands']}")
        logger.info("  Split breakdown:")
        for split_name, split_stats in total_stats['splits'].items():
            logger.info(f"    {split_name}: {split_stats['processed_images']} images, "
                       f"{split_stats['total_hands']} hands")
        logger.info("=" * 80)
        
        logger.info("=" * 80)
        logger.info("Processing pipeline completed successfully!")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

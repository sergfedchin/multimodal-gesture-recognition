"""
Evaluation script for trained gesture recognition models

Loads a trained model checkpoint and evaluates it on the test set,
computing comprehensive metrics and generating visualizations.
F1 score is the primary metric for both hand-level and image-level evaluation.

Usage:
    python evaluate.py --config path/to/config.toml [--checkpoint path/to/checkpoint.pt]
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import torch

from data_loader import create_dataloaders
from evaluator import Evaluator
from models import build_model
from utils import load_config, set_seed, setup_device


def setup_logging():
    """Setup logging to console"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
    return logging.getLogger(__name__)


def main(config_path: str, checkpoint_path: Optional[str] = None):
    """
    Evaluate a trained model on the test set.

    Args:
        config_path: Path to configuration TOML file
        checkpoint_path: Optional path to checkpoint. If None, uses best_model.pt from config
    """
    logger = setup_logging()

    logger.info("=" * 80)
    logger.info("GESTURE RECOGNITION MODEL EVALUATION")
    logger.info("=" * 80)

    # Load configuration
    logger.info(f"Loading config from: {config_path}")
    config = load_config(config_path)

    # Setup paths
    output_dir = Path(config["logging"]["output_dir"])
    experiment_name = config["logging"]["experiment_name"]
    experiment_dir = output_dir / experiment_name

    # Determine checkpoint path
    if checkpoint_path is None:
        checkpoint_path = experiment_dir / "checkpoints" / "best_model.pt"
        logger.info(f"Using default checkpoint: {checkpoint_path}")
    else:
        checkpoint_path = Path(checkpoint_path)
        logger.info(f"Using specified checkpoint: {checkpoint_path}")

    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        logger.error("Please provide a valid checkpoint path or train a model first.")
        return

    # Setup evaluation directory
    eval_dir = experiment_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Evaluation results will be saved to: {eval_dir}")

    # Set seed
    set_seed(config.get("seed", 42))

    # Setup device - ТОЧНО КАК В train.py
    device = setup_device(config)

    # Create dataloaders - ТОЧНО КАК В train.py
    logger.info("Setting up datasets...")
    _, _, _, test_loader, dataset_info = create_dataloaders(config, device=device)

    logger.info("Dataset info:")
    logger.info(f"  Number of classes: {dataset_info['num_classes']}")
    logger.info(f"  Test samples: {dataset_info['test_size']}")
    logger.info(f"  Gesture classes: {dataset_info['gesture_classes']}")

    # Build model - ТОЧНО КАК В train.py
    logger.info("Building model...")
    logger.info(f"  Modality: {config['model']['modality']}")
    logger.info(f"  Architecture: {config['model']['architecture']}")
    if config["model"]["architecture"] == "vssd":
        logger.info(f"  VSSD variant: {config['model']['vssd_variant']}")
    if config["model"]["modality"] == "fusion":
        logger.info(f"  Fusion method: {config['model']['fusion_method']}")

    model = build_model(config)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # Load checkpoint - ВАЖНО: на CPU чтобы избежать дублирования памяти
    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    if "epoch" in checkpoint:
        logger.info(f"  Checkpoint epoch: {checkpoint['epoch']}")
    if "metrics" in checkpoint:
        metrics = checkpoint["metrics"]
        if "f1" in metrics:
            logger.info(f"  Checkpoint val F1: {metrics['f1']:.4f}")
        if "accuracy" in metrics:
            logger.info(f"  Checkpoint val accuracy: {metrics['accuracy']:.4f}")
        if "loss" in metrics:
            logger.info(f"  Checkpoint val loss: {metrics['loss']:.4f}")

    # Create evaluator
    logger.info("Initializing evaluator...")
    evaluator = Evaluator(model, config, device, dataset_info)

    # Run evaluation
    logger.info("=" * 80)
    logger.info("Running evaluation on test set...")
    logger.info("=" * 80)
    results = evaluator.evaluate(test_loader)

    # Run paper-style evaluation (image-level, 33 classes)
    logger.info("=" * 80)
    logger.info("Running paper-style evaluation (image-level, 33 classes)...")
    logger.info("=" * 80)
    paper_results = evaluator.evaluate_paper_style(
        test_parquet_path=config["dataset"]["test_parquet"]
    )
    results.update(paper_results)

    # Print results to terminal
    logger.info("")
    logger.info("=" * 80)
    logger.info("EVALUATION RESULTS SUMMARY")
    logger.info("=" * 80)
    logger.info("")

    logger.info("HAND-LEVEL EVALUATION (33 classes - per individual hand):")
    logger.info(f"  Macro F1 (PRIMARY):       {results['macro_f1']:.4f}")
    logger.info(f"  Weighted F1:              {results['weighted_f1']:.4f}")
    logger.info(f"  Overall Accuracy:         {results['overall_accuracy']:.4f}")
    logger.info(f"  Macro Precision:          {results['macro_precision']:.4f}")
    logger.info(f"  Macro Recall:             {results['macro_recall']:.4f}")
    logger.info("")

    logger.info("IMAGE-LEVEL EVALUATION (33 classes - aggregated per image):")
    # Check if paper-style metrics exist (they should after patch)
    if "paper_style_macro_f1" in paper_results:
        logger.info(f"  Macro F1 (PRIMARY):       {paper_results['paper_style_macro_f1']:.2f}%")
        logger.info(f"  Weighted F1:              {paper_results['paper_style_weighted_f1']:.2f}%")
    else:
        logger.warning("  ⚠ F1 metrics not available - please apply evaluator_patch.txt")
    logger.info(f"  Accuracy:                 {paper_results['paper_style_accuracy']:.2f}%")
    logger.info(f"  Number of test images:    {paper_results['paper_style_num_images']}")
    logger.info("")

    # Print per-class performance
    logger.info("TOP 10 CLASSES BY F1 SCORE (Hand-Level):")
    per_class_df = pd.DataFrame(results["per_class"])
    top_10 = per_class_df.nlargest(10, "f1")
    for idx, row in top_10.iterrows():
        logger.info(
            f"  {row['class']:20s}: F1={row['f1']:.4f}, Acc={row['accuracy']:.4f}, "
            f"P={row['precision']:.4f}, R={row['recall']:.4f}, Count={row['count']}"
        )
    logger.info("")

    # Print worst performing classes
    logger.info("BOTTOM 10 CLASSES BY F1 SCORE (Hand-Level):")
    bottom_10 = per_class_df.nsmallest(10, "f1")
    for idx, row in bottom_10.iterrows():
        logger.info(
            f"  {row['class']:20s}: F1={row['f1']:.4f}, Acc={row['accuracy']:.4f}, "
            f"P={row['precision']:.4f}, R={row['recall']:.4f}, Count={row['count']}"
        )
    logger.info("")

    # Generate visualizations
    logger.info("Generating visualizations...")
    evaluator.visualize_results(results, eval_dir)

    # Save results to JSON
    results_json_path = eval_dir / "results.json"
    evaluator.save_results_json(results, results_json_path)
    logger.info(f"Results saved to: {results_json_path}")

    # Save per-class results to CSV for easy analysis
    per_class_csv_path = eval_dir / "per_class_metrics.csv"
    per_class_df.to_csv(per_class_csv_path, index=False)
    logger.info(f"Per-class metrics saved to: {per_class_csv_path}")

    # Print summary
    logger.info("=" * 80)
    logger.info("EVALUATION COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)
    logger.info(f"Primary Metric (Hand-Level Macro F1): {results['macro_f1']:.4f}")
    if "paper_style_macro_f1" in paper_results:
        logger.info(f"Primary Metric (Image-Level Macro F1): {paper_results['paper_style_macro_f1']:.2f}%")
    logger.info(f"All results saved to: {eval_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate trained gesture recognition model on test set"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration TOML file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (default: uses best_model.pt from experiment directory)",
    )

    args = parser.parse_args()
    main(config_path=args.config, checkpoint_path=args.checkpoint)
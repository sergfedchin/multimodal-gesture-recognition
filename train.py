"""
Main training pipeline orchestrator
Handles config loading, data setup, model training, and evaluation
"""

import logging
import shutil
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

from data_loader import create_dataloaders
from evaluator import Evaluator
from models import build_model
from trainer import Trainer
from utils import load_config, set_seed, setup_device

# ============================================================
# CUDA Memory Configuration - MUST BE SET BEFORE CUDA INIT
# ============================================================
# import os
# os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


def setup_logging(output_dir: Path, experiment_name: str):
    """
    Setup logging to console and file in experiment directory.

    Args:
        output_dir: Base output directory
        experiment_name: Name of the experiment
    """
    # Create experiment directory
    exp_dir = output_dir / experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Log file path
    log_file = exp_dir / "training.log"

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, mode="a"),  # 'a' to append if resuming
        ],
        force=True,  # Override any existing configuration
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Logging to: {log_file}")
    return logger


def save_config_copy(config_path: str, output_dir: Path, experiment_name: str):
    """
    Save a copy of the config file to the experiment directory.

    Args:
        config_path: Path to original config file
        output_dir: Base output directory
        experiment_name: Name of the experiment
    """
    exp_dir = output_dir / experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Copy config file
    config_copy_path = exp_dir / f"config_{experiment_name}.toml"
    shutil.copy2(config_path, config_copy_path)

    logger = logging.getLogger(__name__)
    logger.info(f"Config file copied to: {config_copy_path}")


def main(
    config_path: str = "config.toml",
    checkpoint_path: Optional[str] = None,
):
    """
    Main training pipeline.

    Args:
        config_path: Path to configuration TOML file
        checkpoint_path: Optional path to checkpoint to resume from
    """
    # Load config first (before logging setup)
    config = load_config(config_path)

    # Setup logging to experiment directory
    output_dir = Path(config["logging"]["output_dir"])
    experiment_name = config["logging"]["experiment_name"]
    logger = setup_logging(output_dir, experiment_name)

    logger.info("=" * 80)
    logger.info("Hand Gesture Classification with Multi-Modal VSSD")
    logger.info("=" * 80)
    logger.info(f"Experiment: {experiment_name}")
    logger.info(f"Output directory: {output_dir / experiment_name}")
    logger.info(f"Config: {config_path}")

    # Save copy of config file
    save_config_copy(config_path, output_dir, experiment_name)

    # Set seed
    set_seed(config["training"].get("seed", 42))

    # Setup device
    device = setup_device(config)

    # Create dataloaders
    logger.info("Setting up datasets...")
    train_loader, val_loader, val_metrics_loader, test_loader, dataset_info = (
        create_dataloaders(config, device=device)
    )

    logger.info("Dataset info:")
    logger.info(f"  Number of classes: {dataset_info['num_classes']}")
    logger.info(f"  Training samples: {dataset_info['train_size']}")
    logger.info(f"  Validation samples: {dataset_info['val_size']}")
    logger.info(f"  Validation metrics samples: {dataset_info['val_metrics_size']}")
    logger.info(f"  Test samples: {dataset_info['test_size']}")
    logger.info(f"  Gesture classes: {dataset_info['gesture_classes']}")

    # Build model
    logger.info("Building model...")
    logger.info(f"  Modality: {config['model']['modality']}")
    logger.info(f"  VSSD variant: {config['model']['vssd_variant']}")
    if config["model"]["modality"] == "fusion":
        logger.info(f"  Fusion method: {config['model']['fusion_method']}")

    model = build_model(config)
    logger.info(f"Model architecture:\n{model}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # Create trainer
    logger.info("Initializing trainer...")
    trainer = Trainer(model, config, device, dataset_info)

    # Load checkpoint if provided
    start_epoch = 0
    if checkpoint_path:
        start_epoch = trainer.load_checkpoint(checkpoint_path)
        logger.info(f"Resuming from epoch {start_epoch + 1}")

    # Training
    logger.info("Starting training...")
    trainer.train(train_loader, val_loader, val_metrics_loader, start_epoch=start_epoch)

    # Load best checkpoint for evaluation
    best_checkpoint_path = trainer.checkpoint_dir / "best_model.pt"
    if best_checkpoint_path.exists():
        logger.info("Loading best checkpoint for evaluation...")
        trainer.load_checkpoint(str(best_checkpoint_path))

    # Evaluation
    logger.info("Running final evaluation on test set...")
    evaluator = Evaluator(model, config, device, dataset_info)
    results = evaluator.evaluate(test_loader)

    # Visualize results
    logger.info("Generating visualizations...")
    eval_dir = (
        Path(config["logging"]["output_dir"])
        / config["logging"]["experiment_name"]
        / "evaluation"
    )
    evaluator.visualize_results(results, eval_dir)

    # Save results
    results_json_path = eval_dir / "results.json"
    evaluator.save_results_json(results, results_json_path)

    # Print summary
    logger.info("=" * 80)
    logger.info("FINAL RESULTS SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Overall Accuracy: {results['overall_accuracy']:.4f}")
    logger.info(f"Macro F1: {results['macro_f1']:.4f}")
    logger.info(f"Weighted F1: {results['weighted_f1']:.4f}")
    logger.info(f"Macro Precision: {results['macro_precision']:.4f}")
    logger.info(f"Macro Recall: {results['macro_recall']:.4f}")
    logger.info("Top 5 per-class performance:")
    per_class_df = pd.DataFrame(results["per_class"])
    top_5 = per_class_df.nlargest(5, "accuracy")
    for idx, row in top_5.iterrows():
        logger.info(
            f"  {row['class']}: Acc={row['accuracy']:.4f}, F1={row['f1']:.4f}, Count={row['count']}"
        )

    logger.info(f"Results saved to: {eval_dir}")
    logger.info("=" * 80)
    logger.info("Training pipeline completed successfully!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train multi-modal hand gesture classifier"
    )
    parser.add_argument(
        "--config", type=str, default="config.toml", help="Path to config file"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Path to checkpoint to resume from"
    )

    args = parser.parse_args()

    main(config_path=args.config, checkpoint_path=args.checkpoint)

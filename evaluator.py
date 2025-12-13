"""
Comprehensive evaluation and visualization on test set
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict
import json
import logging
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Comprehensive evaluator for test set with:
    - Multiple metrics (accuracy, F1, precision, recall)
    - Confusion matrix visualization
    - Per-class performance analysis
    - Classification report
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Dict,
        device: torch.device,
        dataset_info: Dict,
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.dataset_info = dataset_info
        self.gesture_classes = dataset_info["gesture_classes"]
        self.idx_to_gesture = dataset_info["idx_to_gesture"]
    
    @torch.no_grad()
    def evaluate(self, test_loader) -> Dict:
        """
        Evaluate on test set and compute all metrics.
        
        Args:
            test_loader: Test dataloader
            
        Returns:
            results: Dictionary with all metrics and predictions
        """
        self.model.eval()
        
        all_preds = []
        all_labels = []
        all_probs = []
        all_metadata = []
        
        logger.info("Running evaluation on test set...")
        
        for batch_idx, batch in enumerate(test_loader):
            # Move batch to device
            for key in batch:
                if key != "metadata" and isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(self.device)
            
            # Forward pass
            outputs = self._forward_batch(batch)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            
            # Collect results
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["label"].cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_metadata.extend(batch["metadata"])
            
            if (batch_idx + 1) % 100 == 0:
                logger.info(f"Processed {batch_idx + 1} batches...")
        
        # Convert to arrays
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        # Compute metrics
        results = self._compute_metrics(all_preds, all_labels, all_probs)
        results["all_preds"] = all_preds
        results["all_labels"] = all_labels
        results["all_probs"] = all_probs
        results["all_metadata"] = all_metadata
        
        logger.info("Evaluation complete!")
        
        return results
    
    def _forward_batch(self, batch: Dict) -> torch.Tensor:
        """Forward pass handling different modalities"""
        modality = self.config["model"]["modality"]
        
        if modality == "rgb":
            outputs = self.model(batch["rgb"])
        elif modality == "depth":
            outputs = self.model(batch["depth"])
        elif modality == "fusion":
            outputs = self.model(batch["rgb"], batch["depth"])
        else:
            raise ValueError(f"Unknown modality: {modality}")
        
        return outputs
    
    def _compute_metrics(
        self,
        preds: np.ndarray,
        labels: np.ndarray,
        probs: np.ndarray,
    ) -> Dict:
        """
        Compute comprehensive metrics.

        Args:
            preds: [N] predictions (int class ids)
            labels: [N] ground truth labels (int class ids)
            probs: [N, C] probability distributions

        Returns:
            metrics: Dictionary with all computed metrics
        """
        metrics: Dict = {}

        num_classes = len(self.gesture_classes)
        label_ids = list(range(num_classes))

        # ---- Overall metrics (standard multiclass) ----
        metrics["overall_accuracy"] = accuracy_score(labels, preds)
        metrics["macro_f1"] = f1_score(labels, preds, average="macro", zero_division=0)
        metrics["weighted_f1"] = f1_score(labels, preds, average="weighted", zero_division=0)
        metrics["macro_precision"] = precision_score(labels, preds, average="macro", zero_division=0)
        metrics["macro_recall"] = recall_score(labels, preds, average="macro", zero_division=0)

        # ---- Per-class metrics (correct multiclass per-label scores) ----
        per_class_precision = precision_score(
            labels, preds, labels=label_ids, average=None, zero_division=0
        )
        per_class_recall = recall_score(
            labels, preds, labels=label_ids, average=None, zero_division=0
        )
        per_class_f1 = f1_score(
            labels, preds, labels=label_ids, average=None, zero_division=0
        )
        per_class_count = np.bincount(labels, minlength=num_classes)

        per_class_metrics = []
        # idx_to_gesture is assumed to be {class_idx: class_name, ...}
        for class_idx, class_name in self.idx_to_gesture.items():
            class_idx = int(class_idx)
            count = int(per_class_count[class_idx])

            # Keep your existing plotting field name "accuracy",
            # but make it explicit: it equals recall for that class (TP / support).
            class_accuracy = float(per_class_recall[class_idx])

            per_class_metrics.append(
                {
                    "class": class_name,
                    "class_idx": class_idx,
                    "accuracy": class_accuracy,
                    "f1": float(per_class_f1[class_idx]),
                    "precision": float(per_class_precision[class_idx]),
                    "recall": float(per_class_recall[class_idx]),
                    "count": count,
                }
            )

        metrics["per_class"] = per_class_metrics

        # ---- Confusion matrix ----
        metrics["confusion_matrix"] = confusion_matrix(labels, preds, labels=label_ids)

        # ---- Classification report (already correct; useful for JSON export) ----
        metrics["classification_report"] = classification_report(
            labels,
            preds,
            target_names=self.gesture_classes,
            zero_division=0,
            output_dict=True,
        )

        logger.info(f"Overall Accuracy: {metrics['overall_accuracy']:.4f}")
        logger.info(f"Macro F1: {metrics['macro_f1']:.4f}")
        logger.info(f"Weighted F1: {metrics['weighted_f1']:.4f}")

        return metrics

    
    def visualize_results(self, results: Dict, output_dir: Path):
        """
        Create and save visualizations.
        
        Args:
            results: Results dictionary from evaluate()
            output_dir: Directory to save visualizations
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Confusion Matrix
        self._plot_confusion_matrix(
            results["confusion_matrix"],
            output_dir / "confusion_matrix.png"
        )
        
        # 2. Per-class metrics
        self._plot_per_class_metrics(
            results["per_class"],
            output_dir / "per_class_metrics.png"
        )
        
        # 3. Top errors
        self._plot_top_errors(
            results["all_preds"],
            results["all_labels"],
            results["all_probs"],
            output_dir / "top_errors.png"
        )
        
        logger.info(f"Visualizations saved to {output_dir}")
    
    def _plot_confusion_matrix(self, cm: np.ndarray, save_path: Path):
        """Plot and save confusion matrix"""
        plt.figure(figsize=(16, 14))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.gesture_classes,
            yticklabels=self.gesture_classes,
            cbar_kws={"label": "Count"}
        )
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved confusion matrix to {save_path}")
    
    def _plot_per_class_metrics(self, per_class: list, save_path: Path):
        """Plot per-class metrics"""
        df = pd.DataFrame(per_class)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Accuracy
        axes[0, 0].barh(df["class"], df["accuracy"])
        axes[0, 0].set_xlabel("Accuracy")
        axes[0, 0].set_title("Per-Class Accuracy")
        axes[0, 0].set_xlim([0, 1])
        
        # F1
        axes[0, 1].barh(df["class"], df["f1"])
        axes[0, 1].set_xlabel("F1 Score")
        axes[0, 1].set_title("Per-Class F1 Score")
        axes[0, 1].set_xlim([0, 1])
        
        # Precision
        axes[1, 0].barh(df["class"], df["precision"])
        axes[1, 0].set_xlabel("Precision")
        axes[1, 0].set_title("Per-Class Precision")
        axes[1, 0].set_xlim([0, 1])
        
        # Recall
        axes[1, 1].barh(df["class"], df["recall"])
        axes[1, 1].set_xlabel("Recall")
        axes[1, 1].set_title("Per-Class Recall")
        axes[1, 1].set_xlim([0, 1])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved per-class metrics to {save_path}")
    
    def _plot_top_errors(self, preds: np.ndarray, labels: np.ndarray, probs: np.ndarray, save_path: Path):
        """Plot top confused class pairs"""
        # Find errors
        errors = preds != labels
        error_probs = probs[errors]
        error_preds = preds[errors]
        error_labels = labels[errors]
        error_confidence = error_probs[np.arange(len(error_probs)), error_preds]
        
        if len(error_confidence) == 0:
            logger.warning("No errors to plot")
            return
        
        # Get top confusions
        confusion_pairs = {}
        for pred, label in zip(error_preds, error_labels):
            pair = (self.idx_to_gesture[label], self.idx_to_gesture[pred])
            confusion_pairs[pair] = confusion_pairs.get(pair, 0) + 1
        
        # Sort by frequency
        top_confusions = sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True)[:10]
        
        if len(top_confusions) > 0:
            pairs, counts = zip(*top_confusions)
            pair_labels = [f"{p[0]} → {p[1]}" for p in pairs]
            
            plt.figure(figsize=(12, 6))
            plt.barh(pair_labels, counts)
            plt.xlabel("Number of Confusions")
            plt.title("Top 10 Most Common Confusions")
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
            logger.info(f"Saved top errors to {save_path}")
    
    def save_results_json(self, results: Dict, save_path: Path):
        """Save results to JSON (excluding arrays)"""
        
        def convert_to_native_types(obj):
            """Recursively convert numpy types to Python native types"""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_to_native_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native_types(item) for item in obj]
            else:
                return obj
        
        save_data = {
            "overall_accuracy": float(results["overall_accuracy"]),
            "macro_f1": float(results["macro_f1"]),
            "weighted_f1": float(results["weighted_f1"]),
            "macro_precision": float(results["macro_precision"]),
            "macro_recall": float(results["macro_recall"]),
            "per_class": convert_to_native_types(results["per_class"]),
            "classification_report": convert_to_native_types(results["classification_report"]),
        }

        with open(save_path, "w") as f:
            json.dump(save_data, f, indent=2)
        
        logger.info(f"Saved results to {save_path}")

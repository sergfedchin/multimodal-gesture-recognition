"""
Comprehensive evaluation and visualization on test set
"""

import json
import logging
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Comprehensive evaluator for test set with:
    - Multiple metrics (accuracy, F1, precision, recall)
    - Confusion matrix visualization
    - Per-class performance analysis
    - Classification report
    - Paper-style image-level evaluation (33 classes)
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
        metrics["weighted_f1"] = f1_score(
            labels, preds, average="weighted", zero_division=0
        )
        metrics["macro_precision"] = precision_score(
            labels, preds, average="macro", zero_division=0
        )
        metrics["macro_recall"] = recall_score(
            labels, preds, average="macro", zero_division=0
        )

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

    # =====================================================================================
    # PAPER-STYLE EVALUATION (IMAGE-LEVEL, 33 CLASSES)
    # =====================================================================================

    class HandLevelDatasetForPaper(Dataset):
        """Dataset for loading hands from test parquet for paper-style evaluation"""

        def __init__(self, test_df, data_root, image_size, modality):
            self.test_df = test_df.reset_index(drop=True)
            self.data_root = Path(data_root)
            self.image_size = image_size
            self.modality = modality

            # Transforms
            self.rgb_transform = transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

            self.depth_transform = transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5], std=[0.5]),
                ]
            )

        def __len__(self):
            return len(self.test_df)

        def __getitem__(self, idx):
            row = self.test_df.iloc[idx]
            output = {"idx": idx}

            # Load RGB if needed
            if self.modality in ["rgb", "fusion"]:
                rgb_path = self.data_root / row["rgb_image_path"]
                try:
                    rgb_image = Image.open(rgb_path).convert("RGB")
                    output["rgb"] = self.rgb_transform(rgb_image)
                except Exception as e:
                    logger.warning(f"Failed to load RGB {rgb_path}: {e}")
                    output["rgb"] = torch.zeros((3, self.image_size, self.image_size))

            # Load Depth if needed
            if self.modality in ["depth", "fusion"]:
                depth_path = self.data_root / row["depth_image_path"]
                try:
                    depth_image = Image.open(depth_path).convert("L")
                    output["depth"] = self.depth_transform(depth_image)
                except Exception as e:
                    logger.warning(f"Failed to load depth {depth_path}: {e}")
                    output["depth"] = torch.zeros((1, self.image_size, self.image_size))

            return output

    @torch.no_grad()
    def evaluate_paper_style(self, test_parquet_path: str) -> Dict:
        """
        Evaluate following the original HaGRID paper methodology:
        - Image-level predictions (not per-hand)
        - 33 classes (excluding no_gesture)
        - Aggregate predictions from multiple hands per image
        - Special handling for thumb_index/thumb_index2

        Args:
            test_parquet_path: Path to test.parquet file

        Returns:
            results: Dictionary with paper-style metrics
        """
        logger.info("=" * 80)
        logger.info("PAPER-STYLE EVALUATION (Image-Level, 33 Classes)")
        logger.info("=" * 80)

        self.model.eval()

        # Load test dataframe
        test_df = pd.read_parquet(test_parquet_path)

        # Filter out no_gesture at image level
        test_df_filtered = test_df[test_df["gesture_class"] != "no_gesture"].copy()
        test_df_filtered = test_df_filtered.reset_index(drop=True)

        logger.info(f"Total test samples (hands): {len(test_df)}")
        logger.info(f"After removing no_gesture images: {len(test_df_filtered)}")
        logger.info(f"Unique images: {test_df_filtered['original_id'].nunique()}")

        # Create idx_to_label mapping
        idx_to_label = self.idx_to_gesture
        label_to_idx = {v: k for k, v in idx_to_label.items()}

        # Step 1: Batched inference on all hands
        logger.info("Step 1/2: Running batched inference...")

        hand_dataset = self.HandLevelDatasetForPaper(
            test_df_filtered,
            self.config["dataset"]["data_root"],
            self.config["model"].get("image_size", 224),
            self.config["model"]["modality"],
        )

        hand_loader = DataLoader(
            hand_dataset,
            batch_size=self.config["training"]["batch_size"],
            shuffle=False,  # IMPORTANT: Preserve order
            num_workers=self.config["dataset"]["num_workers"],
            pin_memory=self.config["hardware"]["pin_memory"],
            drop_last=False,
        )

        all_hand_predictions = []
        all_hand_probabilities = []
        all_indices = []

        for batch in tqdm(hand_loader, desc="Inference"):
            # Move to device
            if "rgb" in batch:
                batch["rgb"] = batch["rgb"].to(self.device)
            if "depth" in batch:
                batch["depth"] = batch["depth"].to(self.device)

            # Forward pass
            outputs = self._forward_batch(batch)
            probabilities = F.softmax(outputs, dim=1)
            pred_indices = outputs.argmax(1)

            # Collect predictions
            for i in range(len(pred_indices)):
                pred_idx = pred_indices[i].item()
                pred_prob = probabilities[i, pred_idx].item()
                pred_label = idx_to_label[pred_idx]

                # Convert thumb_index2 to thumb_index at hand level
                # (should already be merged, but just in case)
                if pred_label == "thumb_index2":
                    pred_label = "thumb_index"
                    # Find thumb_index idx
                    pred_idx = label_to_idx.get("thumb_index", pred_idx)

                all_hand_predictions.append(pred_label)
                all_hand_probabilities.append(pred_prob)
                all_indices.append(batch["idx"][i].item())

        # Verify order preservation
        assert all_indices == list(range(len(test_df_filtered))), "Order not preserved!"

        # Add predictions to dataframe
        test_df_filtered["predicted_label"] = all_hand_predictions
        test_df_filtered["predicted_prob"] = all_hand_probabilities

        # Step 2: Aggregate by original_id
        logger.info("\nStep 2/2: Aggregating predictions per image...")
        grouped = test_df_filtered.groupby("original_id", sort=False)

        all_true_labels = []
        all_pred_labels = []

        for original_id, group in tqdm(grouped, desc="Aggregating"):
            true_label = group.iloc[0]["gesture_class"]

            if true_label == "no_gesture":
                continue

            # Get predictions for each hand
            hand_predictions = group["predicted_label"].tolist()
            hand_probabilities = list(
                zip(group["predicted_label"].tolist(), group["predicted_prob"].tolist())
            )

            # Aggregate with thumb_index special handling
            final_prediction = self._aggregate_hand_predictions(
                hand_predictions, hand_probabilities
            )

            all_true_labels.append(true_label)
            all_pred_labels.append(final_prediction)

        # Get unique labels (33 classes)
        unique_labels = sorted(set(all_true_labels + all_pred_labels))
        if "no_gesture" in unique_labels:
            unique_labels.remove("no_gesture")

        # Calculate accuracy (33 classes)
        correct = sum(
            [1 for true, pred in zip(all_true_labels, all_pred_labels) if true == pred]
        )
        total = len(all_true_labels)
        accuracy = 100 * correct / total

        # Calculate F1 scores (33 classes)
        macro_f1 = f1_score(all_true_labels, all_pred_labels, labels=unique_labels, average='macro', zero_division=0)
        weighted_f1 = f1_score(all_true_labels, all_pred_labels, labels=unique_labels, average='weighted', zero_division=0)

        # Classification report
        report = classification_report(
            all_true_labels,
            all_pred_labels,
            labels=unique_labels,
            target_names=unique_labels,
            digits=4,
            zero_division=0,
        )

        # Confusion matrix
        cm = confusion_matrix(all_true_labels, all_pred_labels, labels=unique_labels)

        results = {
            "paper_style_accuracy": accuracy,
            "paper_style_macro_f1": macro_f1 * 100,  # В процентах для единообразия
            "paper_style_weighted_f1": weighted_f1 * 100,
            "paper_style_num_images": total,
            "paper_style_confusion_matrix": cm,
            "paper_style_labels": unique_labels,
            "paper_style_report": report,
            "paper_style_all_true": all_true_labels,
            "paper_style_all_pred": all_pred_labels,
        }

        logger.info("=" * 80)
        logger.info(f"IMAGE-LEVEL ACCURACY (33 classes): {accuracy:.2f}%")
        logger.info(f"IMAGE-LEVEL MACRO F1 (33 classes): {macro_f1*100:.2f}%")
        logger.info(f"IMAGE-LEVEL WEIGHTED F1 (33 classes): {weighted_f1*100:.2f}%")
        logger.info(f"Number of test images: {total}")
        logger.info("=" * 80)
        logger.info("Classification Report (Image-Level):")
        logger.info(report)

        return results

    @staticmethod
    def _aggregate_hand_predictions(hand_predictions, hand_probabilities):
        """
        Aggregate predictions from multiple hands with special thumb_index handling.

        Rules:
        1. Single hand: return its prediction
        2. Two hands: one no_gesture → return other gesture
        3. Two hands: both thumb_index → thumb_index2
        4. Two hands: same gesture → that gesture
        5. Two hands: different gestures → higher probability
        6. Both no_gesture → higher confidence no_gesture
        """
        num_hands = len(hand_predictions)

        if num_hands == 1:
            return hand_predictions[0]

        elif num_hands == 2:
            pred1, pred2 = hand_predictions
            prob1, prob2 = hand_probabilities[0][1], hand_probabilities[1][1]

            # Both no_gesture
            if pred1 == "no_gesture" and pred2 == "no_gesture":
                return pred1 if prob1 >= prob2 else pred2

            # One is no_gesture
            if pred1 == "no_gesture":
                return pred2
            if pred2 == "no_gesture":
                return pred1

            # SPECIAL: Both hands showing thumb_index → thumb_index2
            if pred1 == "thumb_index" and pred2 == "thumb_index":
                return "thumb_index2"

            # Both same gesture
            if pred1 == pred2:
                return pred1
            else:
                # Different gestures: higher probability
                return pred1 if prob1 >= prob2 else pred2

        else:
            # More than 2 hands (rare)
            non_no_gesture = [
                (label, prob)
                for label, prob in hand_probabilities
                if label != "no_gesture"
            ]

            if non_no_gesture:
                # Check if 2+ hands all predict thumb_index
                if len(non_no_gesture) >= 2 and all(
                    label == "thumb_index" for label, _ in non_no_gesture
                ):
                    return "thumb_index2"

                return max(non_no_gesture, key=lambda x: x[1])[0]
            else:
                return max(hand_probabilities, key=lambda x: x[1])[0]

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
            results["confusion_matrix"], output_dir / "confusion_matrix.png"
        )

        # 2. Per-class metrics
        self._plot_per_class_metrics(
            results["per_class"], output_dir / "per_class_metrics.png"
        )

        # 3. Top errors
        self._plot_top_errors(
            results["all_preds"],
            results["all_labels"],
            results["all_probs"],
            output_dir / "top_errors.png",
        )

        # 4. Paper-style confusion matrix (if available)
        if "paper_style_confusion_matrix" in results:
            self._plot_confusion_matrix(
                results["paper_style_confusion_matrix"],
                output_dir / "confusion_matrix_paper_style.png",
                title="Confusion Matrix (Paper-Style, 33 Classes)",
                labels=results["paper_style_labels"],
            )

        logger.info(f"Visualizations saved to {output_dir}")

    def _plot_confusion_matrix(
        self,
        cm: np.ndarray,
        save_path: Path,
        title: str = "Confusion Matrix",
        labels=None,
    ):
        """Plot and save confusion matrix"""
        if labels is None:
            labels = self.gesture_classes

        plt.figure(figsize=(16, 14))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            cbar_kws={"label": "Count"},
        )

        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title(title)
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

    def _plot_top_errors(
        self, preds: np.ndarray, labels: np.ndarray, probs: np.ndarray, save_path: Path
    ):
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
        top_confusions = sorted(
            confusion_pairs.items(), key=lambda x: x[1], reverse=True
        )[:10]

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
                return {
                    key: convert_to_native_types(value) for key, value in obj.items()
                }
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
            "classification_report": convert_to_native_types(
                results["classification_report"]
            ),
        }

        # Add paper-style results if available
        if "paper_style_accuracy" in results:
            save_data["paper_style_accuracy"] = float(results["paper_style_accuracy"])
            save_data["paper_style_macro_f1"] = float(results.get("paper_style_macro_f1", 0))
            save_data["paper_style_weighted_f1"] = float(results.get("paper_style_weighted_f1", 0))
            save_data["paper_style_num_images"] = int(results["paper_style_num_images"])

        with open(save_path, "w") as f:
            json.dump(save_data, f, indent=2)

        logger.info(f"Saved results to {save_path}")

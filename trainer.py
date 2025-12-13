"""
Training loop with metrics computation, checkpointing, and logging
"""

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    Early stopping to stop training when validation metric stops improving.
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = "min"):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as an improvement
            mode: 'min' for loss (lower is better), 'max' for accuracy (higher is better)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

        logger.info(f"Early stopping initialized: patience={patience}, mode={mode}")

    def __call__(self, current_score: float) -> bool:
        """
        Check if training should stop.

        Args:
            current_score: Current validation metric

        Returns:
            True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = current_score
            return False

        # Check if there's improvement
        if self.mode == "min":
            improved = current_score < (self.best_score - self.min_delta)
        else:  # mode == "max"
            improved = current_score > (self.best_score + self.min_delta)

        if improved:
            self.best_score = current_score
            self.counter = 0
            return False
        else:
            self.counter += 1
            logger.info(f"Early stopping counter: {self.counter}/{self.patience}")

            if self.counter >= self.patience:
                logger.info(
                    f"Early stopping triggered! No improvement for {self.patience} epochs."
                )
                self.early_stop = True
                return True

            return False


class Trainer:
    """
    Training manager with support for:
    - Mixed precision training (AMP)
    - Gradient accumulation
    - Multiple metrics computation
    - Checkpoint saving and loading
    - TensorBoard logging
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

        # Setup optimizer
        self.optimizer = self._build_optimizer()
        logger.info(
            f"Optimizer created with initial LR: {self.optimizer.param_groups[0]['lr']:.2e}"
        )

        # Setup learning rate scheduler
        self.scheduler = self._build_scheduler()
        if self.scheduler is not None:
            logger.info(f"Scheduler type: {type(self.scheduler).__name__}")
            logger.info(
                f"LR after scheduler init: {self.optimizer.param_groups[0]['lr']:.2e}"
            )

        # Setup loss function
        self.criterion = nn.CrossEntropyLoss()

        # Mixed precision training
        self.use_amp = config["hardware"]["use_amp"]
        self.scaler = GradScaler("cuda") if self.use_amp else None

        # Gradient accumulation
        self.accumulation_steps = config["training"]["gradient_accumulation_steps"]
        self.gradient_clip_norm = config["training"].get("gradient_clip_norm", 1.0)
        logger.info(f"Gradient clipping enabled with max norm: {self.gradient_clip_norm}")

        # Directories
        self.output_dir: Path = (
            Path(config["logging"]["output_dir"]) / config["logging"]["experiment_name"]
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)

        # Logging
        self.log_interval = config["logging"]["log_interval"]
        self.save_interval = config["logging"]["save_interval"]
        self.save_best = config["logging"]["save_best"]
        self.checkpoints_keep = config["logging"]["checkpoints_keep"]

        # TensorBoard
        self.writer = SummaryWriter(self.output_dir / "runs")

        # Best model tracking
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.kept_checkpoints = []

        # Early stopping
        early_stopping_config = config["training"].get("early_stopping", {})
        if early_stopping_config.get("enabled", False):
            self.early_stopping = EarlyStopping(
                patience=early_stopping_config.get("patience", 10),
                min_delta=early_stopping_config.get("min_delta", 0.0),
                mode=early_stopping_config.get(
                    "mode", "min"
                ),  # 'min' for loss, 'max' for accuracy
            )
        else:
            self.early_stopping = None
            logger.info("Early stopping is disabled")

        logger.info(f"Trainer initialized. Output dir: {self.output_dir}")

    def _build_optimizer(self) -> optim.Optimizer:
        """Build optimizer from config"""
        opt_name = self.config["training"]["optimizer"].lower()
        lr = self.config["training"]["learning_rate"]
        wd = self.config["training"]["weight_decay"]

        if opt_name == "adam":
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd)
        elif opt_name == "adamw":
            return optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)
        elif opt_name == "sgd":
            return optim.SGD(
                self.model.parameters(), lr=lr, weight_decay=wd, momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_name}")

    def _build_scheduler(self) -> optim.lr_scheduler.LRScheduler:
        """Build learning rate scheduler from config"""
        scheduler_name = self.config["training"]["scheduler"].lower()
        num_epochs = self.config["training"]["num_epochs"]
        warmup_epochs = self.config["training"]["warmup_epochs"]

        # Create linear warmup scheduler
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                # Start from 10% of base LR, linearly increase to 100%
                return 0.1 + 0.9 * (epoch / warmup_epochs)
            
            if scheduler_name == "cosine":
                return 0.5 * (
                    1
                    + np.cos(
                        np.pi * (epoch - warmup_epochs) / (num_epochs - warmup_epochs)
                    )
                )
            elif scheduler_name == "linear":
                return (num_epochs - epoch) / (num_epochs - warmup_epochs)
            elif scheduler_name == "exponential":
                return 0.95 ** (epoch - warmup_epochs)
            elif scheduler_name == "step":
                return 0.1 ** (epoch // (num_epochs // 3))
            else:
                return 1.0

        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def train_epoch(self, train_loader: DataLoader) -> Dict:
        """
        Train for one epoch.

        Returns:
            metrics: Dictionary with loss and other metrics
        """
        self.model.train()

        total_loss = 0.0
        num_batches = 0

        # Create progress bar
        pbar = tqdm(
            train_loader,
            desc=f"[Epoch {self.current_epoch + 1}/{self.config['training']['num_epochs']}]",
            ncols=120,
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            unit="batch",
        )

        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            if isinstance(batch, dict):
                for key in batch:
                    if key != "metadata" and isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device)
            else:
                batch = [
                    x.to(self.device) if isinstance(x, torch.Tensor) else x
                    for x in batch
                ]

            # Forward pass
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                outputs = self._forward_batch(batch)
                loss = self.criterion(outputs, batch["label"])
                loss = loss / self.accumulation_steps

            # Check for NaN/inf
            if torch.isnan(loss) or torch.isinf(loss):
                logger.error(f"NaN/Inf loss detected at batch {batch_idx}!")
                logger.error(f"  Outputs min/max: {outputs.min():.4f}/{outputs.max():.4f}")
                logger.error(f"  Labels: {batch['label'][:10]}")
                # Skip this batch
                continue

            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            total_loss += loss.item() * self.accumulation_steps
            num_batches += 1

            # Optimizer step
            if (batch_idx + 1) % self.accumulation_steps == 0:
                if self.use_amp:
                    # Unscale gradients before clipping
                    self.scaler.unscale_(self.optimizer)
                    
                    # Clip gradients to prevent explosion
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=1.0  # or get from config
                    )
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # Clip gradients for non-AMP training too
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=1.0
                    )
                    
                    self.optimizer.step()
                
                self.optimizer.zero_grad()

            # Update progress bar
            avg_loss = total_loss / num_batches
            current_lr = self.optimizer.param_groups[0]["lr"]
            pbar.set_postfix({"loss": f"{avg_loss:.4f}", "lr": f"{current_lr:.2e}"})

            # Logging - use tqdm.write to avoid interfering with progress bar
            if (batch_idx + 1) % self.log_interval == 0:
                tqdm.write(
                    f"[Epoch {self.current_epoch + 1}/{self.num_epochs}] Batch {batch_idx + 1}/{len(train_loader)}: "
                    f"Loss = {avg_loss:.4f}, LR = {current_lr:.2e}"
                )

        pbar.close()
        avg_loss = total_loss / num_batches
        return {"loss": avg_loss}

    def _forward_batch(self, batch: Dict) -> torch.Tensor:
        """
        Forward pass, handling different modalities.

        Args:
            batch: Dictionary containing batch data

        Returns:
            logits: [B, num_classes]
        """
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

    @torch.no_grad()
    def evaluate(self, val_loader, compute_metrics: bool = True) -> Dict:
        """
        Evaluate on validation set.

        Args:
            val_loader: Validation dataloader
            compute_metrics: Whether to compute detailed metrics

        Returns:
            metrics: Dictionary with loss and optional metrics
        """
        self.model.eval()

        total_loss = 0.0
        num_batches = 0

        all_preds = []
        all_labels = []

        # Create progress bar
        pbar = tqdm(
            val_loader,
            desc="Validation",
            ncols=100,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
            leave=True,
            unit="batch",
        )

        for batch in pbar:
            # Move batch to device
            for key in batch:
                if key != "metadata" and isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(self.device)

            # Forward pass
            with autocast("cuda", enabled=self.use_amp):
                outputs = self._forward_batch(batch)
                loss = self.criterion(outputs, batch["label"])

            total_loss += loss.item()
            num_batches += 1

            # Collect predictions
            preds = outputs.argmax(dim=1).cpu().numpy()
            labels = batch["label"].cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels)

            # Update progress bar
            pbar.set_postfix({"loss": f"{total_loss / num_batches:.4f}"})

        pbar.close()

        avg_loss = total_loss / num_batches
        metrics = {"loss": avg_loss}

        # Compute detailed metrics
        if compute_metrics and len(all_preds) > 0:
            all_preds = np.array(all_preds)
            all_labels = np.array(all_labels)

            accuracy = accuracy_score(all_labels, all_preds)
            f1 = f1_score(
                all_labels,
                all_preds,
                average=self.config["evaluation"]["f1_average"],
                zero_division=0,
            )

            metrics["accuracy"] = accuracy
            metrics["f1"] = f1

        return metrics

    def save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool = False):
        """Save checkpoint"""
        
        # Convert numpy types in metrics to Python native types
        clean_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, np.floating):
                clean_metrics[key] = float(value)
            elif isinstance(value, np.integer):
                clean_metrics[key] = int(value)
            elif isinstance(value, np.ndarray):
                clean_metrics[key] = value.tolist()
            else:
                clean_metrics[key] = value
        
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "metrics": clean_metrics,  # Use cleaned metrics
            "config": self.config,
        }

        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")

        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best checkpoint: {best_path}")

        # Keep only top-k checkpoints
        self.kept_checkpoints.append((checkpoint_path, clean_metrics.get("accuracy", 0)))
        self.kept_checkpoints.sort(key=lambda x: x[1], reverse=True)
        if len(self.kept_checkpoints) > self.checkpoints_keep:
            to_remove = self.kept_checkpoints.pop()
            to_remove[0].unlink()
            logger.info(f"Removed old checkpoint: {to_remove[0]}")


    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint"""
        checkpoint = torch.load(
            checkpoint_path, 
            map_location=self.device,
            weights_only=False  # Required for PyTorch 2.6+ to load metrics with numpy objects
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        return checkpoint["epoch"]

    def train(
        self, train_loader, val_loader, val_metrics_loader: Optional[DataLoader] = None
    ):
        """
        Full training loop.

        Args:
            train_loader: Training dataloader
            val_loader: Full validation dataloader
            val_metrics_loader: Optional smaller validation set for computing metrics during training
        """
        self.num_epochs = self.config["training"]["num_epochs"]
        metrics_interval = self.config["evaluation"]["metrics_compute_interval"]

        logger.info("Starting training...")
        logger.info(f"Total epochs: {self.num_epochs}")
        logger.info(f"Training samples: {self.dataset_info['train_size']}")
        logger.info(f"Validation samples: {self.dataset_info['val_size']}")

        for epoch in range(self.num_epochs):
            self.current_epoch = epoch
            logger.info("=" * 50)
            logger.info(f"Epoch {self.current_epoch + 1}/{self.num_epochs}")
            logger.info("=" * 50)

            # Train
            train_metrics = self.train_epoch(train_loader)
            logger.info(f"Train loss: {train_metrics['loss']:.4f}")
            self.writer.add_scalar("train/loss", train_metrics["loss"], epoch)

            # Step scheduler
            self.scheduler.step()

            # Validate (compute metrics every N epochs or on first/last epoch)
            compute_metrics = (
                (epoch + 1) % metrics_interval == 0
                or epoch == 0
                or epoch == self.num_epochs - 1
            )

            if compute_metrics and val_metrics_loader is not None:
                val_metrics = self.evaluate(val_metrics_loader, compute_metrics=True)
            else:
                val_metrics = self.evaluate(val_loader, compute_metrics=False)

            logger.info(f"Val loss: {val_metrics['loss']:.4f}")
            self.writer.add_scalar("val/loss", val_metrics["loss"], epoch)

            if "accuracy" in val_metrics:
                logger.info(f"Val accuracy: {val_metrics['accuracy']:.4f}")
                logger.info(f"Val F1: {val_metrics['f1']:.4f}")
                self.writer.add_scalar("val/accuracy", val_metrics["accuracy"], epoch)
                self.writer.add_scalar("val/f1", val_metrics["f1"], epoch)

            # Checkpoint
            is_best = False
            if (epoch + 1) % self.save_interval == 0:
                is_best = (
                    self.save_best
                    and val_metrics.get("accuracy", 0) > self.best_val_acc
                )
                if is_best:
                    self.best_val_acc = val_metrics["accuracy"]
                    self.best_epoch = epoch + 1
                self.save_checkpoint(epoch + 1, val_metrics, is_best)

            # Early stopping check
            if self.early_stopping is not None:
                # Determine which metric to monitor based on early stopping mode
                early_stop_config = self.config["training"].get("early_stopping", {})
                monitor_metric = early_stop_config.get(
                    "monitor", "loss"
                )  # 'loss', 'accuracy', or 'f1'

                if monitor_metric == "accuracy":
                    metric_value = val_metrics.get("accuracy", 0.0)
                elif monitor_metric == "f1":
                    metric_value = val_metrics.get("f1", 0.0)
                else:  # Default to loss
                    metric_value = val_metrics["loss"]

                # Check if we should stop
                if self.early_stopping(metric_value):
                    logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                    logger.info(
                        f"Best validation accuracy: {self.best_val_acc:.4f} at epoch {self.best_epoch}"
                    )
                    break

        logger.info("=" * 50)
        logger.info("Training complete!")
        logger.info(
            f"Best val accuracy: {self.best_val_acc:.4f} at epoch {self.best_epoch}"
        )
        logger.info("=" * 50)

        self.writer.close()

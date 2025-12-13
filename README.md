# Multi-Modal Hand Gesture Classification with VSSD

Complete training pipeline for hand gesture classification using VSSD (Vision State Space Duality) with support for RGB images, depth maps, and multi-modal fusion.

## Features

- **Multi-modal support**: RGB-only, depth-only, and fusion architectures
- **Efficient data handling**: Parallelized data loading for massive datasets (~1.5M images)
- **Mixed precision training**: Automatic Mixed Precision (AMP) for faster training and reduced memory
- **Flexible configuration**: TOML-based config for easy experimentation
- **Comprehensive evaluation**: Accuracy, F1, precision, recall, confusion matrices, per-class analysis
- **Checkpoint management**: Best model saving and automatic checkpoint rotation
- **TensorBoard logging**: Real-time training visualization
- **Configurable sampling**: Quickly experiment with data subsets

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Your Data

Ensure your parquet files have the following columns:
- `gesture_class`: Gesture label name
- `rgb_image_path`: Relative path to RGB image
- `depth_image_path`: Relative path to depth map
- `original_id`, `hand_index`, `user_id`, `age`, `gender`, `race`: Metadata

### 3. Configure Your Experiment

Edit `config.toml` with your paths and parameters:

```toml
[dataset]
train_parquet = "path/to/train.parquet"
val_parquet = "path/to/val.parquet"
test_parquet = "path/to/test.parquet"
data_root = "path/to/data"
train_subsample = 0.1  # Use 10% for quick testing

[model]
modality = "fusion"  # "rgb", "depth", or "fusion"
vssd_variant = "tiny"  # "micro", "tiny", "small", "base"
fusion_method = "gated_fusion"

[training]
batch_size = 32
num_epochs = 100
learning_rate = 1e-4
```

### 4. Train the Model

```bash
python train.py --config config.toml
```

Or resume from checkpoint:

```bash
python train.py --config config.toml --checkpoint outputs/experiment_name/checkpoints/best_model.pt
```

## Project Structure

```
.
├── config.toml              # Configuration file
├── train.py                 # Main training script
├── data_loader.py          # Dataset and DataLoader utilities
├── models.py               # Model architectures (RGB, Depth, Fusion)
├── trainer.py              # Training loop and logging
├── evaluator.py            # Comprehensive evaluation and visualization
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Architecture Overview

### RGB-Only Pipeline
```
Input: [B, 3, 224, 224] RGB image
  ↓
VSSD Backbone (ImageNet pretrained)
  ↓
Global Average Pooling: [B, feat_dim]
  ↓
Classification Head (FC + Dropout)
  ↓
Output: [B, 34] logits
```

### Depth-Only Pipeline
```
Input: [B, 1, 224, 224] Depth map
  ↓
Channel Expansion Conv: 1→3 channels
  ↓
VSSD Backbone (ImageNet pretrained)
  ↓
Global Average Pooling: [B, feat_dim]
  ↓
Classification Head
  ↓
Output: [B, 34] logits
```

### Fusion Pipeline (Dual-Branch)
```
RGB Input: [B, 3, 224, 224]        Depth Input: [B, 1, 224, 224]
  ↓                                  ↓
VSSD RGB Branch                     Channel Expansion + VSSD Depth Branch
  ↓                                  ↓
RGB Features: [B, feat_dim, H, W]  Depth Features: [B, feat_dim, H, W]
  ↓                                  ↓
        Fusion Module (Gated/Adaptive/Concat)
                  ↓
        Fused Features: [B, feat_dim, H, W]
                  ↓
        Global Average Pooling: [B, feat_dim]
                  ↓
        Classification Head
                  ↓
        Output: [B, 34] logits
```

## VSSD Backbone Variants

| Variant | Parameters | FLOPs | ImageNet Top-1 |
|---------|-----------|-------|----------------|
| Micro   | 14M       | 2.3G  | 82.5%          |
| Tiny    | 24M       | 4.5G  | 83.6%          |
| Small   | 40M       | 7.4G  | 84.1%          |
| Base    | 89M       | 16.1G | 84.7%          |

## Fusion Methods

### 1. Gated Fusion
Uses learned gating mechanism to adaptively weight RGB and depth features.
```
gate = Sigmoid(Conv(concat(rgb, depth)))
fused = rgb * gate + depth * (1 - gate)
```

### 2. Adaptive Fusion
Channel-wise attention for each modality with normalization.
```
rgb_weight = Sigmoid(ChannelAttention(rgb))
depth_weight = Sigmoid(ChannelAttention(depth))
fused = rgb * (rgb_weight / total) + depth * (depth_weight / total)
```

### 3. Concatenation Fusion
Simple concatenation followed by 1×1 convolution.
```
fused = Conv(concat(rgb, depth))
```

## Configuration Guide

### Data Settings
- `train_subsample`: Fraction of training data to use (0.0-1.0). Use 0.1 for 10% for quick experiments
- `num_workers`: Number of data loading workers. Increase for better parallelism

### Model Settings
- `modality`: "rgb", "depth", or "fusion"
- `vssd_variant`: Backbone size (smaller = faster training)
- `fusion_method`: For fusion mode only
- `pretrained`: Use ImageNet pretrained weights

### Training Settings
- `batch_size`: Larger is faster but uses more memory
- `learning_rate`: 1e-4 is good for pretrained models
- `mixed_precision`: Enable AMP for faster training
- `gradient_accumulation_steps`: Effective batch size = batch_size × accumulation_steps

### Evaluation Settings
- `val_metrics_subsample`: Fraction of validation set to use for metrics during training (0 = disabled)
- `metrics_compute_interval`: Compute metrics every N epochs (saves time)
- `test_metrics_subsample`: Fraction of test set for final evaluation

## Optimization Tips

### For Large Datasets
1. Set `num_workers` to match your CPU cores
2. Enable mixed precision: `use_amp = true`
3. Use gradient accumulation to simulate larger batches
4. Reduce `val_metrics_subsample` if metrics computation is slow
5. Use `train_subsample` during hyperparameter tuning

### For Memory Constraints
1. Reduce `batch_size` (16, 8 if needed)
2. Enable gradient accumulation: `gradient_accumulation_steps = 2`
3. Use smaller `vssd_variant` ("micro" or "tiny")
4. Disable augmentation if memory is critical

### For Faster Iteration
1. Set `train_subsample = 0.1` for 10% data
2. Set `val_metrics_subsample = 0` to skip metrics during training
3. Reduce `num_epochs` for initial testing
4. Use smaller batch size but same learning rate

## Output Structure

```
outputs/experiment_name/
├── checkpoints/
│   ├── best_model.pt
│   ├── checkpoint_epoch_1.pt
│   └── ...
├── evaluation/
│   ├── confusion_matrix.png
│   ├── per_class_metrics.png
│   ├── top_errors.png
│   └── results.json
├── runs/               # TensorBoard logs
└── training.log
```

## Metrics

### Overall Metrics
- **Accuracy**: Fraction of correct predictions
- **Macro F1**: F1 score averaged across all classes (equal weight)
- **Weighted F1**: F1 score weighted by class frequency

### Per-Class Metrics
- **Accuracy**: Per-class accuracy
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1**: Harmonic mean of precision and recall

### Visualizations
- **Confusion Matrix**: Heatmap of predictions vs true labels
- **Per-Class Metrics**: Bar charts for each gesture class
- **Top Errors**: Most common misclassifications

## Memory Usage

Approximate memory requirements (single GPU):

| Variant | Batch Size 32 | Batch Size 64 |
|---------|---------------|---------------|
| Micro   | 4 GB          | 6 GB          |
| Tiny    | 6 GB          | 9 GB          |
| Small   | 8 GB          | 12 GB         |
| Base    | 12 GB         | 16 GB+        |

Note: Fusion requires 2× more memory than single modality.

## Troubleshooting

### Out of Memory
- Reduce batch size
- Use gradient accumulation
- Disable mixed precision
- Use smaller VSSD variant

### Slow Data Loading
- Increase `num_workers`
- Set `pin_memory = true`
- Ensure data is on fast storage (NVMe/SSD)

### Slow Evaluation
- Reduce `val_metrics_subsample`
- Increase `metrics_compute_interval`
- Reduce `test_metrics_subsample` for initial runs

## References

- VSSD Paper: [Vision Mamba with Non-Causal State Space Duality](https://arxiv.org/abs/2407.18559)
- GitHub: https://github.com/YuHengsss/VSSD

## License

This code is provided as-is for research purposes.

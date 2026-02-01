import io
import warnings
from typing import List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


class LRPLinear(nn.Module):
    """LRP rule for Linear layers (ε-rule)."""

    def __init__(self, layer, epsilon=1e-6):
        super().__init__()
        self.layer = layer
        self.epsilon = epsilon

    def forward(self, input_activation, output_activation, output_relevance):
        """
        Backpropagate relevance through linear layer.

        Args:
            input_activation: Input to the layer (saved during forward)
            output_activation: Output of the layer (saved during forward)
            output_relevance: Relevance from the next layer
        """
        # Compute z = W * a + b
        z = self.layer(input_activation)

        # Add epsilon for numerical stability
        z = z + self.epsilon * torch.sign(z)

        # s = R / z
        s = output_relevance / z

        # Compute gradient
        c = torch.autograd.grad(z, input_activation, s, retain_graph=True)[0]

        # R = a * c
        return input_activation * c


class LRPConv2d(nn.Module):
    """LRP rule for Conv2d layers (γ-rule for emphasis on positive contributions)."""

    def __init__(self, layer, gamma=0.25, epsilon=1e-6):
        super().__init__()
        self.layer = layer
        self.gamma = gamma
        self.epsilon = epsilon

    def forward(self, input_activation, output_activation, output_relevance):
        # Create modified layer with γ-weighted weights
        weight = self.layer.weight
        weight_pos = torch.clamp(weight, min=0)
        weight_neg = torch.clamp(weight, max=0)

        # Apply γ-rule: emphasize positive contributions
        weight_modified = weight_pos * (1 + self.gamma) + weight_neg * (1 - self.gamma)

        # Forward with modified weights
        z = F.conv2d(
            input_activation,
            weight_modified,
            bias=self.layer.bias if self.layer.bias is not None else None,
            stride=self.layer.stride,
            padding=self.layer.padding,
            dilation=self.layer.dilation,
            groups=self.layer.groups,
        )

        z = z + self.epsilon * torch.sign(z)
        s = output_relevance / z

        c = torch.autograd.grad(z, input_activation, s, retain_graph=True)[0]
        return input_activation * c


class LRPAttention(nn.Module):
    """LRP rule for attention mechanisms."""

    def __init__(self, epsilon=1e-6):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, query, key, value, attn_weights, output_relevance):
        """
        Propagate relevance through attention.
        Relevance flows proportionally to attention weights.
        """
        # Relevance flows through attention weights to values
        # R_value = R_out * attn_weights^T
        value_relevance = torch.matmul(
            output_relevance.unsqueeze(-1), attn_weights.unsqueeze(-2)
        ).squeeze(-2)

        return value_relevance


class SimpleLRP:
    """
    Simplified LRP for dual-branch models.
    Propagates relevance from output logits back to input pixels.
    """

    def __init__(self, model, epsilon=1e-6):
        self.model = model
        self.epsilon = epsilon
        self.activations = {}

    def compute_rgb_relevance(self, rgb_tensor, depth_tensor, target_class=None):
        """Compute LRP relevance map for RGB input."""
        rgb_tensor = rgb_tensor.clone().requires_grad_(True)
        depth_tensor = depth_tensor.clone().detach()

        # Forward pass
        output = self.model(rgb_tensor, depth_tensor)
        if isinstance(output, tuple):
            output = output[0]

        # Get prediction
        probs = F.softmax(output, dim=1)
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        prediction = target_class
        confidence = probs[0, target_class].item()

        # Initialize relevance at output: R = one-hot for target class
        output_relevance = torch.zeros_like(output)
        output_relevance[0, target_class] = output[0, target_class]

        # Simple LRP: backprop scaled by activation
        # This is a simplified version; full LRP requires layer-by-layer
        self.model.zero_grad()
        output[0, target_class].backward()

        # Get gradient and compute relevance
        grad = rgb_tensor.grad.data

        # LRP-style: multiply gradient by input (relevance = input × gradient)
        # This approximates the ε-rule for the full path
        relevance = (rgb_tensor.detach() * grad).abs()

        # Aggregate across channels
        relevance_map = relevance.sum(dim=1).squeeze().cpu().numpy()

        # Normalize
        relevance_map = (relevance_map - relevance_map.min()) / (
            relevance_map.max() - relevance_map.min() + 1e-8
        )

        return relevance_map, prediction, confidence

    def compute_depth_relevance(self, rgb_tensor, depth_tensor, target_class=None):
        """Compute LRP relevance map for depth input."""
        rgb_tensor = rgb_tensor.clone().detach()
        depth_tensor = depth_tensor.clone().requires_grad_(True)

        output = self.model(rgb_tensor, depth_tensor)
        if isinstance(output, tuple):
            output = output[0]

        probs = F.softmax(output, dim=1)
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        prediction = target_class
        confidence = probs[0, target_class].item()

        self.model.zero_grad()
        output[0, target_class].backward()

        grad = depth_tensor.grad.data
        relevance = (depth_tensor.detach() * grad).abs()
        relevance_map = relevance.squeeze().cpu().numpy()
        relevance_map = (relevance_map - relevance_map.min()) / (
            relevance_map.max() - relevance_map.min() + 1e-8
        )

        return relevance_map, prediction, confidence

    def compute_both(self, rgb_tensor, depth_tensor, target_class=None):
        """Compute relevance for both modalities."""
        rgb_rel, pred, conf = self.compute_rgb_relevance(
            rgb_tensor, depth_tensor, target_class
        )
        depth_rel, _, _ = self.compute_depth_relevance(
            rgb_tensor, depth_tensor, target_class
        )
        return rgb_rel, depth_rel, pred, conf


class OcclusionSensitivity:
    """
    Occlusion-based saliency: mask patches and measure prediction change.
    Model-agnostic and highly reliable.
    """

    def __init__(self, model, patch_size=8, stride=4):
        self.model = model
        self.patch_size = patch_size
        self.stride = stride

    def compute_rgb_sensitivity(self, rgb_tensor, depth_tensor, target_class=None):
        """Compute occlusion sensitivity for RGB."""
        _, _, h, w = rgb_tensor.shape

        # Get baseline prediction
        with torch.no_grad():
            output = self.model(rgb_tensor, depth_tensor)
            if isinstance(output, tuple):
                output = output[0]
            probs = F.softmax(output, dim=1)
            if target_class is None:
                target_class = output.argmax(dim=1).item()
            baseline_score = probs[0, target_class].item()

        # Create sensitivity map
        sensitivity_map = np.zeros((h, w))

        # Iterate over patches
        positions = []
        for y in range(0, h - self.patch_size + 1, self.stride):
            for x in range(0, w - self.patch_size + 1, self.stride):
                positions.append((y, x))

        # Process in batches for efficiency
        batch_size = 32
        for i in range(0, len(positions), batch_size):
            batch_positions = positions[i : i + batch_size]
            batch_tensors = []

            for y, x in batch_positions:
                occluded = rgb_tensor.clone()
                occluded[:, :, y : y + self.patch_size, x : x + self.patch_size] = 0
                batch_tensors.append(occluded)

            batch_rgb = torch.cat(batch_tensors, dim=0)
            batch_depth = depth_tensor.repeat(len(batch_positions), 1, 1, 1)

            with torch.no_grad():
                batch_output = self.model(batch_rgb, batch_depth)
                if isinstance(batch_output, tuple):
                    batch_output = batch_output[0]
                batch_probs = F.softmax(batch_output, dim=1)
                batch_scores = batch_probs[:, target_class].cpu().numpy()

            for j, (y, x) in enumerate(batch_positions):
                importance = baseline_score - batch_scores[j]
                sensitivity_map[y : y + self.patch_size, x : x + self.patch_size] += (
                    importance
                )

        # Normalize
        sensitivity_map = np.maximum(sensitivity_map, 0)
        if sensitivity_map.max() > 0:
            sensitivity_map = sensitivity_map / sensitivity_map.max()

        return sensitivity_map, target_class, baseline_score

    def compute_depth_sensitivity(self, rgb_tensor, depth_tensor, target_class=None):
        """Compute occlusion sensitivity for depth."""
        _, _, h, w = depth_tensor.shape

        with torch.no_grad():
            output = self.model(rgb_tensor, depth_tensor)
            if isinstance(output, tuple):
                output = output[0]
            probs = F.softmax(output, dim=1)
            if target_class is None:
                target_class = output.argmax(dim=1).item()
            baseline_score = probs[0, target_class].item()

        sensitivity_map = np.zeros((h, w))

        positions = []
        for y in range(0, h - self.patch_size + 1, self.stride):
            for x in range(0, w - self.patch_size + 1, self.stride):
                positions.append((y, x))

        batch_size = 32
        for i in range(0, len(positions), batch_size):
            batch_positions = positions[i : i + batch_size]
            batch_tensors = []

            for y, x in batch_positions:
                occluded = depth_tensor.clone()
                occluded[:, :, y : y + self.patch_size, x : x + self.patch_size] = 0
                batch_tensors.append(occluded)

            batch_depth = torch.cat(batch_tensors, dim=0)
            batch_rgb = rgb_tensor.repeat(len(batch_positions), 1, 1, 1)

            with torch.no_grad():
                batch_output = self.model(batch_rgb, batch_depth)
                if isinstance(batch_output, tuple):
                    batch_output = batch_output[0]
                batch_probs = F.softmax(batch_output, dim=1)
                batch_scores = batch_probs[:, target_class].cpu().numpy()

            for j, (y, x) in enumerate(batch_positions):
                importance = baseline_score - batch_scores[j]
                sensitivity_map[y : y + self.patch_size, x : x + self.patch_size] += (
                    importance
                )

        sensitivity_map = np.maximum(sensitivity_map, 0)
        if sensitivity_map.max() > 0:
            sensitivity_map = sensitivity_map / sensitivity_map.max()

        return sensitivity_map, target_class, baseline_score


class SmoothGrad:
    """
    SmoothGrad: Average gradients over noisy inputs.
    Reduces noise in gradient-based visualizations.
    """

    def __init__(self, model, n_samples=25, noise_level=0.15):
        self.model = model
        self.n_samples = n_samples
        self.noise_level = noise_level

    def compute_rgb_smoothgrad(self, rgb_tensor, depth_tensor, target_class=None):
        """Compute SmoothGrad for RGB."""
        accumulated_grads = torch.zeros_like(rgb_tensor)

        # Get target class
        with torch.no_grad():
            output = self.model(rgb_tensor, depth_tensor)
            if isinstance(output, tuple):
                output = output[0]
            if target_class is None:
                target_class = output.argmax(dim=1).item()

        for _ in range(self.n_samples):
            # Add noise
            noise = torch.randn_like(rgb_tensor) * self.noise_level
            noisy_input = rgb_tensor + noise
            noisy_input = torch.clamp(noisy_input, 0, 1)
            noisy_input.requires_grad_(True)

            # Forward
            output = self.model(noisy_input, depth_tensor)
            if isinstance(output, tuple):
                output = output[0]

            # Backward
            self.model.zero_grad()
            output[0, target_class].backward()

            accumulated_grads += noisy_input.grad.data

        # Average
        avg_grads = accumulated_grads / self.n_samples
        saliency = avg_grads.abs().max(dim=1)[0].squeeze().cpu().numpy()
        saliency = (saliency - saliency.min()) / (
            saliency.max() - saliency.min() + 1e-8
        )

        return saliency, target_class

    def compute_depth_smoothgrad(self, rgb_tensor, depth_tensor, target_class=None):
        """Compute SmoothGrad for depth."""
        accumulated_grads = torch.zeros_like(depth_tensor)

        with torch.no_grad():
            output = self.model(rgb_tensor, depth_tensor)
            if isinstance(output, tuple):
                output = output[0]
            if target_class is None:
                target_class = output.argmax(dim=1).item()

        for _ in range(self.n_samples):
            noise = torch.randn_like(depth_tensor) * self.noise_level
            noisy_input = depth_tensor + noise
            noisy_input = torch.clamp(noisy_input, 0, 1)
            noisy_input.requires_grad_(True)

            output = self.model(rgb_tensor, noisy_input)
            if isinstance(output, tuple):
                output = output[0]

            self.model.zero_grad()
            output[0, target_class].backward()

            accumulated_grads += noisy_input.grad.data

        avg_grads = accumulated_grads / self.n_samples
        saliency = avg_grads.abs().squeeze().cpu().numpy()
        saliency = (saliency - saliency.min()) / (
            saliency.max() - saliency.min() + 1e-8
        )

        return saliency, target_class


warnings.filterwarnings("ignore")


class AttentionVisualizer:
    """Класс для визуализации внимания модели классификации жестов."""

    def __init__(
        self,
        model,
        gesture_classes: List[str],
        image_size: int = 128,
        device: Optional[torch.device] = None,
    ):
        """
        Инициализация визуализатора.

        Args:
            model: Готовая модель VSSD Dual Branch (rgb+depth)
            gesture_classes: Список классов жестов
            image_size: Размер входного изображения
            device: Устройство для вычислений (по умолчанию cuda/cpu)
        """
        self.model = model
        self.gesture_classes = gesture_classes
        self.image_size = image_size
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device).eval()

    def _load_image_pair_from_arrays(
        self, rgb_image: np.ndarray, depth_image: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, torch.Tensor, torch.Tensor]:
        """Преобразование numpy изображений в тензоры (аналог load_image_pair)."""
        # RGB
        print(rgb_image.shape)
        if len(rgb_image.shape) == 3:
            rgb_orig = cv2.resize(rgb_image, (self.image_size, self.image_size))
        else:
            rgb_orig = cv2.cvtColor(
                cv2.resize(rgb_image, (self.image_size, self.image_size)),
                cv2.COLOR_GRAY2RGB,
            )
        print(rgb_orig.shape)
        # rgb_orig = cv2.cvtColor(rgb_orig, cv2.COLOR_BGR2RGB)
        rgb_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        rgb_tensor: torch.Tensor = (
            rgb_transform(Image.fromarray(rgb_image)).unsqueeze(0).to(self.device)
        )

        # Depth
        if len(depth_image.shape) == 3:
            depth_orig = cv2.cvtColor(depth_image, cv2.COLOR_BGR2GRAY)
        else:
            depth_orig = depth_image.copy()
        depth_orig = cv2.resize(depth_orig, (self.image_size, self.image_size))
        depth_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )
        depth_tensor = (
            depth_transform(Image.fromarray(depth_image)).unsqueeze(0).to(self.device)
        )

        return rgb_orig, depth_orig, rgb_tensor, depth_tensor

    def _visualize_heatmap(
        self,
        image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.5,
        colormap: int = cv2.COLORMAP_JET,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Наложение heatmap на изображение (аналог visualize_heatmap)."""
        if image.max() > 1.0:
            image = image.astype(np.float32) / 255.0

        if heatmap.shape != image.shape[:2]:
            heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

        # Ensure heatmap is in [0, 1]
        heatmap = np.clip(heatmap, 0, 1)

        # Apply colormap
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), colormap)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB) / 255.0

        # Overlay
        return heatmap_colored * alpha + image * (1 - alpha), heatmap_colored

    def _get_prediction(
        self, rgb_tensor: torch.Tensor, depth_tensor: torch.Tensor
    ) -> Tuple[int, float, torch.Tensor]:
        """Получение предсказания модели."""
        with torch.no_grad():
            output = self.model(rgb_tensor, depth_tensor)
            if isinstance(output, tuple):
                output = output[0]
            probs = F.softmax(output, dim=1)
            confidence, prediction = torch.max(probs, dim=1)
            return prediction.item(), confidence.item(), probs[0]

    def generate_all_visualizations(
        self,
        rgb_image: np.ndarray,
        depth_image: np.ndarray,
        hand_id: int,
        save_path: Optional[str] = None,
    ):
        """
        Генерация всех визуализаций внимания для заданных изображений.

        Args:
            rgb_image: RGB изображение (numpy array)
            depth_image: Depth изображение (numpy array)
            save_path: Путь для сохранения финального изображения (опционально)
        """
        # Load и preprocess
        rgb_orig, depth_orig, rgb_tensor, depth_tensor = (
            self._load_image_pair_from_arrays(rgb_image, depth_image)
        )

        # Get prediction
        pred_idx, confidence, probs = self._get_prediction(rgb_tensor, depth_tensor)
        pred_label = self.gesture_classes[pred_idx]

        print(f"Prediction: {pred_label} (confidence: {confidence:.2%})")
        print("\nTop-5 predictions:")
        top5_probs, top5_indices = torch.topk(probs, 5)
        for prob, idx in zip(top5_probs, top5_indices):
            print(f"  {self.gesture_classes[idx]}: {prob:.2%}")

        # === LRP ===
        print("Computing LRP visualizations...")
        lrp = SimpleLRP(self.model)
        rgb_lrp, depth_lrp, pred, conf = lrp.compute_both(rgb_tensor, depth_tensor)

        rgb_normalized = rgb_orig.astype(np.float32) / 255.0

        depth_rgb = cv2.applyColorMap(depth_orig.astype(np.uint8), cv2.COLORMAP_JET)

        rgb_lrp_vis, rgb_lrp_hm_only = self._visualize_heatmap(
            rgb_normalized, rgb_lrp, alpha=0.5
        )
        depth_lrp_vis, depth_lrp_hm_only = self._visualize_heatmap(
            depth_rgb, depth_lrp, alpha=0.5
        )

        # === Occlusion Sensitivity ===
        print("Computing occlusion sensitivity (this may take 1-2 minutes)...")
        occ = OcclusionSensitivity(self.model, patch_size=16, stride=8)

        rgb_occ, pred_occ, conf_occ = occ.compute_rgb_sensitivity(
            rgb_tensor, depth_tensor
        )
        print("  RGB done")

        depth_occ, _, _ = occ.compute_depth_sensitivity(rgb_tensor, depth_tensor)
        print("  Depth done")

        rgb_occ_vis, rgb_occ_hm_only = self._visualize_heatmap(
            rgb_normalized, rgb_occ, alpha=0.5
        )
        depth_occ_vis, depth_occ_hm_only = self._visualize_heatmap(
            depth_rgb, depth_occ, alpha=0.5
        )

        # === SmoothGrad ===
        print("Computing SmoothGrad visualizations...")
        smooth = SmoothGrad(self.model, n_samples=50, noise_level=0.15)

        rgb_smooth, pred_smooth = smooth.compute_rgb_smoothgrad(
            rgb_tensor, depth_tensor
        )
        depth_smooth, _ = smooth.compute_depth_smoothgrad(rgb_tensor, depth_tensor)

        rgb_smooth_vis, rgb_smooth_hm_only = self._visualize_heatmap(
            rgb_normalized, rgb_smooth, alpha=0.5
        )
        depth_smooth_vis, depth_smooth_hm_only = self._visualize_heatmap(
            depth_rgb, depth_smooth, alpha=0.5
        )

        # === Comprehensive comparison ===

        # Side-by-side comparison
        fig = plt.figure(figsize=(16, 4.3))
        gs = fig.add_gridspec(
            2, 5, hspace=0.05, wspace=0.15, width_ratios=[0.5, 1, 2, 2, 2]
        )  # First col for labels

        # Empty first column for labels
        ax_label1 = fig.add_subplot(gs[0, 0])
        ax_label1.axis("off")
        ax_label1.text(
            0.5,
            0.5,
            "RGB\nизображение",
            fontsize=14,
            fontweight="bold",
            rotation=90,
            va="center",
            ha="center",
            transform=ax_label1.transAxes,
        )

        ax_label2 = fig.add_subplot(gs[1, 0])
        ax_label2.axis("off")
        ax_label2.text(
            0.5,
            0.5,
            "Карта\nглубины",
            fontsize=14,
            fontweight="bold",
            rotation=90,
            va="center",
            ha="center",
            transform=ax_label2.transAxes,
        )

        # Now your images start from column 1 instead of 0
        ax1 = fig.add_subplot(gs[0, 1])  # Changed from [0, 0] to [0, 1]
        ax1.imshow(rgb_orig)
        ax1.axis("off")

        ax2 = fig.add_subplot(gs[1, 1])  # Changed from [1, 0] to [1, 1]
        ax2.imshow(depth_rgb)
        ax2.axis("off")

        # Row 2: LRP - Now these will be properly sized
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.imshow(np.concatenate([rgb_lrp_hm_only, rgb_lrp_vis], axis=1))  # 256x128
        ax3.set_title("LRP", fontsize=12, fontweight="bold")
        ax3.axis("off")

        ax4 = fig.add_subplot(gs[1, 2])
        ax4.imshow(
            np.concatenate([depth_lrp_hm_only, depth_lrp_vis], axis=1)
        )  # 256x128
        ax4.axis("off")

        # Row 3: Occlusion
        ax5 = fig.add_subplot(gs[0, 3])
        ax5.imshow(np.concatenate([rgb_occ_hm_only, rgb_occ_vis], axis=1))  # 256x128
        ax5.set_title("Occlusion Sensitivity", fontsize=12, fontweight="bold")
        ax5.axis("off")

        ax6 = fig.add_subplot(gs[1, 3])
        ax6.imshow(np.concatenate([depth_occ_hm_only, depth_occ_vis], axis=1))
        ax6.axis("off")

        # Row 4: SmoothGrad
        ax7 = fig.add_subplot(gs[0, 4])
        ax7.imshow(
            np.concatenate([rgb_smooth_hm_only, rgb_smooth_vis], axis=1)
        )  # 256x128
        ax7.set_title("SmoothGrad", fontsize=12, fontweight="bold")
        ax7.axis("off")

        ax8 = fig.add_subplot(gs[1, 4])
        ax8.imshow(
            np.concatenate([depth_smooth_hm_only, depth_smooth_vis], axis=1)
        )  # 256x128
        ax8.axis("off")

        plt.tight_layout()

        fig.suptitle(
            f"Визуализация внимания руки #{hand_id}",
            fontsize=18,
            fontweight="bold",
            y=0.98,
        )

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved: {save_path}")

        # Convert to numpy array
        buf.seek(0)
        img_pil = Image.open(buf)
        img_array = np.array(img_pil)

        plt.close(fig)  # Close the figure to free memory

        return img_array

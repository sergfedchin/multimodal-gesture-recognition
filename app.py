"""
Gesture Recognition Demo App
Combines preprocessing (YOLOv13, PPD depth estimation) with VSSD-based classification
"""

import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import rcParams
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image
from torchvision import transforms

from attention_visualizer import AttentionVisualizer

warnings.filterwarnings("ignore")

# Add paths for external dependencies
sys.path.append("preprocess")
sys.path.append("multimodal-gesture-recognition")

# =============================================================================
# CONFIGURATION AND PATHS
# =============================================================================
# Paths to model checkpoints and configs
CONFIG = {
    # Hand detection (YOLOv10x trained on HaGRID)
    "yolov10_weights": "checkpoints/yolov10x_hands.pt",
    # Person detection (YOLOv13L)
    "yolov13_weights": "checkpoints/yolov13l.pt",
    "yolo_conf_threshold": 0.5,
    "yolo_iou_threshold": 0.45,
    # Pixel-Perfect Depth
    "ppd_checkpoint": "checkpoints/ppd.pth",
    "depth_anything_checkpoint": "checkpoints/depth_anything_v2_vitl.pth",
    "ppd_inference_size": (1024, 768),
    # Classification model
    "model_config": "config.toml",
    "model_checkpoint": "checkpoints/best_model.pt",
    # Processing parameters
    "min_hand_confidence": 0.3,
    "person_padding": 0.05,
    "hand_min_size": 32,
    "classification_image_size": 128,
    "display_image_size": (640, 640),  # Standard size for display and annotations
    "detection_font_scale": 1.2,  # Scale for detection labels
    "detection_line_thickness": 3,
    "hand_collage_size": (600, 600),  # Size for the hand collage panel
    "probability_chart_height": 750,  # Height for the probability chart
}

# =============================================================================
# ЗАГРУЗКА ПРИМЕРОВ ПРИ СТАРТЕ
# =============================================================================
EXAMPLES_PATH = Path("./example_images")
EXAMPLE_IMAGES_LIST = []  # Глобальный список для хранения загруженных изображений
example_paths = []

# Собираем все подходящие файлы
for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG", "*.bmp", "*.BMP"]:
    example_paths.extend(EXAMPLES_PATH.glob(ext))

example_paths = sorted(set(example_paths), key=lambda p: p.name)

print(f"📁 Поиск примеров в: {EXAMPLES_PATH.resolve()}")
print(f"🔍 Найдено файлов: {len(example_paths)}")

# Загружаем изображения в память
for p in example_paths:
    try:
        img = cv2.imread(str(p))
        if img is None:
            print(f"⚠️ Пропущен: {p.name} (не удалось загрузить)")
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        EXAMPLE_IMAGES_LIST.append((img_rgb, p.name))  # Сохраняем с меткой (имя файла)
        print(f"✅ Загружен: {p.name}")
    except Exception as e:
        print(f"❌ Ошибка загрузки {p.name}: {e}")

print(f"🖼️ Загружено {len(EXAMPLE_IMAGES_LIST)} примеров для галереи\n")

# Gesture classes (34 classes)
GESTURE_CLASSES = [
    "call",
    "dislike",
    "fist",
    "four",
    "grabbing",
    "grip",
    "hand_heart",
    "hand_heart2",
    "holy",
    "like",
    "little_finger",
    "middle_finger",
    "mute",
    "no_gesture",
    "ok",
    "one",
    "palm",
    "peace",
    "peace_inverted",
    "point",
    "rock",
    "stop",
    "stop_inverted",
    "take_picture",
    "three",
    "three2",
    "three3",
    "three_gun",
    "thumb_index",
    "timeout",
    "two_up",
    "two_up_inverted",
    "xsign",
    "thumb_index2",
]

GESTURE_TRANSLATION = {
    "call": "позвонить",
    "ok": "окей",
}

# =============================================================================
# MODEL LOADING FUNCTIONS
# =============================================================================
class ModelManager:
    """Singleton to manage loaded models and avoid repeated loading"""

    _instance = None
    models_loaded = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load_models(self):
        """Load all required models (called once)"""
        if self.models_loaded:
            return

        print("Loading models...")

        # Load YOLOv10 for hand detection
        print("  Loading YOLOv10x hand detector...")
        try:
            from ultralytics import YOLO

            self.yolov10_hand = YOLO(CONFIG["yolov10_weights"])
        except ImportError:
            raise ImportError("Install ultralytics: pip install ultralytics")

        # Load YOLOv13 for person detection
        print("  Loading YOLOv13L person detector...")
        self.yolov13_person = YOLO(CONFIG["yolov13_weights"])

        # Load Pixel-Perfect Depth
        print("  Loading Pixel-Perfect Depth...")
        self.ppd_model = self._load_ppd_model()

        # Load classification model
        print("  Loading gesture classification model...")
        self.classification_model, self.classification_config = (
            self._load_classification_model()
        )
        # Визуализатор внимания
        self.visualizer = AttentionVisualizer(
            self.classification_model, GESTURE_CLASSES, image_size=128
        )

        self._print_pipeline_parameter_counts()

        self.models_loaded = True
        print("All models loaded successfully!")

    @staticmethod
    def _count_model_parameters(model) -> int:
        """Count model parameters for torch modules and wrappers."""
        if model is None:
            return 0

        torch_model = model.model if hasattr(model, "model") else model
        if not isinstance(torch_model, torch.nn.Module):
            return 0

        return sum(parameter.numel() for parameter in torch_model.parameters())

    def _print_pipeline_parameter_counts(self) -> None:
        """Print total parameter counts for all pipeline models after loading."""
        parameter_counts = {
            "Person detection (YOLOv13L)": self._count_model_parameters(
                self.yolov13_person
            ),
            "Hands detection (YOLOv10x)": self._count_model_parameters(
                self.yolov10_hand
            ),
            "Depth estimation (PPD)": self._count_model_parameters(self.ppd_model),
            "Gesture classification (custom)": self._count_model_parameters(
                self.classification_model
            ),
        }

        total_parameters = sum(parameter_counts.values())

        print("\nPipeline parameter counts:")
        for model_name, count in parameter_counts.items():
            print(f"  - {model_name}: {count:,}")
        print(f"  - Total pipeline parameters: {total_parameters:,}\n")

    def _load_ppd_model(self):
        """Load Pixel-Perfect Depth model"""
        # Note: This requires the ppd module from the preprocessing pipeline
        # You may need to adjust imports based on your setup
        try:
            from preprocess.modules.depth_estimator import DepthEstimator

            # Create a minimal config for PPD
            ppd_config = {
                "pixel_perfect_depth": {
                    "ppd_checkpoint": CONFIG["ppd_checkpoint"],
                    "semantics_checkpoint": CONFIG["depth_anything_checkpoint"],
                    "inference_width": CONFIG["ppd_inference_size"][0],
                    "inference_height": CONFIG["ppd_inference_size"][1],
                    "sampling_steps": 4,
                    "batch_size": 1,
                    "device": "cuda" if torch.cuda.is_available() else "cpu",
                    "use_fp16": True,
                }
            }

            depth_estimator = DepthEstimator(ppd_config)
            return depth_estimator.get_model()
        except ImportError as e:
            print(f"Warning: Could not load PPD model: {e}")
            print("Depth estimation will be skipped")
            return None

    def _load_classification_model(self):
        """Load the trained gesture classification model"""
        try:
            # Import from the model training pipeline
            from multimodal_gesture_recognition.models import build_model
            from multimodal_gesture_recognition.utils import load_config

            # Load config
            config = load_config(CONFIG["model_config"])

            # Update config for inference
            config["model"]["image_size"] = CONFIG["classification_image_size"]
            config["hardware"]["device"] = (
                "cuda" if torch.cuda.is_available() else "cpu"
            )

            # Build model
            model = build_model(config)

            # Load checkpoint
            checkpoint = torch.load(
                CONFIG["model_checkpoint"], map_location="cpu", weights_only=False
            )
            model.load_state_dict(checkpoint["model_state_dict"])

            # Move to device
            device = torch.device(config["hardware"]["device"])
            model = model.to(device)
            model.eval()

            return model, config
        except ImportError as e:
            print(f"Warning: Could not load classification model: {e}")
            print("Using mock predictions for demonstration")
            return None, None


# =============================================================================
# PROCESSING PIPELINE
# =============================================================================
def detect_person_and_hands(image: np.ndarray) -> Tuple[Optional[List], List]:
    """
    Detect person and hands in the image.
    Returns: (person_bbox, hand_bboxes)
    - person_bbox: [x1, y1, x2, y2, confidence] in normalized coordinates (0-1) or None
    - hand_bboxes: list of [x1, y1, x2, y2, confidence] in normalized coordinates
    """
    img_height, img_width = image.shape[:2]
    # Detect person with YOLOv13L
    person_results = manager.yolov13_person(
        image,
        conf=CONFIG["yolo_conf_threshold"],
        iou=CONFIG["yolo_iou_threshold"],
        verbose=False,
    )

    person_bbox = None
    if len(person_results) > 0:
        boxes = person_results[0].boxes
        if boxes is not None and len(boxes) > 0:
            # Filter for person class (class 0 in COCO)
            person_boxes = boxes[boxes.cls == 0]
            if len(person_boxes) > 0:
                # Select person with largest area
                areas = (person_boxes.xyxy[:, 2] - person_boxes.xyxy[:, 0]) * (
                    person_boxes.xyxy[:, 3] - person_boxes.xyxy[:, 1]
                )
                best_idx = areas.argmax()
                bbox = person_boxes.xyxy[best_idx].cpu().numpy()
                conf = person_boxes.conf[best_idx].cpu().numpy()

                # Convert to normalized coordinates
                person_bbox = [
                    bbox[0] / img_width,
                    bbox[1] / img_height,
                    bbox[2] / img_width,
                    bbox[3] / img_height,
                    float(conf),
                ]

    # Detect hands with YOLOv10x
    hand_results = manager.yolov10_hand(
        image,
        conf=CONFIG["min_hand_confidence"],
        iou=CONFIG["yolo_iou_threshold"],
        verbose=False,
    )

    hand_bboxes = []
    if len(hand_results) > 0:
        boxes = hand_results[0].boxes
        if boxes is not None and len(boxes) > 0:
            # Sort by confidence and take top 2
            confidences = boxes.conf.cpu().numpy()
            sorted_indices = np.argsort(confidences)[::-1][:2]

            for idx in sorted_indices:
                bbox = boxes.xyxy[idx].cpu().numpy()
                conf = boxes.conf[idx].cpu().numpy()

                # Convert to normalized coordinates
                hand_bboxes.append(
                    [
                        bbox[0] / img_width,
                        bbox[1] / img_height,
                        bbox[2] / img_width,
                        bbox[3] / img_height,
                        float(conf),
                    ]
                )

    return person_bbox, hand_bboxes


def expand_person_bbox(
    person_bbox: List[float], hand_bboxes: List[List[float]]
) -> List[float]:
    """
    Expand person bbox to include all hand bboxes.
    Follows the same logic as in unified_processor.py
    """
    if not hand_bboxes:
        return person_bbox
    px1, py1, px2, py2, p_conf = person_bbox

    # Find extent of all hand bboxes
    min_x, min_y = px1, py1
    max_x, max_y = px2, py2

    for hand_bbox in hand_bboxes:
        hx1, hy1, hx2, hy2, _ = hand_bbox

        min_x = min(min_x, hx1)
        min_y = min(min_y, hy1)
        max_x = max(max_x, hx2)
        max_y = max(max_y, hy2)

    # Clamp to image bounds
    min_x = max(0.0, min_x)
    min_y = max(0.0, min_y)
    max_x = min(1.0, max_x)
    max_y = min(1.0, max_y)

    return [min_x, min_y, max_x, max_y, p_conf]  # Return with confidence


def estimate_depth(image: np.ndarray) -> Optional[np.ndarray]:
    """Estimate depth map using Pixel-Perfect Depth"""
    if manager.ppd_model is None:
        return None
    try:
        # Convert to BGR (OpenCV format)
        image_cv = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Run inference
        with torch.no_grad():
            depth, _ = manager.ppd_model.infer_image(image_cv)

        # Resize to original image size
        H, W = image.shape[:2]
        depth = torch.nn.functional.interpolate(
            depth, size=(H, W), mode="bilinear", align_corners=False
        )[0, 0]

        # Convert to numpy and normalize for visualization
        depth_np = depth.squeeze().cpu().numpy()
        depth_normalized = (depth_np - depth_np.min()) / (
            depth_np.max() - depth_np.min() + 1e-8
        )

        return (depth_normalized * 255).astype(np.uint8)

    except Exception as e:
        print(f"Depth estimation failed: {e}")
        return None


def crop_hand_from_image(
    image: np.ndarray, bbox: List[float], target_size: int = 128
) -> np.ndarray:
    """
    Crop hand region from image and resize to target size.
    bbox: [x1, y1, x2, y2] in normalized coordinates
    """
    H, W = image.shape[:2]
    # Convert normalized bbox to pixel coordinates
    x1 = int(bbox[0] * W)
    y1 = int(bbox[1] * H)
    x2 = int(bbox[2] * W)
    y2 = int(bbox[3] * H)

    # Ensure minimum size
    if (x2 - x1) < CONFIG["hand_min_size"] or (y2 - y1) < CONFIG["hand_min_size"]:
        # Expand to minimum size
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        half_size = CONFIG["hand_min_size"] // 2
        x1 = max(0, center_x - half_size)
        y1 = max(0, center_y - half_size)
        x2 = min(W, center_x + half_size)
        y2 = min(H, center_y + half_size)

    # Crop and resize
    cropped = image[y1:y2, x1:x2]
    if cropped.size == 0:
        # Fallback: return black image
        cropped = np.zeros((target_size, target_size, 3), dtype=np.uint8)

    # Resize
    cropped_resized = cv2.resize(cropped, (target_size, target_size))

    return cropped_resized


def classify_hand(hand_rgb: np.ndarray, hand_depth: Optional[np.ndarray] = None):
    """
    Classify a hand using the trained model.
    Returns: (predicted_class, confidence, all_probabilities)
    """
    if manager.classification_model is None:
        # Mock prediction for demonstration
        probs = np.random.rand(len(GESTURE_CLASSES))
        probs = probs / probs.sum()
        pred_idx = np.argmax(probs)
        return GESTURE_CLASSES[pred_idx], float(probs[pred_idx]), probs.tolist()
    # Preprocess image

    # RGB preprocessing
    rgb_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    rgb_tensor: torch.Tensor = rgb_transform(Image.fromarray(hand_rgb)).unsqueeze(0)

    # Depth preprocessing (if available)
    if hand_depth is not None and manager.classification_config["model"][
        "modality"
    ] in ["depth", "fusion"]:
        depth_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )
        depth_tensor = depth_transform(Image.fromarray(hand_depth)).unsqueeze(0)
    else:
        depth_tensor = None

    # Move to device
    device = torch.device(manager.classification_config["hardware"]["device"])
    rgb_tensor: torch.Tensor = rgb_tensor.to(device)
    if depth_tensor is not None:
        # print("Depth has been extracted!")
        depth_tensor: torch.Tensor = depth_tensor.to(device)
    # Inference
    with torch.no_grad():
        if manager.classification_config["model"]["modality"] == "rgb":
            logits = manager.classification_model(rgb_tensor)
        elif (
            manager.classification_config["model"]["modality"] == "depth"
            # and depth_tensor is not None
        ):
            logits = manager.classification_model(depth_tensor)
        elif (
            manager.classification_config["model"]["modality"] == "fusion"
            # and depth_tensor is not None
        ):
            logits = manager.classification_model(rgb_tensor, depth_tensor)
        else:
            # Fallback to RGB only
            logits = manager.classification_model(rgb_tensor)

    # Get probabilities
    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    pred_idx = np.argmax(probs)

    return GESTURE_CLASSES[pred_idx], float(probs[pred_idx]), probs.tolist()


def aggregate_predictions(hand_predictions: List[Tuple[str, float]]) -> str:
    """
    Aggregate predictions from multiple hands (paper-style).
    Simplified version of the logic in evaluator.py
    """
    if not hand_predictions:
        return "no_gesture"
    if len(hand_predictions) == 1:
        return hand_predictions[0][0]

    # For two hands
    pred1, conf1 = hand_predictions[0]
    pred2, conf2 = hand_predictions[1]

    # Both thumb_index -> thumb_index2 (special gesture)
    if pred1 == "thumb_index" and pred2 == "thumb_index":
        return "thumb_index2"

    # One no_gesture -> return the other
    if pred1 == "no_gesture":
        return pred2
    if pred2 == "no_gesture":
        return pred1

    # Different gestures -> higher confidence
    return pred1 if conf1 >= conf2 else pred2


# =============================================================================
# GRADIO INTERFACE
# =============================================================================


def scale_image_for_display(image: np.ndarray, target_size: Tuple[int, int]):
    """Scale an image while maintaining aspect ratio."""
    h, w = image.shape[:2]
    target_w, target_h = target_size

    # Calculate scaling factor to fit within target size
    scale_factor = min(target_w / w, target_h / h)

    new_w = int(w * scale_factor)
    new_h = int(h * scale_factor)

    # Resize image
    resized_img = cv2.resize(
        image, (new_w, new_h), interpolation=cv2.INTER_CUBIC
    )  # Changed interpolation

    # Create a canvas of target size and paste the resized image in the center
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    start_x = (target_w - new_w) // 2
    start_y = (target_h - new_h) // 2
    canvas[start_y : start_y + new_h, start_x : start_x + new_w] = resized_img

    return canvas, scale_factor, start_x, start_y


def draw_detections(
    image: np.ndarray, person_bbox: Optional[List], hand_bboxes: List
) -> np.ndarray:
    """Draw detection results on image with scaled font size."""
    # Scale image for consistent display and get scaling factors
    img_scaled, scale_factor, start_x, start_y = scale_image_for_display(
        image.copy(), CONFIG["display_image_size"]
    )

    # Adjust font scale based on the image scaling factor to keep text size visually consistent
    # Base font scale is around 1.0 for a 640x640 image. We adjust it proportionally.
    adjusted_font_scale = (
        max(scale_factor, 0.5) * CONFIG["detection_font_scale"]
    )  # Apply base scale
    thickness = max(int(CONFIG["detection_line_thickness"] * scale_factor), 1)
    line_type = cv2.LINE_AA

    # # Calculate dimensions of the scaled canvas portion where the actual image resides
    # h_actual = img_scaled.shape[0]
    # w_actual = img_scaled.shape[1]

    if person_bbox:
        x1, y1, x2, y2, conf = person_bbox
        # Convert normalized coords to pixel coords on the *original* image first
        orig_img_h, orig_img_w = image.shape[:2]
        x1_orig = int(x1 * orig_img_w)
        y1_orig = int(y1 * orig_img_h)
        x2_orig = int(x2 * orig_img_w)
        y2_orig = int(y2 * orig_img_h)

        # Then map these to the *scaled* canvas coordinates
        # Since the image was centered, we need to account for the offset (start_x, start_y)
        # and the scale factor applied during resizing.
        # The original pixels were mapped like: orig_x -> (orig_x * scale_factor) + start_x
        # So for the scaled canvas, the bbox becomes:
        x1_canvas = int(x1_orig * scale_factor) + start_x
        y1_canvas = int(y1_orig * scale_factor) + start_y
        x2_canvas = int(x2_orig * scale_factor) + start_x
        y2_canvas = int(y2_orig * scale_factor) + start_y

        cv2.rectangle(
            img_scaled,
            (x1_canvas, y1_canvas),
            (x2_canvas, y2_canvas),
            (0, 255, 0),
            thickness,
        )
        cv2.putText(
            img_scaled,
            f"Person ({conf:.2f})",
            (x1_canvas, y1_canvas - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            adjusted_font_scale,
            (0, 255, 0),
            thickness,
            line_type,
        )

    for i, bbox in enumerate(hand_bboxes):
        x1, y1, x2, y2, conf = bbox
        # Same mapping for hand bbox
        orig_img_h, orig_img_w = image.shape[:2]
        x1_orig = int(x1 * orig_img_w)
        y1_orig = int(y1 * orig_img_h)
        x2_orig = int(x2 * orig_img_w)
        y2_orig = int(y2 * orig_img_h)

        x1_canvas = int(x1_orig * scale_factor) + start_x
        y1_canvas = int(y1_orig * scale_factor) + start_y
        x2_canvas = int(x2_orig * scale_factor) + start_x
        y2_canvas = int(y2_orig * scale_factor) + start_y

        cv2.rectangle(
            img_scaled,
            (x1_canvas, y1_canvas),
            (x2_canvas, y2_canvas),
            (255, 0, 0),
            thickness,
        )
        cv2.putText(
            img_scaled,
            f"Hand {i + 1}: {conf:.2f}",
            (x1_canvas, y1_canvas - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            adjusted_font_scale,
            (255, 0, 0),
            thickness,
            line_type,
        )

    return img_scaled


def create_probability_chart_for_hands(
    hand_predictions_data: List[Dict],
) -> List[np.ndarray]:
    """Create a vertical bar chart for gesture probabilities for each hand."""
    charts = []
    for hand_data in hand_predictions_data:
        all_probs = hand_data.get("all_probabilities", [])
        if not all_probs:
            # Return an empty image if no probabilities
            charts.append(np.zeros((400, 600, 3), dtype=np.uint8))
            continue

        # Separate classes and probabilities
        classes = [item[0] for item in all_probs]
        probs = [item[1] for item in all_probs]

        # Sort by probability descending
        sorted_indices = np.argsort(probs)[::-1]  # DESCENDING order
        sorted_classes = [classes[i] for i in sorted_indices]
        sorted_probs = [probs[i] for i in sorted_indices]

        # Take top 10 predictions for better visualization
        top_n = 10
        top_classes = sorted_classes[:top_n]
        top_probs = sorted_probs[:top_n]

        # Set modern matplotlib style
        plt.style.use("dark_background")
        rcParams.update(
            {
                "axes.facecolor": "#111111",
                "figure.facecolor": "#0a0a0a",
                "axes.edgecolor": "#333333",
                "axes.labelcolor": "white",
                "text.color": "white",
                "xtick.color": "white",
                "ytick.color": "white",
            }
        )

        fig, ax = plt.subplots(figsize=(20, 9.13))

        # Create gradient colors
        colors = plt.cm.viridis(np.linspace(0.8, 0.2, top_n))

        bars = ax.bar(
            top_classes, top_probs, color=colors, edgecolor=None, linewidth=0
        )

        ax.grid(False)
        ax.set_xlabel("Классы жестов", fontsize=22, labelpad=10)
        ax.set_ylabel("Вероятность", fontsize=22, labelpad=10)
        ax.set_title(
            f"Распределение вероятностей топ-{top_n} классов руки #{hand_data['hand']}",
            fontsize=26,
            pad=35,
            fontweight="bold",
        )
        ax.set_ylim(0, 1.1)

        # Add value labels on top of bars
        for bar, prob in zip(bars, top_probs):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.02,
                f"{prob:.3f}",
                ha="center",
                va="bottom",
                fontsize=15,
            )

        # Rotate x labels for better readability
        plt.xticks(rotation=45, ha="right", fontsize=15)

        plt.tight_layout()

        # Convert to numpy array
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        buf = canvas.buffer_rgba()
        chart_img = np.frombuffer(buf, dtype=np.uint8).reshape(
            int(canvas.get_renderer().height), int(canvas.get_renderer().width), 4
        )
        chart_img = cv2.cvtColor(chart_img, cv2.COLOR_RGBA2RGB)

        plt.close(fig)

        charts.append(chart_img)

    return charts


def process_image(
    input_image: np.ndarray,
    visualize_attention: bool = True,  # Новая опция
    progress=gr.Progress(),
) -> Dict:
    """
    Main processing pipeline for a single image.
    Returns dictionary with all outputs for Gradio.
    """
    progress(0, desc="Загрузка моделей...")
    # Ensure models are loaded
    manager.load_models()

    # Convert to RGB if needed
    if len(input_image.shape) == 2:
        input_image = cv2.cvtColor(input_image, cv2.COLOR_GRAY2RGB)
    elif input_image.shape[2] == 4:
        input_image = cv2.cvtColor(input_image, cv2.COLOR_RGBA2RGB)

    results = {
        "original_image": input_image.copy(),
        "detections_image": None,
        "depth_map": None,
        "hand_images": [],
        "hand_depth_maps": [],
        "hand_predictions": [],  # Store detailed hand data including probabilities
        "attention_visualizations": [],
        "final_prediction": " ",
        "processing_time": 0,
        "no_hands_found": False,  # Флаг, что руки не найдены
        "visualize_attention": visualize_attention,  # Сохраняем настройку визуализации
    }

    start_time = time.time()

    # Stage 1: Detection
    progress(0.1, desc="Обнаружение человека и рук...")
    person_bbox, hand_bboxes = detect_person_and_hands(input_image)

    # Проверка: если руки не найдены, устанавливаем флаг и завершаем обработку
    if not hand_bboxes:
        results["no_hands_found"] = True
        results["final_prediction"] = (
            "Руки не обнаружены. Загрузите изображение с видимыми руками."
        )

        # Рисуем только детекцию человека (если есть)
        results["detections_image"] = draw_detections(input_image, person_bbox, [])

        # Создаем пустые визуализации
        results["attention_visualizations"] = [np.zeros((400, 600, 3), dtype=np.uint8)]

        results["processing_time"] = time.time() - start_time
        return results

    # Если руки найдены, продолжаем обычную обработку
    # Draw detections on scaled image
    results["detections_image"] = draw_detections(input_image, person_bbox, hand_bboxes)

    if not person_bbox:
        results["final_prediction"] = "no_gesture (человек не обнаружен)"
        results["processing_time"] = time.time() - start_time
        return results

    # Stage 2: Expand person bbox and crop
    progress(0.2, desc="Обрезка области человека...")
    expanded_bbox = expand_person_bbox(
        person_bbox,  # Now includes confidence [x1, y1, x2, y2, conf]
        hand_bboxes,
    )

    # Crop person region
    H, W = input_image.shape[:2]
    px1, py1, px2, py2, p_conf = expanded_bbox  # Unpack with confidence
    px1, py1, px2, py2 = (
        int(px1 * W),
        int(py1 * H),
        int(px2 * W),  # Corrected from px1 + pw
        int(py2 * H),  # Corrected from py1 + ph
    )
    person_crop = input_image[py1:py2, px1:px2]

    if person_crop.size == 0:
        results["final_prediction"] = "Ошибка: Обрезка человека не удалась"
        return results

    # Stage 3: Depth estimation
    progress(0.25, desc="Оценка глубины...")
    depth_map = estimate_depth(person_crop)
    if depth_map is not None:
        results["depth_map"] = depth_map

    # Stage 4: Crop and process hands
    progress(0.35, desc="Обработка обрезанных рук...")
    hand_predictions = []

    for i, hand_bbox in enumerate(hand_bboxes[:2]):  # Max 2 hands
        # Adjust hand bbox relative to person crop
        hx1, hy1, hx2, hy2, conf = hand_bbox
        # Correctly calculate relative coordinates
        hx1_rel = (hx1 * W - px1) / (px2 - px1)
        hy1_rel = (hy1 * H - py1) / (py2 - py1)
        hx2_rel = (hx2 * W - px1) / (px2 - px1)
        hy2_rel = (hy2 * H - py1) / (py2 - py1)

        # Crop hand from RGB
        hand_rgb = crop_hand_from_image(
            person_crop, [hx1_rel, hy1_rel, hx2_rel, hy2_rel]
        )
        results["hand_images"].append(hand_rgb)

        # Crop hand from depth (if available)
        if depth_map is not None:
            hand_depth = crop_hand_from_image(
                cv2.cvtColor(depth_map, cv2.COLOR_GRAY2RGB)
                if len(depth_map.shape) == 2
                else depth_map,
                [hx1_rel, hy1_rel, hx2_rel, hy2_rel],
            )
            results["hand_depth_maps"].append(
                cv2.cvtColor(hand_depth, cv2.COLOR_RGB2GRAY)
            )
        else:
            results["hand_depth_maps"].append(None)

        # Classify hand
        progress(0.4 + 0.2 * i, desc=f"Классификация руки {i + 1}...")
        pred_class, confidence, all_probs = classify_hand(
            hand_rgb, results["hand_depth_maps"][-1] if depth_map is not None else None
        )

        # Визуализация внимания только если включена опция
        if visualize_attention:
            progress(
                0.4 + 0.2 * i + 0.1,
                desc=f"Создание визуализаций внимания для руки {i + 1}...",
            )
            attention_visualization = manager.visualizer.generate_all_visualizations(
                hand_rgb, results["hand_depth_maps"][-1], i + 1
            )
            results["attention_visualizations"].append(attention_visualization)
        else:
            # Если визуализация отключена, добавляем заглушку
            placeholder = np.zeros((400, 600, 3), dtype=np.uint8)
            cv2.putText(
                placeholder,
                "Attention visualization disabled",
                (50, 200),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )
            results["attention_visualizations"].append(placeholder)

        hand_predictions.append((pred_class, confidence))
        # Store detailed data for this hand, including probabilities
        results["hand_predictions"].append(
            {
                "hand": i + 1,
                "class": pred_class,
                "confidence": confidence,
                "all_probabilities": list(zip(GESTURE_CLASSES, all_probs)),
            }
        )

    # Stage 5: Aggregate predictions
    progress(0.9, desc="Агрегация предсказаний...")
    final_prediction = aggregate_predictions(hand_predictions)
    results["final_prediction"] = final_prediction

    results["processing_time"] = time.time() - start_time

    return results


def create_visualization(results: Dict) -> Tuple:
    """Create visualization outputs for Gradio"""
    # Проверяем, были ли найдены руки
    if results.get("no_hands_found", False):
        # Возвращаем только детекцию и сообщение об ошибке
        detection_img = results.get("detections_image", results.get("original_image"))

        # Пустая карта глубины
        depth_display = np.zeros(
            (CONFIG["display_image_size"][1], CONFIG["display_image_size"][0], 3),
            dtype=np.uint8,
        )

        # Пустая визуализация внимания (если включена)
        visualize_attention = results.get("visualize_attention", True)
        if visualize_attention:
            attention_visualization = np.zeros((600, 600, 3), dtype=np.uint8)
            attention_height = 600
        else:
            attention_visualization = None
            attention_height = 0

        # Пустая диаграмма вероятностей
        probability_chart_display = np.zeros(
            (600, 800, 3),
            dtype=np.uint8,
        )

        # Сообщение об ошибке
        prediction_text = results.get("final_prediction", "Руки не обнаружены")

        return (
            detection_img,
            depth_display,
            attention_visualization,
            probability_chart_display,
            prediction_text,
            gr.update(height=600),  # Высота для диаграммы вероятностей
            gr.update(height=attention_height),  # Высота для визуализации внимания
        )

    # Если руки найдены, продолжаем обычную визуализацию
    detection_img = results.get("detections_image", results.get("original_image"))

    # Depth map (if available)
    depth_img = results.get("depth_map")
    if depth_img is not None:
        depth_display, _, _, _ = scale_image_for_display(
            cv2.applyColorMap(depth_img, cv2.COLORMAP_JET), CONFIG["display_image_size"]
        )
    else:
        depth_display = np.zeros(
            (CONFIG["display_image_size"][1], CONFIG["display_image_size"][0], 3),
            dtype=np.uint8,
        )

    hand_predictions_data = results.get("hand_predictions", [])

    # Generate probability chart(s) for each hand
    probability_charts = create_probability_chart_for_hands(hand_predictions_data)

    # Calculate dynamic height based on number of hands для диаграммы вероятностей
    if not probability_charts:
        probability_chart_display = np.zeros((600, 800, 3), dtype=np.uint8)
        probability_height = 600
    elif len(probability_charts) == 1:
        probability_chart_display = probability_charts[0]
        probability_height = 600
    else:
        # Stack charts vertically for two hands
        probability_chart_display = np.vstack(probability_charts)
        probability_height = 1200  # Double height for two hands

    # Calculate dynamic height for attention visualization
    visualize_attention = results.get("visualize_attention", True)
    attention_visualizations = results["attention_visualizations"]
    
    if not visualize_attention:
        attention_visualization = None
        attention_height = 0
    elif len(attention_visualizations) > 1:
        # Stack attention visualizations vertically for two hands
        attention_visualization = np.vstack(attention_visualizations)
        attention_height = 1200  # Double height for two hands
    else:
        attention_visualization = attention_visualizations[0]
        attention_height = 600

    # Prediction text
    prediction_text = (
        f"# Итоговое предсказание: {GESTURE_TRANSLATION.get((g := results.get('final_prediction', 'неизвестен')), g)}\n\n"
        # f"Время анализа: {results.get('processing_time', 0):.2f}с\n\n"
    )

    # Add hand predictions summary
    hand_predictions = results.get("hand_predictions", [])
    for hp in hand_predictions:
        prediction_text += (
            f"### Рука {hp['hand']}: {hp['class']} (уверенность: {hp['confidence']:.3f})\n"
        )

    return (
        detection_img,
        depth_display,
        attention_visualization,
        probability_chart_display,
        prediction_text,
        gr.update(height=probability_height),  # Динамическая высота для диаграммы вероятностей
        gr.update(height=attention_height),    # Динамическая высота для визуализации внимания
    )


# =============================================================================
# GRADIO APP DEFINITION
# =============================================================================
# Initialize model manager
manager = ModelManager()

# Create Gradio interface
with gr.Blocks(
    title="Мультимодальное распознавание жестов", theme=gr.themes.Soft()
) as demo:
    gr.Markdown("# Мультимодальное распознавание жестов")
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(
                label="Исходное изображение",
                type="numpy",
                height=600,
            )

            visualize_attention_checkbox = gr.Checkbox(
                label="Визуализировать внимание",
                value=True,
                info="Это может занять несколько минут.",
            )

            process_btn = gr.Button(
                "🚀 Проанализировать изображение", variant="primary"
            )
            clear_btn = gr.Button("🔄 Очистить")

        with gr.Column():
            detection_output = gr.Image(
                label="Детекции (зеленым: человек, красным: руки)", height=600
            )
        with gr.Column():
            depth_output = gr.Image(label="Карта глубины", height=600)

    with gr.Row():
        with gr.Column(scale=2.5):
            probability_chart_output = gr.Image(
                label="Распределение вероятностей",
                height=600,  # Начальная высота
            )
        with gr.Column(scale=1):
            prediction_output = gr.Markdown(
                label="Предсказания",
                # lines=10,
                # max_lines=20
            )

    # Компонент визуализации внимания
    attention_output = gr.Image(
        label="Визуализация внимания модели",
        height=400,  # Начальная высота
        visible=True,
    )

    # Измененная функция, которая возвращает результаты и обновления для компонентов
    def process_and_display(image, visualize_attention):
        if image is None:
            # Возвращаем None для изображений, пустую строку для текста, и update для видимости
            return (
                None,
                None,
                gr.update(visible=visualize_attention, value=None, height=0),
                None,
                "",
                gr.update(height=600),
                gr.update(height=0),
            )

        results = process_image(image, visualize_attention)
        (
            detection_img,
            depth_display,
            attention_visualization,
            probability_chart_display,
            prediction_text,
            probability_height_update,
            attention_height_update,
        ) = create_visualization(results)

        # Определяем видимость компонента внимания
        attention_visible = visualize_attention and attention_visualization is not None
        
        return (
            detection_img,
            depth_display,
            gr.update(
                value=attention_visualization, 
                visible=attention_visible,
                height=attention_height_update["height"] if attention_visible else 0
            ),
            probability_chart_display,
            prediction_text,
            probability_height_update,
            attention_height_update,
        )

    process_btn.click(
        fn=process_and_display,
        inputs=[input_image, visualize_attention_checkbox],
        outputs=[
            detection_output,
            depth_output,
            attention_output,
            probability_chart_output,
            prediction_output,
            probability_chart_output,  # Обновляем высоту диаграммы вероятностей
            attention_output,          # Обновляем высоту визуализации внимания
        ],
        show_progress='hidden'
    )

    def clear_all():
        return (
            None,
            None,
            None,
            gr.update(visible=False, value=None, height=0),
            None,
            "",
            gr.update(height=600),
            gr.update(height=0),
        )

    clear_btn.click(
        fn=clear_all,
        inputs=[],
        outputs=[
            input_image,
            detection_output,
            depth_output,
            attention_output,
            probability_chart_output,
            prediction_output,
            probability_chart_output,  # Обновляем высоту диаграммы вероятностей
            attention_output,          # Обновляем высоту визуализации внимания
        ],
    )

    # Examples
    # Галерея примеров (после объявления всех основных компонентов)
    gallery = gr.Gallery(
        value=EXAMPLE_IMAGES_LIST,  # Список кортежей (изображение, метка)
        label="Примеры для анализа",
        columns=6,
        rows=2.5,
        height=500,
        preview=False,
        allow_preview=False,
        show_label=True,
    )

    # Обработчик клика по галерее
    def load_from_gallery(evt: gr.SelectData):
        """Загружает выбранное изображение из предзагруженного списка"""
        index = evt.index  # Получаем индекс выбранного элемента
        if 0 <= index < len(EXAMPLE_IMAGES_LIST):
            img_array, _ = EXAMPLE_IMAGES_LIST[index]
            return img_array
        return None

    gallery.select(
        fn=load_from_gallery,
        inputs=None,
        outputs=[input_image],
    )

# =============================================================================
# LAUNCH SCRIPT
# =============================================================================
if __name__ == "__main__":
    # Parse command line arguments
    import argparse

    parser = argparse.ArgumentParser(description="Gesture Recognition Demo")
    parser.add_argument(
        "--share", action="store_true", help="Create a public share link"
    )
    parser.add_argument(
        "--server-name", type=str, default="127.0.0.1", help="Server address"
    )
    parser.add_argument("--server-port", type=int, default=7860, help="Server port")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    print("=" * 60)
    print("Gesture Recognition Demo")
    print("=" * 60)
    print("\nStarting application...")
    print(f"Server: http://{args.server_name}:{args.server_port}")
    print("Press Ctrl+C to stop\n")

    # Launch Gradio app
    demo.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
        debug=args.debug,
    )

"""
Gesture Recognition Demo App
Combines preprocessing (YOLOv13, PPD depth estimation) with VSSD-based classification
"""

import sys
import time
import warnings
from typing import Dict, List, Optional, Tuple

import cv2
import gradio as gr
import numpy as np
import torch

warnings.filterwarnings("ignore")

# Add paths for external dependencies
sys.path.append("preprocess")
sys.path.append("multimodal-gesture-recognition")

# ============================================================================
# CONFIGURATION AND PATHS
# ============================================================================

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
    "model_config": "multimodal_gesture_recognition/outputs/vssd_micro_small_cross_attention_fusion/config_vssd_micro_small_cross_attention_fusion.toml",
    "model_checkpoint": "checkpoints/best_model.pt",
    # Processing parameters
    "min_hand_confidence": 0.3,
    "person_padding": 0.05,
    "hand_min_size": 32,
    "classification_image_size": 128,
}

# Gesture classes (33 classes, excluding no_gesture)
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
    "thumb_index2",
    "timeout",
    "two_up",
    "two_up_inverted",
    "xsign",
]

# ============================================================================
# MODEL LOADING FUNCTIONS
# ============================================================================


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

        self.models_loaded = True
        print("All models loaded successfully!")

    def _load_ppd_model(self):
        """Load Pixel-Perfect Depth model"""
        # Note: This requires the ppd module from the preprocessing pipeline
        # You may need to adjust imports based on your setup
        try:
            from modules.config_loader import ConfigLoader
            from modules.depth_estimator import DepthEstimator

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
            import toml

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


# ============================================================================
# PROCESSING PIPELINE
# ============================================================================


def detect_person_and_hands(image: np.ndarray) -> Tuple[Optional[List], List]:
    """
    Detect person and hands in the image.
    Returns: (person_bbox, hand_bboxes)
    - person_bbox: [x1, y1, x2, y2] in normalized coordinates (0-1) or None
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

                # Convert to normalized coordinates
                person_bbox = [
                    bbox[0] / img_width,
                    bbox[1] / img_height,
                    bbox[2] / img_width,
                    bbox[3] / img_height,
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

    px1, py1, px2, py2 = person_bbox

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

    return [min_x, min_y, max_x - min_x, max_y - min_y]


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
    from PIL import Image
    from torchvision import transforms

    # RGB preprocessing
    rgb_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    rgb_tensor = rgb_transform(Image.fromarray(hand_rgb)).unsqueeze(0)

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
    rgb_tensor = rgb_tensor.to(device)
    if depth_tensor is not None:
        print("Depth has been extracted!")
        depth_tensor = depth_tensor.to(device)    # Inference
    with torch.no_grad():
        print(f"Modality: {manager.classification_config['model']['modality']}")
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


# ============================================================================
# GRADIO INTERFACE
# ============================================================================


def draw_detections(
    image: np.ndarray, person_bbox: Optional[List], hand_bboxes: List
) -> np.ndarray:
    """Draw detection results on image"""
    img = image.copy()
    H, W = img.shape[:2]

    # Draw person bbox (green)
    if person_bbox:
        x1, y1, w, h = person_bbox
        x1, y1, x2, y2 = int(x1 * W), int(y1 * H), int((x1 + w) * W), int((y1 + h) * H)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(
            img, "Person", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2
        )

    # Draw hand bboxes (red)
    for i, bbox in enumerate(hand_bboxes):
        x1, y1, x2, y2, conf = bbox
        x1, y1, x2, y2 = int(x1 * W), int(y1 * H), int(x2 * W), int(y2 * H)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(
            img,
            f"Hand {i + 1}: {conf:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2,
        )

    return img


def process_image(input_image: np.ndarray, progress=gr.Progress()) -> Dict:
    """
    Main processing pipeline for a single image.
    Returns dictionary with all outputs for Gradio.
    """
    progress(0, desc="Loading models...")

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
        "hand_predictions": [],
        "final_prediction": "",
        "all_probabilities": [],
        "processing_time": 0,
    }

    start_time = time.time()

    # Stage 1: Detection
    progress(0.1, desc="Detecting person and hands...")
    person_bbox, hand_bboxes = detect_person_and_hands(input_image)

    # Draw detections
    results["detections_image"] = draw_detections(input_image, person_bbox, hand_bboxes)

    if not person_bbox or not hand_bboxes:
        results["final_prediction"] = "no_gesture (no person/hands detected)"
        results["processing_time"] = time.time() - start_time
        return results

    # Stage 2: Expand person bbox and crop
    progress(0.3, desc="Cropping person region...")
    expanded_bbox = expand_person_bbox(
        [
            person_bbox[0],
            person_bbox[1],
            person_bbox[0] + person_bbox[2],
            person_bbox[1] + person_bbox[3],
        ],
        hand_bboxes,
    )

    # Crop person region
    H, W = input_image.shape[:2]
    px1, py1, pw, ph = expanded_bbox
    px1, py1, px2, py2 = (
        int(px1 * W),
        int(py1 * H),
        int((px1 + pw) * W),
        int((py1 + ph) * H),
    )
    person_crop = input_image[py1:py2, px1:px2]

    if person_crop.size == 0:
        results["final_prediction"] = "Error: Person crop failed"
        return results

    # Stage 3: Depth estimation
    progress(0.5, desc="Estimating depth...")
    depth_map = estimate_depth(person_crop)
    if depth_map is not None:
        results["depth_map"] = depth_map

    # Stage 4: Crop and process hands
    progress(0.7, desc="Processing hand crops...")
    hand_predictions = []

    for i, hand_bbox in enumerate(hand_bboxes[:2]):  # Max 2 hands
        # Adjust hand bbox relative to person crop
        hx1, hy1, hx2, hy2, conf = hand_bbox
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
        progress(0.7 + 0.1 * (i + 1), desc=f"Classifying hand {i + 1}...")
        pred_class, confidence, _ = classify_hand(
            hand_rgb, results["hand_depth_maps"][-1] if depth_map is not None else None
        )

        hand_predictions.append((pred_class, confidence))
        results["hand_predictions"].append(
            {"hand": i + 1, "class": pred_class, "confidence": confidence}
        )

    # Stage 5: Aggregate predictions
    progress(0.9, desc="Aggregating predictions...")
    final_prediction = aggregate_predictions(hand_predictions)
    results["final_prediction"] = final_prediction

    # Get full probability distribution (average of hand predictions)
    if manager.classification_model is not None and hand_predictions:
        # Get probabilities for the first hand (or average if we had a better method)
        _, _, all_probs = classify_hand(results["hand_images"][0], results["hand_depth_maps"][0])
        results["all_probabilities"] = list(zip(GESTURE_CLASSES, all_probs))

    results["processing_time"] = time.time() - start_time

    return results


def create_visualization(results: Dict) -> Tuple:
    """Create visualization outputs for Gradio"""
    # Original image with detections
    detection_img = results.get("detections_image", results.get("original_image"))

    # Depth map (if available)
    depth_img = results.get("depth_map")
    if depth_img is not None:
        depth_display = cv2.applyColorMap(depth_img, cv2.COLORMAP_VIRIDIS)
    else:
        depth_display = np.zeros((256, 256, 3), dtype=np.uint8)

    # Hand crops
    hand_images = results.get("hand_images", [])
    hand_depth_maps = results.get("hand_depth_maps", [])

    # Create a grid for hand images
    max_hands = 2
    hand_grid = []

    for i in range(max_hands):
        if i < len(hand_images):
            rgb_img = hand_images[i]
            depth_img = hand_depth_maps[i] if i < len(hand_depth_maps) else None

            # Create composite image
            if depth_img is not None:
                # Resize depth to match RGB
                depth_resized = cv2.resize(
                    depth_img, (rgb_img.shape[1], rgb_img.shape[0])
                )
                if len(depth_resized.shape) == 2:
                    depth_resized = cv2.cvtColor(depth_resized, cv2.COLOR_GRAY2RGB)

                # Stack horizontally
                composite = np.hstack([rgb_img, depth_resized])
            else:
                composite = rgb_img

            hand_grid.append(composite)
        else:
            # Placeholder
            hand_grid.append(np.zeros((128, 256, 3), dtype=np.uint8))

    # Stack hand images vertically
    if hand_grid:
        hand_display = np.vstack(hand_grid) if len(hand_grid) > 1 else hand_grid[0]
    else:
        hand_display = np.zeros((256, 256, 3), dtype=np.uint8)

    # Prediction text
    prediction_text = (
        f"Final Prediction: {results.get('final_prediction', 'Unknown')}\n"
    )
    prediction_text += f"Processing Time: {results.get('processing_time', 0):.2f}s\n\n"

    # Add hand predictions
    hand_predictions = results.get("hand_predictions", [])
    for hp in hand_predictions:
        prediction_text += (
            f"Hand {hp['hand']}: {hp['class']} (confidence: {hp['confidence']:.3f})\n"
        )

    # Add probability distribution for top 5 classes
    all_probs = results.get("all_probabilities", [])
    if all_probs:
        sorted_probs = sorted(all_probs, key=lambda x: x[1], reverse=True)[:5]
        prediction_text += "\nTop 5 Predictions:\n"
        for class_name, prob in sorted_probs:
            prediction_text += f"  {class_name}: {prob:.3f}\n"

    return detection_img, depth_display, hand_display, prediction_text


# ============================================================================
# GRADIO APP DEFINITION
# ============================================================================

# Initialize model manager
manager = ModelManager()

# Create Gradio interface
with gr.Blocks(title="Gesture Recognition Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎯 Hand Gesture Recognition Demo
    
    This app detects hands in images/video, estimates depth maps, and classifies gestures using a multi-modal VSSD model.
    
    **How to use:**
    1. Upload an image or use your webcam
    2. The system will detect person and hands using YOLOv13 and YOLOv10
    3. Depth maps are estimated using Pixel-Perfect Depth
    4. Hand crops are classified using a trained VSSD model
    5. Final prediction is aggregated from multiple hands
    """)

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(
                label="Input Image",
                sources=["upload", "webcam"],
                type="numpy",
                height=400,
            )

            process_btn = gr.Button("🚀 Process Image", variant="primary")
            clear_btn = gr.Button("🔄 Clear")

        with gr.Column():
            detection_output = gr.Image(
                label="Detections (Green: Person, Red: Hands)", height=400
            )

    with gr.Row():
        with gr.Column():
            depth_output = gr.Image(label="Depth Map", height=300)

        with gr.Column():
            hands_output = gr.Image(
                label="Hand Crops (Left: RGB, Right: Depth)", height=300
            )

    with gr.Row():
        prediction_output = gr.Textbox(label="Predictions", lines=10, max_lines=20)

    # Processing pipeline
    def process_and_display(image):
        if image is None:
            return None, None, None, "Please upload an image first"

        results = process_image(image)
        return create_visualization(results)

    process_btn.click(
        fn=process_and_display,
        inputs=[input_image],
        outputs=[detection_output, depth_output, hands_output, prediction_output],
    )

    # Clear function
    def clear_all():
        return None, None, None, ""

    clear_btn.click(
        fn=clear_all,
        inputs=[],
        outputs=[
            input_image,
            detection_output,
            depth_output,
            hands_output,
            prediction_output,
        ],
    )

    # Examples
    gr.Examples(
        examples=[
            ["example_images/gesture1.jpg"],
            ["example_images/gesture2.jpg"],
            ["example_images/gesture3.jpg"],
        ],
        inputs=[input_image],
        outputs=[detection_output, depth_output, hands_output, prediction_output],
        fn=process_and_display,
        cache_examples=False,
        label="Try these examples:",
    )

    gr.Markdown("""
    ---
    
    **Technical Details:**
    - Person Detection: YOLOv13L
    - Hand Detection: YOLOv10x (trained on HaGRID)
    - Depth Estimation: Pixel-Perfect Depth with Depth Anything V2
    - Classification: VSSD (Vision Mamba 2) with multi-modal fusion
    
    **Note:** First run will load all models (may take 1-2 minutes).
    """)

# ============================================================================
# LAUNCH SCRIPT
# ============================================================================

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

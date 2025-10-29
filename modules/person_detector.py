"""YOLOv13-based person detection module with batched inference."""


import gc
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
from tqdm import tqdm
import cv2


try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError("ultralytics package not found. Install with: pip install ultralytics")

try:
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    raise ImportError("PyTorch not found. Install with: pip install torch")


logger = logging.getLogger('gesture_processing')


class ImageDataset(Dataset):
    """Dataset for loading images with their paths and annotations."""

    def __init__(self, image_paths: List[Path], annotations: Dict[str, Any]):
        """
        Initialize dataset.

        Args:
            image_paths: List of image file paths
            annotations: Dictionary of annotations with hand bboxes
        """
        self.image_paths = image_paths
        self.annotations = annotations

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, Path, Dict[str, Any]]:
        """
        Load image and return with metadata.

        Args:
            idx: Index of image to load

        Returns:
            Tuple of (image_array, image_path, annotation)
        """
        img_path = self.image_paths[idx]
        img_stem = img_path.stem
        annotation = self.annotations.get(img_stem, {})

        # Load image using OpenCV (BGR format, same as YOLO expects)
        image = cv2.imread(str(img_path))

        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")

        return image, img_path, annotation


def collate_fn(batch: List[Tuple[np.ndarray, Path, Dict[str, Any]]]) -> Tuple[List[np.ndarray], List[Path], List[Dict[str, Any]]]:
    """
    Custom collate function to handle variable-sized images.

    Args:
        batch: List of (image, path, annotation) tuples

    Returns:
        Tuple of (images_list, paths_list, annotations_list)
    """
    images, paths, annotations = zip(*batch)
    return list(images), list(paths), list(annotations)


class PersonDetector:
    """Handles person detection using YOLOv13 with batched inference."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize YOLOv13 person detector.

        Args:
            config: Configuration dictionary
        """
        self.config = config['yolov13']
        self.show_progress = config['processing'].get('show_progress', True)
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        """Load YOLOv13 model."""
        model_path = self.config['model_weights']

        logger.info(f"Loading YOLOv13 model from: {model_path}")

        try:
            self.model = YOLO(str(model_path))
            logger.info("YOLOv13 model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load YOLOv13 model: {str(e)}")
            raise

    def detect_persons_batch(
        self,
        image_paths: List[Path],
        annotations: Dict[str, Any]
    ) -> Dict[str, List[float]]:
        """
        Detect persons in a batch of images using DataLoader.

        Args:
            image_paths: List of image file paths
            annotations: Dictionary of annotations with hand bboxes

        Returns:
            Dictionary mapping image stems to person bboxes [x, y, w, h] (normalized)
        """
        assert isinstance(self.model, YOLO)
        person_bboxes = {}
        batch_size = self.config['batch_size']

        logger.info(f"Processing {len(image_paths)} images with YOLOv13 (batch_size={batch_size})")

        # Create dataset and dataloader
        dataset = ImageDataset(image_paths, annotations)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.config.get('num_workers', 0),
            collate_fn=collate_fn,
            pin_memory=False
        )

        # Create progress bar
        if self.show_progress:
            pbar = tqdm(
                total=len(image_paths),
                desc="Detecting persons",
                unit="img",
                dynamic_ncols=True,
                smoothing=0
            )

        # Process batches
        for i, (batch_images, batch_paths, batch_annotations) in enumerate(dataloader):
            # Run inference on loaded images
            results = self.model.predict(
                source=batch_images,  # Pass numpy arrays directly
                conf=self.config['confidence_threshold'],
                iou=self.config['iou_threshold'],
                imgsz=self.config['image_size'],
                device=self.config['device'],
                verbose=False
            )

            # Process results
            for img_path, result, annotation in zip(batch_paths, results, batch_annotations):
                img_stem = img_path.stem
                bbox = self._process_single_detection(
                    result,
                    img_stem,
                    annotation
                )
                person_bboxes[img_stem] = bbox

            # Update progress bar
            if self.show_progress:
                pbar.update(len(batch_images))
            
            if i % 50 == 0:
                gc.collect()

        if self.show_progress:
            pbar.close()

        return person_bboxes

    def _process_single_detection(
        self,
        result: Any,
        img_stem: str,
        annotation: Dict[str, Any]
    ) -> List[float]:
        """
        Process detection result for a single image.

        Args:
            result: YOLO detection result
            img_stem: Image filename stem
            annotation: Annotation data for this image

        Returns:
            Person bbox [x, y, w, h] (normalized)
        """
        # Get image dimensions
        img_height, img_width = result.orig_shape

        # Filter for person class (class 0 in COCO)
        boxes = result.boxes
        person_mask = boxes.cls == 0

        if not person_mask.any():
            # No person detected - use full image
            logger.warning(f"No person detected in {img_stem}, using full image")
            return [0.0, 0.0, 1.0, 1.0]

        person_boxes = boxes[person_mask]

        if len(person_boxes) == 1:
            # Single person detected
            bbox_xyxy = person_boxes.xyxy[0].cpu().numpy()
            return self._xyxy_to_xywh_normalized(bbox_xyxy, img_width, img_height)

        else:
            # Multiple persons detected - select best match with hand bboxes
            logger.warning(f"Multiple persons detected in {img_stem}, selecting best match")
            return self._select_best_person_bbox(
                person_boxes,
                annotation.get('bboxes', []),
                img_width,
                img_height
            )

    def _select_best_person_bbox(
        self,
        person_boxes: Any,
        hand_bboxes: List[List[float]],
        img_width: int,
        img_height: int
    ) -> List[float]:
        """
        Select person bbox with maximum overlap with hand bboxes.

        Args:
            person_boxes: Detected person boxes
            hand_bboxes: Hand bounding boxes from annotations (normalized xywh)
            img_width: Image width
            img_height: Image height

        Returns:
            Best matching person bbox [x, y, w, h] (normalized)
        """
        if not hand_bboxes:
            # No hand bboxes - return largest person bbox
            areas = (person_boxes.xyxy[:, 2] - person_boxes.xyxy[:, 0]) * \
                    (person_boxes.xyxy[:, 3] - person_boxes.xyxy[:, 1])
            best_idx = areas.argmax()
            bbox_xyxy = person_boxes.xyxy[best_idx].cpu().numpy()
            return self._xyxy_to_xywh_normalized(bbox_xyxy, img_width, img_height)

        # Calculate overlap with each person bbox
        max_overlap = 0
        best_person_bbox = None

        for person_box in person_boxes:
            person_bbox_xyxy = person_box.xyxy[0].cpu().numpy()
            person_bbox_norm = self._xyxy_to_xywh_normalized(
                person_bbox_xyxy, img_width, img_height
            )

            total_overlap = 0
            for hand_bbox in hand_bboxes:
                overlap = self._calculate_overlap(person_bbox_norm, hand_bbox)
                total_overlap += overlap

            if total_overlap > max_overlap:
                max_overlap = total_overlap
                best_person_bbox = person_bbox_norm

        return best_person_bbox if best_person_bbox is not None else [0.0, 0.0, 1.0, 1.0]

    @staticmethod
    def _xyxy_to_xywh_normalized(
        bbox_xyxy: np.ndarray,
        img_width: int,
        img_height: int
    ) -> List[float]:
        """
        Convert bbox from xyxy to normalized xywh format.

        Args:
            bbox_xyxy: Bbox in [x1, y1, x2, y2] format
            img_width: Image width
            img_height: Image height

        Returns:
            Bbox in normalized [x, y, w, h] format
        """
        x1, y1, x2, y2 = bbox_xyxy
        x = x1 / img_width
        y = y1 / img_height
        w = (x2 - x1) / img_width
        h = (y2 - y1) / img_height
        return [float(x), float(y), float(w), float(h)]

    @staticmethod
    def _calculate_overlap(bbox1: List[float], bbox2: List[float]) -> float:
        """
        Calculate IoU overlap between two normalized bboxes.

        Args:
            bbox1: First bbox [x, y, w, h] (normalized)
            bbox2: Second bbox [x, y, w, h] (normalized)

        Returns:
            IoU overlap ratio
        """
        x1_1, y1_1, w1, h1 = bbox1
        x1_2, y1_2, w2, h2 = bbox2

        x2_1, y2_1 = x1_1 + w1, y1_1 + h1
        x2_2, y2_2 = x1_2 + w2, y1_2 + h2

        # Calculate intersection
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection = (x_right - x_left) * (y_bottom - y_top)
        union = w1 * h1 + w2 * h2 - intersection

        return intersection / union if union > 0 else 0.0
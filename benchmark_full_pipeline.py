"""
GPU benchmark for the full Gradio-style pipeline (YOLO person/hands → person crop →
batched PPD → hand crops → gesture classifier).

Disk reads are timed separately (optional overlap with ``--prefetch-disk``, using ``dataset.num_workers`` from each config as the prefetch thread pool size). GPU work is timed wall-to-wall and split into stages; see ``disk_load_seconds_total``, ``gpu_pipeline_wall_seconds_total``, and ``stage_seconds_total`` in the JSON output.

Usage (from repo root, with venv activated):
  python benchmark_full_pipeline.py --configs-dir path/to/configs --output-json out.json

Or with PYTHONPATH including multimodal_gesture_recognition, this file can be run similarly
to benchmark_configs_speed.py.

Further speed ideas (not all implemented here): ``--prefetch-disk`` overlaps decode with GPU (thread count = ``dataset.num_workers``);
a PyTorch ``DataLoader`` or NVIDIA DALI can pipeline more aggressively; ``torch.compile`` /
TensorRT on YOLO or the classifier; on-GPU resize/normalize (e.g. Kornia) reduces
``cpu_hand_prepare``; lowering YOLO ``imgsz`` trades accuracy for throughput.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import random
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.amp import autocast
from torchvision import transforms

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent


def _setup_sys_path() -> None:
    sys.path.insert(0, str(REPO_ROOT / "multimodal_gesture_recognition"))
    sys.path.insert(0, str(REPO_ROOT / "preprocess"))


_setup_sys_path()


def _ensure_modules_pkg_stub() -> None:
    # Import modules.depth_estimator without running preprocess/modules/__init__.py
    if "modules" in sys.modules:
        return
    import types

    pkg = types.ModuleType("modules")
    pkg.__path__ = [str(REPO_ROOT / "preprocess" / "modules")]
    sys.modules["modules"] = pkg


_ensure_modules_pkg_stub()

from models import build_model  # noqa: E402
from utils import load_config, setup_device  # noqa: E402
from modules.depth_estimator import DepthEstimator  # noqa: E402


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


def flush_results_json(output_path: Path, results: List[Dict[str, Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(results, indent=2, ensure_ascii=False)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(output_path)


def collect_config_paths(configs_dir: Path) -> List[Path]:
    paths = sorted(p for p in configs_dir.glob("*.toml") if p.is_file())
    return [p for p in paths if p.name.lower() != "pyproject.toml"]


def collect_example_paths(example_dir: Path) -> List[Path]:
    paths: List[Path] = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG", "*.bmp", "*.BMP"):
        paths.extend(example_dir.glob(ext))
    return sorted(set(paths), key=lambda p: p.name)


def apply_benchmark_config(config: Dict) -> Dict:
    c = copy.deepcopy(config)
    m = c.setdefault("model", {})
    m["pretrained"] = False
    m["rgb_pretrained"] = False
    m["depth_pretrained"] = False
    m["pretrained_ckpt"] = ""
    m["rgb_pretrained_ckpt"] = ""
    m["depth_pretrained_ckpt"] = ""
    return c


def extract_report_row(config: Dict) -> Dict[str, Any]:
    model = config["model"]
    modality = model["modality"]
    arch = model["architecture"]

    if modality == "fusion":
        rgb_arch = model.get("rgb_architecture", arch)
        depth_arch = model.get("depth_architecture", arch)
        fusion_strategy = model["fusion_method"]
    elif modality == "rgb":
        rgb_arch = arch
        depth_arch = None
        fusion_strategy = None
    elif modality == "depth":
        rgb_arch = None
        depth_arch = arch
        fusion_strategy = None
    else:
        raise ValueError(f"Unknown modality: {modality}")

    vssd_default = model.get("vssd_variant")
    if rgb_arch == "vssd":
        rgb_vssd = model.get("rgb_vssd_variant", vssd_default)
    else:
        rgb_vssd = None

    if depth_arch == "vssd":
        depth_vssd = model.get("depth_vssd_variant", vssd_default)
    else:
        depth_vssd = None

    return {
        "rgb_architecture": rgb_arch,
        "depth_architecture": depth_arch,
        "rgb_vssd_variant": rgb_vssd,
        "depth_vssd_variant": depth_vssd,
        "modality": modality,
        "fusion_strategy": fusion_strategy,
        "batch_size": int(config["training"]["batch_size"]),
        "image_size": int(model.get("image_size", 224)),
    }


def configure_cudnn_benchmark(deterministic: bool, device: torch.device) -> None:
    if device.type != "cuda":
        return
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


def set_benchmark_seed(seed: int, deterministic: bool) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _autocast_cm(device: torch.device, use_amp: bool):
    if device.type == "cuda":
        return autocast("cuda", enabled=use_amp)
    return nullcontext()




def build_hand_transforms() -> Tuple[Callable, Callable]:
    """Shared torchvision transforms (once per config, not per batch)."""
    rgb_tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    depth_tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )
    return rgb_tf, depth_tf


def _empty_stage_times() -> Dict[str, float]:
    return {
        "yolo_person": 0.0,
        "yolo_hands": 0.0,
        "cpu_layout": 0.0,
        "ppd": 0.0,
        "cpu_hand_prepare": 0.0,
        "classifier": 0.0,
    }


def _largest_person_bbox_norm(result, img_w: int, img_h: int) -> Optional[List[float]]:
    if result.boxes is None or len(result.boxes) == 0:
        return None
    boxes = result.boxes
    person = boxes[boxes.cls == 0]
    if len(person) == 0:
        return None
    areas = (person.xyxy[:, 2] - person.xyxy[:, 0]) * (person.xyxy[:, 3] - person.xyxy[:, 1])
    j = int(areas.argmax())
    bbox = person.xyxy[j].cpu().numpy()
    conf = float(person.conf[j].cpu().numpy())
    return [
        float(bbox[0] / img_w),
        float(bbox[1] / img_h),
        float(bbox[2] / img_w),
        float(bbox[3] / img_h),
        conf,
    ]


def _all_hand_bboxes_norm(
    result, img_w: int, img_h: int, min_conf: float
) -> List[List[float]]:
    if result.boxes is None or len(result.boxes) == 0:
        return []
    boxes = result.boxes
    confidences = boxes.conf.cpu().numpy()
    order = np.argsort(confidences)[::-1]
    out: List[List[float]] = []
    for idx in order:
        c = float(confidences[idx])
        if c < min_conf:
            continue
        bbox = boxes.xyxy[int(idx)].cpu().numpy()
        out.append(
            [
                float(bbox[0] / img_w),
                float(bbox[1] / img_h),
                float(bbox[2] / img_w),
                float(bbox[3] / img_h),
                c,
            ]
        )
    return out


def expand_person_bbox(
    person_bbox: List[float], hand_bboxes: List[List[float]]
) -> List[float]:
    if not hand_bboxes:
        return person_bbox
    px1, py1, px2, py2, p_conf = person_bbox
    min_x, min_y = px1, py1
    max_x, max_y = px2, py2
    for hand_bbox in hand_bboxes:
        hx1, hy1, hx2, hy2, _ = hand_bbox
        min_x = min(min_x, hx1)
        min_y = min(min_y, hy1)
        max_x = max(max_x, hx2)
        max_y = max(max_y, hy2)
    min_x = max(0.0, min_x)
    min_y = max(0.0, min_y)
    max_x = min(1.0, max_x)
    max_y = min(1.0, max_y)
    return [min_x, min_y, max_x, max_y, p_conf]


def crop_hand_from_image(
    image: np.ndarray,
    bbox: Sequence[float],
    target_size: int,
    hand_min_size: int,
) -> np.ndarray:
    H, W = image.shape[:2]
    x1 = int(bbox[0] * W)
    y1 = int(bbox[1] * H)
    x2 = int(bbox[2] * W)
    y2 = int(bbox[3] * H)
    if (x2 - x1) < hand_min_size or (y2 - y1) < hand_min_size:
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        half_size = hand_min_size // 2
        x1 = max(0, center_x - half_size)
        y1 = max(0, center_y - half_size)
        x2 = min(W, center_x + half_size)
        y2 = min(H, center_y + half_size)
    cropped = image[y1:y2, x1:x2]
    if cropped.size == 0:
        cropped = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    return cv2.resize(cropped, (target_size, target_size))


def load_shared_detectors(
    yolov10_weights: Path,
    yolov13_weights: Path,
    ppd_cfg: Dict[str, Any],
    device: torch.device,
):
    from ultralytics import YOLO

    yolo_hand = YOLO(str(yolov10_weights))
    yolo_person = YOLO(str(yolov13_weights))
    depth_estimator = DepthEstimator(ppd_cfg)
    ppd_model = depth_estimator.get_model()
    if device.type == "cuda":
        yolo_hand.to(device)
        yolo_person.to(device)
    return yolo_hand, yolo_person, ppd_model




def run_pipeline_batch_gpu(
    images_rgb: List[np.ndarray],
    yolo_hand: Any,
    yolo_person: Any,
    ppd_model: nn.Module,
    clf: nn.Module,
    bench_cfg: Dict,
    device: torch.device,
    yolo_conf: float,
    yolo_iou: float,
    min_hand_conf: float,
    hand_min_size: int,
    use_amp: bool,
    rgb_tf: Callable,
    depth_tf: Callable,
) -> Dict[str, float]:
    """Run one batch; returns per-stage seconds (CUDA sync around GPU segments)."""
    times = _empty_stage_times()
    B = len(images_rgb)
    if B == 0:
        return times

    _sync_if_cuda(device)
    t0 = time.perf_counter()
    person_res = yolo_person.predict(
        images_rgb,
        conf=yolo_conf,
        iou=yolo_iou,
        verbose=False,
        batch=B,
    )
    _sync_if_cuda(device)
    times["yolo_person"] = time.perf_counter() - t0

    _sync_if_cuda(device)
    t0 = time.perf_counter()
    hand_res = yolo_hand.predict(
        images_rgb,
        conf=min_hand_conf,
        iou=yolo_iou,
        verbose=False,
        batch=B,
    )
    _sync_if_cuda(device)
    times["yolo_hands"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    modality = bench_cfg["model"]["modality"]
    image_size = int(bench_cfg["model"]["image_size"])

    ppd_inputs_bgr: List[np.ndarray] = []
    ppd_targets: List[Tuple[int, int]] = []
    ppd_src_idx: List[int] = []

    person_crops: List[Optional[np.ndarray]] = [None] * B
    px_boxes: List[Optional[Tuple[int, int, int, int]]] = [None] * B
    hand_lists: List[List[List[float]]] = [[] for _ in range(B)]
    full_hw: List[Tuple[int, int]] = []

    for i in range(B):
        img = images_rgb[i]
        H, W = img.shape[:2]
        full_hw.append((H, W))
        pr = person_res[i]
        hr = hand_res[i]
        person_bbox = _largest_person_bbox_norm(pr, W, H)
        hands = _all_hand_bboxes_norm(hr, W, H, min_hand_conf)
        hand_lists[i] = hands
        if not hands or person_bbox is None:
            continue
        expanded = expand_person_bbox(person_bbox, hands)
        px1, py1, px2, py2, _ = expanded
        ix1, iy1, ix2, iy2 = (
            int(px1 * W),
            int(py1 * H),
            int(px2 * W),
            int(py2 * H),
        )
        crop = img[iy1:iy2, ix1:ix2]
        if crop.size == 0:
            continue
        person_crops[i] = crop
        px_boxes[i] = (ix1, iy1, ix2, iy2)
        ppd_inputs_bgr.append(cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
        ppd_targets.append((crop.shape[0], crop.shape[1]))
        ppd_src_idx.append(i)

    times["cpu_layout"] = time.perf_counter() - t0

    depth_maps: List[Optional[np.ndarray]] = [None] * B
    _sync_if_cuda(device)
    t0 = time.perf_counter()
    if ppd_inputs_bgr:
        maps = ppd_model.infer_images_batched(ppd_inputs_bgr, ppd_targets, use_fp16=True)
        for k, src_i in enumerate(ppd_src_idx):
            depth_maps[src_i] = maps[k]
    _sync_if_cuda(device)
    times["ppd"] = time.perf_counter() - t0

    rgb_tensors: List[torch.Tensor] = []
    depth_tensors: List[torch.Tensor] = []

    t0 = time.perf_counter()
    for i in range(B):
        img = images_rgb[i]
        H, W = full_hw[i]
        hands = hand_lists[i]
        pbox = px_boxes[i]
        pc = person_crops[i]
        dmap = depth_maps[i]
        if not hands or pbox is None or pc is None:
            continue
        ix1, iy1, ix2, iy2 = pbox
        cw = max(ix2 - ix1, 1)
        ch = max(iy2 - iy1, 1)
        for hand in hands:
            hx1, hy1, hx2, hy2, _ = hand
            hx1_rel = (hx1 * W - ix1) / cw
            hy1_rel = (hy1 * H - iy1) / ch
            hx2_rel = (hx2 * W - ix1) / cw
            hy2_rel = (hy2 * H - iy1) / ch
            hand_rgb = crop_hand_from_image(
                pc, [hx1_rel, hy1_rel, hx2_rel, hy2_rel], image_size, hand_min_size
            )
            if modality != "depth":
                rgb_tensors.append(rgb_tf(Image.fromarray(hand_rgb)))
            if modality in ("depth", "fusion"):
                if dmap is None:
                    hand_d = np.zeros((image_size, image_size), dtype=np.uint8)
                else:
                    if dmap.ndim == 2:
                        d3 = cv2.cvtColor(dmap, cv2.COLOR_GRAY2RGB)
                    else:
                        d3 = dmap
                    hand_d_rgb = crop_hand_from_image(
                        d3,
                        [hx1_rel, hy1_rel, hx2_rel, hy2_rel],
                        image_size,
                        hand_min_size,
                    )
                    hand_d = cv2.cvtColor(hand_d_rgb, cv2.COLOR_RGB2GRAY)
                depth_tensors.append(depth_tf(Image.fromarray(hand_d)))

    times["cpu_hand_prepare"] = time.perf_counter() - t0

    if modality == "rgb" and not rgb_tensors:
        return times
    if modality == "depth" and not depth_tensors:
        return times
    if modality == "fusion" and (not rgb_tensors or not depth_tensors):
        return times

    _sync_if_cuda(device)
    t0 = time.perf_counter()
    rgb_batch = (
        torch.stack(rgb_tensors, dim=0).to(device, non_blocking=True)
        if rgb_tensors
        else None
    )
    if modality == "rgb":
        with torch.no_grad():
            with _autocast_cm(device, use_amp):
                _ = clf(rgb_batch)
        _sync_if_cuda(device)
        times["classifier"] = time.perf_counter() - t0
        return times

    if modality == "depth":
        depth_batch = torch.stack(depth_tensors, dim=0).to(device, non_blocking=True)
        with torch.no_grad():
            with _autocast_cm(device, use_amp):
                _ = clf(depth_batch)
        _sync_if_cuda(device)
        times["classifier"] = time.perf_counter() - t0
        return times

    depth_batch = torch.stack(depth_tensors, dim=0).to(device, non_blocking=True)
    with torch.no_grad():
        with _autocast_cm(device, use_amp):
            _ = clf(rgb_batch, depth_batch)
    _sync_if_cuda(device)
    times["classifier"] = time.perf_counter() - t0
    return times



def benchmark_one_config(
    cfg_path: Path,
    bench_cfg: Dict,
    device: torch.device,
    yolo_hand: Any,
    yolo_person: Any,
    ppd_model: nn.Module,
    example_paths: List[Path],
    warmup_batches: int,
    max_batches: int,
    deterministic: bool,
    yolo_conf: float,
    yolo_iou: float,
    min_hand_conf: float,
    hand_min_size: int,
    bench_seed: int,
    prefetch_disk: bool,
) -> Dict[str, Any]:
    row = extract_report_row(bench_cfg)
    row["config_path"] = str(cfg_path.resolve())

    B = int(bench_cfg["training"]["batch_size"])
    use_amp = bool(bench_cfg["hardware"].get("use_amp", False)) and device.type == "cuda"
    num_workers = int(bench_cfg.get("dataset", {}).get("num_workers", 0))

    set_benchmark_seed(bench_seed, deterministic)
    configure_cudnn_benchmark(deterministic, device)

    clf: nn.Module = build_model(bench_cfg).to(device)
    clf.eval()

    rgb_tf, depth_tf = build_hand_transforms()

    rng = random.Random(bench_seed)

    def sample_batch_paths() -> List[Path]:
        return [rng.choice(example_paths) for _ in range(B)]

    def load_batch_cpu(paths: List[Path]) -> List[np.ndarray]:
        out: List[np.ndarray] = []
        for p in paths:
            bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if bgr is None:
                raise RuntimeError(f"Failed to read image: {p}")
            out.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        return out

    def run_gpu(cpu_imgs: List[np.ndarray]) -> Dict[str, float]:
        _sync_if_cuda(device)
        return run_pipeline_batch_gpu(
            cpu_imgs,
            yolo_hand,
            yolo_person,
            ppd_model,
            clf,
            bench_cfg,
            device,
            yolo_conf,
            yolo_iou,
            min_hand_conf,
            hand_min_size,
            use_amp,
            rgb_tf,
            depth_tf,
        )

    for _ in range(max(0, warmup_batches)):
        paths = sample_batch_paths()
        cpu_imgs = load_batch_cpu(paths)
        _sync_if_cuda(device)
        _ = run_gpu(cpu_imgs)
        _sync_if_cuda(device)

    total_images = 0
    gpu_wall_seconds = 0.0
    disk_seconds = 0.0
    stage_totals: Dict[str, float] = defaultdict(float)
    timed_limit = max_batches if max_batches > 0 else 50

    if prefetch_disk and num_workers > 0:
        ex = ThreadPoolExecutor(max_workers=num_workers)
        try:
            next_paths = sample_batch_paths()
            fut = ex.submit(load_batch_cpu, next_paths)
            for _ in range(timed_limit):
                t_disk0 = time.perf_counter()
                cpu_imgs = fut.result()
                disk_seconds += time.perf_counter() - t_disk0
                next_paths = sample_batch_paths()
                fut = ex.submit(load_batch_cpu, next_paths)
                _sync_if_cuda(device)
                t_gpu0 = time.perf_counter()
                st = run_gpu(cpu_imgs)
                _sync_if_cuda(device)
                gpu_wall_seconds += time.perf_counter() - t_gpu0
                for k, v in st.items():
                    stage_totals[k] += v
                total_images += B
        finally:
            ex.shutdown(wait=True, cancel_futures=False)
    else:
        for _ in range(timed_limit):
            paths = sample_batch_paths()
            t_disk0 = time.perf_counter()
            cpu_imgs = load_batch_cpu(paths)
            disk_seconds += time.perf_counter() - t_disk0
            _sync_if_cuda(device)
            t_gpu0 = time.perf_counter()
            st = run_gpu(cpu_imgs)
            _sync_if_cuda(device)
            gpu_wall_seconds += time.perf_counter() - t_gpu0
            for k, v in st.items():
                stage_totals[k] += v
            total_images += B

    del clf
    if device.type == "cuda":
        torch.cuda.empty_cache()


    row["prefetch_num_workers"] = int(num_workers) if prefetch_disk else 0
    if total_images == 0 or gpu_wall_seconds <= 0:
        row["images_per_second"] = None
        row["error"] = "No timed batches or zero elapsed time"
        return row

    row["images_per_second"] = float(total_images / gpu_wall_seconds)
    row["gpu_pipeline_wall_seconds_total"] = float(gpu_wall_seconds)
    row["disk_load_seconds_total"] = float(disk_seconds)
    row["stage_seconds_total"] = {k: float(stage_totals[k]) for k in sorted(stage_totals)}
    row["stage_ms_per_image"] = {
        k: float(stage_totals[k]) / total_images * 1e3 for k in sorted(stage_totals)
    }
    row["disk_ms_per_image"] = float(disk_seconds) / total_images * 1e3
    gpu_only = sum(stage_totals[k] for k in ("yolo_person", "yolo_hands", "ppd", "classifier"))
    row["gpu_stages_sum_seconds_total"] = float(gpu_only)
    if gpu_only > 0:
        row["images_per_second_gpu_stages_only"] = float(total_images / gpu_only)
    return row

def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Benchmark full gesture pipeline on GPU")
    parser.add_argument("--configs-dir", type=str, required=True)
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument(
        "--example-images-dir",
        type=str,
        default=str(REPO_ROOT / "example_images"),
    )
    parser.add_argument("--warmup-batches", type=int, default=3)
    parser.add_argument(
        "--max-batches",
        type=int,
        default=10,
        help="Timed batches per config (each loads B images from disk). Use 0 for default 50.",
    )
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument(
        "--prefetch-disk",
        action="store_true",
        help="Prefetch next batch while GPU runs current batch (uses dataset.num_workers threads; 0 disables overlap)",
    )
    parser.add_argument(
        "--verify-ppd-parity",
        action="store_true",
        help="Run B=1 PPD batched vs single forward check on first example image, then exit 0",
    )
    parser.add_argument("--yolov10-weights", type=str, default="checkpoints/yolov10x_hands.pt")
    parser.add_argument("--yolov13-weights", type=str, default="checkpoints/yolov13l.pt")
    parser.add_argument("--ppd-checkpoint", type=str, default="checkpoints/ppd.pth")
    parser.add_argument(
        "--depth-anything-checkpoint",
        type=str,
        default="checkpoints/depth_anything_v2_vitl.pth",
    )
    parser.add_argument("--yolo-conf", type=float, default=0.5)
    parser.add_argument("--yolo-iou", type=float, default=0.45)
    parser.add_argument("--min-hand-confidence", type=float, default=0.3)
    parser.add_argument("--hand-min-size", type=int, default=32)
    parser.add_argument("--ppd-sampling-steps", type=int, default=4)
    parser.add_argument(
        "--ppd-inference-size",
        type=str,
        default="1024,768",
        help="W,H for DepthEstimator config (matches app)",
    )
    args = parser.parse_args()

    configs_dir = Path(args.configs_dir)
    output_path = Path(args.output_json)
    example_dir = Path(args.example_images_dir)

    if not configs_dir.is_dir():
        logger.error("configs-dir is not a directory: %s", configs_dir)
        sys.exit(1)

    cfg_paths = collect_config_paths(configs_dir)
    if not cfg_paths:
        logger.error("No .toml files in %s", configs_dir)
        sys.exit(1)

    example_paths = collect_example_paths(example_dir)
    if not example_paths:
        logger.error("No example images under %s", example_dir)
        sys.exit(1)

    first_cfg = load_config(str(cfg_paths[0]))
    device = setup_device(first_cfg)
    if device.type != "cuda":
        logger.error("This benchmark expects CUDA (GPU-only timing).")
        sys.exit(1)

    w_str, h_str = args.ppd_inference_size.split(",")
    ppd_cfg = {
        "pixel_perfect_depth": {
            "ppd_checkpoint": args.ppd_checkpoint,
            "semantics_checkpoint": args.depth_anything_checkpoint,
            "inference_width": int(w_str.strip()),
            "inference_height": int(h_str.strip()),
            "sampling_steps": int(args.ppd_sampling_steps),
            "batch_size": 1,
            "device": "cuda",
            "use_fp16": True,
        }
    }

    yolo_hand, yolo_person, ppd_model = load_shared_detectors(
        Path(args.yolov10_weights),
        Path(args.yolov13_weights),
        ppd_cfg,
        device,
    )

    if args.verify_ppd_parity:
        from modules.ppd.models.ppd import verify_infer_images_batched_parity  # noqa: E402

        p0 = example_paths[0]
        bgr = cv2.imread(str(p0))
        if bgr is None:
            logger.error("Cannot read %s for parity check", p0)
            sys.exit(1)
        th, tw = bgr.shape[0], bgr.shape[1]
        verify_infer_images_batched_parity(ppd_model, bgr, (th, tw))
        logger.info("PPD batched parity check passed for %s", p0)
        sys.exit(0)

    results: List[Dict[str, Any]] = []
    for cfg_path in cfg_paths:
        logger.info("Benchmarking %s", cfg_path)
        row: Optional[Dict[str, Any]] = None
        try:
            raw_cfg = load_config(str(cfg_path))
            bench_cfg = apply_benchmark_config(raw_cfg)
            bench_cfg.setdefault("hardware", {})["device"] = "cuda"
            bench_seed = int(
                bench_cfg.get("seed", bench_cfg.get("training", {}).get("seed", 42))
            )
            row = benchmark_one_config(
                cfg_path,
                bench_cfg,
                device,
                yolo_hand,
                yolo_person,
                ppd_model,
                example_paths,
                warmup_batches=max(0, args.warmup_batches),
                max_batches=max(0, args.max_batches),
                deterministic=args.deterministic,
                yolo_conf=args.yolo_conf,
                yolo_iou=args.yolo_iou,
                min_hand_conf=args.min_hand_confidence,
                hand_min_size=args.hand_min_size,
                bench_seed=bench_seed,
                prefetch_disk=args.prefetch_disk,
            )
            ips = row.get("images_per_second")
            logger.info(
                "  images_per_second=%s",
                f"{ips:.2f}" if ips is not None else str(ips),
            )
        except Exception as e:
            logger.exception("Failed on %s", cfg_path)
            if row is None:
                try:
                    raw_cfg = load_config(str(cfg_path))
                    row = extract_report_row(raw_cfg)
                    row["config_path"] = str(cfg_path.resolve())
                except Exception:
                    row = {"config_path": str(cfg_path.resolve())}
                row["images_per_second"] = None
                row["error"] = str(e)
        finally:
            # Persist after every config so partial results survive crashes on later configs.
            if row is not None:
                results.append(row)
                flush_results_json(output_path, results)
                logger.info("Wrote %d row(s) to %s", len(results), output_path)


if __name__ == "__main__":
    main()

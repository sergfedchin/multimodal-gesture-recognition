"""
Benchmark inference speed for multiple experiment configs (untrained weights).

Scans a directory of TOML configs, builds each model with random initialization
(no checkpoints), runs GPU-timed forward passes on the validation set defined in
each config, and writes a JSON list of results. Only one model is built at a time; each is
released before the next config. After each config finishes (success or error), the JSON file
is rewritten so partial results survive crashes or later failures.

On CPU, timed regions omit CUDA synchronization (there is no GPU); throughput is wall time
around the forward pass only.

Usage:
    .venv/bin/python benchmark_configs_speed.py --configs-dir path/to/configs --output-json results.json
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.amp import autocast

from data_loader import HandGestureDataset
from models import build_model
from utils import load_config, setup_device

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


def benchmark_seed(config: Dict) -> int:
    return int(config.get("seed", config.get("training", {}).get("seed", 42)))


def apply_benchmark_config(config: Dict) -> Dict:
    """Deep copy config and disable all pretrained / checkpoint loading."""
    c = copy.deepcopy(config)
    m = c.setdefault("model", {})
    m["pretrained"] = False
    m["rgb_pretrained"] = False
    m["depth_pretrained"] = False
    m["pretrained_ckpt"] = ""
    m["rgb_pretrained_ckpt"] = ""
    m["depth_pretrained_ckpt"] = ""
    return c


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
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def extract_report_fields(config: Dict) -> Dict[str, Any]:
    """Map config to JSON fields (architecture / fusion / batch / image size)."""
    model = config["model"]
    modality = model["modality"]
    arch = model["architecture"]

    if modality == "fusion":
        rgb_arch = model.get("rgb_architecture", arch)
        depth_arch = model.get("depth_architecture", arch)
        fusion_method = model["fusion_method"]
    elif modality == "rgb":
        rgb_arch = arch
        depth_arch = None
        fusion_method = None
    elif modality == "depth":
        rgb_arch = None
        depth_arch = arch
        fusion_method = None
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
        "fusion_method": fusion_method,
        "eval_batch_size": int(config["training"]["batch_size"]),
        "image_size": int(model.get("image_size", 224)),
    }


def create_val_loader_only(config: Dict, device: torch.device) -> DataLoader:
    """
    Validation DataLoader only (matches data_loader val path; no train/test IO).
    """
    val_df = pd.read_parquet(config["dataset"]["val_parquet"])
    seed = benchmark_seed(config)

    if config["dataset"]["val_subsample"] < 1.0:
        val_size = int(len(val_df) * config["dataset"]["val_subsample"])
        val_df, _ = train_test_split(
            val_df,
            train_size=val_size,
            stratify=val_df["label"],
            random_state=seed,
        )
        logger.info("Val set stratified subsample: %d samples", len(val_df))

    dev_str = str(device)
    val_dataset = HandGestureDataset(
        metadata_df=val_df,
        data_root=config["dataset"]["data_root"],
        modality=config["model"]["modality"],
        image_size=config["model"].get("image_size"),
        augmentation_cfg=None,
        device=dev_str,
        merge_thumb_index=True,
    )

    return DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["dataset"]["num_workers"],
        pin_memory=config["hardware"]["pin_memory"],
        drop_last=False,
    )


def _move_batch_tensors(batch: Dict, device: torch.device) -> None:
    for key in batch:
        if key != "metadata" and isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(device, non_blocking=True)


def _forward_only(model: nn.Module, batch: Dict, modality: str) -> torch.Tensor:
    if modality == "rgb":
        return model(batch["rgb"])
    if modality == "depth":
        return model(batch["depth"])
    if modality == "fusion":
        return model(batch["rgb"], batch["depth"])
    raise ValueError(f"Unknown modality: {modality}")


def _batch_size(batch: Dict, modality: str) -> int:
    if modality == "rgb":
        return int(batch["rgb"].shape[0])
    if modality == "depth":
        return int(batch["depth"].shape[0])
    return int(batch["rgb"].shape[0])


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _autocast_cm(device: torch.device, use_amp: bool):
    if device.type == "cuda":
        return autocast("cuda", enabled=use_amp)
    return nullcontext()


def warmup_forward(
    val_loader: DataLoader,
    model: nn.Module,
    device: torch.device,
    modality: str,
    use_amp: bool,
    warmup_batches: int,
) -> None:
    if warmup_batches <= 0:
        return
    count = 0
    while count < warmup_batches:
        saw_batch = False
        for batch in val_loader:
            saw_batch = True
            _move_batch_tensors(batch, device)
            _sync_if_cuda(device)
            with torch.no_grad():
                with _autocast_cm(device, use_amp):
                    _ = _forward_only(model, batch, modality)
            _sync_if_cuda(device)
            count += 1
            if count >= warmup_batches:
                return
        if not saw_batch:
            raise RuntimeError("Validation loader is empty (cannot warmup)")


def benchmark_one_config(
    config_path: Path,
    config: Dict,
    device: torch.device,
    warmup_batches: int,
    max_batches: int,
    deterministic: bool,
) -> Dict[str, Any]:
    """Run warmup + timed forwards; return report row including config_path."""
    row = extract_report_fields(config)
    row["config_path"] = str(config_path.resolve())

    bench_config = apply_benchmark_config(config)
    modality = bench_config["model"]["modality"]
    use_amp = bool(bench_config["hardware"].get("use_amp", False)) and device.type == "cuda"

    set_benchmark_seed(benchmark_seed(config), deterministic)
    configure_cudnn_benchmark(deterministic, device)

    val_loader = create_val_loader_only(bench_config, device)
    model: nn.Module | None = None
    try:
        model = build_model(bench_config).to(device)
        model.eval()

        warmup_forward(
            val_loader, model, device, modality, use_amp, warmup_batches
        )

        total_images = 0
        total_seconds = 0.0
        timed_batches = 0

        for batch in val_loader:
            if max_batches > 0 and timed_batches >= max_batches:
                break
            _move_batch_tensors(batch, device)
            _sync_if_cuda(device)
            t0 = time.perf_counter()
            with torch.no_grad():
                with _autocast_cm(device, use_amp):
                    _ = _forward_only(model, batch, modality)
            _sync_if_cuda(device)
            t1 = time.perf_counter()
            total_seconds += t1 - t0
            total_images += _batch_size(batch, modality)
            timed_batches += 1

        if total_images == 0 or total_seconds <= 0:
            row["images_per_second"] = None
            row["error"] = "No timed batches or zero elapsed time"
            return row

        row["images_per_second"] = float(total_images / total_seconds)
        return row
    finally:
        if model is not None:
            del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


def collect_config_paths(configs_dir: Path) -> List[Path]:
    paths = sorted(p for p in configs_dir.glob("*.toml") if p.is_file())
    # Exclude Python packaging metadata when scanning repo roots by mistake
    return [p for p in paths if p.name.lower() != "pyproject.toml"]


def flush_results_json(output_path: Path, results: List[Dict[str, Any]]) -> None:
    """Atomically write the full results list so partial progress is not lost on failure."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(results, indent=2, ensure_ascii=False)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(output_path)


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Benchmark inference speed for multiple TOML configs (untrained weights)"
    )
    parser.add_argument(
        "--configs-dir",
        type=str,
        required=True,
        help="Directory containing .toml config files",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        required=True,
        help="Path where the JSON list of results will be written",
    )
    parser.add_argument(
        "--warmup-batches",
        type=int,
        default=3,
        help="Number of forward-only warmup batches (not included in timing)",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=0,
        help="Max validation batches to time after warmup (0 = all)",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic cuDNN settings (slower, more reproducible)",
    )
    args = parser.parse_args()

    configs_dir = Path(args.configs_dir)
    output_path = Path(args.output_json)
    if not configs_dir.is_dir():
        logger.error("configs-dir is not a directory: %s", configs_dir)
        sys.exit(1)

    config_paths = collect_config_paths(configs_dir)
    if not config_paths:
        logger.error("No .toml files found in %s", configs_dir)
        sys.exit(1)

    results: List[Dict[str, Any]] = []

    for cfg_path in config_paths:
        logger.info("Processing %s", cfg_path)
        try:
            config = load_config(str(cfg_path))
            device = setup_device(config)
            row = benchmark_one_config(
                cfg_path,
                config,
                device,
                warmup_batches=max(0, args.warmup_batches),
                max_batches=max(0, args.max_batches),
                deterministic=args.deterministic,
            )
            if "error" not in row:
                ips = row["images_per_second"]
                logger.info(
                    "  images_per_second=%s",
                    f"{ips:.2f}" if ips is not None else str(ips),
                )
        except Exception as e:
            logger.exception("Failed on %s", cfg_path)
            try:
                config = load_config(str(cfg_path))
                row = extract_report_fields(config)
                row["config_path"] = str(cfg_path.resolve())
            except Exception:
                row = {
                    "rgb_architecture": None,
                    "depth_architecture": None,
                    "rgb_vssd_variant": None,
                    "depth_vssd_variant": None,
                    "fusion_method": None,
                    "eval_batch_size": None,
                    "image_size": None,
                    "config_path": str(cfg_path.resolve()),
                }
            row["images_per_second"] = None
            row["error"] = str(e)

        results.append(row)
        flush_results_json(output_path, results)
        logger.info("Saved %d result(s) to %s", len(results), output_path)

    logger.info("Finished; %d entries in %s", len(results), output_path)


if __name__ == "__main__":
    main()

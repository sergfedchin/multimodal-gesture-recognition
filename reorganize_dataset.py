import gc
import json
import shutil
from pathlib import Path
from typing import Dict
import polars as pl
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from uuid import uuid4

# ============================================================================
# CONFIGURATION - Modify these paths according to your setup
# ============================================================================

PREPROCESSED_DIR = Path("/mnt/e/HSE/thesis/processed_hands")
ORIGINAL_ANNOTATIONS_DIR = Path("/mnt/e/HSE/thesis/HaGRIDv2-1M/annotations")
OUTPUT_DIR = Path("/home/serg_fedchn/Homework/thesis/data/processed_hands")
OUTPUT_PARQUET_DIR = OUTPUT_DIR / "parquet"
OUTPUT_IMAGES_DIR = OUTPUT_DIR / "images"
TEMP_PARQUET_DIR = OUTPUT_DIR / "temp_parquet"  # Temporary per-gesture files

GESTURE_CLASSES = [
    "dislike", "call", "fist", "four", "grabbing", "grip", "hand_heart",
    "hand_heart2", "holy", "like", "little_finger", "middle_finger",
    "mute", "no_gesture", "ok", "one", "palm", "peace", "peace_inverted",
    "point", "rock", "stop", "stop_inverted", "take_picture", "three",
    "three_gun", "three2", "three3", "thumb_index", "thumb_index2",
    "timeout", "two_up", "two_up_inverted", "xsign",
]

MAX_WORKERS = 8
MAX_GESTURE_NAME_LEN = max(len(g) for g in GESTURE_CLASSES)

# ============================================================================
# GLOBAL VARIABLE - Loaded once, shared via fork (copy-on-write)
# ============================================================================

ORIGINAL_ANNOTATIONS = None

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_json(path: Path) -> Dict:
    """Load JSON file with error handling."""
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        tqdm.write(f"Error loading {path}: {e}")
        return {}


def copy_image_files(gesture: str, original_id: str, hand_idx: int, split: str):
    """Copy RGB and depth images to new directory structure."""
    try:
        rgb_src = (
            PREPROCESSED_DIR
            / gesture
            / "rgb"
            / "hands"
            / "png"
            / f"{original_id}_{hand_idx}.png"
        )
        depth_src = (
            PREPROCESSED_DIR
            / gesture
            / "depth"
            / "hands"
            / "png"
            / f"{original_id}_{hand_idx}.png"
        )

        rgb_dst_dir = OUTPUT_IMAGES_DIR / split / "rgb"
        depth_dst_dir = OUTPUT_IMAGES_DIR / split / "depth"
        rgb_dst_dir.mkdir(parents=True, exist_ok=True)
        depth_dst_dir.mkdir(parents=True, exist_ok=True)

        if rgb_src.exists():
            shutil.copy2(rgb_src, rgb_dst_dir / f"{original_id}_{hand_idx}.png")
        if depth_src.exists():
            shutil.copy2(depth_src, depth_dst_dir / f"{original_id}_{hand_idx}.png")

    except Exception as e:
        tqdm.write(f'Error while copying images: {str(e)}')
        # pass  # Silent to avoid output clutter


def load_all_original_annotations() -> Dict[str, Dict[str, Dict]]:
    """Load all original annotations into memory."""
    tqdm.write("Loading original annotations...")
    annotations = {}

    for split in ["train", "test", "val"]:
        annotations[split] = {}
        for gesture in tqdm(
            GESTURE_CLASSES, desc=f"Loading {split} annotations", leave=False
        ):
            ann_path = ORIGINAL_ANNOTATIONS_DIR / split / f"{gesture}.json"
            annotations[split][gesture] = load_json(ann_path)

    return annotations


def process_gesture_with_progress(args: tuple) -> tuple[str, str, str]:
    """Process gesture and write to two temporary parquet files."""
    gesture, worker_id = args
    
    global ORIGINAL_ANNOTATIONS
    
    records = []
    landmarks_records = []  # Separate storage for landmarks

    try:
        ann_path = PREPROCESSED_DIR / gesture / "annotations.json"
        preprocessed_anns = load_json(ann_path)

        if not preprocessed_anns:
            tqdm.write(f"Warning: No preprocessed annotations for {gesture}")
            return gesture, None, None

        pbar = tqdm(
            preprocessed_anns.items(),
            desc=f"[W{worker_id}] {gesture:<{MAX_GESTURE_NAME_LEN}}",
            position=worker_id + 1,
            leave=False,
            smoothing=0
        )

        for i, (original_id, preproc_data) in enumerate(pbar, 1):
            try:
                split = preproc_data.get("split", "train")
                orig_data = ORIGINAL_ANNOTATIONS[split][gesture].get(original_id, {})
                num_hands = len((preproc_data.get("adjusted_hands_bboxes") or []))

                for hand_idx in range(num_hands):
                    try:
                        # Main record WITHOUT landmarks
                        record = {
                            "original_id": original_id,
                            "hand_index": hand_idx,
                            "gesture_class": gesture,
                            "split": split,
                            "rgb_image_path": f"{split}/rgb/{original_id}_{hand_idx}.png",
                            "depth_image_path": f"{split}/depth/{original_id}_{hand_idx}.png",
                            "body_bbox": preproc_data.get("body_bbox"),
                            "adjusted_hand_bbox": (
                                (preproc_data.get("adjusted_hands_bboxes") or []) + [None]
                            )[hand_idx],
                            "original_hand_bbox": ((orig_data.get("bboxes") or []) + [None])[hand_idx],
                            "label": ((orig_data.get("labels") or []) + [None])[hand_idx],
                            "user_id": orig_data.get("user_id"),
                        }
                        
                        # Add metadata
                        meta = orig_data.get("meta", {}) or {}
                        record["age"] = ((meta.get("age") or []) + [None])[0]
                        record["gender"] = ((meta.get("gender") or []) + [None])[0]
                        record["race"] = ((meta.get("race") or []) + [None])[0]
                        
                        records.append(record)
                        
                        # Separate landmarks record
                        landmarks = ((orig_data.get("hand_landmarks") or []) + [None])[hand_idx]
                        if landmarks:
                            landmarks_records.append({
                                "original_id": original_id,
                                "hand_index": hand_idx,
                                "hand_landmarks": landmarks
                            })
                        
                        copy_image_files(gesture, original_id, hand_idx, split)
                        
                    except Exception as e:
                        tqdm.write(f'Error while creating hand record: {str(e)}')
                        pass  # Silent to avoid clutter

                if i % 1000 == 0:
                    gc.collect()

            except Exception as e:
                tqdm.write(f'Error while processing image: {str(e)}')
                pass  # Silent to avoid clutter

        pbar.close()

        # Write both files immediately with unique names
        temp_main_file = None
        temp_landmarks_file = None
        
        if records:
            temp_main_file = TEMP_PARQUET_DIR / f"main_{gesture}_{uuid4().hex[:8]}.parquet"
            df = pl.DataFrame(records, strict=False)
            df.write_parquet(temp_main_file)
        
        if landmarks_records:
            temp_landmarks_file = TEMP_PARQUET_DIR / f"landmarks_{gesture}_{uuid4().hex[:8]}.parquet"
            df_landmarks = pl.DataFrame(landmarks_records, strict=False)
            df_landmarks.write_parquet(temp_landmarks_file)
        
        return gesture, str(temp_main_file) if temp_main_file else None, str(temp_landmarks_file) if temp_landmarks_file else None

    except Exception as e:
        tqdm.write(f"Error processing gesture {gesture}: {e}")
        return gesture, None, None


# ============================================================================
# MAIN PROCESSING
# ============================================================================

def main():
    """Main function to reorganize the dataset."""
    global ORIGINAL_ANNOTATIONS

    tqdm.write("Starting dataset reorganization...")

    # Create output directories
    OUTPUT_PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_PARQUET_DIR.mkdir(parents=True, exist_ok=True)

    # Load annotations ONCE in main process (will be forked, not copied)
    ORIGINAL_ANNOTATIONS = load_all_original_annotations()
    tqdm.write("Annotations loaded. Memory will be shared via fork.\n")

    # Prepare work items with worker IDs
    work_items = [(gesture, i % MAX_WORKERS) for i, gesture in enumerate(GESTURE_CLASSES)]

    tqdm.write(f"Processing {len(GESTURE_CLASSES)} gestures with {MAX_WORKERS} workers...\n")
    temp_main_files = []
    temp_landmarks_files = []

    # Use fork context for copy-on-write memory sharing
    mp_context = mp.get_context('fork')
    
    with ProcessPoolExecutor(
        max_workers=MAX_WORKERS,
        mp_context=mp_context
    ) as executor:
        # Submit all tasks
        future_to_gesture = {
            executor.submit(process_gesture_with_progress, item): item[0]
            for item in work_items
        }

        # Main progress bar at position 0
        main_pbar = tqdm(
            total=len(GESTURE_CLASSES),
            desc="Overall progress",
            position=0,
        )

        # Process results as they complete
        for future in as_completed(future_to_gesture):
            gesture = future_to_gesture[future]
            try:
                gesture_name, temp_main, temp_landmarks = future.result()
                if temp_main:
                    temp_main_files.append(temp_main)
                if temp_landmarks:
                    temp_landmarks_files.append(temp_landmarks)
                main_pbar.set_postfix({"last": gesture_name})
            except Exception as e:
                tqdm.write(f"Error in gesture {gesture}: {e}")
            finally:
                main_pbar.update(1)

        main_pbar.close()

    # Combine main files by split
    tqdm.write("\nCombining main parquet files by split...")
    for split in ["train", "test", "val"]:
        tqdm.write(f"  Processing {split}...")
        if temp_main_files:
            df = pl.scan_parquet(temp_main_files).filter(pl.col("split") == split).collect()
            if len(df) > 0:
                output_path = OUTPUT_PARQUET_DIR / f"{split}.parquet"
                df.write_parquet(output_path)
                tqdm.write(f"    {split}: {len(df)} records saved to {output_path}")
            else:
                tqdm.write(f"    {split}: No records found")
        else:
            tqdm.write(f"    {split}: No data to process")

    # Combine all landmarks files
    tqdm.write("\nCombining landmarks parquet files...")
    if temp_landmarks_files:
        df_landmarks = pl.scan_parquet(temp_landmarks_files).collect()
        landmarks_path = OUTPUT_PARQUET_DIR / "hand_landmarks.parquet"
        df_landmarks.write_parquet(landmarks_path)
        tqdm.write(f"  Landmarks: {len(df_landmarks)} records saved to {landmarks_path}")
    else:
        tqdm.write("  No landmarks data to process")

    # Clean up temporary files
    tqdm.write("\nCleaning up temporary files...")
    shutil.rmtree(TEMP_PARQUET_DIR)

    tqdm.write("\nDataset reorganization complete!")
    tqdm.write(f"Parquet files saved to: {OUTPUT_PARQUET_DIR}")
    tqdm.write(f"  - train.parquet, test.parquet, val.parquet (main data)")
    tqdm.write(f"  - hand_landmarks.parquet (can join with main data)")
    tqdm.write(f"Images saved to: {OUTPUT_IMAGES_DIR}")


if __name__ == "__main__":
    main()

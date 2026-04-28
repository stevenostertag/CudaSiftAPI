"""
optimize_extractor.py - Use Optuna to find CuSift extract parameters that
ensure all ground truth keypoints are found while minimizing the number of
non-matching keypoints using a single objective score.

This script processes a directory of images, each with a corresponding ground
truth file. It uses a single-value objective function where trials are heavily
penalized for failing to find all ground truth keypoints. The goal is to find
parameters that successfully detect all truth points across all images while
producing the minimum number of excess keypoints.

Usage:
    python optimize_extractor.py <image_dir> <truth_dir> <output_dir> \\[options\\]
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any, Union, Optional

import numpy as np
import optuna
import psutil
from PIL import Image
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

from cusift import CuSift, CuSiftError, ExtractOptions, KeypointList

# --- MODIFICATION: Define a large penalty for failing the primary objective ---
FAILURE_PENALTY = 1_000_000

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_image_grayscale(path: Path) -> np.ndarray:
    """Loads an image from a path and converts it to a float32 grayscale NumPy array."""
    img = Image.open(path).convert("L")
    return np.asarray(img, dtype=np.float32)

def _preprocess_image(image: np.ndarray, gauss_sigma: float = 1.0) -> np.ndarray:
    """Replaces zeros/NaNs with the median and applies a Gaussian blur."""
    valid_pixels = image[(image != 0) & ~np.isnan(image)]
    if valid_pixels.size == 0:
        return image
    median_background = np.median(valid_pixels)
    processed_image = np.copy(image)
    processed_image[np.isnan(processed_image) | (processed_image == 0)] = median_background
    if gauss_sigma > 0:
        gaussian_filter(processed_image, sigma=gauss_sigma, output=processed_image)
    return processed_image

def _load_ground_truth(path: Path) -> List[Tuple[float, float]]:
    """Loads ground truth keypoint locations from a JSON file."""
    with open(path, "r") as f:
        data = json.load(f)
    return [(kp['x'], kp['y']) for kp in data]

def _find_image_truth_pairs(image_dir: Path, truth_dir: Path) -> List[Tuple[Path, Path]]:
    """Finds matching image and ground truth file pairs."""
    pairs = []
    for img_path in image_dir.glob("*.tiff"):
        truth_filename = img_path.stem + ".json"
        truth_path = truth_dir / truth_filename
        if truth_path.exists():
            pairs.append((img_path, truth_path))
        else:
            print(f"Warning: No ground truth file found for {img_path.name}", file=sys.stderr)
    if not pairs:
        print(f"Error: No matching image-truth pairs found in the specified directories.", file=sys.stderr)
    return pairs

def _estimate_and_check_memory(image_paths: List[Path]):
    """Estimates memory required to load all images and checks against available memory."""
    print("Estimating memory requirements...")
    total_bytes = 0
    for img_path in image_paths:
        try:
            with Image.open(img_path) as img:
                total_bytes += img.width * img.height * 4
        except Exception as e:
            print(
                f"Warning: Could not read dimensions from {img_path.name}. Memory estimation may be inaccurate. Error: {e}",
                file=sys.stderr,
            )
    estimated_gb = total_bytes / (1024**3)
    available_gb = psutil.virtual_memory().available / (1024**3)
    print(f"Estimated memory to load all images: {estimated_gb:.2f} GB")
    print(f"Available system memory: {available_gb:.2f} GB")
    if estimated_gb > available_gb * 0.9:
        print(
            "Error: Estimated memory requirement exceeds 90% of available system memory.",
            file=sys.stderr,
        )
        sys.exit(1)

def _match_keypoints_to_truth(
    extracted_kps: KeypointList, truth_kps: List[Tuple[float, float]], tolerance: float
) -> Tuple[int, int]:
    """Matches extracted keypoints to ground truth, returning found count and non-matching count."""
    found_truth_indices = set()
    if not extracted_kps:
        return 0, 0
    
    extracted_locations = np.array([(kp.x, kp.y) for kp in extracted_kps])
    truth_locations = np.array(truth_kps)
    
    # For each truth keypoint, see if any extracted keypoint is within tolerance
    for i, truth_loc in enumerate(truth_locations):
        distances = np.sqrt(np.sum((extracted_locations - truth_loc)**2, axis=1))
        if np.min(distances) <= tolerance:
            found_truth_indices.add(i)

    # For each extracted keypoint, check if it's a "non-match"
    num_non_matching = 0
    for ext_loc in extracted_locations:
        distances_to_truth = np.sqrt(np.sum((truth_locations - ext_loc)**2, axis=1))
        if np.min(distances_to_truth) > tolerance:
            num_non_matching += 1
            
    return len(found_truth_indices), num_non_matching

def _report_study_results(study: optuna.Study, t_opt: float | None = None) -> None:
    """Prints a summary of the optimization study results."""
    print("=" * 60)
    if t_opt is not None:
        print(f"Optimisation complete ({len(study.trials)} trials in {t_opt:.1f} s)")
    else:
        print(f"Study Review ({len(study.trials)} trials found)")
    
    try:
        best_trial = study.best_trial
        print(f"Best trial: #{best_trial.number}")
        print(f"  - Value: {best_trial.value:.2f} (lower is better)")
        
        if best_trial.value >= FAILURE_PENALTY:
             failed_images = (best_trial.value - FAILURE_PENALTY) // FAILURE_PENALTY
             print("  - Status: FAILED. Did not find all ground truth keypoints.")
             if failed_images > 0:
                 print(f"  - Failed on at least {int(failed_images)} image(s).")
        else:
             print("  - Status: SUCCESS. Found all ground truth keypoints.")
             print(f"  - Total non-matching keypoints: {int(best_trial.value)}")

        print("  - Parameters:")
        for k, v in best_trial.params.items():
            print(f"    {k}: {v}")
            
    except ValueError:
        print("Study finished without finding any successful trials.")
        
    print("=" * 60)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="optimize_extractor",
        description=(
            "Use Optuna to optimise CuSift extraction parameters to find all "
            "ground truth keypoints while minimizing excess detections."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("image_dir", type=str, help="Path to the directory with images.")
    p.add_argument("truth_dir", type=str, help="Path to the directory with ground truth JSON files.")
    p.add_argument("output_dir", type=str, help="Directory for study results.")
    p.add_argument("--lib", type=str, default=None, metavar="PATH", help="Explicit path to the cusift shared library.")
    p.add_argument("--n-trials", type=int, default=100, help="Number of Optuna trials (default: 100).")
    p.add_argument("--timeout", type=float, default=None, help="Maximum optimisation time in seconds (default: unlimited).")
    p.add_argument("--seed", type=int, default=42, help="Sampler seed for reproducibility (default: 42).")
    p.add_argument("--match-tolerance", type=float, default=2.0, help="Pixel distance tolerance for matching keypoints (default: 2.0).")
    p.add_argument("--review", action="store_true", help="Review results of an existing study and exit.")
    return p.parse_args(argv)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    
    img_dir, truth_dir, out_dir = Path(args.image_dir), Path(args.truth_dir), Path(args.output_dir)

    if not img_dir.is_dir() or not truth_dir.is_dir():
        print("Error: Image and ground truth directories must exist.", file=sys.stderr)
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    
    image_truth_pairs = _find_image_truth_pairs(img_dir, truth_dir)
    if not image_truth_pairs: sys.exit(1)

    _estimate_and_check_memory([p[0] for p in image_truth_pairs])

    storage_path = out_dir / "extractor_study.db"
    storage = f"sqlite:///{storage_path}"
    study_name = "cusift_extractor_optimize_single_obj"

    if args.review:
        try:
            study = optuna.load_study(study_name=study_name, storage=storage)
            _report_study_results(study)
        except KeyError:
            print(f"Error: Study '{study_name}' not found in {storage_path}", file=sys.stderr)
            sys.exit(1)
        sys.exit(0)

    print("Pre-loading and pre-processing data...")
    preloaded_data = []
    for img_path, truth_path in tqdm(image_truth_pairs, desc="Pre-loading"):
        try:
            preloaded_data.append({
                "image": _preprocess_image(_load_image_grayscale(img_path)),
                "truth": _load_ground_truth(truth_path),
                "name": img_path.name
            })
        except Exception as e:
            print(f"Warning: Could not load/process {img_path.name}. Skipping. Error: {e}", file=sys.stderr)
    
    if not preloaded_data:
        print("Error: Failed to load any valid image-truth pairs.", file=sys.stderr)
        sys.exit(1)

    print(f"Successfully pre-loaded {len(preloaded_data)} pairs.")
    sift = CuSift(dll_path=args.lib)
    print("[OK] CuSift initialized.")

    def objective(trial: optuna.Trial) -> float:
        extract_opts = ExtractOptions(
            thresh=trial.suggest_float("thresh", 0.1, 2.0),
            lowest_scale=trial.suggest_float("lowest_scale", 0.0, 5.0),
            edge_thresh=trial.suggest_float("edge_thresh", 5.0, 25.0),
            init_blur=trial.suggest_float("init_blur", 0.5, 2.0),
            num_octaves=trial.suggest_int("num_octaves", 4, 8),
            max_keypoints=trial.suggest_int("max_keypoints", 8192, 65536)
        )
        
        total_score = 0.0
        failed_image_count = 0

        for data in preloaded_data:
            try:
                extracted_kps = sift.extract(data["image"], options=extract_opts)
                num_found, num_non_matching = _match_keypoints_to_truth(
                    extracted_kps, data["truth"], args.match_tolerance
                )
                extracted_kps.free()

                if num_found == len(data["truth"]):
                    total_score += num_non_matching
                else:
                    failed_image_count += 1
            
            except CuSiftError as e:
                if "out of memory" in str(e) or "an illegal memory access" in str(e):
                    print(f"\nFATAL: CUDA error on image {data['name']}: {e}", file=sys.stderr)
                    raise e # Stop the study
                failed_image_count += 1 # Treat other errors as failure

        # If any image failed, apply a large penalty for each failure.
        if failed_image_count > 0:
            return (failed_image_count * FAILURE_PENALTY) + total_score

        return total_score

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    # --- MODIFICATION: Changed to single-objective minimization ---
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
    )

    n_existing = len(study.trials)
    if n_existing > 0: print(f"Resuming study. Found {n_existing} existing trials.")

    n_new = args.n_trials - n_existing
    t0 = time.perf_counter()

    if n_new > 0:
        print(f"Starting Optuna optimisation (running {n_new} new trials)...")
        try:
            study.optimize(objective, n_trials=n_new, timeout=args.timeout, show_progress_bar=True)
        except CuSiftError:
            print("\nOptimization stopped due to a fatal CUDA error.", file=sys.stderr)
    else:
        print("Optimisation already complete.")

    t_opt = time.perf_counter() - t0
    _report_study_results(study, t_opt)

    try:
        if study.best_trial.value < FAILURE_PENALTY:
            best_params_path = out_dir / "best_extractor_params.json"
            print(f"Saving best trial's parameters to: {best_params_path}")
            best_params_path.write_text(json.dumps(study.best_trial.params, indent=2))
        else:
            print("No trial succeeded in finding all ground truth keypoints.")
    except ValueError:
        print("Could not determine best parameters as no successful trials were found.")

    sys.exit(0)

if __name__ == "__main__":
    main()

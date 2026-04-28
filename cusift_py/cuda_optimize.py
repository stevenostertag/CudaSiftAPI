"""
cuda_optimize.py - Use Optuna to find CuSift parameters that minimise the
mean absolute difference (MAD) between overlapping pixels of warped images
from a directory of pairs.

This version uses a shared memory buffer to store pre-loaded images, allowing
all parallel worker processes to access the data without duplicating it in RAM,
leading to significant memory savings and faster startup times.

It runs optimization studies in parallel, tracks progress with a high-level
progress bar, and performs an efficient coverage-based sort on the results.

This version includes a --debug flag. When enabled, it saves detailed
artifacts (parameters, warped images, match visualizations, and the input
images themselves) for failing image pairs to help diagnose optimization problems.

Usage:
    python cuda_optimize.py <image_dir> <output_dir> --system <SystemName> [options]
"""
from cusift import CuSift, CuSiftError, ExtractOptions, HomographyOptions
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
from PIL import Image as PILImage
import shutil
import math
import re
from multiprocessing import shared_memory

try:
    from PIL import Image, ImageDraw
    import optuna
    from scipy.ndimage import gaussian_filter
    from tqdm import tqdm
    import psutil
    from joblib import Parallel, delayed
except ImportError:
    print(
        "Error: Required libraries are missing. Install with: pip install optuna Pillow scipy tqdm psutil joblib",
        file=sys.stderr,
    )
    sys.exit(1)

# ---------------------------------------------------------------------------
# Optimization Parameter Space
# ---------------------------------------------------------------------------
PARAMETER_SPACE = {
    "thresh": ("float", 0.5, 2.6, {"step": 0.1}),
    "lowest_scale": ("float", 0.0, 5.0),
    "edge_thresh": ("float", 8.0, 20.0),
    "init_blur": ("float", 1.1, 1.6),
    "num_octaves": ("int", 6, 7),
    "num_loops": ("int", 20000, 20000, {"step": 1000}),
    "max_ambiguity": ("float", 0.8, 1.2),
    "ransac_thresh": ("float", 1.5, 3.5),
    "improve_max_ambiguity": ("float", 0.5, 1.0),
    "improve_thresh": ("float", 4.0, 5.5),
    "improve_num_loops": ("int", 1000, 1000, {"step": 100}),
}
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _suggest_parameters(trial: optuna.Trial, parameter_space: dict) -> dict:
    params = {}
    for name, config in parameter_space.items():
        param_type, low, high, *kwargs = config
        extra_args = kwargs[0] if kwargs else {}
        if param_type == "float": params[name] = trial.suggest_float(name, low, high, **extra_args)
        elif param_type == "int": params[name] = trial.suggest_int(name, low, high, **extra_args)
    return params

def _load_image_grayscale(path: Path) -> np.ndarray:
    img = Image.open(path).convert("L")
    return np.asarray(img, dtype=np.float32)

def _find_image_pairs(directory: Path) -> list[tuple[str, str]]:
    pairs, h, m = [], {}, {}
    for f in directory.glob("*.tiff"):
        name = f.name
        if name.startswith("ACD_Historic_"): h[name.split("_")[-1]] = str(f)
        elif name.startswith("ACD_Mission_"): m[name.split("_")[-1]] = str(f)
    for s, p in m.items():
        if s in h: pairs.append((h[s], p))
    if not pairs: print(f"Warning: No image pairs found in {directory}", file=sys.stderr)
    return pairs

def _overlap_mad(warped1: np.ndarray, warped2: np.ndarray) -> float:
    valid1, valid2 = np.isfinite(warped1) & (warped1 != 0), np.isfinite(warped2) & (warped2 != 0)
    overlap = valid1 & valid2
    if not np.any(overlap): return float("inf")
    r1, r2 = warped1[overlap].astype(np.float64), warped2[overlap].astype(np.float64)
    mu1, s1 = r1.mean(), r1.std()
    mu2, s2 = r2.mean(), r2.std()
    if s1 > 1e-6: r1 = (r1 - mu1) / s1
    else: r1 -= mu1
    if s2 > 1e-6: r2 = (r2 - mu2) / s2
    else: r2 -= mu2
    return float(np.mean(np.abs(r1 - r2)))

def _report_study_results(study: optuna.Study) -> None:
    if not study.best_trial:
        print(f"Study '{study.study_name}' finished without finding any successful trials.")
        return
    best = study.best_trial
    score = best.value
    if score > 0:
        successful_pairs = math.ceil(score)
        median_mad = successful_pairs - score
    else:
        successful_pairs, median_mad = 0, float('inf')
    
    print("\n--- Best Trial Summary ---")
    print(f"  - Study: {study.study_name}")
    print(f"  - Score: {score:.4f} | Successful Pairs: {successful_pairs} | Median MAD: {median_mad:.4f}")
    print("  - Parameters:")
    if 'thresh' not in best.params and (fixed_thresh := study.user_attrs.get("fixed_thresh")) is not None:
        print(f"    thresh: {fixed_thresh}")
    for k, v in best.params.items():
        print(f"    {k}: {v}")
    print("-" * 26)

def _draw_matches(img1: np.ndarray, img2: np.ndarray, kp1: np.ndarray, kp2: np.ndarray, matches: np.ndarray) -> Image.Image:
    """Draws keypoint matches between two images."""
    h1, w1 = img1.shape
    h2, w2 = img2.shape
    out_img = Image.new('L', (w1 + w2, max(h1, h2)))
    out_img.paste(PILImage.fromarray(img1.astype(np.uint8)), (0, 0))
    out_img.paste(PILImage.fromarray(img2.astype(np.uint8)), (w1, 0))
    draw = ImageDraw.Draw(out_img)
    for i in range(matches.shape[0]):
        idx1, idx2 = int(matches[i, 0]), int(matches[i, 1])
        x1, y1 = kp1[idx1, 0], kp1[idx1, 1]
        x2, y2 = kp2[idx2, 0], kp2[idx2, 1]
        draw.line([(x1, y1), (x2 + w1, y2)], fill=255, width=1)
    return out_img
# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="cuda_optimize", description="Use Optuna to optimise CuSift parameters.", formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("image_dir", type=str, help="Path to the directory with image pairs.")
    p.add_argument("output_dir", type=str, help="Directory for results.")
    p.add_argument("--system", type=str, required=True, help="Name of the system for the output filename.")
    p.add_argument("--param-file", type=str, default=None, metavar="PATH", help="Path to a JSON file defining the Optuna parameter space.")
    p.add_argument("--lib", type=str, default=None, metavar="PATH", help="Explicit path to the cusift shared library.")
    p.add_argument("--n-trials", type=int, default=100, help="Number of Optuna trials (default: 100).")
    p.add_argument("--timeout", type=float, default=None, help="Maximum optimisation time in seconds (default: unlimited).")
    p.add_argument("--seed", type=int, default=42, help="Sampler seed for reproducibility (default: 42).")
    p.add_argument("--n-jobs", type=int, default=-1, help="Number of parallel jobs. -1 uses all available cores (default: -1).")
    p.add_argument("--debug", action="store_true", help="Enable debug mode to save failure artifacts.")
    p.add_argument("--skip-studies", action="store_true", help="Skip optimization and run recalculation/sort on existing results.")
    return p.parse_args(argv)
# ---------------------------------------------------------------------------
# Main Logic
# ---------------------------------------------------------------------------
def run_optimization_run(run_number: int, fixed_thresh_value: float, shm_name: str, image_metadata: dict,
                         pairs_to_process_indices: list, sift_lib_path: str | None, args: argparse.Namespace,
                         run_out_dir: Path, parameter_space: dict):
    sift = CuSift(dll_path=sift_lib_path)
    try:
        existing_shm = shared_memory.SharedMemory(name=shm_name)
    except FileNotFoundError:
        print(f"[Run {run_number}] FATAL: Worker could not attach to shared memory block '{shm_name}'.", file=sys.stderr)
        return None
    preloaded_image_data = []
    for i, j in pairs_to_process_indices:
        meta1, meta2 = image_metadata[i], image_metadata[j]
        img1 = np.ndarray(meta1['shape'], dtype=meta1['dtype'], buffer=existing_shm.buf, offset=meta1['offset'])
        img2 = np.ndarray(meta2['shape'], dtype=meta2['dtype'], buffer=existing_shm.buf, offset=meta2['offset'])
        preloaded_image_data.append(((img1, meta1['name']), (img2, meta2['name'])))
    run_param_space = {k: v for k, v in parameter_space.items() if k != 'thresh'}
    def objective(trial: optuna.Trial) -> float:
        params = _suggest_parameters(trial, run_param_space)
        params['thresh'] = fixed_thresh_value
        
        extract_params = {k: v for k, v in params.items() if k in ExtractOptions.__annotations__}
        homography_params = {k: v for k, v in params.items() if k in HomographyOptions.__annotations__}
        extract_opts = ExtractOptions(**extract_params, max_keypoints=65536)
        homography_opts = HomographyOptions(**homography_params, seed=0)
        
        mad_values = []
        for i, ((img1_data, name1), (img2_data, name2)) in enumerate(preloaded_image_data):
            failure_reason = ""
            debug_dir = Path(args.output_dir) / "debug" / f"trial_{run_number}_{trial.number}_pair_{Path(name1).stem}"
            
            try:
                if args.debug:
                    debug_dir.mkdir(parents=True, exist_ok=True)
                    PILImage.fromarray(img1_data).save(debug_dir / "input_1.tiff")
                    PILImage.fromarray(img2_data).save(debug_dir / "input_2.tiff")
                
                kp1_c, kp2_c, matches_c, H, n, w1, w2 = sift.extract_and_match_and_find_homography_and_warp(img1_data, img2_data, extract_options=extract_opts, homography_options=homography_opts)
                
                if n < 4:
                    failure_reason = f"Not enough inliers: {n} < 4"
                else:
                    mad = _overlap_mad(w1, w2)
                    if mad == float("inf"):
                        failure_reason = "No valid overlap"
                    else:
                        s_limit, sh_limit = 0.2, 0.2
                        s_good = (abs(1.0 - H[0,0]) < s_limit or abs(1.0 - H[1,1]) < s_limit)
                        sh_good = (abs(H[1,0]) < sh_limit or abs(H[0,1]) < sh_limit)
                        if not s_good or not sh_good:
                            failure_reason = "Bad geometry"
                        else:
                            mad_values.append(mad)
                if args.debug and failure_reason:
                    (debug_dir / "parameters.json").write_text(json.dumps(params, indent=2))
                    (debug_dir / "failure_reason.txt").write_text(failure_reason)
                    PILImage.fromarray(w1).save(debug_dir / "warped_1.tiff")
                    PILImage.fromarray(w2).save(debug_dir / "warped_2.tiff")
                    
                    kp1_np = np.array([[kp.x, kp.y] for kp in kp1_c])
                    kp2_np = np.array([[kp.x, kp.y] for kp in kp2_c])
                    matches_np = np.array([[m.m1, m.m2] for m in matches_c] if hasattr(matches_c, '__iter__') and len(matches_c) > 0 and hasattr(matches_c[0], 'm1') else [])
                    
                    if matches_np.shape[0] > 0:
                        _draw_matches(img1_data, img2_data, kp1_np, kp2_np, matches_np).save(debug_dir / "matches.png")
                kp1_c.free(); kp2_c.free()
            except (CuSiftError, ValueError) as e:
                if args.debug:
                    debug_dir.mkdir(parents=True, exist_ok=True)
                    (debug_dir / "parameters.json").write_text(json.dumps(params, indent=2))
                    (debug_dir / "exception.txt").write_text(str(e))
                continue
        
        if not mad_values: return -1.0
        return len(mad_values) - min(np.median(mad_values), 1.0)
    run_out_dir.mkdir(parents=True, exist_ok=True)
    storage, study_name = f"sqlite:///{run_out_dir / 'cusift_study.db'}", f"cusift_run_{run_number}"
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=args.seed), study_name=study_name, storage=storage, load_if_exists=True)
    study.set_user_attr("fixed_thresh", fixed_thresh_value)
    if (n_new := args.n_trials - len(study.trials)) > 0:
        study.optimize(objective, n_trials=n_new, timeout=args.timeout)
    
    existing_shm.close()
    if not study.best_trial or study.best_trial.value < 0: return None
    
    best_params = study.best_trial.params
    best_params['thresh'] = study.user_attrs["fixed_thresh"]
    return {"run_number": run_number, "best_params": best_params}

def verify_and_split_pairs(run_result: dict, image_data: list, pairs_to_process: list, sift: CuSift, successful_results: dict):
    if not run_result: return pairs_to_process, None, set()
    bp, run_number = run_result['best_params'], run_result['run_number']
    
    extract_opts = ExtractOptions(**{k: v for k, v in bp.items() if k in ExtractOptions.__annotations__}, max_keypoints=65536)
    homography_opts = HomographyOptions(**{k: v for k, v in bp.items() if k in HomographyOptions.__annotations__}, seed=0)
    
    failed_pairs, succeeded_keys = [], set()
    for i, (img1, img2) in enumerate(image_data):
        pair_tuple, pair_key = pairs_to_process[i], f"{Path(pairs_to_process[i][0]).name}__{Path(pairs_to_process[i][1]).name}"
        is_successful = False
        try:
            kp1, kp2, _, H, n, w1, w2 = sift.extract_and_match_and_find_homography_and_warp(img1, img2, extract_options=extract_opts, homography_options=homography_opts)
            if n >= 4 and (mad := _overlap_mad(w1, w2)) != float("inf"):
                s_limit, sh_limit = 0.2, 0.2
                if (abs(1.0 - H[0,0]) < s_limit or abs(1.0 - H[1,1]) < s_limit) and \
                   (abs(H[1,0]) < sh_limit or abs(H[0,1]) < sh_limit):
                    is_successful = True
                    succeeded_keys.add(pair_key)
                    if pair_key not in successful_results:
                        successful_results[pair_key] = {"run_number": run_number, "best_params": bp, "mad": mad}
            kp1.free(); kp2.free()
        except (CuSiftError, ValueError): pass
        if not is_successful: failed_pairs.append(pair_tuple)
        
    return failed_pairs, bp, succeeded_keys

def recalculate_run_coverage(worker_id: int, run_best_params: dict, shm_name: str, image_metadata: dict,
                             pair_indices: list, all_image_path_pairs: list, sift_lib_path: str | None) -> dict:
    """Worker function to test one parameter set against the full dataset using shared memory."""
    sift = CuSift(dll_path=sift_lib_path)
    run_number = run_best_params['run_number']
    params = run_best_params['params']
    extract_opts = ExtractOptions(**{k: v for k, v in params.items() if k in ExtractOptions.__annotations__}, max_keypoints=65536)
    homography_opts = HomographyOptions(**{k: v for k, v in params.items() if k in HomographyOptions.__annotations__}, seed=0)
    
    try:
        shm = shared_memory.SharedMemory(name=shm_name)
    except FileNotFoundError:
        print(f"[Recalc Worker] FATAL: Could not attach to shared memory block '{shm_name}'.", file=sys.stderr)
        return {"run_number": run_number, "succeeded_keys": set()}
    succeeded_keys = set()
    # Use tqdm for the inner loop, positioned by the worker_id
    for i, (idx1, idx2) in enumerate(tqdm(pair_indices, desc=f"Recalc Run {run_number}", position=worker_id + 1, leave=False)):
        meta1, meta2 = image_metadata[idx1], image_metadata[idx2]
        img1 = np.ndarray(meta1['shape'], dtype=meta1['dtype'], buffer=shm.buf, offset=meta1['offset'])
        img2 = np.ndarray(meta2['shape'], dtype=meta2['dtype'], buffer=shm.buf, offset=meta2['offset'])
        try:
            kp1, kp2, _, H, n, w1, w2 = sift.extract_and_match_and_find_homography_and_warp(img1, img2, extract_options=extract_opts, homography_options=homography_opts)
            if n >= 4 and _overlap_mad(w1, w2) != float("inf"):
                s_limit, sh_limit = 0.2, 0.2
                if (abs(1.0 - H[0,0]) < s_limit or abs(1.0 - H[1,1]) < s_limit) and \
                   (abs(H[1,0]) < sh_limit or abs(H[0,1]) < sh_limit):
                    pair_key = f"{Path(all_image_path_pairs[i][0]).name}__{Path(all_image_path_pairs[i][1]).name}"
                    succeeded_keys.add(pair_key)
            kp1.free(); kp2.free()
        except (CuSiftError, ValueError):
            continue
            
    shm.close()
    return {"run_number": run_number, "succeeded_keys": succeeded_keys}

def run_final_coverage_sort(all_run_best_params: list, all_run_succeeded_pairs: dict, out_dir: Path, system_name: str):
    if not all_run_best_params: return
    print("\n" + "="*80 + "\nPERFORMING COVERAGE-BASED SORT\n" + "="*80)
    
    eval_results = []
    for r in all_run_best_params:
        run_num = r['run_number']
        succeeded_set = all_run_succeeded_pairs.get(run_num, set())
        eval_results.append({
            "source_run_number": run_num, 
            "succeeded_pairs": list(succeeded_set), 
            "success_count": len(succeeded_set), 
            "parameters": r['params']
        })
    sorted_by_count = sorted(eval_results, key=lambda x: x['success_count'], reverse=True)
    
    final_params, processed_pairs = [], set()
    
    for r in sorted_by_count: 
        r['newly_covered_count'] = len(set(r['succeeded_pairs']) - processed_pairs)
    while any(r.get('newly_covered_count', -1) > 0 for r in sorted_by_count):
        best = max(sorted_by_count, key=lambda x: x.get('newly_covered_count', -1))
        final_params.append(best['parameters'])
        processed_pairs.update(set(best['succeeded_pairs']))
        best['newly_covered_count'] = -1
        
        for r in sorted_by_count:
            if r.get('newly_covered_count', -1) != -1:
                r['newly_covered_count'] = len(set(r['succeeded_pairs']) - processed_pairs)
    final_path = out_dir / f"{system_name}.json"
    final_path.write_text(json.dumps(final_params, indent=2))
    print(f"Saved final coverage-sorted parameter list to: {final_path}")

def _load_all_best_params_from_disk(output_dir: Path) -> list[dict]:
    """Scans the output directory for all study databases and extracts the best parameters from each."""
    print("Scanning for all completed study databases...")
    study_dbs = list(output_dir.glob("**/cusift_study.db"))
    if not study_dbs:
        print("No study databases found.")
        return []
    best_params_list = []
    for db_path in tqdm(sorted(study_dbs), desc="Loading Best Params"):
        # Infer run number from the parent directory name, e.g., "iter_1_run_15" -> 15
        match = re.search(r'run_(\d+)', str(db_path.parent.name))
        if not match:
            continue
        
        run_number = int(match.group(1))
        study_name = f"cusift_run_{run_number}"
        storage_url = f"sqlite:///{db_path}"
        try:
            study = optuna.load_study(study_name=study_name, storage=storage_url)
            if study.best_trial:
                best_params = study.best_trial.params
                # Retrieve the fixed thresh value from user attributes
                best_params['thresh'] = study.user_attrs.get("fixed_thresh", 0.0)
                best_params_list.append({"run_number": run_number, "params": best_params})
        except (KeyError, ValueError):
            # Ignore studies that can't be loaded or don't exist
            continue
            
    print(f"Found and loaded best parameters from {len(best_params_list)} studies.")
    return best_params_list

def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.debug:
        (out_dir / "debug").mkdir(parents=True, exist_ok=True)
        print("!!! DEBUG MODE ENABLED. Failure artifacts will be saved. This may be slow. !!!")
    try:
        parameter_space = json.loads(Path(args.param_file).read_text()) if args.param_file else PARAMETER_SPACE
    except (json.JSONDecodeError, FileNotFoundError) as e: sys.exit(f"Error loading parameter file: {e}")
    img_dir = Path(args.image_dir)
    if not img_dir.is_dir(): sys.exit(f"Error: image directory not found: {img_dir}")
    all_image_path_pairs = _find_image_pairs(img_dir)
    if not all_image_path_pairs: sys.exit("Exiting: No image pairs to process.")
    sift = CuSift(dll_path=args.lib)
    print("[OK] CuSift initialized.")
    # --- This is the start of the new logic ---
    if not args.skip_studies:
        thresh_cfg = parameter_space["thresh"]
        base_thresh_values = [round(v, 4) for v in np.arange(thresh_cfg[1], thresh_cfg[2] + thresh_cfg[3].get("step", 0.1), thresh_cfg[3].get("step", 0.1))]
        pairs_to_process = all_image_path_pairs
        successful_results, all_run_best_params, all_run_succeeded_pairs = {}, [], {}
        iteration = 1
        shm = None
        n_jobs = args.n_jobs if args.n_jobs > 0 else psutil.cpu_count(logical=True)
        try:
            while pairs_to_process:
                print(f"\n{'='*80}\nSTARTING GLOBAL ITERATION #{iteration} | Pairs remaining: {len(pairs_to_process)}\n{'='*80}")
                unique_paths = sorted(list(set(p for pair in pairs_to_process for p in pair)))
                path_to_idx = {path: i for i, path in enumerate(unique_paths)}
                print(f"Loading {len(unique_paths)} unique images into shared memory...")
                loaded_images = {i: _load_image_grayscale(Path(p)) for i, p in tqdm(enumerate(unique_paths), total=len(unique_paths), desc="Loading Images")}
                total_size = sum(arr.nbytes for arr in loaded_images.values())
                shm = shared_memory.SharedMemory(create=True, size=total_size)
                print(f"Created shared memory block: {total_size / (1024**2):.2f} MB")
                shm_flat_view = np.ndarray((total_size,), dtype=np.uint8, buffer=shm.buf)
                current_offset, image_metadata = 0, {}
                for i, arr in loaded_images.items():
                    byte_data = arr.tobytes()
                    shm_flat_view[current_offset : current_offset + len(byte_data)] = np.frombuffer(byte_data, dtype=np.uint8)
                    image_metadata[i] = {'name': Path(unique_paths[i]).name, 'shape': arr.shape, 'dtype': arr.dtype, 'offset': current_offset}
                    current_offset += len(byte_data)
                del loaded_images
                pairs_indices = [(path_to_idx[p1], path_to_idx[p2]) for p1, p2 in pairs_to_process]
                run_start = len(all_run_best_params) + 1
                tasks = (delayed(run_optimization_run)(run_start + i, thresh, shm.name, image_metadata, pairs_indices, args.lib, args, out_dir / f"iter_{iteration}_run_{run_start+i}", parameter_space) for i, thresh in enumerate(base_thresh_values))
                parallel_results = Parallel(n_jobs=n_jobs)(tqdm(tasks, total=len(base_thresh_values), desc=f"Iter {iteration} Studies"))
                valid_results = [res for res in parallel_results if res is not None]
                if not valid_results:
                    print("No runs produced a valid result. Halting optimization.")
                    shm.close(); shm.unlink(); shm = None
                    break
                main_proc_data = [(_load_image_grayscale(Path(p[0])), _load_image_grayscale(Path(p[1]))) for p in tqdm(pairs_to_process, desc="Verification Pre-load")]
                newly_succeeded = set()
                for run_result in valid_results:
                    study_path = out_dir / f"iter_{iteration}_run_{run_result['run_number']}" / 'cusift_study.db'
                    study = optuna.load_study(study_name=f"cusift_run_{run_result['run_number']}", storage=f"sqlite:///{study_path}")
                    _report_study_results(study)
                    _, best_params, succeeded_keys = verify_and_split_pairs(run_result, main_proc_data, pairs_to_process, sift, successful_results)
                    if best_params:
                        all_run_best_params.append({"run_number": run_result['run_number'], "params": best_params})
                        all_run_succeeded_pairs[run_result['run_number']] = succeeded_keys
                        newly_succeeded.update(succeeded_keys)
                shm.close(); shm.unlink(); shm = None
                print(f"\n{'='*80}\nGLOBAL ITERATION #{iteration} SUMMARY | Newly Succeeded: {len(newly_succeeded)}\n{'='*80}")
                if not newly_succeeded:
                    print("No further progress made. Halting.")
                    break
                pairs_to_process = [p for p in pairs_to_process if f"{Path(p[0]).name}__{Path(p[1]).name}" not in newly_succeeded]
                iteration += 1
        finally:
            if shm is not None and shm._name:
                print("Final cleanup: Unlinking shared memory block.")
                shm.close()
                shm.unlink()
        
        print("\n" + "="*80 + "\nITERATIVE OPTIMIZATION COMPLETE\n" + "="*80)
        if not pairs_to_process: print("All image pairs were successfully processed.")
        else:
            print(f"{len(pairs_to_process)} pairs remain unprocessed.")
            (out_dir / "unprocessed_pairs.json").write_text(json.dumps(pairs_to_process, indent=2))
            
        print(f"Total unique successful pairs: {len(successful_results)} out of {len(all_image_path_pairs)}")
        if successful_results: (out_dir / "successful_results.json").write_text(json.dumps(successful_results, indent=2))
    else:
        # This block executes if --skip-studies is used
        print("\n--skip-studies flag detected. Bypassing optimization studies.")
        # Initialize variables needed for the recalculation step
        all_run_best_params, all_run_succeeded_pairs = [], {}
        n_jobs = args.n_jobs if args.n_jobs > 0 else psutil.cpu_count(logical=True)
    # --- The new logic ends here ---
    # --- RECALCULATION STEP ---
    recalc_shm = None
    try:
        # !!! MODIFICATION START !!!
        # Load the best parameters directly from all saved study files to ensure completeness.
        all_run_best_params = _load_all_best_params_from_disk(out_dir)
        # !!! MODIFICATION END !!!
        
        if all_run_best_params:
            print("\n" + "="*80 + "\nRE-EVALUATING BEST PARAMETERS ON FULL DATASET\n" + "="*80)
            
            # Create a new shared memory block for the full dataset
            unique_paths = sorted(list(set(p for pair in all_image_path_pairs for p in pair)))
            path_to_idx = {path: i for i, path in enumerate(unique_paths)}
            
            print(f"Loading {len(unique_paths)} unique images into shared memory for final evaluation...")
            loaded_images = {i: _load_image_grayscale(Path(p)) for i, p in tqdm(enumerate(unique_paths), total=len(unique_paths), desc="Loading Full Dataset")}
            total_size = sum(arr.nbytes for arr in loaded_images.values())
            
            recalc_shm = shared_memory.SharedMemory(create=True, size=total_size, name="recalc_shm")
            print(f"Created final evaluation shared memory block: {total_size / (1024**2):.2f} MB")
            
            shm_flat_view = np.ndarray((total_size,), dtype=np.uint8, buffer=recalc_shm.buf)
            current_offset, image_metadata = 0, {}
            for i, arr in loaded_images.items():
                byte_data = arr.tobytes()
                shm_flat_view[current_offset : current_offset + len(byte_data)] = np.frombuffer(byte_data, dtype=np.uint8)
                image_metadata[i] = {'name': Path(unique_paths[i]).name, 'shape': arr.shape, 'dtype': arr.dtype, 'offset': current_offset}
                current_offset += len(byte_data)
            del loaded_images
            
            all_pair_indices = [(path_to_idx[p1], path_to_idx[p2]) for p1, p2 in all_image_path_pairs]
            
            recalc_tasks = (delayed(recalculate_run_coverage)(i, bp, recalc_shm.name, image_metadata, all_pair_indices, all_image_path_pairs, args.lib) for i, bp in enumerate(all_run_best_params))
            
            recalc_results = Parallel(n_jobs=n_jobs)(tqdm(recalc_tasks, total=len(all_run_best_params), desc="Recalculating Coverage", position=0))
            
            recalculated_succeeded_pairs = {res['run_number']: res['succeeded_keys'] for res in recalc_results}
            print("[OK] Full dataset re-evaluation complete.")
            all_run_succeeded_pairs = recalculated_succeeded_pairs
    finally:
        if recalc_shm is not None:
            print("Final cleanup: Unlinking final evaluation shared memory block.")
            recalc_shm.close()
            recalc_shm.unlink()
            
    # --- FINAL SORT STEP ---
    run_final_coverage_sort(all_run_best_params, all_run_succeeded_pairs, out_dir, args.system)
    
    sys.exit(0)

if __name__ == "__main__":
    main()

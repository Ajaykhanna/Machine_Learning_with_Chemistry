"""
Extract NACT, Energy, and Geometry Data from NEXMD Frame Directories

Reads frame directories containing:
  - nact.out:  time + flattened N×N NACT matrix (antisymmetric)
  - pes.out:   time + S0 + S1-S20 energies (eV)
  - *.xyz:     atomic coordinates

Features:
  - Multiprocessing with configurable chunk sizes
  - TQDM progress bars
  - Configurable number of excited states and pair selection
  - Robust leftover handling — no data lost

Outputs:
  - acn_Z.npy, acn_R.npy, acn_E.npy
  - acn_NACT{i}{j}.npy, acn_dENACT{i}{j}.npy per pair
  - extraction_report.json
"""

import os
import sys
import json
import glob
import time
import numpy as np
from pathlib import Path
from itertools import combinations
from multiprocessing import Pool, cpu_count
from functools import partial

try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("WARNING: tqdm not installed. Install with: pip install tqdm")
    print("         Falling back to basic progress reporting.\n")

# ============================================================================
# Configuration
# ============================================================================
BASE_DIR = "/vast/home/akhanna2/scratch/nexmd/acetylacetone_diketone/gsoptd_md/extracted_frames_gsoptd_md"
OUTPUT_DIR = "./50K_nact_data"
OUTPUT_PREFIX = "acn_"

N_FRAMES = 50000  # Total frames to extract (frame_1 to frame_N)
N_ATOMS = 15  # Acetylacetone
FRAME_PREFIX = "frame_"  # Subdirectory naming

# --- Configurable excited states ---
# The nact.out contains a 20×20 matrix, but you can choose to use fewer states.
# N_STATES_IN_FILE: actual matrix size in nact.out (must match file)
# N_STATES_TO_USE:  how many states to extract (1..N_STATES_TO_USE)
N_STATES_IN_FILE = 20  # NACT matrix is always 20×20 in the file
N_STATES_TO_USE = 3  # Extract pairs for states S1, S2, S3

# Pair selection: automatically computed from N_STATES_TO_USE
# States are 1-indexed: S1, S2, ..., S_{N_STATES_TO_USE}
# Pairs: (1,2), (1,3), ..., up to all upper-triangle combinations

# --- Multiprocessing ---
N_WORKERS = min(16, cpu_count())  # Number of parallel workers
CHUNK_SIZE = 500  # Frames per chunk (tune for your I/O)

# ============================================================================
# Element symbol → atomic number mapping
# ============================================================================
ELEMENT_TO_Z = {
    "H": 1,
    "He": 2,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Ne": 10,
    "Na": 11,
    "Mg": 12,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Ar": 18,
    "K": 19,
    "Ca": 20,
}


# ============================================================================
# Single-frame parsing functions
# ============================================================================
def parse_xyz_file(filepath, n_atoms):
    """Parse XYZ file → (symbols, coords[n_atoms, 3])."""
    with open(filepath, "r") as f:
        lines = f.readlines()

    n = int(lines[0].strip())
    if n != n_atoms:
        raise ValueError(f"Expected {n_atoms} atoms, got {n} in {filepath}")

    symbols = []
    coords = []
    for line in lines[2 : 2 + n_atoms]:
        parts = line.split()
        symbols.append(parts[0])
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])

    return symbols, np.array(coords, dtype=np.float64)


def parse_nact_file(filepath, n_states_in_file):
    """Parse nact.out → (time, nact_matrix[n_states, n_states])."""
    data = np.loadtxt(filepath)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    time_val = data[0, 0]
    nact_flat = data[0, 1:]

    expected = n_states_in_file * n_states_in_file
    if len(nact_flat) != expected:
        raise ValueError(
            f"Expected {expected} values for {n_states_in_file}×{n_states_in_file} "
            f"matrix, got {len(nact_flat)} in {filepath}"
        )

    return time_val, nact_flat.reshape(n_states_in_file, n_states_in_file)


def parse_pes_file(filepath):
    """Parse pes.out → (time, energies[n_states+1])."""
    data = np.loadtxt(filepath)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    return data[0, 0], data[0, 1:]


def process_single_frame(
    frame_num, base_dir, frame_prefix, n_atoms, n_states_in_file, n_states_to_use, pairs
):
    """Process one frame directory. Returns a result dict or error info.

    This function is called by each worker process.
    """
    frame_dir = os.path.join(base_dir, f"{frame_prefix}{frame_num}")

    result = {
        "frame_num": frame_num,
        "status": "ok",
        "error": None,
        "coords": None,
        "symbols": None,
        "energies": None,
        "nact_raw": None,
        "denact": None,
        "antisym_error": 0.0,
    }

    # Check directory exists
    if not os.path.isdir(frame_dir):
        result["status"] = "missing_dir"
        return result

    # --- XYZ ---
    xyz_files = glob.glob(os.path.join(frame_dir, "*.xyz"))
    if not xyz_files:
        result["status"] = "missing_xyz"
        return result

    try:
        symbols, coords = parse_xyz_file(xyz_files[0], n_atoms)
        result["coords"] = coords
        result["symbols"] = symbols
    except Exception as e:
        result["status"] = "bad_xyz"
        result["error"] = str(e)
        return result

    # --- PES ---
    pes_file = os.path.join(frame_dir, "pes.out")
    if not os.path.isfile(pes_file):
        result["status"] = "missing_pes"
        return result

    try:
        _, energies = parse_pes_file(pes_file)
        result["energies"] = energies
    except Exception as e:
        result["status"] = "bad_pes"
        result["error"] = str(e)
        return result

    # --- NACT ---
    nact_file = os.path.join(frame_dir, "nact.out")
    if not os.path.isfile(nact_file):
        result["status"] = "missing_nact"
        return result

    try:
        _, nact_matrix = parse_nact_file(nact_file, n_states_in_file)

        # Check antisymmetry
        result["antisym_error"] = float(np.max(np.abs(nact_matrix + nact_matrix.T)))

        # Extract upper triangle for requested pairs and compute scaled NACTs
        nact_vals = {}
        denact_vals = {}

        for i, j in pairs:
            raw_val = nact_matrix[i - 1, j - 1]  # 0-indexed in matrix
            nact_vals[f"{i}_{j}"] = raw_val

            # dENACT = NACT × (E_j - E_i)
            # energies[0] = S0, energies[i] = Si
            delta_E = energies[j] - energies[i]
            denact_vals[f"{i}_{j}"] = raw_val * delta_E

        result["nact_raw"] = nact_vals
        result["denact"] = denact_vals

    except Exception as e:
        result["status"] = "bad_nact"
        result["error"] = str(e)
        return result

    return result


def process_chunk(
    frame_numbers,
    base_dir,
    frame_prefix,
    n_atoms,
    n_states_in_file,
    n_states_to_use,
    pairs,
):
    """Process a chunk of frames sequentially (called within a worker)."""
    results = []
    for fn in frame_numbers:
        r = process_single_frame(
            fn,
            base_dir,
            frame_prefix,
            n_atoms,
            n_states_in_file,
            n_states_to_use,
            pairs,
        )
        results.append(r)
    return results


# ============================================================================
# Main
# ============================================================================
def main():
    start_time = time.time()

    # Compute pairs from N_STATES_TO_USE
    state_indices = list(range(1, N_STATES_TO_USE + 1))
    all_pairs = list(combinations(state_indices, 2))
    n_pairs = len(all_pairs)

    print("=" * 80)
    print("NEXMD NACT/Energy/Geometry Extraction")
    print("=" * 80)
    print(f"Base directory:      {BASE_DIR}")
    print(f"Output directory:    {OUTPUT_DIR}")
    print(f"Frames to extract:   {N_FRAMES} (frame_1 to frame_{N_FRAMES})")
    print(
        f"States in file:      {N_STATES_IN_FILE} ({N_STATES_IN_FILE}×{N_STATES_IN_FILE} matrix)"
    )
    print(
        f"States to use:       {N_STATES_TO_USE} → S{state_indices[0]}..S{state_indices[-1]}"
    )
    print(f"Pairs to extract:    {n_pairs}")
    if n_pairs <= 20:
        print(f"  Pairs: {all_pairs}")
    else:
        print(f"  First 5: {all_pairs[:5]}")
        print(f"  Last 5:  {all_pairs[-5:]}")
    print(f"Atoms:               {N_ATOMS}")
    print(f"Workers:             {N_WORKERS}")
    print(f"Chunk size:          {CHUNK_SIZE}")
    print()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ========================================================================
    # Build chunks — sequential frame numbers, robust leftover handling
    # ========================================================================
    all_frame_numbers = list(range(1, N_FRAMES + 1))  # frame_1 to frame_N

    chunks = []
    for start in range(0, N_FRAMES, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, N_FRAMES)
        chunks.append(all_frame_numbers[start:end])

    n_chunks = len(chunks)
    total_in_chunks = sum(len(c) for c in chunks)

    print(f"Chunking: {n_chunks} chunks, {total_in_chunks} frames total")
    assert total_in_chunks == N_FRAMES, (
        f"FATAL: Chunk accounting error! "
        f"Chunks contain {total_in_chunks} frames but expected {N_FRAMES}. "
        f"Last chunk has {len(chunks[-1])} frames."
    )

    # Report chunk breakdown including leftover
    if N_FRAMES % CHUNK_SIZE == 0:
        print(f"  All chunks: {CHUNK_SIZE} frames × {n_chunks} chunks (no leftover)")
    else:
        n_full = n_chunks - 1
        leftover = len(chunks[-1])
        print(f"  Full chunks:    {CHUNK_SIZE} frames × {n_full} chunks")
        print(f"  Leftover chunk: {leftover} frames × 1 chunk")

    print(f"  Verified: {total_in_chunks} == {N_FRAMES} ✓")
    print()

    # ========================================================================
    # Parallel extraction
    # ========================================================================
    print(f"Extracting with {N_WORKERS} workers...")

    worker_fn = partial(
        process_chunk,
        base_dir=BASE_DIR,
        frame_prefix=FRAME_PREFIX,
        n_atoms=N_ATOMS,
        n_states_in_file=N_STATES_IN_FILE,
        n_states_to_use=N_STATES_TO_USE,
        pairs=all_pairs,
    )

    all_results = []

    with Pool(processes=N_WORKERS) as pool:
        if HAS_TQDM:
            iterator = pool.imap(worker_fn, chunks)
            pbar = tqdm(iterator, total=n_chunks, desc="  Chunks", unit="chunk")
            for chunk_results in pbar:
                all_results.extend(chunk_results)
                n_ok = sum(1 for r in all_results if r["status"] == "ok")
                pbar.set_postfix(ok=n_ok, total=len(all_results))
            pbar.close()
        else:
            for idx, chunk_results in enumerate(pool.imap(worker_fn, chunks)):
                all_results.extend(chunk_results)
                if (idx + 1) % max(1, n_chunks // 20) == 0 or idx == n_chunks - 1:
                    n_ok = sum(1 for r in all_results if r["status"] == "ok")
                    elapsed = time.time() - start_time
                    print(
                        f"  Chunk {idx + 1}/{n_chunks}: "
                        f"{n_ok}/{len(all_results)} ok  [{elapsed:.0f}s]"
                    )

    extract_time = time.time() - start_time

    # ========================================================================
    # Verify completeness — CRITICAL: no frames lost
    # ========================================================================
    print(f"\n  Extraction done in {extract_time:.1f}s")
    print(f"  Results collected: {len(all_results)}")

    assert len(all_results) == N_FRAMES, (
        f"FATAL: Lost frames during multiprocessing! "
        f"Got {len(all_results)} results but expected {N_FRAMES}."
    )

    # Sort by frame number to restore sequential order
    all_results.sort(key=lambda r: r["frame_num"])

    # Verify every frame_num from 1..N_FRAMES is present exactly once
    result_frame_nums = [r["frame_num"] for r in all_results]
    expected_frame_nums = list(range(1, N_FRAMES + 1))
    if result_frame_nums != expected_frame_nums:
        missing = set(expected_frame_nums) - set(result_frame_nums)
        duplicate = [fn for fn in result_frame_nums if result_frame_nums.count(fn) > 1]
        print(f"  FATAL: Frame number mismatch!")
        if missing:
            print(f"    Missing: {sorted(missing)[:20]}...")
        if duplicate:
            print(f"    Duplicate: {sorted(set(duplicate))[:20]}...")
        sys.exit(1)

    print(f"  Frame integrity check: all {N_FRAMES} frames accounted for ✓")

    # ========================================================================
    # Categorize results
    # ========================================================================
    status_counts = {}
    for r in all_results:
        s = r["status"]
        status_counts[s] = status_counts.get(s, 0) + 1

    print(f"\n  Status breakdown:")
    for status, count in sorted(status_counts.items()):
        pct = 100.0 * count / N_FRAMES
        marker = "✓" if status == "ok" else "✗"
        print(f"    {marker} {status:20s}: {count:6d} ({pct:.1f}%)")

    # Collect valid results
    ok_results = [r for r in all_results if r["status"] == "ok"]
    n_valid = len(ok_results)

    if n_valid == 0:
        print("\nFATAL: No valid frames extracted!")
        sys.exit(1)

    # Get Z from first valid frame
    Z_all = np.array(
        [ELEMENT_TO_Z.get(s, 0) for s in ok_results[0]["symbols"]],
        dtype=np.int32,
    )
    print(f"\n  Atomic numbers: {Z_all}")
    print(f"  Elements: {ok_results[0]['symbols']}")

    # ========================================================================
    # Assemble arrays
    # ========================================================================
    print(f"\n  Assembling arrays for {n_valid} valid configurations...")

    R_all = np.zeros((n_valid, N_ATOMS, 3), dtype=np.float64)
    E_all = np.zeros((n_valid, len(ok_results[0]["energies"])), dtype=np.float64)

    pair_keys = [f"{i}_{j}" for (i, j) in all_pairs]
    nact_arrays = {k: np.zeros(n_valid, dtype=np.float64) for k in pair_keys}
    denact_arrays = {k: np.zeros(n_valid, dtype=np.float64) for k in pair_keys}

    antisym_errors = []

    if HAS_TQDM:
        iterator = tqdm(
            enumerate(ok_results), total=n_valid, desc="  Assembling", unit="config"
        )
    else:
        iterator = enumerate(ok_results)

    for idx, r in iterator:
        R_all[idx] = r["coords"]
        E_all[idx] = r["energies"]

        for k in pair_keys:
            nact_arrays[k][idx] = r["nact_raw"][k]
            denact_arrays[k][idx] = r["denact"][k]

        if r["antisym_error"] > 1e-8:
            antisym_errors.append((r["frame_num"], r["antisym_error"]))

    # ========================================================================
    # Statistics
    # ========================================================================
    print("\n" + "=" * 80)
    print("DATA STATISTICS")
    print("=" * 80)

    print(f"\n  Valid configurations: {n_valid}")
    print(f"  Coordinates R: shape {R_all.shape}")
    print(f"    Range: [{R_all.min():.4f}, {R_all.max():.4f}] Å")

    print(f"\n  Energies E: shape {E_all.shape}")
    for s in range(min(N_STATES_TO_USE + 1, E_all.shape[1])):
        label = "S0" if s == 0 else f"S{s}"
        print(f"    {label}: [{E_all[:, s].min():.4f}, {E_all[:, s].max():.4f}] eV")

    if antisym_errors:
        max_err = max(e for _, e in antisym_errors)
        print(
            f"\n  Antisymmetry violations: {len(antisym_errors)} "
            f"(max error: {max_err:.2e})"
        )

    print(f"\n  Raw NACT statistics:")
    print(
        f"    {'Pair':>10s} | {'min':>12s} | {'max':>12s} | "
        f"{'mean':>12s} | {'std':>12s}"
    )
    print(f"    {'-'*10}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}")

    for key in pair_keys:
        i, j = key.split("_")
        v = nact_arrays[key]
        print(
            f"    S{i}-S{j:>2s}    | {v.min():>12.6f} | {v.max():>12.6f} | "
            f"{v.mean():>12.6f} | {v.std():>12.6f}"
        )

    print(f"\n  Scaled NACT (dENACT) statistics:")
    print(
        f"    {'Pair':>10s} | {'min':>12s} | {'max':>12s} | "
        f"{'mean':>12s} | {'std':>12s}"
    )
    print(f"    {'-'*10}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}")

    for key in pair_keys:
        i, j = key.split("_")
        v = denact_arrays[key]
        print(
            f"    S{i}-S{j:>2s}    | {v.min():>12.6f} | {v.max():>12.6f} | "
            f"{v.mean():>12.6f} | {v.std():>12.6f}"
        )

    # ========================================================================
    # Save data
    # ========================================================================
    print("\n" + "=" * 80)
    print("SAVING DATA")
    print("=" * 80)
    print(f"  Output directory: {OUTPUT_DIR}")

    # Z — tiled to (n_valid, n_atoms)
    Z_tiled = np.tile(Z_all, (n_valid, 1))
    z_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}Z.npy")
    np.save(z_path, Z_tiled)
    print(f"  Z:  {z_path}  shape={Z_tiled.shape}")

    # R
    r_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}R.npy")
    np.save(r_path, R_all.astype(np.float32))
    print(f"  R:  {r_path}  shape={R_all.shape}")

    # E
    e_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}E.npy")
    np.save(e_path, E_all.astype(np.float32))
    print(f"  E:  {e_path}  shape={E_all.shape}")

    # Raw NACTs
    print(f"\n  Saving {n_pairs} raw NACT files...")
    for key in pair_keys:
        i, j = key.split("_")
        fpath = os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}NACT{i}{j}.npy")
        np.save(fpath, nact_arrays[key].astype(np.float32))

    # Scaled NACTs
    print(f"  Saving {n_pairs} scaled NACT (dENACT) files...")
    for key in pair_keys:
        i, j = key.split("_")
        fpath = os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}dENACT{i}{j}.npy")
        np.save(fpath, denact_arrays[key].astype(np.float32))

    print(f"  Saved {2 * n_pairs} NACT/dENACT files total")

    # ========================================================================
    # Final verification — count output files
    # ========================================================================
    npy_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith(".npy")]
    expected_npy = 3 + 2 * n_pairs  # Z + R + E + NACT + dENACT per pair
    print(f"\n  Output .npy files: {len(npy_files)} (expected {expected_npy})")
    if len(npy_files) != expected_npy:
        print(f"  WARNING: File count mismatch!")

    # ========================================================================
    # Save report
    # ========================================================================
    bad_frames = {
        status: [r["frame_num"] for r in all_results if r["status"] == status]
        for status in status_counts
        if status != "ok"
    }

    report = {
        "base_dir": BASE_DIR,
        "output_dir": OUTPUT_DIR,
        "n_frames_requested": N_FRAMES,
        "n_valid": n_valid,
        "n_states_in_file": N_STATES_IN_FILE,
        "n_states_to_use": N_STATES_TO_USE,
        "n_atoms": N_ATOMS,
        "n_pairs": n_pairs,
        "pairs": [list(p) for p in all_pairs],
        "atomic_numbers": Z_all.tolist(),
        "energy_unit": "eV",
        "coordinate_unit": "Angstrom",
        "n_workers": N_WORKERS,
        "chunk_size": CHUNK_SIZE,
        "extraction_time_s": float(extract_time),
        "status_counts": status_counts,
        "bad_frames": {k: v[:50] for k, v in bad_frames.items()},
        "n_antisym_violations": len(antisym_errors),
        "nact_stats": {},
        "denact_stats": {},
    }

    for key in pair_keys:
        i, j = key.split("_")
        report["nact_stats"][f"NACT{i}{j}"] = {
            "min": float(nact_arrays[key].min()),
            "max": float(nact_arrays[key].max()),
            "mean": float(nact_arrays[key].mean()),
            "std": float(nact_arrays[key].std()),
        }
        report["denact_stats"][f"dENACT{i}{j}"] = {
            "min": float(denact_arrays[key].min()),
            "max": float(denact_arrays[key].max()),
            "mean": float(denact_arrays[key].mean()),
            "std": float(denact_arrays[key].std()),
        }

    report_path = os.path.join(OUTPUT_DIR, "extraction_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report: {report_path}")

    # ========================================================================
    # Summary
    # ========================================================================
    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print("EXTRACTION COMPLETE")
    print("=" * 80)
    print(f"  Frames processed:  {N_FRAMES}")
    print(f"  Valid configs:     {n_valid}")
    print(f"  States used:       S1..S{N_STATES_TO_USE} ({n_pairs} pairs)")
    print(f"  Output files:      {len(npy_files)} .npy + report")
    print(
        f"  Total time:        {total_time:.1f}s "
        f"({N_FRAMES / total_time:.0f} frames/s)"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()

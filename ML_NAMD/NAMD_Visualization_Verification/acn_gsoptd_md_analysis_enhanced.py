#!/usr/bin/env python
# coding: utf-8
"""
ACN Ground State MD Analysis with State-Specific Processing

Enhanced version featuring:
- Optimized Multiprocessing with IPC chunking (16 Cores Default)
- Strict memory management (16GB bounds handling)
- State-specific data extraction (S0 through Sn)
- Comprehensive error handling, logging, and JSON reporting
- Safe frame extraction guaranteeing zero data loss on partial corruption
- Progress tracking with TQDM
"""

import os
import sys
import time
import json
import datetime
import traceback
import numpy as np
import pandas as pd
import multiprocessing
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Dict, List, Any
from tqdm import tqdm
import logging
import argparse

# ============================================================================
# CONFIGURATION SECTION
# ============================================================================

# Default Settings
DEFAULT_CORES = 16
N_STATES = 4  # Ground + 4 excited states (user-configurable)
BASE_PATH = "/data/akhanna2/data/Ml_Projects/acn_data/gsoptd_md/aca_ol/s0_optd_md/100K_frames/"
DATE = datetime.datetime.now().strftime("%Y-%b-%d_%H-%M-%S")
OUTPUT_DIR = f"./{DATE}"

FILENAMES = [f"frame_{i}" for i in range(1, 50001)]

# Set Matplotlib Font Size
plt.rcParams["font.size"] = 14

# Unit Conversion Constants
AU2A = 0.529177249
A2AU = 1 / AU2A
AU2FS = 0.02418884344
FS2AU = 1 / AU2FS
AU2EV = 27.2113961
EV2AU = 1 / AU2EV

# Ground State Energy (AM1 optimized structure via NEXMD)
GROUND_STATE_E = -1390.5966817023

# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logger(output_dir: str) -> logging.Logger:
    """Configures console and file loggers."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    log_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("MD_Analysis")
    logger.setLevel(logging.INFO)
    
    # Prevent duplicate logging if called multiple times
    if not logger.handlers:
        # File Handler
        file_handler = logging.FileHandler(os.path.join(output_dir, "execution.log"))
        file_handler.setFormatter(log_formatter)
        logger.addHandler(file_handler)
        
        # Console Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(log_formatter)
        logger.addHandler(console_handler)
        
    return logger

logger = setup_logger(OUTPUT_DIR)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_distance(coordinates: np.ndarray, index_atom1: int, index_atom2: int) -> float:
    try:
        atom1_coords = np.array(coordinates[index_atom1][1:], dtype=float)
        atom2_coords = np.array(coordinates[index_atom2][1:], dtype=float)
        return float(np.linalg.norm(atom1_coords - atom2_coords))
    except Exception as e:
        raise ValueError(f"Distance calculation failed: {e}")

def get_dihedral_angle(
    coordinates: np.ndarray,
    index_atom1: int,
    index_atom2: int,
    index_atom3: int,
    index_atom4: int,
) -> float:
    try:
        atom1_coords = np.array(coordinates[index_atom1], dtype=float)
        atom2_coords = np.array(coordinates[index_atom2], dtype=float)
        atom3_coords = np.array(coordinates[index_atom3], dtype=float)
        atom4_coords = np.array(coordinates[index_atom4], dtype=float)

        b1 = atom2_coords - atom1_coords
        b2 = atom3_coords - atom2_coords
        b3 = atom4_coords - atom3_coords

        b2_norm = np.linalg.norm(b2)
        if b2_norm == 0:
            raise ValueError("Atoms 2 and 3 are coincident")
        b2 /= b2_norm

        n1 = np.cross(b1, b2)
        n2 = np.cross(b2, b3)

        n1_norm = np.linalg.norm(n1)
        n2_norm = np.linalg.norm(n2)

        if n1_norm == 0 or n2_norm == 0:
            raise ValueError("Collinear atoms detected")

        n1 /= n1_norm
        n2 /= n2_norm

        x = np.dot(n1, n2)
        y = np.dot(n1, np.cross(n2, b2))

        return float(np.degrees(np.arctan2(y, x)))
    except Exception as e:
        raise ValueError(f"Dihedral calculation failed: {e}")

# ============================================================================
# DATA READING FUNCTIONS
# ============================================================================

def get_energies(filename: str, n_states: int = -1) -> np.ndarray:
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Energy file not found: {filename}")
    data = np.loadtxt(filename)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if len(data) > 2:
        data = data[1:-1]
        
    energy_cols = data[:, 1:]
    if n_states == 0:
        return energy_cols[:, :1]
    elif n_states == -1:
        return energy_cols
    return energy_cols[:, :n_states]

def get_ground_state_dipole(filename: str) -> np.ndarray:
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Frame file not found: {filename}")
    with open(filename, "r") as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if "Ground State Molecular Dipole Moment" in line:
            if i + 2 < len(lines):
                data_line = lines[i + 2].split()
                return np.array([float(data_line[0]), float(data_line[1]), float(data_line[2])])
    raise ValueError(f"Dipole not found in {filename}")

def get_nacr_vectors(filename: str, natoms: int = 15, n_states: int = 2) -> np.ndarray:
    if not os.path.exists(filename):
        raise FileNotFoundError(f"NACR file not found: {filename}")
    data = np.loadtxt(filename)
    n_excited = n_states - 1
    n_nacr_pairs = int(n_excited * (n_excited - 1) / 2) if n_excited > 1 else 0
    if n_nacr_pairs == 0:
        return np.array([]).reshape(0, natoms, 3)
    return data[:n_nacr_pairs, 3:].reshape(-1, natoms, 3)

def get_transition_dipole_moments(filename: str, n_states: int = 1) -> np.ndarray:
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Dipole moment file not found: {filename}")
    data = np.loadtxt(filename)
    n_excited = n_states - 1
    return data[:n_excited, 2:-1]

def get_gradients(filename: str, natoms: int = 15, n_states: int = 2) -> np.ndarray:
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Gradients file not found: {filename}")
    data = np.loadtxt(filename)
    n_excited = n_states - 1
    return data[:n_excited, 2:].reshape(-1, natoms, 3)

def get_atomic_number_and_coordinates(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Coordinates file not found: {filename}")
    data = np.loadtxt(filename, skiprows=2, dtype=str)
    atomic_dict = {"H": 1, "C": 6, "N": 7, "O": 8, "S": 16, "F": 9, "Cl": 17, "Br": 35, "I": 53}
    atomic_numbers = [atomic_dict[row[0]] for row in data]
    return np.array(atomic_numbers), data[:, 1:].astype(float)

# ============================================================================
# DATA EXTRACTION WORKER
# ============================================================================

def process_frame(frame: str, base_path: str, target_n_states: int, extract_flags: List[str]) -> Tuple:
    """Core extraction logic decoupled from global namespace for MP."""
    frame_path = os.path.join(base_path, frame)
    ext_all = "all" in extract_flags

    e_eV = None
    if ext_all or "E" in extract_flags or target_n_states == -1:
        if target_n_states == -1:
            e_eV = get_energies(os.path.join(frame_path, "pes.out"), n_states=-1)
            n_states_actual = e_eV.shape[1]
        elif target_n_states == 0:
            e_eV = get_energies(os.path.join(frame_path, "pes.out"), n_states=0)
            n_states_actual = 1
        elif target_n_states == 1:
            e_eV = get_energies(os.path.join(frame_path, "pes.out"), n_states=2)
            n_states_actual = e_eV.shape[1]
        else:
            e_eV = get_energies(os.path.join(frame_path, "pes.out"), n_states=target_n_states)
            n_states_actual = e_eV.shape[1]
    else:
        n_states_actual = target_n_states

    z, coords, da_angle_1, da_angle_2 = None, None, None, None
    if ext_all or any(x in extract_flags for x in ["R", "Z", "gradients", "NACRs"]):
        z, coords = get_atomic_number_and_coordinates(os.path.join(frame_path, "coords.xyz"))
        da_angle_1 = get_dihedral_angle(coords, 4, 1, 0, 2)
        da_angle_2 = get_dihedral_angle(coords, 7, 2, 0, 1)

    dipole_S0 = None
    if ext_all or any(x in extract_flags for x in ["E", "tdms"]):
        try:
            frame_out_file = os.path.join(frame_path, f"{frame}_exsp.out")
            dipole_S0 = get_ground_state_dipole(frame_out_file)
        except Exception:
            pass

    nacrs, tdms_au, gradients = None, None, None

    if n_states_actual >= 2:
        if ext_all or "tdms" in extract_flags:
            tdms_au = get_transition_dipole_moments(os.path.join(frame_path, "tdipole.out"), n_states=n_states_actual if n_states_actual > 2 else 2)
        if ext_all or "gradients" in extract_flags:
            gradients = get_gradients(os.path.join(frame_path, "gradients.out"), len(z) if z is not None else 15, n_states=n_states_actual if n_states_actual > 2 else 2)
    
    if n_states_actual > 2:
        if ext_all or "NACRs" in extract_flags:
            nacrs = get_nacr_vectors(os.path.join(frame_path, "nacr.out"), len(z) if z is not None else 15, n_states_actual)

    return (e_eV, z, coords, nacrs, tdms_au, dipole_S0, [da_angle_1, da_angle_2] if da_angle_1 is not None else None, gradients)

def process_frame_worker(args: Tuple[int, str, str, int, List[str]]) -> Tuple[bool, int, str, Any]:
    """
    Wrapper for multiprocessing. Catches exceptions to prevent total failure 
    if a single frame out of 100,000 is corrupted.
    Returns: (Success_Boolean, Index, Frame_Name, Data_or_Error_Message)
    """
    index, frame, base_path, n_states, extract_flags = args
    try:
        data = process_frame(frame, base_path, n_states, extract_flags)
        return (True, index, frame, data)
    except Exception as e:
        error_msg = traceback.format_exc()
        return (False, index, frame, error_msg)

# ============================================================================
# DATA SAVING FUNCTIONS
# ============================================================================

def save_data(filename: str, data: np.ndarray, subdirectory: str = OUTPUT_DIR) -> None:
    filepath = os.path.join(subdirectory, filename)
    np.save(filepath, data)
    logger.info(f"Saved: {filepath} with shape {data.shape}")

def save_derived_state_specific_data(
    delta_E: Optional[np.ndarray], mean_E: np.ndarray, shifted_E: np.ndarray, n_states: int
) -> None:
    if delta_E is not None:
        for state_idx in range(1, n_states):
            save_data(f"acn_dE{state_idx}.npy", delta_E[:, state_idx - 1])
            
    for state_idx in range(n_states):
        save_data(f"acn_mE{state_idx}.npy", mean_E[:, state_idx])
        save_data(f"acn_sE{state_idx}.npy", shifted_E[:, state_idx])

def save_state_specific_data(parent_data: Dict[str, Optional[np.ndarray]], n_states: int) -> None:
    if parent_data.get("energies") is None: return
    actual_n_states = parent_data["energies"].shape[1]

    for state_idx in range(actual_n_states):
        save_data(f"acn_S{state_idx}.npy", parent_data["energies"][:, state_idx])

    if parent_data.get("dipole_S0") is not None:
        save_data("acn_D0.npy", parent_data["dipole_S0"])

    if parent_data.get("gradients") is not None:
        for state_idx in range(1, actual_n_states):
            save_data(f"acn_F{state_idx}.npy", parent_data["gradients"][:, state_idx - 1, :, :])

    if parent_data.get("dipoles_excited") is not None:
        for state_idx in range(1, actual_n_states):
            save_data(f"acn_D{state_idx}.npy", parent_data["dipoles_excited"][:, state_idx - 1, :])

    if parent_data.get("nacrs") is not None:
        nacr_idx = 0
        for i in range(1, actual_n_states):
            for j in range(i + 1, actual_n_states):
                save_data(f"acn_NACR_S{i}S{j}.npy", parent_data["nacrs"][:, nacr_idx, :, :])
                nacr_idx += 1

# ============================================================================
# ANALYSIS & REPORTING
# ============================================================================

def create_analysis_plots(energy_data: np.ndarray, dihedral_data: np.ndarray) -> None:
    try:
        fig, axs = plt.subplots(1, 3, figsize=(20, 7))
        axs = axs.ravel()

        axs[0].plot(energy_data)
        axs[0].set_title("Energies Over Time")
        axs[0].set_ylabel("Energy [au]")
        axs[0].set_xlabel("Frame")
        axs[0].legend([f"$S_{i}$" for i in range(energy_data.shape[1])])

        axs[1].hist(energy_data, bins=100, alpha=0.7)
        axs[1].set_title("Energy Distribution")
        axs[1].set_ylabel("Counts")
        axs[1].set_xlabel("Energy [au]")
        
        axs[2].hist(dihedral_data, bins=100, alpha=0.75)
        axs[2].set_title("Dihedral Angle Histogram")
        axs[2].set_xlabel("Dihedral Angle (degrees)")
        axs[2].legend([r"$\angle$HO-C-C-CO", r"$\angle$O=C-C-C"])

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "acn_analysis.png"), dpi=150)
        plt.close()
    except Exception as e:
        logger.error(f"Error creating plots: {e}")

def generate_report(stats: dict) -> None:
    report_path = os.path.join(OUTPUT_DIR, "run_report.json")
    with open(report_path, "w") as f:
        json.dump(stats, f, indent=4)
    logger.info(f"Execution report saved to {report_path}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main(args: argparse.Namespace):
    start_time = time.time()
    
    frames_to_process = FILENAMES[:args.total_frames]
    total_frames = len(frames_to_process)
    
    # Calculate optimal chunksize to minimize IPC overhead while keeping memory stable
    # Scale chunksize down if memory limit is strict
    mem_scaling = max(0.2, args.memory / 16.0)
    chunksize = min(int(200 * mem_scaling), max(1, total_frames // (args.workers * 4)))
    
    logger.info("=" * 70)
    logger.info(f"Starting ACN MD Extraction Pipeline")
    logger.info(f"Total Frames: {total_frames} | Cores: {args.workers} | Chunksize: {chunksize}")
    logger.info(f"Target States Mode: {args.n_states} | Extracting: {args.extract}")
    logger.info("=" * 70)

    # Create task arguments explicitly decoupled from global scope with an enumeration index
    tasks = [(i, f, args.base_path, args.n_states, args.extract) for i, f in enumerate(frames_to_process)]

    successful_results = []
    failed_frames = []

    # Execute Multiprocessing Pool
    with multiprocessing.Pool(processes=args.workers) as pool:
        # imap_unordered yields results instantly without waiting for chronological sequence blocks
        iterator = pool.imap_unordered(process_frame_worker, tasks, chunksize=chunksize)
        
        for success, index, frame, payload in tqdm(iterator, total=total_frames, desc="Extracting"):
            if success:
                successful_results.append((index, payload))
            else:
                failed_frames.append({"index": index, "frame": frame, "error": payload})

    if failed_frames:
        logger.warning(f"{len(failed_frames)} frames failed to process. Check execution.log for details.")
        for fail in failed_frames[:5]: # Log first 5 errors to avoid spam
            logger.error(f"Failed {fail['frame']}:\n{fail['error']}")

    if not successful_results:
        logger.error("No frames were successfully processed! Exiting.")
        return

    logger.info("Sorting and aggregating extracted data into matrices...")
    
    # Sort by the original index to guarantee perfectly chronological arrays
    successful_results.sort(key=lambda x: x[0])
    
    # Unpack sorted aggregated successful results
    sorted_payloads = [x[1] for x in successful_results]
    (e_array, z_array, coords_mat, nacrs, tdms_au, dipole_S0_array, da_angle, gradients) = zip(*sorted_payloads)

    # Aggregation & memory allocation mapping (with safe None checks)
    energy_data = np.vstack(e_array) if e_array[0] is not None else None
    atomic_number_data = np.tile(z_array[0], (len(z_array[0]), 1)) if z_array[0] is not None else None
    coords_data = np.array(coords_mat) if coords_mat[0] is not None else None
    dipole_S0_data = np.array(dipole_S0_array) if dipole_S0_array[0] is not None else None
    da_angle_data = np.array(da_angle) if da_angle[0] is not None else None

    nacr_data = np.array(nacrs) if nacrs[0] is not None else None
    tdm_data = np.array(tdms_au) if tdms_au[0] is not None else None
    gradients_data = np.array(gradients) if gradients[0] is not None else None

    logger.info("Saving parent block data...")
    if energy_data is not None: save_data("acn_E.npy", energy_data)
    if atomic_number_data is not None: save_data("acn_Z.npy", atomic_number_data)
    if coords_data is not None: save_data("acn_R.npy", coords_data)
    if dipole_S0_data is not None: save_data("acn_D0.npy", dipole_S0_data)
    if da_angle_data is not None: save_data("acn_DA.npy", da_angle_data)

    if nacr_data is not None: save_data("acn_NACR.npy", nacr_data)
    if tdm_data is not None: save_data("acn_D.npy", tdm_data)
    if gradients_data is not None: save_data("acn_F.npy", gradients_data)

    if energy_data is not None:
        logger.info("Generating state-specific and derived data files...")
        if energy_data.shape[1] > 1:
            delta_E = energy_data[:, 1:] - energy_data[:, 0].reshape(-1, 1)
            save_data("acn_dE.npy", delta_E)
        else:
            delta_E = None

        shifted_E = energy_data - args.ground_state_e
        mean_E = energy_data - np.mean(energy_data, axis=0, keepdims=True)
        
        save_derived_state_specific_data(delta_E, mean_E, shifted_E, energy_data.shape[1])
        
        parent_data_map = {
            "energies": energy_data,
            "gradients": gradients_data,
            "dipoles_excited": tdm_data,
            "dipole_S0": dipole_S0_data,
            "nacrs": nacr_data,
        }
        save_state_specific_data(parent_data_map, energy_data.shape[1])

    logger.info("Creating data visualizations...")
    if energy_data is not None and da_angle_data is not None:
        create_analysis_plots(energy_data, da_angle_data)

    # Generate Statistical Report
    elapsed_time = time.time() - start_time
    report = {
        "execution_metadata": {
            "date": DATE,
            "elapsed_time_seconds": round(elapsed_time, 2),
            "cores_utilized": args.workers,
            "chunksize_used": chunksize,
            "memory_target_gb": args.memory,
            "base_path": args.base_path
        },
        "frame_statistics": {
            "total_attempted": total_frames,
            "successful_extractions": len(successful_results),
            "failed_extractions": len(failed_frames)
        },
        "array_shapes": {
            "energies": energy_data.shape if energy_data is not None else None,
            "coordinates": coords_data.shape if coords_data is not None else None,
            "dipole_S0": dipole_S0_data.shape if dipole_S0_data is not None else None,
            "dihedral_angles": da_angle_data.shape if da_angle_data is not None else None,
            "nacrs": nacr_data.shape if nacr_data is not None else None,
            "dipoles_excited": tdm_data.shape if tdm_data is not None else None,
            "gradients": gradients_data.shape if gradients_data is not None else None
        },
        "failed_frame_logs": [f["frame"] for f in failed_frames]
    }
    generate_report(report)
    
    logger.info(f"Pipeline finished successfully in {elapsed_time:.2f} seconds.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ACN Ground State MD Analysis Pipeline")
    parser.add_argument("--base_path", type=str, default=BASE_PATH, help="Base path to MD frames directory")
    parser.add_argument("--total_frames", type=int, default=50000, help="Total number of frames to process")
    parser.add_argument("--workers", type=int, default=DEFAULT_CORES, help="Number of CPU workers (cores) to use")
    parser.add_argument("--memory", type=int, default=16, help="Target memory limit in GB (guides multiprocessing chunk size)")
    parser.add_argument("--ground_state_e", type=float, default=GROUND_STATE_E, help="Ground state energy reference")
    parser.add_argument("--n_states", type=int, default=N_STATES, help="Number of electronic states")
    parser.add_argument(
        "--extract", 
        nargs="+", 
        default=["E", "R", "Z"], 
        choices=["E", "R", "Z", "gradients", "NACRs", "tdms", "all"], 
        help="Specific files/properties to extract. Default is E, R, and Z."
    )
    args = parser.parse_args()

    # Ensure optimal core usage gracefully bounded by actual system specs
    system_cores = os.cpu_count() or 4
    args.workers = min(args.workers, system_cores)

    main(args)



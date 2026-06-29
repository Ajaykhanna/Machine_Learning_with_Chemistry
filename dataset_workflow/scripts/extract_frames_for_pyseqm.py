#!/usr/bin/env python
# coding: utf-8
"""
Extract Atomic Numbers (Z), Coordinates (R), and Velocities (V) from XYZ Frames

This script extracts atomic numbers, Cartesian coordinates, and velocities from 
individual frame directories (frame_1 through frame_N) with optional
sorting functionality.

Features:
- Parallel processing with multiprocessing
- Progress tracking with TQDM
- Comprehensive error handling and logging
- Optional Z/R/V sorting (descending atomic number)
- Support for frame_*.vel or velocity.out files
- Validation system
"""

import os
import argparse
import datetime
import numpy as np
import multiprocessing
from typing import Tuple, Optional
from tqdm import tqdm
import logging

# ============================================================================
# CONFIGURATION SECTION
# ============================================================================

# Default Settings
BASE_PATH = "./"  # Base path to frame XYZ files (user-configurable)
# Format date with timestamp to ensure a single unique directory per run
DATE = datetime.datetime.now().strftime("%Y-%b-%d_%H-%M-%S")

# Create output directory first so the logger can use it
if not os.path.exists(f"./{DATE}"):
    os.makedirs(f"./{DATE}")

# Configure logging to write to execution.log in the newly created directory 
# as well as outputting to the console
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(f"./{DATE}/execution.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Generate frame filenames (frame_1/frame_1.xyz through frame_100000/frame_100000.xyz)
# (Filenames are now generated dynamically in main() using start_frame and end_frame)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def get_atomic_number_and_coordinates(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reads the atomic numbers and coordinates from a XYZ file.

    Parameters:
    -----------
    filename : str
        Path to the XYZ file containing atomic coordinates

    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        - atomic_numbers: Array of atomic numbers (shape: n_atoms)
        - atomic_coordinates: Array of Cartesian coordinates (shape: n_atoms, 3)
    """
    try:
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Coordinates file not found: {filename}")

        data = np.loadtxt(filename, skiprows=2, dtype=str)

        # Map atomic symbols to atomic numbers
        atomic_symbols = ["H", "C", "N", "O", "S", "F", "Cl", "Br", "I"]
        atomic_numbers_map = [1, 6, 7, 8, 16, 9, 17, 35, 53]
        atomic_dict = dict(zip(atomic_symbols, atomic_numbers_map))

        # Extract atomic numbers
        atomic_numbers = []
        for row in data:
            symbol = row[0]
            if symbol not in atomic_dict:
                raise ValueError(f"Unknown atomic symbol: {symbol}")
            atomic_numbers.append(atomic_dict[symbol])

        # Extract coordinates
        atomic_coordinates = data[:, 1:].astype(float)

        return np.array(atomic_numbers), atomic_coordinates
    except Exception as e:
        logger.error(f"Error reading atomic data from {filename}: {e}")
        raise


def get_velocities(xyz_filename: str) -> np.ndarray:
    """
    Reads the velocities from a .vel or velocity.out file in the same directory.

    Parameters:
    -----------
    xyz_filename : str
        Path to the XYZ file (used to locate the frame directory and frame index)

    Returns:
    --------
    np.ndarray
        Array of Cartesian velocities (shape: n_atoms, 3)
    """
    frame_dir = os.path.dirname(xyz_filename)
    basename = os.path.basename(xyz_filename)
    
    # Attempt to extract frame index from "frame_1.xyz" -> "1"
    try:
        frame_idx_str = basename.split('_')[1].split('.')[0]
    except IndexError:
        frame_idx_str = None

    # Priority 1: frame_*.vel, Priority 2: velocity.out
    vel_path = os.path.join(frame_dir, f"frame_{frame_idx_str}.vel") if frame_idx_str else None
    out_path = os.path.join(frame_dir, "velocity.out")

    if vel_path and os.path.exists(vel_path):
        try:
            # Assuming XYZ-like format: 2 header lines (num atoms, frame title)
            return np.loadtxt(vel_path, skiprows=2, dtype=float)
        except Exception as e:
            logger.error(f"Error reading {vel_path}: {e}")
            raise
    elif os.path.exists(out_path):
        try:
            velocities = []
            in_veloc = False
            with open(out_path, 'r') as f:
                for line in f:
                    if "$VELOC" in line:
                        in_veloc = True
                        continue
                    if "$ENDVELOC" in line:
                        break
                    if in_veloc:
                        parts = line.strip().split()
                        if len(parts) == 3:
                            velocities.append([float(p) for p in parts])
            if not velocities:
                raise ValueError(f"No velocity data found in {out_path} between $VELOC and $ENDVELOC.")
            return np.array(velocities, dtype=float)
        except Exception as e:
            logger.error(f"Error reading {out_path}: {e}")
            raise
    else:
        raise FileNotFoundError(f"No velocity file (frame_*.vel or velocity.out) found for {xyz_filename} in {frame_dir}")


def check_if_sorted_descending(Z: np.ndarray) -> bool:
    """Checks if atomic numbers array is sorted in descending order for all frames."""
    for frame_idx in range(Z.shape[0]):
        frame_Z = Z[frame_idx]
        if not np.all(frame_Z[:-1] >= frame_Z[1:]):
            return False
    return True


def sort_Z_R_V(
    Z: np.ndarray, R: np.ndarray, V: np.ndarray, subdirectory: str = DATE, prefix: str = "molecule"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Sorts atomic numbers in descending order and rearranges coordinates and velocities accordingly.

    Parameters:
    -----------
    Z : np.ndarray
        Atomic numbers array with shape (n_frames, n_atoms)
    R : np.ndarray
        Coordinates array with shape (n_frames, n_atoms, 3)
    V : np.ndarray
        Velocities array with shape (n_frames, n_atoms, 3)
    subdirectory : str
        Subdirectory to save sorted files
    prefix: str
        Prefix for naming output files

    Returns:
    --------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Sorted Z, Sorted R, Sorted V, and sorting indices
    """
    logger.info("Sorting Z, R, and V data by descending atomic number...")

    # Get descending sort indices for each frame
    desc_indices = np.argsort(-Z, axis=1)

    # Sort Z using the indices
    sorted_Z = np.take_along_axis(Z, desc_indices, axis=1)

    # Sort R and V using the same indices (expand to 3D for coordinates/velocities)
    sorted_R = np.take_along_axis(R, desc_indices[:, :, None], axis=1)
    sorted_V = np.take_along_axis(V, desc_indices[:, :, None], axis=1)

    # Save sorted data
    logger.info("Saving sorted data files...")
    save_data(f"sorted_{prefix}_Z.npy", sorted_Z, subdirectory)
    save_data(f"sorted_{prefix}_R.npy", sorted_R, subdirectory)
    save_data(f"sorted_{prefix}_V.npy", sorted_V, subdirectory)
    save_data(f"sorted_{prefix}_Z_indices.npy", desc_indices, subdirectory)

    logger.info(f"Sorted data saved:")
    logger.info(f"  sorted_{prefix}_Z.npy: {sorted_Z.shape}")
    logger.info(f"  sorted_{prefix}_R.npy: {sorted_R.shape}")
    logger.info(f"  sorted_{prefix}_V.npy: {sorted_V.shape}")

    return sorted_Z, sorted_R, sorted_V, desc_indices


# ============================================================================
# DATA EXTRACTION AND PROCESSING
# ============================================================================


def process_frame(frame: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Processes a single frame and extracts atomic numbers, coordinates, and velocities.

    Parameters:
    -----------
    frame : str
        Frame filename (e.g., "frame_1/frame_1.xyz")

    Returns:
    --------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        atomic_numbers, coordinates, velocities
    """
    try:
        frame_path = os.path.join(BASE_PATH, frame)
        z, coords = get_atomic_number_and_coordinates(frame_path)
        vels = get_velocities(frame_path)
        
        # Guard against mismatching atomic rows
        if len(z) != len(vels):
            raise ValueError(f"Atom count mismatch in {frame}: Z({len(z)}) != V({len(vels)})")

        return z, coords, vels
    except Exception as e:
        logger.error(f"Error processing frame {frame}: {e}")
        raise


# ============================================================================
# DATA SAVING FUNCTIONS
# ============================================================================


def save_data(filename: str, data: np.ndarray, subdirectory: str = DATE) -> None:
    """Saves numpy array to .npy file."""
    try:
        filepath = f"{subdirectory}/{filename}"
        np.save(filepath, data)
        logger.info(f"Saved: {filepath} with shape {data.shape}")
    except IOError as e:
        logger.error(f"Error saving {filename}: {e}")
        raise


# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================


def validate_data(
    Z: np.ndarray, R: np.ndarray, V: np.ndarray, filenames: list, num_samples: int = 5
) -> None:
    """
    Validates extracted Z, R, and V data against source files.
    """
    logger.info(f"Validating data with {num_samples} random samples...")

    # Select random frame indices
    num_frames = len(filenames)
    sample_indices = np.random.choice(num_frames, size=num_samples, replace=False)

    all_valid = True
    for idx in sample_indices:
        frame = filenames[idx]
        frame_path = os.path.join(BASE_PATH, frame)

        try:
            # Re-read data from file
            z_file, r_file = get_atomic_number_and_coordinates(frame_path)
            v_file = get_velocities(frame_path)

            # Compare atomic numbers
            if not np.array_equal(Z[idx], z_file):
                logger.error(f"Validation failed for {frame}: Z mismatch")
                all_valid = False
                continue

            # Compare coordinates (with tolerance for floating point errors)
            if not np.allclose(R[idx], r_file, rtol=1e-10, atol=1e-10):
                logger.error(f"Validation failed for {frame}: R mismatch")
                all_valid = False
                continue

            # Compare velocities (with tolerance)
            if not np.allclose(V[idx], v_file, rtol=1e-10, atol=1e-10):
                logger.error(f"Validation failed for {frame}: V mismatch")
                all_valid = False
                continue

            logger.info(f"âœ“ Frame {idx} ({frame}): Validation passed")

        except Exception as e:
            logger.error(f"Validation error for {frame}: {e}")
            all_valid = False

    if all_valid:
        logger.info("âœ“ All validation checks passed!")
    else:
        logger.warning("âš  Some validation checks failed!")


# ============================================================================
# MAIN EXTRACTION PIPELINE
# ============================================================================


def main(
    base_path: str = "./",
    start_frame: int = 1,
    end_frame: int = 100001,
    num_frames: Optional[int] = None,
    validate: bool = True,
    sort_data: bool = False,
    prefix: str = "molecule",
) -> dict:
    """
    Main extraction pipeline for Z, R, and V data from XYZ frames.
    """
    global BASE_PATH
    BASE_PATH = base_path

    logger.info("=" * 70)
    logger.info("Z, R, and V Extraction Pipeline")
    logger.info("=" * 70)
    logger.info(f"Base path: {BASE_PATH}")
    logger.info(f"Output directory: {DATE}/")

    # Determine frames to process dynamically based on start and end frames
    frames_to_process = [f"frame_{i}/frame_{i}.xyz" for i in range(start_frame, end_frame)]

    if num_frames is not None:
        frames_to_process = frames_to_process[:num_frames]

    # Check that at least one frame exists
    first_frame_path = os.path.join(BASE_PATH, frames_to_process[0])
    if not os.path.exists(first_frame_path):
        raise FileNotFoundError(
            f"First frame not found: {first_frame_path}\n"
            f"Please check BASE_PATH and file naming convention."
        )

    # Process frames with multiprocessing and progress bar
    logger.info(f"Processing {len(frames_to_process)} frames...")
    with multiprocessing.Pool() as pool:
        results = list(
            tqdm(
                pool.imap(process_frame, frames_to_process),
                total=len(frames_to_process),
                desc="Extracting Z, R, and V",
            )
        )

    # Unpack results: (atomic_numbers, coordinates, velocities)
    z_array, coords_array, vels_array = zip(*results)

    # Convert to numpy arrays
    atomic_number_data = np.array(z_array)
    coords_data = np.array(coords_array)
    vels_data = np.array(vels_array)

    logger.info("Extracted data shapes:")
    logger.info(f"  Atomic numbers (Z): {atomic_number_data.shape}")
    logger.info(f"  Coordinates (R): {coords_data.shape}")
    logger.info(f"  Velocities (V): {vels_data.shape}")

    # Save parent data files
    logger.info("Saving data files...")
    save_data(f"{prefix}_Z.npy", atomic_number_data)
    save_data(f"{prefix}_R.npy", coords_data)
    save_data(f"{prefix}_V.npy", vels_data)

    # Perform validation if requested
    if validate:
        logger.info("Performing validation...")
        validate_data(
            atomic_number_data,
            coords_data,
            vels_data,
            frames_to_process,
            num_samples=min(5, len(frames_to_process)),
        )

    # Prepare return dictionary
    result_dict = {
        "atomic_numbers": atomic_number_data,
        "coordinates": coords_data,
        "velocities": vels_data,
    }

    # Sort data if requested
    if sort_data:
        logger.info("Checking if Z data is already sorted...")
        if check_if_sorted_descending(atomic_number_data):
            logger.info("âœ“ Z data is already sorted in descending order!")
            logger.info("  Skipping sorting operation.")
        else:
            logger.info("Z data is not sorted. Starting sorting operation...")
            sorted_Z, sorted_R, sorted_V, sort_indices = sort_Z_R_V(
                atomic_number_data, coords_data, vels_data, prefix=prefix
            )
            result_dict["sorted_atomic_numbers"] = sorted_Z
            result_dict["sorted_coordinates"] = sorted_R
            result_dict["sorted_velocities"] = sorted_V
            result_dict["sort_indices"] = sort_indices

    logger.info("=" * 70)
    logger.info(f"Processing complete! Data saved to {DATE}/")
    logger.info("=" * 70)

    return result_dict


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Extract atomic numbers (Z), coordinates (R), and velocities (V) from XYZ frames",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract Z, R, V from all frames in current directory
  python extract_ZRV_from_frames.py

  # Extract using custom prefix (will output prefix_V.npy using the requested molecule prefix)
  python extract_ZRV_from_frames.py --prefix my_mol

  # Extract and sort by descending atomic number
  python extract_ZRV_from_frames.py --sort_ZRV
        """,
    )

    parser.add_argument(
        "--base_path",
        type=str,
        default="./",
        help="Base directory containing frame_*.xyz files (default: current directory)",
    )

    parser.add_argument(
        "--start_frame",
        type=int,
        default=1,
        help="Starting frame number (default: 1)",
    )

    parser.add_argument(
        "--end_frame",
        type=int,
        default=100001,
        help="Ending frame number (exclusive, default: 100001). Reminder: Add +1 to your intended final frame number!",
    )

    parser.add_argument(
        "--num_frames",
        type=int,
        default=None,
        help="Number of frames to process (default: all frames from 1 to 100000)",
    )
    
    parser.add_argument(
        "--prefix",
        type=str,
        default="molecule",
        help="Prefix for saved numpy arrays (e.g., molecule_Z.npy, molecule_V.npy). Default: molecule",
    )

    parser.add_argument(
        "--no_validate",
        action="store_true",
        help="Skip validation step (faster but less safe)",
    )

    # Added --sort_ZR alias for backwards compatibility with any existing scripts you have
    parser.add_argument(
        "--sort_ZRV", "--sort_ZR",
        action="store_true",
        dest="sort_ZRV",
        help="Sort Z by descending atomic number and rearrange R and V accordingly",
    )

    args = parser.parse_args()

    # Run main extraction pipeline
    print("=" * 70)
    print("Z, R, and V Extraction from XYZ Frames")
    print("=" * 70)
    print(f"Base path: {args.base_path}")
    print(f"Frames Range: {args.start_frame} to {args.end_frame - 1} (exclusive limit: {args.end_frame})")
    print(f"Number of frames limit: {args.num_frames if args.num_frames else 'None'}")
    print(f"Output Prefix: {args.prefix}")
    print(f"Validation: {not args.no_validate}")
    print(f"Sort Z, R, and V: {args.sort_ZRV}")
    print("=" * 70)

    extracted_data = main(
        base_path=args.base_path,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        num_frames=args.num_frames,
        validate=not args.no_validate,
        sort_data=args.sort_ZRV,
        prefix=args.prefix
    )

    print("\n" + "=" * 70)
    print("Processing Complete!")
    print("=" * 70)
    print(f"Output directory: {DATE}/")
    print(f"Data keys: {list(extracted_data.keys())}")
    print("=" * 70)




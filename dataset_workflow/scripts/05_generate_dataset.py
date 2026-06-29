#!/usr/bin/env python3
"""
Molecule-Agnostic Multi-State Molecular Dynamics Data Processing Pipeline - PARALLEL VERSION

This script provides parallelized processing of molecular dynamics simulation data
using multiprocessing. It is designed for production use with datasets ranging from
100K to millions of data points.

Key Features:
    - Parallel processing using multiprocessing Pool
    - Configurable number of worker processes (default: 8)
    - Smart chunk sizing to ensure no data loss
    - Data integrity validation after processing
    - Memory-efficient processing with numpy memory mapping
    - Progress tracking with tqdm
    - Comprehensive logging

Processing Steps (parallelized where beneficial):
    1. Load input files (I/O bound - not parallelized)
    2. Process energies (parallelized by sample chunks)
    3. Process dipoles (parallelized by sample chunks)
    4. Process forces (parallelized by sample chunks)
    5. Process NACRs (parallelized by sample chunks)
    6. Validate data integrity
    7. Chunk and save outputs

Usage:
    python 05_generate_dataset.py --input-dir ./data --output-dir ./output --prefix molecule
    python 05_generate_dataset.py --input-dir . --output-dir . --n-workers 16 --prefix molecule
    python 05_generate_dataset.py --input-dir . --output-dir . --n-states 10 --n-workers 8 --prefix molecule

Author: MLIP Data Generation Workflow
Version: 2.0.0 (Parallel)
Date: November 2025
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import multiprocessing as mp
import os
import sys
import time
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

# =============================================================================
# CONSTANTS
# =============================================================================

# Number of electronic states (1 ground + 10 excited)
N_STATES: int = 11

# Default atom count; overwritten after molecule-specific reference forces are loaded
N_ATOMS: int = 30

# Number of NAC couplings between excited states: (N_STATES-1)*(N_STATES-2)//2
N_COUPLINGS: int = 45

# Reference energy for shifted calculations (eV)
OPTIMIZED_ENERGY: float = -2370.6278390611951181

# Reference forces for shifted calculations (N_ATOMS x 3 coordinates)
OPTIMIZED_FORCES: NDArray[np.float64] = np.array(
[[-2.0969337023283019e-03,  2.2960033227181903e-03,  9.5463975440549659e-04],
 [-8.5941417922958535e-04,  1.0764659481724663e-03,  9.7531720812667525e-04],
 [ 9.1921647847888199e-04, -1.9487616914598549e-04,  2.3980601989006308e-04],
 [-2.4662775409283189e-03, -1.0097631474046409e-03,  3.5105742864362591e-04],
 [ 2.0269828029237935e-03,  4.1194554018821883e-03,  2.6521872933687411e-04],
 [-1.8570549838035078e-03, -5.7249049411955966e-05, -1.1623260772954946e-03],
 [ 2.6499330474873961e-03,  1.2639946700550020e-03, -2.2753737648547028e-04],
 [ 1.5249886961568038e-03,  4.2329757912100696e-03, -1.9100520826432089e-04],
 [-4.9329576172161271e-03, -3.2567499072353456e-03, -3.3550409012339912e-04],
 [ 8.9195102781403079e-04, -1.1849552482585679e-03, -2.4335312207773852e-04],
 [ 1.5528676839954292e-03, -9.6138876933116180e-04, -6.1901287188693196e-04],
 [ 1.4781221592690130e-03, -3.8704075128035420e-03,  3.9646193374052053e-04],
 [ 2.8751342780086730e-04, -1.4277049959212640e-03, -2.0941990326094409e-04],
 [-8.1675944616399487e-04, -1.0486492410144432e-03, -6.4520405966903872e-04],
 [ 6.3780004483993125e-04, -1.5265165261174607e-04, -1.5754660240162744e-04],
 [-2.4157660592924457e-05,  9.3993934636300847e-04, -4.3069787476104362e-05],
 [-2.5652487761234667e-04, -4.2058093105354355e-05,  4.6020544822585486e-04],
 [ 3.0472568345288331e-04,  1.8462839422251492e-04,  2.2876209263234600e-04],
 [ 4.2760936649094416e-04,  1.0125198912370528e-03,  2.5803942701746550e-04],
 [ 1.2027020214778767e-04,  8.9780752692197874e-05,  4.7196336507576795e-05],
 [ 6.6025157145511992e-04, -9.9626166005831185e-04,  4.8409012823810455e-04],
 [-2.0157563091990904e-04, -2.2504893180164132e-04, -1.4495517834039046e-03],
 [-2.8911825707269093e-04, -6.9444452148271964e-04,  2.9906655384218036e-04],
 [-3.7309639908652409e-04,  8.4520295170645993e-05,  2.1068421031666951e-05],
 [ 4.3914340455759870e-05, -9.6639251978102372e-05,  3.0252780955454528e-04],
 [ 2.3508272004433119e-04, -3.8810911267450621e-05,  8.6887073760588684e-05],
 [ 2.3856221562084485e-04, -5.0150192221320999e-05, -6.4362302605591726e-05],
 [ 1.7407882652170636e-04,  7.5254413301716061e-06, -2.2451180002831805e-05]],
    dtype=np.float64,
)

# Keep downstream shape checks molecule-specific.
N_ATOMS: int = int(OPTIMIZED_FORCES.shape[0])

# Chunking configuration for output
CHOP_START: int = 10000
CHOP_END: int = 100001  # exclusive (max chunk = 100K)
CHOP_STEP: int = 10000

# Parallel processing configuration
DEFAULT_N_WORKERS: int = 8
# Minimum samples per worker chunk (to avoid overhead for small chunks)
MIN_CHUNK_SIZE: int = 1000
# Maximum samples per worker chunk (for memory efficiency)
MAX_CHUNK_SIZE: int = 25000

# Input file specifications (required files - excluding configurable coordinate files)
INPUT_FILES: Dict[str, Tuple[str | int, ...]] = {
    "energies_all_states.npy": ("N", N_STATES),
    "total_energy.npy": ("N", 1),
    "transition_dipoles.npy": ("N", N_STATES - 1, 3),
    "gs_dipoles.npy": ("N", 3),
    "forces_all_states.npy": ("N", N_STATES, N_ATOMS, 3),
    "nacrs_all_states.npy": (N_COUPLINGS, "N", N_ATOMS, 3),
}

# Default coordinate input files (can be overridden via CLI)
DEFAULT_COORDS_R_FILE: str = "molecule_R.npy"  # Atomic coordinates (N, N_ATOMS, 3)
DEFAULT_COORDS_Z_FILE: str = "molecule_Z.npy"  # Atomic numbers (N, N_ATOMS)

# Optional file for atomic reordering (if forces/NACRs are sorted by Z)
OPTIONAL_SORTED_Z_FILE: str = "sorted_Z_indices.npy"


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class ParallelConfig:
    """Configuration for the parallel data processing pipeline.

    Attributes:
        input_dir: Directory containing input .npy files
        output_dir: Base directory for output files
        log_file: Path to log file
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        chop_start: First chunk size for output
        chop_end: Maximum chunk size (exclusive)
        chop_step: Increment between chunk sizes
        split_nacr: Whether to save individual NACR state-pair files
        n_states: Number of excited states to process
        n_workers: Number of parallel worker processes
        validate: Whether to run data integrity validation
    """

    input_dir: Path
    output_dir: Path
    log_file: Path
    log_level: str = "INFO"
    chop_start: int = CHOP_START
    chop_end: int = CHOP_END
    chop_step: int = CHOP_STEP
    split_nacr: bool = False
    n_states: int = N_STATES - 1  # Default: 10 excited states (11 total)
    n_workers: int = DEFAULT_N_WORKERS
    validate: bool = True
    sorted_z_file: Optional[Path] = None  # Optional file for atomic reordering
    coords_r_file: str = DEFAULT_COORDS_R_FILE  # Input file for atomic coordinates
    coords_z_file: str = DEFAULT_COORDS_Z_FILE  # Input file for atomic numbers
    output_prefix: str = "molecule"
    reference_energy_file: Optional[Path] = None
    reference_forces_file: Optional[Path] = None

    @property
    def total_states(self) -> int:
        """Total number of states (ground + excited)."""
        return self.n_states + 1

    @property
    def n_couplings(self) -> int:
        """Number of NACR couplings for the selected states."""
        return self.n_states * (self.n_states - 1) // 2

    def get_chop_sizes(self) -> NDArray[np.int64]:
        """Get array of chunk sizes for output."""
        sizes = np.arange(self.chop_start, self.chop_end, self.chop_step)
        max_size = self.chop_end - 1 if self.chop_end > 0 else 0
        if len(sizes) == 0 or sizes[-1] < max_size:
            sizes = np.append(sizes, max_size)
        return sizes

    def get_chunk_dirs(self) -> List[str]:
        """Get list of chunk directory names."""
        return [f"{size // 1000}K" for size in self.get_chop_sizes()]


@dataclass
class ChunkInfo:
    """Information about a processing chunk.

    Attributes:
        chunk_id: Unique identifier for the chunk
        start_idx: Starting sample index (inclusive)
        end_idx: Ending sample index (exclusive)
        n_samples: Number of samples in this chunk
    """

    chunk_id: int
    start_idx: int
    end_idx: int

    @property
    def n_samples(self) -> int:
        return self.end_idx - self.start_idx


@dataclass
class ValidationResult:
    """Result of data integrity validation.

    Attributes:
        is_valid: Whether all validations passed
        total_samples: Expected total sample count
        actual_samples: Actual sample count in output
        checksum_match: Whether input/output checksums match
        shape_valid: Whether all output shapes are correct
        errors: List of validation error messages
    """

    is_valid: bool
    total_samples: int
    actual_samples: int
    checksum_match: bool
    shape_valid: bool
    errors: List[str] = field(default_factory=list)


# =============================================================================
# LOGGING SETUP
# =============================================================================


def setup_logging(log_file: Path, log_level: str) -> logging.Logger:
    """Configure dual logging to console and file."""
    logger = logging.getLogger("molecule_dataset_processor")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    # File handler
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(funcName)s - %(message)s"
    )
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_format = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    return logger


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def print_header(logger: logging.Logger) -> None:
    """Print the pipeline header."""
    header = """
================================================================================
Molecular Dataset Processing Pipeline - PARALLEL VERSION
================================================================================
"""
    logger.info(header)


def print_table(
    title: str, headers: List[str], rows: List[List[str]], logger: logging.Logger
) -> None:
    """Print a formatted table."""
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))

    separator = "â”œ" + "â”¼".join("â”€" * (w + 2) for w in col_widths) + "â”¤"
    top_border = "â”Œ" + "â”¬".join("â”€" * (w + 2) for w in col_widths) + "â”"
    bottom_border = "â””" + "â”´".join("â”€" * (w + 2) for w in col_widths) + "â”˜"

    def format_row(cells: List[str]) -> str:
        return "â”‚" + "â”‚".join(f" {c:<{w}} " for c, w in zip(cells, col_widths)) + "â”‚"

    logger.info(f"\n[{title}]")
    logger.info(top_border)
    logger.info(format_row(headers))
    logger.info(separator)
    for row in rows:
        logger.info(format_row(row))
    logger.info(bottom_border)


def format_shape(shape: Tuple[int, ...]) -> str:
    """Format a shape tuple as a string."""
    return f"({', '.join(str(s) for s in shape)})"


def format_time(seconds: float) -> str:
    """Format seconds as a human-readable string."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def compute_array_checksum(arr: NDArray) -> str:
    """Compute MD5 checksum of an array for validation."""
    return hashlib.md5(arr.tobytes()).hexdigest()


def get_nacr_state_pairs(n_excited_states: int = N_STATES - 1) -> List[Tuple[int, int]]:
    """Generate list of NACR state pairs (i, j) where 1 <= i < j <= n_excited_states."""
    pairs = []
    total_states = n_excited_states + 1
    for i in range(1, total_states):
        for j in range(i + 1, total_states):
            pairs.append((i, j))
    return pairs


def get_nacr_indices_for_subset(n_excited_states: int) -> List[int]:
    """Get indices of NACR couplings to extract for a subset of states."""
    full_pairs = get_nacr_state_pairs(N_STATES - 1)
    full_pair_to_idx = {pair: idx for idx, pair in enumerate(full_pairs)}
    subset_pairs = get_nacr_state_pairs(n_excited_states)
    indices = [full_pair_to_idx[pair] for pair in subset_pairs]
    return indices


# =============================================================================
# CHUNK MANAGEMENT
# =============================================================================


def calculate_optimal_chunks(
    n_samples: int,
    n_workers: int,
    min_chunk: int = MIN_CHUNK_SIZE,
    max_chunk: int = MAX_CHUNK_SIZE,
) -> List[ChunkInfo]:
    """Calculate optimal chunk distribution ensuring no data loss.

    This function ensures:
    1. All samples are covered (no gaps)
    2. Chunks are evenly distributed across workers
    3. Any remainder samples are distributed to early chunks
    4. Chunk sizes respect min/max constraints

    Args:
        n_samples: Total number of samples
        n_workers: Number of worker processes
        min_chunk: Minimum samples per chunk
        max_chunk: Maximum samples per chunk

    Returns:
        List of ChunkInfo objects covering all samples
    """
    # Calculate base chunk size
    base_chunk_size = max(min_chunk, n_samples // n_workers)
    base_chunk_size = min(base_chunk_size, max_chunk)

    # Calculate number of chunks needed
    n_chunks = max(1, (n_samples + base_chunk_size - 1) // base_chunk_size)

    # Recalculate chunk size to distribute evenly
    chunk_size = n_samples // n_chunks
    remainder = n_samples % n_chunks

    chunks = []
    current_idx = 0

    for i in range(n_chunks):
        # Add one extra sample to first 'remainder' chunks
        extra = 1 if i < remainder else 0
        this_chunk_size = chunk_size + extra
        end_idx = current_idx + this_chunk_size

        chunks.append(
            ChunkInfo(
                chunk_id=i,
                start_idx=current_idx,
                end_idx=end_idx,
            )
        )
        current_idx = end_idx

    # Verify no data loss
    total_covered = sum(c.n_samples for c in chunks)
    assert (
        total_covered == n_samples
    ), f"Data loss detected: {total_covered} != {n_samples}"

    # Verify no gaps
    for i in range(1, len(chunks)):
        assert (
            chunks[i].start_idx == chunks[i - 1].end_idx
        ), f"Gap detected at chunk {i}"

    return chunks


# =============================================================================
# PARALLEL PROCESSING WORKER FUNCTIONS
# =============================================================================


def process_energies_chunk(
    chunk: ChunkInfo,
    energies: NDArray[np.float64],
    total_energy: NDArray[np.float64],
    n_total_states: int,
    optimized_energy: float,
) -> Tuple[int, NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Process energy data for a single chunk.

    Returns:
        Tuple of (chunk_id, dataset_S, dataset_sE, dataset_dE) for this chunk
    """
    start, end = chunk.start_idx, chunk.end_idx

    # Slice to selected states
    e_chunk = energies[start:end, :n_total_states]
    te_chunk = total_energy[start:end]

    # Raw energies
    dataset_S = e_chunk.copy()

    # Shifted energies
    dataset_sE = e_chunk - optimized_energy

    # Delta energies
    dataset_dE = e_chunk - te_chunk

    return (chunk.chunk_id, dataset_S, dataset_sE, dataset_dE)


def process_dipoles_chunk(
    chunk: ChunkInfo,
    gs_dipoles: NDArray[np.float64],
    transition_dipoles: NDArray[np.float64],
    n_excited_states: int,
) -> Tuple[int, NDArray[np.float64]]:
    """Process dipole data for a single chunk.

    Returns:
        Tuple of (chunk_id, dataset_D) for this chunk
    """
    start, end = chunk.start_idx, chunk.end_idx

    # Ground state dipoles: (chunk_size, 3) -> (chunk_size, 1, 3)
    gs_expanded = gs_dipoles[start:end, np.newaxis, :]

    # Transition dipoles: slice to selected excited states
    trans_chunk = transition_dipoles[start:end, :n_excited_states, :]

    # Concatenate
    dataset_D = np.concatenate([gs_expanded, trans_chunk], axis=1)

    return (chunk.chunk_id, dataset_D)


def process_forces_chunk(
    chunk: ChunkInfo,
    forces: NDArray[np.float64],
    sort_indices: Optional[NDArray[np.int64]],
    n_total_states: int,
    optimized_forces: NDArray[np.float64],
) -> Tuple[int, NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Process force data for a single chunk.

    Args:
        chunk: Chunk information with start/end indices
        forces: Force array (n_samples, n_states, n_atoms, 3)
        sort_indices: Optional sorted Z indices for atomic reordering.
                      If None, forces are assumed to be in correct order.
        n_total_states: Number of states to process

    Returns:
        Tuple of (chunk_id, dataset_F, dataset_sF, dataset_dF) for this chunk
    """
    start, end = chunk.start_idx, chunk.end_idx

    # Slice to selected states
    f_chunk = forces[start:end, :n_total_states, :, :]

    if sort_indices is not None:
        # Compute unsort indices to restore original atomic order
        si_chunk = sort_indices[start:end]
        unsort_indices = np.argsort(si_chunk, axis=1)
        unsort_indices_4d = unsort_indices[:, None, :, None]

        # Restore atomic order
        dataset_F = np.take_along_axis(f_chunk, unsort_indices_4d, axis=2)

        # Shifted forces (apply unsorting after shift)
        shifted_raw = f_chunk - optimized_forces[None, None, :, :]
        dataset_sF = np.take_along_axis(shifted_raw, unsort_indices_4d, axis=2)
    else:
        # No reordering needed - data is already in correct order
        dataset_F = f_chunk.copy()
        dataset_sF = f_chunk - optimized_forces[None, None, :, :]

    # Delta forces (difference from ground state)
    ground_state_forces = dataset_F[:, 0:1, :, :]
    dataset_dF = dataset_F - ground_state_forces

    return (chunk.chunk_id, dataset_F, dataset_sF, dataset_dF)


def process_nacrs_chunk(
    chunk: ChunkInfo,
    nacrs: NDArray[np.float64],
    sort_indices: Optional[NDArray[np.int64]],
    n_excited_states: int,
    nacr_indices: Optional[List[int]] = None,
) -> Tuple[int, NDArray[np.float64]]:
    """Process NACR data for a single chunk.

    Note: NACRs have shape (n_couplings, N, 15, 3), so we slice on axis 1.

    Args:
        chunk: Chunk information with start/end indices
        nacrs: NACR array (n_couplings, n_samples, n_atoms, 3)
        sort_indices: Optional sorted Z indices for atomic reordering.
                      If None, NACRs are assumed to be in correct order.
        n_excited_states: Number of excited states
        nacr_indices: Optional list of coupling indices to extract

    Returns:
        Tuple of (chunk_id, dataset_NACR) for this chunk
    """
    start, end = chunk.start_idx, chunk.end_idx

    # Extract relevant couplings if subset
    if nacr_indices is not None:
        nacrs_subset = nacrs[nacr_indices, start:end, :, :]
    else:
        nacrs_subset = nacrs[:, start:end, :, :]

    # Transpose: (n_couplings, chunk_size, 15, 3) -> (chunk_size, n_couplings, 15, 3)
    nacrs_transposed = np.moveaxis(nacrs_subset, 0, 1)

    if sort_indices is not None:
        # Compute unsort indices to restore original atomic order
        si_chunk = sort_indices[start:end]
        unsort_indices = np.argsort(si_chunk, axis=1)
        unsort_indices_4d = unsort_indices[:, None, :, None]

        # Restore atomic order
        dataset_NACR = np.take_along_axis(nacrs_transposed, unsort_indices_4d, axis=2)
    else:
        # No reordering needed - data is already in correct order
        dataset_NACR = nacrs_transposed.copy()

    return (chunk.chunk_id, dataset_NACR)


def process_denacr_chunk(
    chunk: ChunkInfo,
    dataset_NACR: NDArray[np.float64],
    energies: NDArray[np.float64],
    nacr_pairs: List[Tuple[int, int]],
) -> Tuple[int, NDArray[np.float64]]:
    """Process scaled NACR (dENACR) data for a single chunk.

    Computes dENACR = NACR * (E_j - E_i) based on Hellmann-Feynman theorem.
    For NACR_ij = <psi_i|nabla_R H|psi_j> / (E_i - E_j), we recover the
    derivative coupling by multiplying by the energy gap (E_j - E_i).

    Args:
        chunk: Chunk information with start/end indices
        dataset_NACR: Processed NACR array (n_samples, n_couplings, n_atoms, 3)
        energies: Energy array (n_samples, n_states) - column 0 is ground state,
                  columns 1+ are excited states S1, S2, ...
        nacr_pairs: List of (i, j) state pairs where i < j (1-indexed for excited states)

    Returns:
        Tuple of (chunk_id, dataset_dENACR) for this chunk
    """
    start, end = chunk.start_idx, chunk.end_idx
    n_chunk = end - start
    n_couplings = len(nacr_pairs)
    n_atoms = dataset_NACR.shape[2]

    # Get NACR chunk
    nacr_chunk = dataset_NACR[start:end]  # (chunk_size, n_couplings, n_atoms, 3)

    # Get energies chunk
    e_chunk = energies[start:end]  # (chunk_size, n_states)

    # Pre-allocate dENACR
    dataset_dENACR = np.empty((n_chunk, n_couplings, n_atoms, 3), dtype=np.float64)

    # For each coupling pair, compute dENACR = NACR * (E_j - E_i)
    for pair_idx, (state_i, state_j) in enumerate(nacr_pairs):
        # State indices in energy array: state_i and state_j are 1-indexed (S1=1, S2=2, ...)
        # In energies array: column 0 = ground, column 1 = S1, column 2 = S2, etc.
        # So state_i maps to column state_i, state_j maps to column state_j
        E_i = e_chunk[:, state_i]  # (chunk_size,)
        E_j = e_chunk[:, state_j]  # (chunk_size,)

        # Energy gap: E_j - E_i (positive since j > i and typically E_j > E_i)
        delta_E = E_j - E_i  # (chunk_size,)

        # Scale NACR by energy gap
        # nacr_chunk[:, pair_idx] has shape (chunk_size, n_atoms, 3)
        # delta_E needs to be broadcast: (chunk_size,) -> (chunk_size, 1, 1)
        dataset_dENACR[:, pair_idx, :, :] = (
            nacr_chunk[:, pair_idx, :, :] * delta_E[:, None, None]
        )

    return (chunk.chunk_id, dataset_dENACR)


# =============================================================================
# PARALLEL PROCESSING ORCHESTRATION
# =============================================================================


def run_parallel_processing(
    energies: NDArray[np.float64],
    total_energy: NDArray[np.float64],
    gs_dipoles: NDArray[np.float64],
    transition_dipoles: NDArray[np.float64],
    forces: NDArray[np.float64],
    sort_indices: Optional[NDArray[np.int64]],
    nacrs: NDArray[np.float64],
    config: ParallelConfig,
    logger: logging.Logger,
) -> Dict[str, NDArray[np.float64]]:
    """Run all processing steps in parallel.

    Args:
        energies: Energy array (n_samples, n_states)
        total_energy: Total energy array (n_samples, 1)
        gs_dipoles: Ground state dipoles (n_samples, 3)
        transition_dipoles: Transition dipoles (n_samples, n_states-1, 3)
        forces: Forces array (n_samples, n_states, n_atoms, 3)
        sort_indices: Optional sorted Z indices for atomic reordering.
                      If None, forces/NACRs are assumed to be in correct order.
        nacrs: NACR array (n_couplings, n_samples, n_atoms, 3)
        config: Processing configuration
        logger: Logger instance

    Returns:
        Dictionary of output arrays
    """
    n_samples = energies.shape[0]
    n_workers = config.n_workers

    logger.info("\n[PARALLEL PROCESSING]")
    logger.info(f"  Total samples: {n_samples:,}")
    logger.info(f"  Worker processes: {n_workers}")
    logger.info(
        f"  Processing {config.total_states} states (ground + {config.n_states} excited)"
    )
    logger.info(f"  NACR couplings: {config.n_couplings}")
    logger.info(
        f"  Atomic reordering: {'enabled' if sort_indices is not None else 'disabled'}"
    )

    # Calculate optimal chunks
    chunks = calculate_optimal_chunks(n_samples, n_workers)
    logger.info(f"  Processing chunks: {len(chunks)}")
    logger.info(
        f"  Chunk sizes: {chunks[0].n_samples:,} to {chunks[-1].n_samples:,} samples"
    )

    # Prepare NACR indices only when the input contains the full 10-excited-state
    # coupling set and the requested dataset uses a smaller state subset. Q-Chem
    # 5-root inputs already contain exactly the 10 S1-S5 pairs and must not be
    # indexed as though they were stored in the old 45-pair layout.
    nacr_indices = None
    input_n_couplings = int(nacrs.shape[0])
    if input_n_couplings == config.n_couplings:
        nacr_indices = None
    elif config.n_states < N_STATES - 1 and input_n_couplings >= N_COUPLINGS:
        nacr_indices = get_nacr_indices_for_subset(config.n_states)
    else:
        raise ValueError(
            f"NACR coupling count mismatch: input has {input_n_couplings}, "
            f"requested dataset expects {config.n_couplings}."
        )

    # Pre-allocate output arrays
    dataset_S = np.empty((n_samples, config.total_states), dtype=np.float64)
    dataset_sE = np.empty((n_samples, config.total_states), dtype=np.float64)
    dataset_dE = np.empty((n_samples, config.total_states), dtype=np.float64)
    dataset_D = np.empty((n_samples, config.total_states, 3), dtype=np.float64)
    dataset_F = np.empty((n_samples, config.total_states, N_ATOMS, 3), dtype=np.float64)
    dataset_sF = np.empty((n_samples, config.total_states, N_ATOMS, 3), dtype=np.float64)
    dataset_dF = np.empty((n_samples, config.total_states, N_ATOMS, 3), dtype=np.float64)
    dataset_NACR = np.empty((n_samples, config.n_couplings, N_ATOMS, 3), dtype=np.float64)
    dataset_dENACR = np.empty((n_samples, config.n_couplings, N_ATOMS, 3), dtype=np.float64)

    # Get NACR state pairs for dENACR computation
    nacr_pairs = get_nacr_state_pairs(config.n_states)

    # Track processed samples for validation
    processed_samples = set()

    with mp.Pool(processes=n_workers) as pool:
        # Process Energies
        logger.info("\n  Step 1/5: Processing energies...")
        start_time = time.time()

        energy_func = partial(
            process_energies_chunk,
            energies=energies,
            total_energy=total_energy,
            n_total_states=config.total_states,
            optimized_energy=OPTIMIZED_ENERGY,
        )

        for chunk_id, s_chunk, se_chunk, de_chunk in tqdm(
            pool.imap_unordered(energy_func, chunks),
            total=len(chunks),
            desc="    Energies",
            unit="chunk",
        ):
            chunk = chunks[chunk_id]
            start, end = chunk.start_idx, chunk.end_idx
            dataset_S[start:end] = s_chunk
            dataset_sE[start:end] = se_chunk
            dataset_dE[start:end] = de_chunk
            processed_samples.update(range(start, end))

        elapsed = time.time() - start_time
        logger.info(f"    âœ“ Energies processed ({elapsed:.2f}s)")

        # Verify all samples processed
        if len(processed_samples) != n_samples:
            raise RuntimeError(
                f"Energy processing incomplete: {len(processed_samples)}/{n_samples} samples"
            )
        processed_samples.clear()

        # Process Dipoles
        logger.info("  Step 2/5: Processing dipoles...")
        start_time = time.time()

        dipole_func = partial(
            process_dipoles_chunk,
            gs_dipoles=gs_dipoles,
            transition_dipoles=transition_dipoles,
            n_excited_states=config.n_states,
        )

        for chunk_id, d_chunk in tqdm(
            pool.imap_unordered(dipole_func, chunks),
            total=len(chunks),
            desc="    Dipoles",
            unit="chunk",
        ):
            chunk = chunks[chunk_id]
            start, end = chunk.start_idx, chunk.end_idx
            dataset_D[start:end] = d_chunk
            processed_samples.update(range(start, end))

        elapsed = time.time() - start_time
        logger.info(f"    âœ“ Dipoles processed ({elapsed:.2f}s)")

        if len(processed_samples) != n_samples:
            raise RuntimeError(
                f"Dipole processing incomplete: {len(processed_samples)}/{n_samples} samples"
            )
        processed_samples.clear()

        # Process Forces
        logger.info("  Step 3/5: Processing forces...")
        start_time = time.time()

        force_func = partial(
            process_forces_chunk,
            forces=forces,
            sort_indices=sort_indices,
            n_total_states=config.total_states,
            optimized_forces=OPTIMIZED_FORCES,
        )

        for chunk_id, f_chunk, sf_chunk, df_chunk in tqdm(
            pool.imap_unordered(force_func, chunks),
            total=len(chunks),
            desc="    Forces",
            unit="chunk",
        ):
            chunk = chunks[chunk_id]
            start, end = chunk.start_idx, chunk.end_idx
            dataset_F[start:end] = f_chunk
            dataset_sF[start:end] = sf_chunk
            dataset_dF[start:end] = df_chunk
            processed_samples.update(range(start, end))

        elapsed = time.time() - start_time
        logger.info(f"    âœ“ Forces processed ({elapsed:.2f}s)")

        if len(processed_samples) != n_samples:
            raise RuntimeError(
                f"Force processing incomplete: {len(processed_samples)}/{n_samples} samples"
            )
        processed_samples.clear()

        # Process NACRs
        logger.info("  Step 4/5: Processing NACRs...")
        start_time = time.time()

        nacr_func = partial(
            process_nacrs_chunk,
            nacrs=nacrs,
            sort_indices=sort_indices,
            n_excited_states=config.n_states,
            nacr_indices=nacr_indices,
        )

        for chunk_id, nacr_chunk in tqdm(
            pool.imap_unordered(nacr_func, chunks),
            total=len(chunks),
            desc="    NACRs",
            unit="chunk",
        ):
            chunk = chunks[chunk_id]
            start, end = chunk.start_idx, chunk.end_idx
            dataset_NACR[start:end] = nacr_chunk
            processed_samples.update(range(start, end))

        elapsed = time.time() - start_time
        logger.info(f"    âœ“ NACRs processed ({elapsed:.2f}s)")

        if len(processed_samples) != n_samples:
            raise RuntimeError(
                f"NACR processing incomplete: {len(processed_samples)}/{n_samples} samples"
            )
        processed_samples.clear()

        # Process dENACR (scaled NACRs)
        logger.info("  Step 5/5: Processing dENACR (scaled NACRs)...")
        start_time = time.time()

        denacr_func = partial(
            process_denacr_chunk,
            dataset_NACR=dataset_NACR,
            energies=energies,
            nacr_pairs=nacr_pairs,
        )

        for chunk_id, denacr_chunk in tqdm(
            pool.imap_unordered(denacr_func, chunks),
            total=len(chunks),
            desc="    dENACR",
            unit="chunk",
        ):
            chunk = chunks[chunk_id]
            start, end = chunk.start_idx, chunk.end_idx
            dataset_dENACR[start:end] = denacr_chunk
            processed_samples.update(range(start, end))

        elapsed = time.time() - start_time
        logger.info(f"    âœ“ dENACR processed ({elapsed:.2f}s)")

        if len(processed_samples) != n_samples:
            raise RuntimeError(
                f"dENACR processing incomplete: {len(processed_samples)}/{n_samples} samples"
            )

    return {
        "dataset_S.npy": dataset_S,
        "dataset_sE.npy": dataset_sE,
        "dataset_dE.npy": dataset_dE,
        "dataset_D.npy": dataset_D,
        "dataset_F.npy": dataset_F,
        "dataset_sF.npy": dataset_sF,
        "dataset_dF.npy": dataset_dF,
        "dataset_NACR.npy": dataset_NACR,
        "dataset_dENACR.npy": dataset_dENACR,
    }


# =============================================================================
# DATA VALIDATION
# =============================================================================


def validate_data_integrity(
    input_energies: NDArray[np.float64],
    input_total_energy: NDArray[np.float64],
    output_data: Dict[str, NDArray[np.float64]],
    config: ParallelConfig,
    logger: logging.Logger,
) -> ValidationResult:
    """Validate data integrity after processing.

    Performs:
    1. Sample count validation
    2. Shape validation
    3. Checksum validation for reversible operations
    4. NaN/Inf detection

    Args:
        input_energies: Original energy data
        input_total_energy: Original total energy data
        output_data: Processed output arrays
        config: Processing configuration
        logger: Logger instance

    Returns:
        ValidationResult with validation status and details
    """
    logger.info("\n[DATA VALIDATION]")

    errors = []
    n_samples = input_energies.shape[0]
    n_total = config.total_states
    n_couplings = config.n_couplings

    # Expected shapes
    expected_shapes = {
        "dataset_S.npy": (n_samples, n_total),
        "dataset_sE.npy": (n_samples, n_total),
        "dataset_dE.npy": (n_samples, n_total),
        "dataset_D.npy": (n_samples, n_total, 3),
        "dataset_F.npy": (n_samples, n_total, N_ATOMS, 3),
        "dataset_sF.npy": (n_samples, n_total, N_ATOMS, 3),
        "dataset_dF.npy": (n_samples, n_total, N_ATOMS, 3),
        "dataset_NACR.npy": (n_samples, n_couplings, N_ATOMS, 3),
        "dataset_dENACR.npy": (n_samples, n_couplings, N_ATOMS, 3),
        "dataset_R.npy": (n_samples, N_ATOMS, 3),
        "dataset_Z.npy": (n_samples, N_ATOMS),
    }

    # Shape validation
    shape_valid = True
    for name, expected in expected_shapes.items():
        actual = output_data[name].shape
        if actual != expected:
            shape_valid = False
            errors.append(
                f"Shape mismatch for {name}: expected {expected}, got {actual}"
            )
            logger.error(f"  âœ— {name}: shape {actual} != {expected}")
        else:
            logger.debug(f"  âœ“ {name}: shape {actual} OK")

    # Sample count validation
    actual_samples = output_data["dataset_S.npy"].shape[0]
    sample_count_valid = actual_samples == n_samples

    if not sample_count_valid:
        errors.append(
            f"Sample count mismatch: expected {n_samples}, got {actual_samples}"
        )

    # Checksum validation for dataset_S (should match input energies sliced)
    input_slice = input_energies[:, :n_total]
    input_checksum = compute_array_checksum(input_slice)
    output_checksum = compute_array_checksum(output_data["dataset_S.npy"])
    checksum_match = input_checksum == output_checksum

    if not checksum_match:
        errors.append("Checksum mismatch for dataset_S (energies)")
        logger.warning("  âš  Energy checksum mismatch - investigating...")
        # Check if values match even if checksum differs (floating point issues)
        if np.allclose(input_slice, output_data["dataset_S.npy"]):
            logger.info("    Values match within tolerance (floating point)")
            checksum_match = True
            errors.pop()  # Remove the error
    else:
        logger.info("  âœ“ Energy checksum validation passed")

    # Validate shifted energies computation
    expected_sE = input_energies[:, :n_total] - OPTIMIZED_ENERGY
    if not np.allclose(expected_sE, output_data["dataset_sE.npy"]):
        errors.append("Shifted energy computation incorrect")
        logger.error("  âœ— Shifted energy validation failed")
    else:
        logger.info("  âœ“ Shifted energy validation passed")

    # Validate delta energies computation
    expected_dE = input_energies[:, :n_total] - input_total_energy
    if not np.allclose(expected_dE, output_data["dataset_dE.npy"]):
        errors.append("Delta energy computation incorrect")
        logger.error("  âœ— Delta energy validation failed")
    else:
        logger.info("  âœ“ Delta energy validation passed")

    # NaN/Inf detection
    for name, arr in output_data.items():
        if np.any(np.isnan(arr)):
            errors.append(f"NaN values detected in {name}")
            logger.error(f"  âœ— {name}: contains NaN values")
        if np.any(np.isinf(arr)):
            errors.append(f"Inf values detected in {name}")
            logger.error(f"  âœ— {name}: contains Inf values")

    is_valid = len(errors) == 0

    if is_valid:
        logger.info("\n  âœ“ All validation checks passed!")
    else:
        logger.error(f"\n  âœ— Validation failed with {len(errors)} error(s)")

    return ValidationResult(
        is_valid=is_valid,
        total_samples=n_samples,
        actual_samples=actual_samples,
        checksum_match=checksum_match,
        shape_valid=shape_valid,
        errors=errors,
    )


# =============================================================================
# SAVING FUNCTIONS
# =============================================================================


def save_chunked_outputs(
    output_data: Dict[str, NDArray[np.float64]],
    config: ParallelConfig,
    logger: logging.Logger,
) -> None:
    def out_name(name: str) -> str:
        return name.replace("dataset_", f"{config.output_prefix}_", 1) if name.startswith("dataset_") else name
    """Save processed data in chunks to size-based directories."""
    chop_sizes = config.get_chop_sizes()
    chunk_dirs = config.get_chunk_dirs()

    # Print output dimensions
    table_rows = [
        [out_name(name).replace(".npy", ""), format_shape(arr.shape)]
        for name, arr in output_data.items()
    ]
    print_table("OUTPUT DIMENSIONS", ["Array", "Full Shape"], table_rows, logger)

    # Get NACR state pairs
    nacr_pairs = get_nacr_state_pairs(config.n_states)
    max_state = config.n_states
    logger.info(
        f"\n  NACR state pairs: {len(nacr_pairs)} pairs (S1-S2 through S{max_state-1}-S{max_state})"
    )

    # Reshape dENACR arrays: (N, n_couplings, 15, 3) -> (N, n_couplings, 45)
    logger.info("\n[RESHAPING dENACR ARRAYS]")
    original_shape = output_data["dataset_dENACR.npy"].shape
    n_samples_full = original_shape[0]
    n_couplings = original_shape[1]
    logger.info(
        f"  Reshaping {out_name('dataset_dENACR.npy')}: {format_shape(original_shape)} -> ({n_samples_full}, {n_couplings}, {N_ATOMS * 3})"
    )
    output_data["dataset_dENACR.npy"] = output_data["dataset_dENACR.npy"].reshape(
        n_samples_full, n_couplings, N_ATOMS * 3
    )
    logger.info(
        f"  OK {out_name('dataset_dENACR.npy')} reshaped to {format_shape(output_data['dataset_dENACR.npy'].shape)}"
    )

    logger.info("\n[CHUNKING PROGRESS]")

    # Create all directories
    for chunk_dir in chunk_dirs:
        dir_path = config.output_dir / chunk_dir
        dir_path.mkdir(parents=True, exist_ok=True)

    # Save chunks
    for chop_size, chunk_dir in tqdm(
        zip(chop_sizes, chunk_dirs),
        total=len(chop_sizes),
        desc="Saving chunks",
        unit="chunk",
    ):
        output_path = config.output_dir / chunk_dir

        # Save unified arrays
        for filename, arr in output_data.items():
            chunked_arr = arr[:chop_size]
            np.save(output_path / out_name(filename), chunked_arr)
            logger.debug(
                f"Saved {output_path / out_name(filename)}: {format_shape(chunked_arr.shape)}"
            )

        # Save state-wise arrays
        for state_idx in range(config.total_states):
            np.save(
                output_path / f"{config.output_prefix}_S{state_idx}.npy",
                output_data["dataset_S.npy"][:chop_size, state_idx],
            )
            np.save(
                output_path / f"{config.output_prefix}_sE{state_idx}.npy",
                output_data["dataset_sE.npy"][:chop_size, state_idx],
            )
            np.save(
                output_path / f"{config.output_prefix}_dE{state_idx}.npy",
                output_data["dataset_dE.npy"][:chop_size, state_idx],
            )
            np.save(
                output_path / f"{config.output_prefix}_D{state_idx}.npy",
                output_data["dataset_D.npy"][:chop_size, state_idx, :],
            )
            np.save(
                output_path / f"{config.output_prefix}_F{state_idx}.npy",
                output_data["dataset_F.npy"][:chop_size, state_idx, :, :],
            )
            np.save(
                output_path / f"{config.output_prefix}_sF{state_idx}.npy",
                output_data["dataset_sF.npy"][:chop_size, state_idx, :, :],
            )
            np.save(
                output_path / f"{config.output_prefix}_dF{state_idx}.npy",
                output_data["dataset_dF.npy"][:chop_size, state_idx, :, :],
            )

        # Save NACR and dENACR state-pair arrays if requested
        if config.split_nacr:
            for pair_idx, (state_i, state_j) in enumerate(nacr_pairs):
                # Save NACR pair
                nacr_pair_data = output_data["dataset_NACR.npy"][:chop_size, pair_idx, :, :]
                np.save(output_path / f"{config.output_prefix}_NACR{state_i}{state_j}.npy", nacr_pair_data)
                # Save dENACR pair - already reshaped to (N, n_couplings, 45), extract and keep as (N, 45)
                denacr_pair_data = output_data["dataset_dENACR.npy"][
                    :chop_size, pair_idx, :
                ]
                np.save(
                    output_path / f"{config.output_prefix}_dENACR{state_i}{state_j}.npy", denacr_pair_data
                )

    # Summary
    unified_files = len(output_data)
    state_files = config.total_states * 7
    nacr_pair_files = (
        len(nacr_pairs) * 2 if config.split_nacr else 0
    )  # NACR + dENACR pairs
    total_files = unified_files + state_files + nacr_pair_files

    logger.info(
        f"\n  âœ“ Saved {len(chop_sizes)} chunks ({chunk_dirs[0]} to {chunk_dirs[-1]})"
    )
    logger.info(f"  âœ“ {total_files} files per chunk:")
    logger.info(f"      - {unified_files} unified arrays")
    logger.info(
        f"      - {state_files} state-wise arrays ({config.total_states} states Ã— 7 properties)"
    )
    if config.split_nacr:
        logger.info(
            f"      - {nacr_pair_files} NACR/dENACR state-pair arrays ({len(nacr_pairs)} pairs Ã— 2)"
        )

    # Print final shapes of all created arrays
    logger.info("\n[FINAL ARRAY SHAPES]")
    logger.info("  Unified arrays:")
    for name, arr in output_data.items():
        logger.info(f"    {out_name(name)}: {format_shape(arr.shape)}")

    if config.split_nacr:
        logger.info("\n  State-pair arrays (per chunk):")
        for pair_idx, (state_i, state_j) in enumerate(nacr_pairs):
            # NACR pair shape
            nacr_shape = (chop_sizes[-1], N_ATOMS, 3)
            logger.info(
                f"    {config.output_prefix}_NACR{state_i}{state_j}.npy: {format_shape(nacr_shape)}"
            )
            # dENACR pair shape (flattened)
            denacr_shape = (chop_sizes[-1], N_ATOMS * 3)
            logger.info(
                f"    {config.output_prefix}_dENACR{state_i}{state_j}.npy: {format_shape(denacr_shape)}"
            )


# =============================================================================
# DATA LOADING
# =============================================================================


def load_input_data(config: ParallelConfig, logger: logging.Logger) -> Tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    Optional[NDArray[np.int64]],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.int64],
    int,
]:
    """Load all input files.

    Returns:
        Tuple of (energies, total_energy, gs_dipoles, transition_dipoles,
                  forces, sort_indices, nacrs, coords_R, coords_Z, n_samples)
        Note: sort_indices is None if --sorted-z was not provided.
    """
    logger.info("\n[LOADING INPUT FILES]")

    # Build complete list of required files (including configurable coordinate files)
    required_files = list(INPUT_FILES.keys()) + [
        config.coords_r_file,
        config.coords_z_file,
    ]

    # Check for missing required files
    missing = []
    for filename in required_files:
        if not (config.input_dir / filename).exists():
            missing.append(filename)

    if missing:
        raise FileNotFoundError(f"Missing required input files: {missing}")

    # Load required files
    loaded = {}
    table_rows = []

    for filename in tqdm(required_files, desc="Loading files", unit="file"):
        filepath = config.input_dir / filename
        data = np.load(filepath)
        loaded[filename] = data
        table_rows.append([filename, format_shape(data.shape)])
        logger.debug(f"Loaded {filename}: {data.shape}")

    # Load optional sorted_Z_indices file if specified
    sort_indices: Optional[NDArray[np.int64]] = None
    if config.sorted_z_file is not None:
        if not config.sorted_z_file.exists():
            raise FileNotFoundError(
                f"Sorted Z indices file not found: {config.sorted_z_file}"
            )
        sort_indices = np.load(config.sorted_z_file).astype(np.int64)
        table_rows.append([config.sorted_z_file.name, format_shape(sort_indices.shape)])
        logger.debug(f"Loaded {config.sorted_z_file.name}: {sort_indices.shape}")

    print_table("INPUT DIMENSIONS", ["File", "Shape"], table_rows, logger)

    n_samples = loaded["energies_all_states.npy"].shape[0]
    n_files_loaded = len(INPUT_FILES) + (1 if sort_indices is not None else 0)
    logger.info(f"\n  âœ“ All {n_files_loaded} input files loaded")
    logger.info(f"  âœ“ Detected {n_samples:,} samples/timesteps")
    if sort_indices is not None:
        logger.info(
            f"  âœ“ Atomic reordering enabled (using {config.sorted_z_file.name})"
        )
    else:
        logger.info(
            "  âœ“ Atomic reordering disabled (forces/NACRs assumed in correct order)"
        )

    return (
        loaded["energies_all_states.npy"].astype(np.float64),
        loaded["total_energy.npy"].astype(np.float64),
        loaded["gs_dipoles.npy"].astype(np.float64),
        loaded["transition_dipoles.npy"].astype(np.float64),
        loaded["forces_all_states.npy"].astype(np.float64),
        sort_indices,
        loaded["nacrs_all_states.npy"].astype(np.float64),
        loaded[config.coords_r_file].astype(np.float64),
        loaded[config.coords_z_file].astype(np.int64),
        n_samples,
    )


# =============================================================================
# ARGUMENT PARSING
# =============================================================================


def parse_arguments() -> ParallelConfig:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Molecule-agnostic multi-state MD dataset processing pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python 05_generate_dataset.py --input-dir ./data --output-dir ./output --prefix molecule
  python 05_generate_dataset.py --input-dir . --output-dir . --n-workers 16 --prefix molecule
  python 05_generate_dataset.py --n-states 10 --n-workers 8 --no-validate --prefix molecule
  python 05_generate_dataset.py --sorted-z sorted_Z_indices.npy --prefix molecule  # Enable atomic reordering

Performance Tips:
  - Use n_workers equal to number of CPU cores for optimal performance
  - For very large datasets (>1M samples), consider increasing MAX_CHUNK_SIZE
  - Use --no-validate to skip validation for faster processing
        """,
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("."),
        help="Directory containing input .npy files (default: current directory)",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Base directory for output files (default: current directory)",
    )

    parser.add_argument(
        "--log-file",
        type=Path,
        default=Path("processing_parallel.log"),
        help="Path to log file (default: processing_parallel.log)",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )

    parser.add_argument(
        "--chop-start",
        type=int,
        default=CHOP_START,
        help=f"First chunk size (default: {CHOP_START})",
    )

    parser.add_argument(
        "--chop-end",
        type=int,
        default=CHOP_END,
        help=f"Maximum chunk size, exclusive (default: {CHOP_END})",
    )

    parser.add_argument(
        "--chop-step",
        type=int,
        default=CHOP_STEP,
        help=f"Increment between chunk sizes (default: {CHOP_STEP})",
    )

    parser.add_argument(
        "--split-nacr",
        action="store_true",
        default=False,
        help="Save individual NACR state-pair files",
    )

    parser.add_argument(
        "--n-states",
        type=int,
        default=N_STATES - 1,
        help=f"Number of excited states to process (default: {N_STATES - 1})",
    )

    parser.add_argument(
        "--n-workers",
        type=int,
        default=DEFAULT_N_WORKERS,
        help=f"Number of parallel worker processes (default: {DEFAULT_N_WORKERS})",
    )

    parser.add_argument(
        "--no-validate",
        action="store_true",
        default=False,
        help="Skip data integrity validation (faster but less safe)",
    )

    parser.add_argument(
        "--sorted-z",
        type=Path,
        default=None,
        metavar="FILE",
        help="Optional file with sorted Z indices for atomic reordering. "
        "If provided, forces and NACRs will be reordered from Z-sorted "
        "order back to original molecular order. If not provided, data "
        "is assumed to already be in the correct atomic order.",
    )

    parser.add_argument(
        "--coords-r",
        type=str,
        default=DEFAULT_COORDS_R_FILE,
        metavar="FILE",
        help=f"Input file for atomic coordinates (N, {N_ATOMS}, 3). "
        f"(default: {DEFAULT_COORDS_R_FILE})",
    )

    parser.add_argument(
        "--prefix",
        type=str,
        default="molecule",
        help="Prefix for output dataset files (default: molecule)",
    )

    parser.add_argument(
        "--coords-z",
        type=str,
        default=DEFAULT_COORDS_Z_FILE,
        metavar="FILE",
        help=f"Input file for atomic numbers (N, {N_ATOMS}). "
        f"(default: {DEFAULT_COORDS_Z_FILE})",
    )

    parser.add_argument(
        "--reference-energy",
        type=Path,
        default=None,
        metavar="FILE",
        help="Molecule-specific optimized reference energy text file.",
    )

    parser.add_argument(
        "--reference-forces",
        type=Path,
        default=None,
        metavar="FILE",
        help="Molecule-specific optimized reference forces .npy file.",
    )
    args = parser.parse_args()

    return ParallelConfig(
        input_dir=args.input_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        log_file=args.log_file.resolve(),
        log_level=args.log_level,
        chop_start=args.chop_start,
        chop_end=args.chop_end,
        chop_step=args.chop_step,
        split_nacr=args.split_nacr,
        n_states=args.n_states,
        n_workers=args.n_workers,
        validate=not args.no_validate,
        sorted_z_file=args.sorted_z.resolve() if args.sorted_z else None,
        coords_r_file=args.coords_r,
        coords_z_file=args.coords_z,
        output_prefix=args.prefix,
        reference_energy_file=args.reference_energy.resolve() if args.reference_energy else None,
        reference_forces_file=args.reference_forces.resolve() if args.reference_forces else None,
    )



def load_reference_constants(energy_file: Optional[Path], forces_file: Optional[Path], logger: logging.Logger) -> None:
    """Load molecule-specific optimized reference energy/forces at runtime."""
    global OPTIMIZED_ENERGY, OPTIMIZED_FORCES, N_ATOMS

    if energy_file is None and forces_file is None:
        logger.info(f"\n  Reference energy: {OPTIMIZED_ENERGY:.16f} eV")
        logger.info(f"  Reference forces: embedded constants with shape {OPTIMIZED_FORCES.shape}")
        return

    if energy_file is None or forces_file is None:
        raise ValueError("Provide both --reference-energy and --reference-forces, or neither.")
    if not energy_file.exists():
        raise FileNotFoundError(f"Reference energy file not found: {energy_file}")
    if not forces_file.exists():
        raise FileNotFoundError(f"Reference forces file not found: {forces_file}")

    OPTIMIZED_ENERGY = float(np.loadtxt(energy_file))
    OPTIMIZED_FORCES = np.load(forces_file).astype(np.float64)
    if OPTIMIZED_FORCES.ndim != 2 or OPTIMIZED_FORCES.shape[1] != 3:
        raise ValueError(
            f"Reference forces must have shape (n_atoms, 3), got {OPTIMIZED_FORCES.shape}"
        )
    N_ATOMS = int(OPTIMIZED_FORCES.shape[0])
    logger.info(f"\n  Reference energy: {OPTIMIZED_ENERGY:.16f} eV from {energy_file}")
    logger.info(f"  Reference forces: {OPTIMIZED_FORCES.shape} from {forces_file}")

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================


def main() -> int:
    """Main entry point."""
    total_start_time = time.time()

    # Parse arguments
    config = parse_arguments()

    # Setup logging
    logger = setup_logging(config.log_file, config.log_level)

    try:
        print_header(logger)
        load_reference_constants(config.reference_energy_file, config.reference_forces_file, logger)

        # Log configuration
        logger.info(f"  Input directory:  {config.input_dir}")
        logger.info(f"  Output directory: {config.output_dir}")
        logger.info(f"  Workers:          {config.n_workers}")
        logger.info(
            f"  States:           {config.total_states} (ground + {config.n_states} excited)"
        )
        logger.info(
            f"  Validation:       {'enabled' if config.validate else 'disabled'}"
        )
        logger.info(
            f"  Atomic reorder:   {'enabled (' + config.sorted_z_file.name + ')' if config.sorted_z_file else 'disabled'}"
        )

        # Load input data
        (
            energies,
            total_energy,
            gs_dipoles,
            transition_dipoles,
            forces,
            sort_indices,
            nacrs,
            dataset_R,
            dataset_Z,
            n_samples,
        ) = load_input_data(config, logger)

        # Run parallel processing
        output_data = run_parallel_processing(
            energies=energies,
            total_energy=total_energy,
            gs_dipoles=gs_dipoles,
            transition_dipoles=transition_dipoles,
            forces=forces,
            sort_indices=sort_indices,
            nacrs=nacrs,
            config=config,
            logger=logger,
        )

        # Add coordinate arrays to output data (passthrough, no processing needed)
        output_data["dataset_R.npy"] = dataset_R
        output_data["dataset_Z.npy"] = dataset_Z

        # Validate if enabled
        if config.validate:
            validation = validate_data_integrity(
                input_energies=energies,
                input_total_energy=total_energy,
                output_data=output_data,
                config=config,
                logger=logger,
            )

            if not validation.is_valid:
                logger.error("Data validation failed!")
                for error in validation.errors:
                    logger.error(f"  - {error}")
                return 1

        # Save outputs
        save_chunked_outputs(output_data, config, logger)

        # Summary
        total_elapsed = time.time() - total_start_time
        chunk_dirs = config.get_chunk_dirs()

        logger.info("\n[SUMMARY]")
        logger.info(f"  Total processing time: {format_time(total_elapsed)}")
        logger.info(f"  Samples processed:     {n_samples:,}")
        logger.info(
            f"  Chunks created:        {len(chunk_dirs)} ({chunk_dirs[0]} to {chunk_dirs[-1]})"
        )
        logger.info(f"  Output directory:      {config.output_dir}")
        logger.info(f"  Log file:              {config.log_file}")

        # Performance stats
        samples_per_second = n_samples / total_elapsed
        logger.info(f"\n  Performance: {samples_per_second:,.0f} samples/second")

        logger.info(
            """
================================================================================
Processing complete!
================================================================================"""
        )

        return 0

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return 1
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    # Required for multiprocessing on some platforms
    mp.set_start_method("spawn", force=True)
    sys.exit(main())

#!/usr/bin/env python3
"""
State Crossing Analysis Script for Non-Adiabatic Molecular Dynamics
Identifies Trivial vs. Non-Trivial (Avoided) Crossings using Force Extrapolation
"""

import os
import numpy as np
import multiprocessing
import argparse
from tqdm import tqdm

# Constants
BOHR_TO_A = 0.529177249
HA_TO_EV = 27.2113961


def get_frame_data(frame_idx, base_dir, n_states=5):
    """Extracts Coordinates, Energies, and Gradients for a specific frame."""
    frame_dir = os.path.join(base_dir, f"frame_{frame_idx}")
    if not os.path.exists(frame_dir):
        return None

    try:
        # 1. Coordinates
        coords_data = np.loadtxt(
            os.path.join(frame_dir, "coords.xyz"), skiprows=2, dtype=str
        )
        coords = coords_data[:, 1:].astype(float)

        # 2. Energies (pes.out: col 0=time, col 1=S0, col 2=S1...)
        pes_data = np.loadtxt(os.path.join(frame_dir, "pes.out"), ndmin=2)
        # Get S1 to S5 (indices 2 to 6)
        energies = pes_data[-1, 2 : 2 + n_states]

        # 3. Gradients (gradients.out: row 0=S1, row 1=S2...)
        grad_data = np.loadtxt(os.path.join(frame_dir, "gradients.out"))
        natoms = len(coords)
        gradients = grad_data[:n_states, 2:].reshape(n_states, natoms, 3)

        return frame_idx, coords, energies, gradients
    except Exception as e:
        return None


def main(args):
    print(f"--- Analyzing MD Trajectory for Crossings ({args.frames} frames) ---")

    # Multiprocessing Extraction
    pool_args = [(i, args.base_dir, args.n_states) for i in range(1, args.frames + 1)]
    results = []
    with multiprocessing.Pool(args.cores) as pool:
        for res in tqdm(
            pool.starmap(get_frame_data, pool_args),
            total=args.frames,
            desc="Extracting",
        ):
            if res is not None:
                results.append(res)

    results.sort(key=lambda x: x[0])

    # Store trajectory arrays
    all_coords = np.array([r[1] for r in results])
    all_energies = np.array([r[2] for r in results])
    all_grads = np.array([r[3] for r in results])
    valid_frames = [r[0] for r in results]

    crossings_log = []

    print("\n--- Scanning for Crossings ---")
    print(f"{'Frame':<8} | {'Pair':<8} | {'Gap (eV)':<10} | {'Type'}")
    print("-" * 50)

    # Loop through trajectory (t to t+1)
    for t in range(len(valid_frames) - 1):
        frame_num = valid_frames[t]

        # Calculate coordinate displacement in Bohr
        dR_angstrom = all_coords[t + 1] - all_coords[t]
        dR_bohr = dR_angstrom / BOHR_TO_A

        # Check adjacent pairs (S1-S2, S2-S3, S3-S4, S4-S5)
        for i in range(args.n_states - 1):
            j = i + 1

            # Adiabatic Gap at time t
            dE_ha = all_energies[t, j] - all_energies[t, i]
            gap_ev = abs(dE_ha) * HA_TO_EV

            # Threshold Check
            if gap_ev < args.threshold_ev:
                # F-Tracking Extrapolation Logic
                # dE_extrap = dE(t) + sum((g_j(t) - g_i(t)) * dR)
                g_i = all_grads[t, i]
                g_j = all_grads[t, j]
                dE_extrap_ha = dE_ha + np.sum((g_j - g_i) * dR_bohr)

                # Check for sign reversal
                if np.sign(dE_ha) != np.sign(dE_extrap_ha):
                    cross_type = "Trivial Crossing"
                else:
                    cross_type = "Avoided (Non-Trivial)"

                crossings_log.append((frame_num, i + 1, j + 1, gap_ev, cross_type))
                print(
                    f"{frame_num:<8} | S{i+1}-S{j+1:<6} | {gap_ev:<10.4f} | {cross_type}"
                )

    if not crossings_log:
        print("No crossings detected below the specified threshold.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find Trivial and Avoided Crossings")
    parser.add_argument(
        "--base_dir",
        type=str,
        required=True,
        help="Path containing frame_X directories",
    )
    parser.add_argument(
        "--frames", type=int, default=1000, help="Number of frames to scan"
    )
    parser.add_argument(
        "--n_states",
        type=int,
        default=5,
        help="Number of excited states to track (e.g., 5 for S1-S5)",
    )
    parser.add_argument(
        "--threshold_ev",
        type=float,
        default=0.1,
        help="Energy gap threshold in eV to flag a crossing",
    )
    parser.add_argument("--cores", type=int, default=8, help="CPU cores for extraction")
    args = parser.parse_args()
    main(args)

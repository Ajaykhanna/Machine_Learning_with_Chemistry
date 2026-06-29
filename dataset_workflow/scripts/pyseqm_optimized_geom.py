#!/usr/bin/env python3
"""Optimize a ground-state geometry with PySEQM and export reference S0 data.

This step is used before dataset generation so shifted energies/forces are
computed relative to the PySEQM AM1 ground-state minimum for the same molecule.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from seqm.seqm_functions.constants import Constants
from seqm.Molecule import Molecule
from seqm.geometryOptimization import geomeTRIC_optimization


torch.set_default_dtype(torch.float64)

SYMBOL_TO_Z: Dict[str, int] = {
    "H": 1,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "P": 15,
    "S": 16,
    "CL": 17,
}
Z_TO_SYMBOL: Dict[int, str] = {z: sym.title() for sym, z in SYMBOL_TO_Z.items()}


def parse_xyz(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Read the first XYZ frame from path and return atomic numbers and coords."""
    lines = path.read_text().splitlines()
    if not lines:
        raise ValueError(f"Empty XYZ file: {path}")

    try:
        n_atoms = int(lines[0].strip())
    except ValueError as exc:
        raise ValueError(f"First line of {path} must be the atom count") from exc

    atom_lines = lines[2 : 2 + n_atoms]
    if len(atom_lines) != n_atoms:
        raise ValueError(f"Expected {n_atoms} atom lines in {path}, found {len(atom_lines)}")

    species: List[int] = []
    coords: List[List[float]] = []
    for line in atom_lines:
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"Malformed XYZ atom line: {line!r}")
        symbol = parts[0].upper()
        if symbol not in SYMBOL_TO_Z:
            raise ValueError(f"Unsupported element symbol {parts[0]!r} in {path}")
        species.append(SYMBOL_TO_Z[symbol])
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])

    return np.asarray(species, dtype=np.int64), np.asarray(coords, dtype=np.float64)


def sort_species_coordinates(species: np.ndarray, coordinates: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Sort atomic numbers high-to-low and apply the same order to coordinates."""
    order = np.argsort(-species, kind="stable")
    return species[order], coordinates[order]


def save_optimized_geometry(filename: Path, species: np.ndarray, coordinates: np.ndarray, energy: float) -> None:
    """Save optimized geometry to XYZ."""
    with filename.open("w", encoding="utf-8") as handle:
        handle.write(f"{len(species)}\n")
        handle.write(f"PySEQM AM1 optimized geometry; Total Energy: {energy:.12f} eV\n")
        for z, xyz in zip(species, coordinates):
            symbol = Z_TO_SYMBOL.get(int(z), "X")
            handle.write(f"{symbol:<2} {xyz[0]:15.12f} {xyz[1]:15.12f} {xyz[2]:15.12f}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PySEQM AM1 ground-state geometry optimization")
    parser.add_argument("--xyz", required=True, type=Path, help="Input XYZ geometry, e.g. NEXMD coords.xyz")
    parser.add_argument("--output-dir", type=Path, default=Path("pyseqm_gsopt"), help="Output directory")
    parser.add_argument("--prefix", default=None, help="Molecule prefix for output files; defaults to input XYZ parent name")
    parser.add_argument("--method", default="AM1", help="SEQM method: AM1, PM3, MNDO, ...")
    parser.add_argument("--scf-eps", type=float, default=1e-10, help="SCF convergence threshold")
    parser.add_argument("--max-iter", type=int, default=1000, help="Maximum geomeTRIC optimization iterations")
    parser.add_argument("--tol", type=float, default=1e-6, help="Geometry optimization tolerance")
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Execution device. auto uses CUDA when available, cuda requires CUDA, cpu never uses CUDA.",
    )
    parser.add_argument("--sort-z", action="store_true", help="Sort atoms by descending atomic number before optimization")
    return parser.parse_args()


def resolve_device(policy: str) -> torch.device:
    if policy == "cpu":
        return torch.device("cpu")
    if policy == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested, but CUDA is not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inferred_prefix = args.xyz.parent.parent.name if args.xyz.name == "coords.xyz" else args.xyz.stem
    prefix = args.prefix or inferred_prefix

    device = resolve_device(args.device)
    species_np, coords_np = parse_xyz(args.xyz)
    if args.sort_z:
        species_np, coords_np = sort_species_coordinates(species_np, coords_np)

    species = torch.as_tensor(species_np, dtype=torch.int64, device=device).unsqueeze(0)
    coordinates = torch.as_tensor(coords_np, dtype=torch.float64, device=device).unsqueeze(0)

    const = Constants().to(device)
    elements = [0] + sorted(set(species_np.tolist()))
    seqm_parameters = {
        "method": args.method,
        "scf_eps": args.scf_eps,
        "scf_converger": [1, 0.0],
        "sp2": [False, 1.0e-5],
        "elements": elements,
        "learned": [],
        "pair_outer_cutoff": 1.0e10,
        "eig": True,
    }

    molecule = Molecule(const, seqm_parameters, coordinates, species).to(device)
    geomeTRIC_optimization(molecule, max_iter=args.max_iter, tol=args.tol)

    energy = float(molecule.Etot.detach().cpu().numpy()[0])
    forces = molecule.force.detach().cpu().numpy()[0]
    opt_coords = molecule.coordinates.detach().cpu().numpy()[0]

    np.savetxt(args.output_dir / f"{prefix}_optimized_reference_energy.txt", np.asarray([energy]), fmt="%.16f")
    np.save(args.output_dir / f"{prefix}_optimized_reference_forces.npy", forces)
    np.save(args.output_dir / f"{prefix}_optimized_reference_coordinates.npy", opt_coords)
    np.save(args.output_dir / f"{prefix}_optimized_reference_species.npy", species_np)
    save_optimized_geometry(args.output_dir / f"{prefix}_optimized_gs_geometry.xyz", species_np, opt_coords, energy)

    print("Molecule Properties After Ground State Geometry Optimization:")
    print(f"Total Atoms: {len(species_np)}")
    print(f"Device: {device}")
    print(f"Method: {args.method}")
    print(f"Total Energy (eV): {energy:.16f}")
    print("Ground State Forces (eV/Angstrom):")
    print(np.array2string(forces, precision=16, separator=", ", max_line_width=200))
    print(f"Optimized geometry written to: {args.output_dir / f'{prefix}_optimized_gs_geometry.xyz'}")
    print(f"Reference force array written to: {args.output_dir / f'{prefix}_optimized_reference_forces.npy'}")


if __name__ == "__main__":
    main()




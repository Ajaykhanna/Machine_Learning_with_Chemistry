#!/usr/bin/env python3
"""Q-Chem backend for the dataset workflow.

This script provides three production-oriented entry points:

* ``--gsopt``: ground-state DFT optimization plus reference force extraction.
* ``--frame``: one SLURM-array task over a chunk of prepared configurations.
* ``--collect``: validate frame shards and assemble canonical dataset inputs.

The collector writes the same file names consumed by ``05_generate_dataset.py``.
All energies are eV, forces are eV/Angstrom, and NACRs are Angstrom^-1.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

HARTREE_TO_EV = 27.21138602
BOHR_TO_ANGSTROM = 0.529177210903
FORCE_AU_TO_EV_PER_ANGSTROM = HARTREE_TO_EV / BOHR_TO_ANGSTROM

Z_TO_SYMBOL = {
    1: "H",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
    15: "P",
    16: "S",
    17: "Cl",
    35: "Br",
    53: "I",
}

BASIS_ALIASES = {
    "6-31g*": "6-31G*",
    "6-31g(d)": "6-31G*",
    "6-31g": "6-31G",
}


def qchem_basis(value: str) -> str:
    return BASIS_ALIASES.get(value.lower(), value)


def read_xyz(path: Path) -> Tuple[List[str], np.ndarray]:
    lines = path.read_text().splitlines()
    natoms = int(lines[0].strip())
    symbols: List[str] = []
    coords: List[List[float]] = []
    for line in lines[2 : 2 + natoms]:
        parts = line.split()
        if len(parts) < 4:
            continue
        symbols.append(parts[0])
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if len(symbols) != natoms:
        raise ValueError(f"Expected {natoms} atoms in {path}, parsed {len(symbols)}")
    return symbols, np.asarray(coords, dtype=np.float64)


def write_xyz(path: Path, symbols: List[str], coords: np.ndarray, comment: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"{len(symbols)}\n{comment}\n")
        for sym, (x, y, z) in zip(symbols, coords):
            handle.write(f"{sym:<3} {x: .10f} {y: .10f} {z: .10f}\n")


def symbols_from_z(z_row: np.ndarray) -> List[str]:
    symbols: List[str] = []
    for z in z_row.astype(int):
        if int(z) not in Z_TO_SYMBOL:
            raise ValueError(f"Unsupported atomic number for Q-Chem input: {int(z)}")
        symbols.append(Z_TO_SYMBOL[int(z)])
    return symbols


def molecule_block(symbols: List[str], coords: np.ndarray, charge: int, multiplicity: int) -> str:
    lines = ["$molecule", f"{charge} {multiplicity}"]
    for sym, (x, y, z) in zip(symbols, coords):
        lines.append(f"{sym:<3} {x: .10f} {y: .10f} {z: .10f}")
    lines.append("$end")
    return "\n".join(lines)


def build_gsopt_input(symbols: List[str], coords: np.ndarray, args: argparse.Namespace) -> str:
    mol = molecule_block(symbols, coords, args.charge, args.multiplicity)
    basis = qchem_basis(args.basis)
    return f"""$comment
Q-Chem GS optimization: {args.method}/{basis}
$end

{mol}

$rem
JOBTYPE              OPT
METHOD               {args.method}
BASIS                {basis}
GEOM_OPT_PRINT       3
GEOM_OPT_MAX_CYCLES  {args.geom_opt_max_cycles}
MAX_SCF_CYCLES       {args.max_scf_cycles}
MEM_TOTAL            {args.qchem_mem_total_mb}
SYM_IGNORE           TRUE
$end

@@@

$comment
Ground-state force at optimized geometry.
$end

$molecule
READ
$end

$rem
JOBTYPE              FORCE
METHOD               {args.method}
BASIS                {basis}
SCF_GUESS            READ
MAX_SCF_CYCLES       {args.max_scf_cycles}
MEM_TOTAL            {args.qchem_mem_total_mb}
SYM_IGNORE           TRUE
$end
"""


def build_sp_input(symbols: List[str], coords: np.ndarray, args: argparse.Namespace) -> str:
    mol = molecule_block(symbols, coords, args.charge, args.multiplicity)
    basis = qchem_basis(args.basis)
    return f"""$comment
Q-Chem TDA single point: {args.method}/{basis}, {args.n_roots} singlet states
$end

{mol}

$rem
JOBTYPE              SP
METHOD               {args.method}
BASIS                {basis}
CIS_N_ROOTS          {args.n_roots}
CIS_SINGLETS         TRUE
CIS_TRIPLETS         FALSE
RPA                  FALSE
MAX_CIS_CYCLES       {args.max_cis_cycles}
MAX_SCF_CYCLES       {args.max_scf_cycles}
MEM_TOTAL            {args.qchem_mem_total_mb}
SYM_IGNORE           TRUE
$end
"""


def build_force_input(
    symbols: List[str],
    coords: np.ndarray,
    args: argparse.Namespace,
    state: int,
    use_scf_read: bool = False,
) -> str:
    mol = molecule_block(symbols, coords, args.charge, args.multiplicity)
    basis = qchem_basis(args.basis)
    scf_guess = "SCF_GUESS            READ" if use_scf_read else ""
    if state == 0:
        state_lines = ""
        title = "ground state"
    else:
        state_lines = f"""CIS_N_ROOTS          {args.n_roots}
CIS_SINGLETS         TRUE
CIS_TRIPLETS         FALSE
RPA                  FALSE
CIS_STATE_DERIV      {state}
MAX_CIS_CYCLES       {args.max_cis_cycles}"""
        title = f"S{state}"
    return f"""$comment
Q-Chem FORCE {title}: {args.method}/{basis}
$end

{mol}

$rem
JOBTYPE              FORCE
METHOD               {args.method}
BASIS                {basis}
{scf_guess}
{state_lines}
MAX_SCF_CYCLES       {args.max_scf_cycles}
MEM_TOTAL            {args.qchem_mem_total_mb}
SYM_IGNORE           TRUE
$end
"""


def build_nac_input(symbols: List[str], coords: np.ndarray, args: argparse.Namespace) -> str:
    mol = molecule_block(symbols, coords, args.charge, args.multiplicity)
    basis = qchem_basis(args.basis)
    states = list(range(1, args.n_roots + 1))
    dc_block = "\n".join(["$derivative_coupling", "   all excited-state pairs", "   " + " ".join(map(str, states)), "$end"])
    return f"""$comment
Q-Chem NACs: all pairs among S1-S{args.n_roots}
$end

{mol}

$rem
JOBTYPE              SP
METHOD               {args.method}
BASIS                {basis}
CIS_N_ROOTS          {args.n_roots}
CIS_SINGLETS         TRUE
CIS_TRIPLETS         FALSE
RPA                  FALSE
CALC_NAC             TRUE
CIS_DER_NUMSTATE     {args.n_roots}
SET_QUADRATIC        FALSE
MAX_CIS_CYCLES       {args.max_cis_cycles}
MAX_SCF_CYCLES       {args.max_scf_cycles}
MEM_TOTAL            {args.qchem_mem_total_mb}
SYM_IGNORE           TRUE
$end

{dc_block}
"""


def run_qchem(input_file: Path, output_file: Path, save_name: Optional[str], save: Optional[bool], args: argparse.Namespace) -> int:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    scratch = Path(args.qchem_scratch).resolve()
    scratch.mkdir(parents=True, exist_ok=True)
    env_prefix = f"source {shlex.quote(args.qchem_env_file)}; " if args.qchem_env_file else ""
    cmd = env_prefix + f"export QCSCRATCH={shlex.quote(str(scratch))}; "
    cmd += f"{shlex.quote(args.qchem_exe)} -nt {int(args.ncpu)} "
    if save_name and save is True:
        cmd += f"-save {shlex.quote(str(input_file))} {shlex.quote(str(output_file))} {shlex.quote(save_name)}"
    elif save_name and save is False:
        cmd += f"{shlex.quote(str(input_file))} {shlex.quote(str(output_file))} {shlex.quote(save_name)}"
    else:
        cmd += f"{shlex.quote(str(input_file))} {shlex.quote(str(output_file))}"
    print(f"CMD: {cmd}")
    proc = subprocess.run(cmd, shell=True, executable="/bin/bash", text=True, stderr=subprocess.PIPE)
    if proc.stderr:
        print(proc.stderr, file=sys.stderr)
    return int(proc.returncode)


def read_output(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def parse_scf_energy(text: str) -> Optional[float]:
    patterns = [
        r"Total energy in the final basis set\s*=\s*([-+]?\d+\.\d+)",
        r"SCF\s+energy in the final basis set\s*=\s*([-+]?\d+\.\d+)",
        r"DFT\s+total energy\s*=\s*([-+]?\d+\.\d+)",
    ]
    value: Optional[float] = None
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            value = float(match.group(1))
    return value


def parse_state_total_energies(text: str) -> Dict[int, float]:
    values: Dict[int, float] = {}
    for match in re.finditer(r"Total energy for state\s+(\d+):\s+([-+]?\d+\.\d+)\s+au", text, re.IGNORECASE):
        values[int(match.group(1))] = float(match.group(2))
    return values


def parse_excitation_energies_ev(text: str) -> Dict[int, float]:
    values: Dict[int, float] = {}
    for match in re.finditer(r"Excited state\s+(\d+)[:\s]+excitation energy \(eV\)\s*=\s*([-+]?\d+\.\d+)", text, re.IGNORECASE):
        values.setdefault(int(match.group(1)), float(match.group(2)))
    return values


def parse_transition_dipoles(text: str, n_roots: int) -> np.ndarray:
    dipoles = np.zeros((n_roots, 3), dtype=np.float64)
    current_state: Optional[int] = None
    state_re = re.compile(r"Excited state\s+(\d+)", re.IGNORECASE)
    tm_re = re.compile(r"Trans\.\s*Mom\.\s*:\s*([-+]?\d+\.\d+)\s+X\s+([-+]?\d+\.\d+)\s+Y\s+([-+]?\d+\.\d+)\s+Z", re.IGNORECASE)
    for line in text.splitlines():
        state_match = state_re.search(line)
        if state_match:
            current_state = int(state_match.group(1))
            continue
        tm_match = tm_re.search(line)
        if tm_match and current_state is not None and 1 <= current_state <= n_roots:
            dipoles[current_state - 1] = [float(tm_match.group(i)) for i in range(1, 4)]
    return dipoles


def parse_oscillator_strengths(text: str) -> Dict[int, float]:
    values: Dict[int, float] = {}
    current_state: Optional[int] = None
    for line in text.splitlines():
        state_match = re.search(r"Excited state\s+(\d+)", line, re.IGNORECASE)
        if state_match:
            current_state = int(state_match.group(1))
            continue
        strength_match = re.search(r"Strength\s*:\s*([-+]?\d+\.\d+(?:[Ee][-+]?\d+)?)", line, re.IGNORECASE)
        if strength_match and current_state is not None:
            values[current_state] = float(strength_match.group(1))
    return values


def parse_ground_dipole_debye(text: str) -> Optional[np.ndarray]:
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if re.search(r"Dipole Moment.*Debye", line, re.IGNORECASE):
            for candidate in lines[i + 1 : i + 8]:
                nums = re.findall(r"[-+]?\d+\.\d+(?:[Ee][-+]?\d+)?", candidate)
                if len(nums) >= 3:
                    return np.asarray([float(nums[0]), float(nums[1]), float(nums[2])], dtype=np.float64)
    for match in re.finditer(r"X\s+([-+]?\d+\.\d+)\s+Y\s+([-+]?\d+\.\d+)\s+Z\s+([-+]?\d+\.\d+)", text, re.IGNORECASE):
        return np.asarray([float(match.group(1)), float(match.group(2)), float(match.group(3))], dtype=np.float64)
    return None


def parse_gradient_matrix(text: str) -> Optional[np.ndarray]:
    lines = text.splitlines()
    grad_re = re.compile(r"Gradient of (?:the state energy|(?:DFT|SCF|CIS|HF|MP2|TDDFT|TDA|QM)\s+Energy)", re.IGNORECASE)
    last_idx: Optional[int] = None
    for i, line in enumerate(lines):
        if grad_re.search(line):
            last_idx = i
    if last_idx is None:
        return None
    components = {1: [], 2: [], 3: []}
    i = last_idx + 1
    while i < min(last_idx + 800, len(lines)):
        tokens = lines[i].split()
        if tokens and all(token.lstrip("-").isdigit() for token in tokens):
            ncols = len(tokens)
            i += 1
            for comp in range(1, 4):
                row = lines[i].split() if i < len(lines) else []
                i += 1
                if row and int(row[0]) == comp:
                    components[comp].extend(float(v) for v in row[1 : 1 + ncols])
        else:
            if components[1]:
                break
            i += 1
    if not (components[1] and components[2] and components[3]):
        return None
    natoms = len(components[1])
    grad = np.zeros((natoms, 3), dtype=np.float64)
    grad[:, 0] = components[1][:natoms]
    grad[:, 1] = components[2][:natoms]
    grad[:, 2] = components[3][:natoms]
    return grad


def parse_optimized_geometry(text: str) -> Tuple[Optional[List[str]], Optional[np.ndarray]]:
    lines = text.splitlines()
    conv_idx: Optional[int] = None
    for i, line in enumerate(lines):
        if "OPTIMIZATION CONVERGED" in line.upper():
            conv_idx = i
    if conv_idx is None:
        return None, None
    symbols: List[str] = []
    coords: List[List[float]] = []
    in_table = False
    for line in lines[conv_idx : min(conv_idx + 400, len(lines))]:
        match = re.match(r"\s+\d+\s+([A-Z][a-z]?)\s+([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)", line)
        if match:
            in_table = True
            symbols.append(match.group(1))
            coords.append([float(match.group(2)), float(match.group(3)), float(match.group(4))])
        elif in_table:
            break
    if not symbols:
        return None, None
    return symbols, np.asarray(coords, dtype=np.float64)


def parse_nac_vectors(text: str, natoms: int) -> Dict[Tuple[int, int], np.ndarray]:
    vectors: Dict[Tuple[int, int], np.ndarray] = {}
    lines = text.splitlines()
    pair_re = re.compile(r"between states\s+(\d+)\s+and\s+(\d+)", re.IGNORECASE)
    etf_re = re.compile(r"CIS derivative coupling (with|without) ETF", re.IGNORECASE)
    current_pair: Optional[Tuple[int, int]] = None
    i = 0
    while i < len(lines):
        pair_match = pair_re.search(lines[i])
        if pair_match:
            a, b = int(pair_match.group(1)), int(pair_match.group(2))
            current_pair = (min(a, b), max(a, b))
            i += 1
            continue
        etf_match = etf_re.search(lines[i])
        if etf_match and current_pair is not None:
            etf_type = etf_match.group(1).lower()
            while i < len(lines) and not re.search(r"Atom\s+X\s+Y\s+Z", lines[i], re.IGNORECASE):
                i += 1
            i += 1
            while i < len(lines) and re.match(r"^-{5,}", lines[i].strip()):
                i += 1
            rows: List[List[float]] = []
            for _ in range(natoms):
                if i >= len(lines):
                    break
                parts = lines[i].split()
                i += 1
                if len(parts) >= 4:
                    try:
                        rows.append([float(parts[1]), float(parts[2]), float(parts[3])])
                    except ValueError:
                        pass
            if len(rows) == natoms and (etf_type == "with" or current_pair not in vectors):
                vectors[current_pair] = np.asarray(rows, dtype=np.float64)
            continue
        i += 1
    return vectors


def require(value: Optional[object], message: str) -> object:
    if value is None:
        raise RuntimeError(message)
    return value


def qchem_state_pairs(n_roots: int) -> List[Tuple[int, int]]:
    return list(itertools.combinations(range(1, n_roots + 1), 2))


def parse_sp_products(sp_text: str, n_roots: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    gs_energy_h = require(parse_scf_energy(sp_text), "Could not parse Q-Chem ground-state energy")
    state_totals_h = parse_state_total_energies(sp_text)
    exc_ev = parse_excitation_energies_ev(sp_text)
    energies = np.zeros((n_roots + 1,), dtype=np.float64)
    total_energy = np.asarray([float(gs_energy_h) * HARTREE_TO_EV], dtype=np.float64)
    energies[0] = total_energy[0]
    for state in range(1, n_roots + 1):
        if state in state_totals_h:
            energies[state] = state_totals_h[state] * HARTREE_TO_EV
        elif state in exc_ev:
            energies[state] = total_energy[0] + exc_ev[state]
        else:
            raise RuntimeError(f"Missing Q-Chem energy for excited state S{state}")
    gs_dipole = parse_ground_dipole_debye(sp_text)
    if gs_dipole is None:
        raise RuntimeError("Could not parse Q-Chem ground-state dipole")
    transition_dipoles = parse_transition_dipoles(sp_text, n_roots)
    return energies, total_energy, gs_dipole, transition_dipoles


def run_gsopt(args: argparse.Namespace) -> int:
    symbols, coords = read_xyz(Path(args.xyz))
    out_dir = Path(args.output_dir).resolve()
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    qcin = raw_dir / f"{args.prefix}_qchem_gsopt.in"
    qcout = raw_dir / f"{args.prefix}_qchem_gsopt.out"
    qcin.write_text(build_gsopt_input(symbols, coords, args), encoding="utf-8", newline="\n")
    if args.prepare_only:
        print(f"Prepared {qcin}")
        return 0
    rc = run_qchem(qcin, qcout, f"{args.prefix}_qchem_gsopt", True, args)
    text = read_output(qcout)
    if rc != 0:
        raise RuntimeError(f"Q-Chem GS optimization failed with return code {rc}: {qcout}")
    energy_h = require(parse_scf_energy(text), "Could not parse Q-Chem optimized energy")
    grad = require(parse_gradient_matrix(text), "Could not parse Q-Chem optimized ground-state gradient")
    opt_symbols, opt_coords = parse_optimized_geometry(text)
    if opt_symbols is None or opt_coords is None:
        opt_symbols, opt_coords = symbols, coords
    forces = -np.asarray(grad, dtype=np.float64) * FORCE_AU_TO_EV_PER_ANGSTROM
    energy_ev = float(energy_h) * HARTREE_TO_EV
    write_xyz(out_dir / f"{args.prefix}_qchem_optimized_gs_geometry.xyz", opt_symbols, opt_coords, f"Q-Chem {args.method}/{args.basis} E={energy_ev:.16f} eV")
    np.savetxt(out_dir / f"{args.prefix}_optimized_reference_energy.txt", np.asarray([energy_ev]), fmt="%.16f")
    np.save(out_dir / f"{args.prefix}_optimized_reference_forces.npy", forces)
    summary = {
        "software": "Q-Chem",
        "method": args.method,
        "basis": qchem_basis(args.basis),
        "energy_ev": energy_ev,
        "forces_shape": list(forces.shape),
        "qchem_output": str(qcout),
    }
    (out_dir / f"{args.prefix}_qchem_gsopt_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote Q-Chem reference files in {out_dir}")
    return 0


def run_one_frame(frame_idx: int, coords: np.ndarray, z: np.ndarray, args: argparse.Namespace) -> Path:
    symbols = symbols_from_z(z)
    frame_dir = Path(args.shard_dir).resolve() / f"frame_{frame_idx:06d}"
    frame_dir.mkdir(parents=True, exist_ok=True)
    save_name = f"{args.prefix}_frame_{frame_idx:06d}"
    sp_in = frame_dir / "es_sp.in"
    sp_out = frame_dir / "es_sp.out"
    sp_in.write_text(build_sp_input(symbols, coords, args), encoding="utf-8", newline="\n")
    if args.prepare_only:
        return frame_dir / "prepared.marker"
    rc = run_qchem(sp_in, sp_out, save_name, True, args)
    if rc != 0:
        raise RuntimeError(f"Q-Chem SP failed for frame {frame_idx} with rc={rc}")
    sp_text = read_output(sp_out)
    energies, total_energy, gs_dipole, transition_dipoles = parse_sp_products(sp_text, args.n_roots)
    forces = np.zeros((args.n_roots + 1, len(symbols), 3), dtype=np.float64)
    for state in range(0, args.n_roots + 1):
        force_in = frame_dir / f"force_S{state}.in"
        force_out = frame_dir / f"force_S{state}.out"
        force_in.write_text(build_force_input(symbols, coords, args, state), encoding="utf-8", newline="\n")
        frc = run_qchem(force_in, force_out, None, None, args)
        if frc != 0:
            raise RuntimeError(f"Q-Chem FORCE S{state} failed for frame {frame_idx} with rc={frc}")
        grad = require(parse_gradient_matrix(read_output(force_out)), f"Could not parse FORCE S{state} gradient for frame {frame_idx}")
        forces[state] = -np.asarray(grad, dtype=np.float64) * FORCE_AU_TO_EV_PER_ANGSTROM
    nac_in = frame_dir / "nac.in"
    nac_out = frame_dir / "nac.out"
    nac_in.write_text(build_nac_input(symbols, coords, args), encoding="utf-8", newline="\n")
    nrc = run_qchem(nac_in, nac_out, None, None, args)
    if nrc != 0:
        raise RuntimeError(f"Q-Chem NAC failed for frame {frame_idx} with rc={nrc}")
    nac_raw = parse_nac_vectors(read_output(nac_out), len(symbols))
    nacrs = np.zeros((len(qchem_state_pairs(args.n_roots)), len(symbols), 3), dtype=np.float64)
    for pair_idx, pair in enumerate(qchem_state_pairs(args.n_roots)):
        if pair not in nac_raw:
            raise RuntimeError(f"Missing NAC pair S{pair[0]}-S{pair[1]} for frame {frame_idx}")
        nacrs[pair_idx] = nac_raw[pair] / BOHR_TO_ANGSTROM
    shard = Path(args.shard_dir).resolve() / f"shard_{frame_idx:06d}.npz"
    np.savez_compressed(
        shard,
        frame_idx=np.asarray([frame_idx], dtype=np.int64),
        energies=energies[None, :],
        total_energy=total_energy[None, :],
        gs_dipoles=gs_dipole[None, :],
        transition_dipoles=transition_dipoles[None, :, :],
        forces=forces[None, :, :, :],
        nacrs=nacrs[:, None, :, :],
    )
    return shard


def run_frame_chunk(args: argparse.Namespace) -> int:
    coords_all = np.load(args.coords)
    z_all = np.load(args.species)
    n_frames = int(coords_all.shape[0])
    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", args.array_task_id))
    chunk_size = int(args.chunk_size)
    start = task_id * chunk_size
    end = min(start + chunk_size, n_frames)
    if start >= n_frames:
        print(f"Array task {task_id} has no frames; n_frames={n_frames}")
        return 0
    for frame_idx in range(start, end):
        shard = run_one_frame(frame_idx, coords_all[frame_idx], z_all[frame_idx], args)
        print(f"Wrote {shard}")
    return 0


def run_collect(args: argparse.Namespace) -> int:
    coords = np.load(args.coords)
    z = np.load(args.species)
    n_frames, n_atoms = int(coords.shape[0]), int(coords.shape[1])
    n_roots = int(args.n_roots)
    shard_dir = Path(args.shard_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    energies = np.zeros((n_frames, n_roots + 1), dtype=np.float64)
    total_energy = np.zeros((n_frames, 1), dtype=np.float64)
    gs_dipoles = np.zeros((n_frames, 3), dtype=np.float64)
    transition_dipoles = np.zeros((n_frames, n_roots, 3), dtype=np.float64)
    forces = np.zeros((n_frames, n_roots + 1, n_atoms, 3), dtype=np.float64)
    nacrs = np.zeros((len(qchem_state_pairs(n_roots)), n_frames, n_atoms, 3), dtype=np.float64)
    missing: List[int] = []
    for frame_idx in range(n_frames):
        shard = shard_dir / f"shard_{frame_idx:06d}.npz"
        if not shard.exists():
            missing.append(frame_idx)
            continue
        data = np.load(shard)
        energies[frame_idx] = data["energies"][0]
        total_energy[frame_idx] = data["total_energy"][0]
        gs_dipoles[frame_idx] = data["gs_dipoles"][0]
        transition_dipoles[frame_idx] = data["transition_dipoles"][0]
        forces[frame_idx] = data["forces"][0]
        nacrs[:, frame_idx, :, :] = data["nacrs"][:, 0, :, :]
    if missing:
        preview = ", ".join(map(str, missing[:20]))
        raise RuntimeError(f"Missing {len(missing)} Q-Chem shard(s); first missing frame indices: {preview}")
    np.save(output_dir / "energies_all_states.npy", energies)
    np.save(output_dir / "total_energy.npy", total_energy)
    np.save(output_dir / "gs_dipoles.npy", gs_dipoles)
    np.save(output_dir / "transition_dipoles.npy", transition_dipoles)
    np.save(output_dir / "forces_all_states.npy", forces)
    np.save(output_dir / "nacrs_all_states.npy", nacrs)
    summary = {
        "software": "Q-Chem",
        "n_frames": n_frames,
        "n_atoms": n_atoms,
        "n_roots": n_roots,
        "energies_shape": list(energies.shape),
        "forces_shape": list(forces.shape),
        "nacrs_shape": list(nacrs.shape),
    }
    (output_dir / "qchem_collect_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Q-Chem backend for dataset_workflow")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--gsopt", action="store_true")
    mode.add_argument("--frame", action="store_true")
    mode.add_argument("--collect", action="store_true")
    parser.add_argument("--xyz", type=Path, help="Input XYZ for Q-Chem GS optimization")
    parser.add_argument("--coords", type=Path, help="Prepared R.npy coordinate file")
    parser.add_argument("--species", type=Path, help="Prepared Z.npy species file")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-dir", type=Path, default=Path("qchem_shards"))
    parser.add_argument("--prefix", default="molecule")
    parser.add_argument("--method", default="CAM-B3LYP")
    parser.add_argument("--basis", default="6-31G*")
    parser.add_argument("--n-roots", type=int, default=5)
    parser.add_argument("--charge", type=int, default=0)
    parser.add_argument("--multiplicity", type=int, default=1)
    parser.add_argument("--ncpu", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=1)
    parser.add_argument("--array-task-id", type=int, default=0)
    parser.add_argument("--qchem-exe", default="/usr/projects/ml4chem/Programs/qchem/bin/qchem")
    parser.add_argument("--qchem-env-file", default="")
    parser.add_argument("--qchem-scratch", default="./qchem_scratch")
    parser.add_argument("--qchem-mem-total-mb", type=int, default=125000)
    parser.add_argument("--max-scf-cycles", type=int, default=250)
    parser.add_argument("--max-cis-cycles", type=int, default=100)
    parser.add_argument("--geom-opt-max-cycles", type=int, default=200)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.gsopt and not args.xyz:
        parser.error("--gsopt requires --xyz")
    if (args.frame or args.collect) and (not args.coords or not args.species):
        parser.error("--frame/--collect require --coords and --species")
    return args


def main() -> int:
    args = parse_args()
    t0 = time.time()
    if args.gsopt:
        rc = run_gsopt(args)
    elif args.frame:
        rc = run_frame_chunk(args)
    else:
        rc = run_collect(args)
    print(f"Q-Chem backend finished in {time.time() - t0:.2f}s")
    return rc


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)

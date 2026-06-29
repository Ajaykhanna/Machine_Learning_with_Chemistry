#!/usr/bin/env python3
"""NEXMD ground-state optimization and dynamics setup/submission."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

from geometry_parser import extract_optimized_geometry, write_xyz_validation_file

ATOMIC_NUMBERS = {
    "H": 1,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Br": 35,
    "I": 53,
}


def atoms_from_xyz(xyz_path: Path) -> list[tuple[int, str]]:
    if not xyz_path.exists():
        raise FileNotFoundError(f"Missing XYZ geometry: {xyz_path}")
    atoms = []
    for line in xyz_path.read_text().splitlines()[2:]:
        parts = line.split()
        if len(parts) < 4:
            continue
        token = parts[0]
        atomic_number = int(token) if token.isdigit() else ATOMIC_NUMBERS[token]
        atoms.append((atomic_number, f"{atomic_number:>3}   {float(parts[1]):>12.6f}  {float(parts[2]):>12.6f}  {float(parts[3]):>12.6f}"))
    if not atoms:
        raise ValueError(f"No atoms parsed from {xyz_path}")
    return sorted(atoms, key=lambda item: item[0], reverse=True)


def atoms_from_gaussian(mol_dir: Path) -> list[tuple[int, str]]:
    log_file = mol_dir / f"{mol_dir.name}_gsopt.log"
    if not log_file.exists():
        matches = sorted(mol_dir.glob("**/*_gsopt.log"))
        if matches:
            log_file = matches[-1]
    atoms = extract_optimized_geometry(str(log_file)) if log_file.exists() else None
    if not atoms:
        raise RuntimeError(f"Could not extract optimized Gaussian geometry from {mol_dir}")
    write_xyz_validation_file(str(log_file.parent / f"{mol_dir.name}_gaussian_gsoptd.xyz"), atoms)
    return atoms


def update_input_ceon(template_path: Path, output_path: Path, sorted_atoms: list[tuple[int, str]]) -> None:
    if not template_path.exists():
        raise FileNotFoundError(f"Missing NEXMD template: {template_path}")

    num_atoms = len(sorted_atoms)
    lines = template_path.read_text().splitlines(keepends=True)
    new_lines = []
    skip_coord = False
    skip_veloc = False
    coord_found = False
    veloc_found = False

    for line in lines:
        stripped = line.strip()
        if "natoms=" in line:
            line = re.sub(r"natoms=\d+", f"natoms={num_atoms}", line)

        if stripped.startswith("&coord"):
            new_lines.append(line)
            for _, atom_line in sorted_atoms:
                new_lines.append(f"\t{atom_line}\n")
            skip_coord = True
            coord_found = True
            continue
        if stripped.startswith("&endcoord"):
            skip_coord = False
            new_lines.append(line)
            continue
        if skip_coord:
            continue

        if stripped.startswith("&veloc"):
            new_lines.append(line)
            for _ in range(num_atoms):
                new_lines.append("\t 0.000000000    0.000000000    0.000000000\n")
            skip_veloc = True
            veloc_found = True
            continue
        if stripped.startswith("&endveloc"):
            skip_veloc = False
            new_lines.append(line)
            continue
        if skip_veloc:
            continue

        new_lines.append(line)

    if not coord_found or not veloc_found:
        raise RuntimeError(f"Template must contain &coord and &veloc blocks: {template_path}")
    output_path.write_text("".join(new_lines))


def write_sbatch(template_sbatch: Path, output_sbatch: Path, molecule: str, mode: str) -> None:
    if not template_sbatch.exists():
        raise FileNotFoundError(f"Missing NEXMD sbatch template: {template_sbatch}")
    content = template_sbatch.read_text(encoding="utf-8-sig")
    job_name = f"{molecule}_{mode}"
    stage = "gsopt" if mode == "gs_opt" else "gsdyn"
    output_name = f"{molecule}_{stage}.out"
    content = re.sub(r"#SBATCH --job-name=.*", f"#SBATCH --job-name={job_name}", content)
    content = re.sub(r"^PREFIX=.*$", f"PREFIX=\"${{PREFIX:-{molecule}}}\"", content, flags=re.MULTILINE)
    content = re.sub(r"^NEXMD_OUTPUT=.*$", f"NEXMD_OUTPUT=\"${{NEXMD_OUTPUT:-{output_name}}}\"", content, flags=re.MULTILINE)
    output_sbatch.write_text(content)


def submit_job(sbatch_path: Path, cwd: Path, prepare_only: bool) -> None:
    if prepare_only:
        print(f"-> Prepared {sbatch_path.name}; not submitting.")
        return
    result = subprocess.run(["sbatch", sbatch_path.name], cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip())
    print(f"-> {result.stdout.strip()}")


def prepare_gs_opt(mol_dir: Path, out_dir: Path, base_dir: Path, prepare_only: bool) -> None:
    atoms = atoms_from_gaussian(mol_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    update_input_ceon(base_dir / "gsopt_input.ceon", out_dir / "input.ceon", atoms)
    write_sbatch(base_dir / "standard_nexmd_md.sbatch", out_dir / "submit_gs_opt.sbatch", mol_dir.name, "gs_opt")
    print(f"-> Wrote NEXMD GS optimization input in {out_dir}")
    submit_job(out_dir / "submit_gs_opt.sbatch", out_dir, prepare_only)


def prepare_gs_dyn(mol_dir: Path, out_dir: Path, base_dir: Path, prepare_only: bool) -> None:
    source_xyz = mol_dir / "nexmd" / "coords.xyz"
    atoms = atoms_from_xyz(source_xyz)
    out_dir.mkdir(parents=True, exist_ok=True)
    update_input_ceon(base_dir / "gsdyn_input.ceon", out_dir / "input.ceon", atoms)
    write_sbatch(base_dir / "standard_nexmd_md.sbatch", out_dir / "submit_gs_dyn.sbatch", mol_dir.name, "gs_dyn")
    print(f"-> Wrote NEXMD GS dynamics input in {out_dir}")
    submit_job(out_dir / "submit_gs_dyn.sbatch", out_dir, prepare_only)


def main() -> int:
    parser = argparse.ArgumentParser(description="NEXMD ground-state pipeline")
    parser.add_argument("-p", "--path", required=True, help="Molecule folder, e.g. ./example_molecule")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--gs_opt", action="store_true", help="Prepare/submit NEXMD AM1 GS optimization")
    group.add_argument("--gs_dyn", action="store_true", help="Prepare/submit NEXMD AM1 GS dynamics")
    parser.add_argument("--out_dir", help="Output directory. Defaults to molecule/nexmd or molecule/nexmd/gs_dyn")
    parser.add_argument("--prepare-only", action="store_true", help="Write files but do not call sbatch")
    args = parser.parse_args()

    package_root = Path(__file__).resolve().parents[1]
    base_dir = package_root / "templates"
    if not base_dir.exists():
        base_dir = Path.cwd()
    mol_dir = Path(args.path).resolve()
    if not mol_dir.is_dir():
        raise FileNotFoundError(f"Molecule directory not found: {mol_dir}")

    if args.gs_opt:
        out_dir = Path(args.out_dir).resolve() if args.out_dir else mol_dir / "nexmd"
        prepare_gs_opt(mol_dir, out_dir, base_dir, args.prepare_only)
    else:
        out_dir = Path(args.out_dir).resolve() if args.out_dir else mol_dir / "nexmd" / "gs_dyn"
        prepare_gs_dyn(mol_dir, out_dir, base_dir, args.prepare_only)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)












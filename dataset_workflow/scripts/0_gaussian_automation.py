#!/usr/bin/env python3
"""Gaussian screening automation for MLIP dataset candidate molecules."""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

from pipeline_config import (
    BRIGHT_OSCILLATOR_THRESHOLD,
    DARK_OSCILLATOR_THRESHOLD,
    DEFAULT_EXCITED_STATES,
    NEAR_DEGENERATE_GAP_EV,
    SCREENING_MIN_STATES,
)

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


@dataclass
class ExcitedState:
    state: int
    energy_ev: float
    wavelength_nm: float | None
    oscillator_strength: float


def read_sorted_unopt_xyz(xyz_path: Path) -> str:
    if not xyz_path.exists():
        raise FileNotFoundError(f"Missing required geometry file: {xyz_path}")

    rows = []
    for line in xyz_path.read_text().splitlines()[2:]:
        parts = line.split()
        if len(parts) < 4:
            continue
        symbol = parts[0]
        if symbol not in ATOMIC_NUMBERS:
            raise ValueError(f"Unsupported atom symbol '{symbol}' in {xyz_path}")
        rows.append((ATOMIC_NUMBERS[symbol], symbol, parts[1], parts[2], parts[3]))

    if not rows:
        raise ValueError(f"No coordinates found in {xyz_path}")

    rows.sort(key=lambda item: item[0], reverse=True)
    return "\n".join(
        f"{symbol:<2} {float(x):>16.8f} {float(y):>16.8f} {float(z):>16.8f}"
        for _, symbol, x, y, z in rows
    )


def gsopt_com_template(chk_name: str, title_name: str, coordinates_block: str) -> str:
    return f"""%chk={chk_name}
# opt freq b3lyp/6-31g(d)

{title_name} GS Opt Freq

0 1
{coordinates_block}

"""


def exsp_com_template(oldchk_name: str, chk_name: str, title_name: str, nstates: int) -> str:
    return f"""%oldchk={oldchk_name}
%chk={chk_name}
# tda=(nstates={nstates},root=1) cam-b3lyp/6-31g(d) geom=allcheck guess=read

{title_name} Excited State Calculations


"""


def gaussian_sbatch_template(job_name: str, com_name: str, log_name: str, err_name: str, qlog_name: str, mem_gb: int, cpus: int, walltime: str) -> str:
    return f"""#!/bin/csh
#SBATCH --partition=shared-spr
#SBATCH --qos=long
#SBATCH --time={walltime}
#SBATCH --nodes=1
#SBATCH --ntasks={cpus}
#SBATCH --mem={mem_gb}GB
#SBATCH --job-name={job_name}
#SBATCH --error={err_name}
#SBATCH --output={qlog_name}

setenv MY_SCRATCH /tmp/akhanna2/GAUSSIAN_SCR/${{SLURM_JOBID}}/
mkdir -p ${{MY_SCRATCH}}
setenv GAUSS_SCRDIR ${{MY_SCRATCH}}

setenv g16root /usr/projects/cint/Gaussian/g16A03
source ${{g16root}}/g16/bsd/g16.login

echo "Job started at `date`" > {qlog_name}
echo "Job ID: ${{SLURM_JOBID}}" >> {qlog_name}
echo "Working directory: ${{PWD}}" >> {qlog_name}
echo "Scratch directory: ${{GAUSS_SCRDIR}}" >> {qlog_name}
echo "Memory requested: {mem_gb}GB" >> {qlog_name}
echo "CPUs requested: {cpus}" >> {qlog_name}

g16 -m={mem_gb}GB -p={cpus} < {com_name} > {log_name}
set status_code=$status
rm -rf ${{MY_SCRATCH}}
exit $status_code
"""



def stage_file(mol_dir: Path, stage: str, suffix: str) -> Path:
    return mol_dir / f"{mol_dir.name}_{stage}.{suffix}"


def first_existing(paths: list[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]

def run_command(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, capture_output=True, text=True)


def submit_job(sbatch_name: str, cwd: Path) -> str:
    result = run_command(["sbatch", sbatch_name], cwd)
    if result.returncode != 0:
        raise RuntimeError(f"sbatch failed:\n{result.stderr.strip()}")
    match = re.search(r"Submitted batch job\s+(\d+)", result.stdout)
    if not match:
        raise RuntimeError(f"Could not parse job id from sbatch output: {result.stdout}")
    return match.group(1)


def slurm_job_state(job_id: str, cwd: Path) -> str | None:
    squeue = run_command(["squeue", "-h", "-j", job_id, "-o", "%T"], cwd)
    if squeue.returncode == 0 and squeue.stdout.strip():
        return squeue.stdout.strip().splitlines()[0]
    sacct = run_command(["sacct", "-j", job_id, "--format=State", "--noheader"], cwd)
    if sacct.returncode == 0 and sacct.stdout.strip():
        states = [line.strip().split()[0] for line in sacct.stdout.splitlines() if line.strip()]
        if states:
            return states[0]
    return None


def wait_for_job(job_id: str, cwd: Path, poll_seconds: int) -> str:
    terminal_prefixes = ("COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY")
    while True:
        state = slurm_job_state(job_id, cwd)
        if state and state.startswith(terminal_prefixes):
            return state
        print(f"  -> Job {job_id} state: {state or 'unknown'}; polling again in {poll_seconds}s")
        time.sleep(poll_seconds)


def gaussian_completed(log_path: Path) -> bool:
    return log_path.exists() and "Normal termination of Gaussian" in log_path.read_text(errors="ignore")


def parse_frequencies(log_path: Path) -> list[float]:
    frequencies = []
    for line in log_path.read_text(errors="ignore").splitlines():
        if "Frequencies --" in line:
            frequencies.extend(float(value) for value in line.split("--", 1)[1].split())
    return frequencies


def collect_gsopt(mol_dir: Path) -> bool:
    log_path = stage_file(mol_dir, "gsopt", "log")
    if not gaussian_completed(log_path):
        raise RuntimeError(f"Gaussian GS optimization did not terminate normally: {log_path}")
    frequencies = parse_frequencies(log_path)
    if not frequencies:
        raise RuntimeError(f"No frequencies found in {log_path}")
    freq_path = mol_dir / "mol_freq.txt"
    has_negative = any(freq < 0.0 for freq in frequencies)
    freq_path.write_text("\n".join([
        f"source_log: {log_path.name}",
        f"minimum_frequency_cm-1: {min(frequencies):.8f}",
        f"has_negative_frequency: {has_negative}",
        "frequencies_cm-1:",
        *[f"{freq:.8f}" for freq in frequencies],
    ]) + "\n")
    if has_negative:
        print(f"  -> Negative frequency detected. Wrote {freq_path}; stopping this molecule.")
        return False
    print(f"  -> Frequency validation passed. Wrote {freq_path}")
    return True


def parse_excited_states(log_path: Path) -> list[ExcitedState]:
    states = []
    pattern = re.compile(r"Excited State\s+(\d+):.*?([\d.]+)\s+eV(?:\s+([\d.]+)\s+nm)?.*?f=([\d.Ee+-]+)")
    for line in log_path.read_text(errors="ignore").splitlines():
        match = pattern.search(line)
        if match:
            states.append(ExcitedState(
                state=int(match.group(1)),
                energy_ev=float(match.group(2)),
                wavelength_nm=float(match.group(3)) if match.group(3) else None,
                oscillator_strength=float(match.group(4)),
            ))
    return states


def classify_state(oscillator_strength: float) -> str:
    if oscillator_strength > BRIGHT_OSCILLATOR_THRESHOLD:
        return "bright"
    if oscillator_strength < DARK_OSCILLATOR_THRESHOLD:
        return "dark"
    return "partially_allowed"


def write_exsp_summary(mol_dir: Path, states: list[ExcitedState]) -> dict[str, object]:
    txt_path = mol_dir / "mol_exsp.txt"
    lines = ["state energy_ev wavelength_nm oscillator_strength class"]
    for state in states:
        wavelength = "NA" if state.wavelength_nm is None else f"{state.wavelength_nm:.4f}"
        lines.append(f"{state.state} {state.energy_ev:.8f} {wavelength} {state.oscillator_strength:.8f} {classify_state(state.oscillator_strength)}")
    txt_path.write_text("\n".join(lines) + "\n")
    gaps = [states[idx + 1].energy_ev - states[idx].energy_ev for idx in range(min(len(states) - 1, SCREENING_MIN_STATES - 1))]
    near_degenerate_pairs = sum(gap < NEAR_DEGENERATE_GAP_EV for gap in gaps)
    bright_count = sum(state.oscillator_strength > BRIGHT_OSCILLATOR_THRESHOLD for state in states[:SCREENING_MIN_STATES])
    dark_count = sum(state.oscillator_strength < DARK_OSCILLATOR_THRESHOLD for state in states[:SCREENING_MIN_STATES])
    return {
        "molecule": mol_dir.name,
        "n_states_parsed": len(states),
        "first_state_ev": states[0].energy_ev if states else "",
        "min_adjacent_gap_first5_ev": min(gaps) if gaps else "",
        "near_degenerate_pairs_first5": near_degenerate_pairs,
        "bright_states_first5": bright_count,
        "dark_states_first5": dark_count,
        "screening_rank_score": near_degenerate_pairs * 10 + bright_count + dark_count,
        "passes_basic_interest": len(states) >= SCREENING_MIN_STATES and near_degenerate_pairs > 0,
    }


def plot_exsp(mol_dir: Path, states: list[ExcitedState]) -> None:
    if plt is None or not states:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    labels = [state.state for state in states]
    energies = [state.energy_ev for state in states]
    oscillator_strengths = [state.oscillator_strength for state in states]
    axes[0].bar(labels, energies, color="#2f6f8f")
    axes[0].set_xlabel("Excited state")
    axes[0].set_ylabel("Transition energy (eV)")
    axes[0].set_title("Excitation energies")
    colors = ["#c0392b" if value > BRIGHT_OSCILLATOR_THRESHOLD else "#7f8c8d" if value < DARK_OSCILLATOR_THRESHOLD else "#d4a017" for value in oscillator_strengths]
    axes[1].bar(labels, oscillator_strengths, color=colors)
    axes[1].axhline(BRIGHT_OSCILLATOR_THRESHOLD, color="#c0392b", linestyle="--", linewidth=1)
    axes[1].axhline(DARK_OSCILLATOR_THRESHOLD, color="#7f8c8d", linestyle=":", linewidth=1)
    axes[1].set_xlabel("Excited state")
    axes[1].set_ylabel("Oscillator strength")
    axes[1].set_title("Bright/dark screening")
    fig.tight_layout()
    fig.savefig(mol_dir / "mol_exsp_screening.png", dpi=200)
    plt.close(fig)


def update_screening_summary(base_dir: Path, row: dict[str, object]) -> None:
    csv_path = base_dir / "screening_summary.csv"
    rows = []
    if csv_path.exists():
        with csv_path.open(newline="") as handle:
            rows = [existing for existing in csv.DictReader(handle) if existing["molecule"] != row["molecule"]]
    rows.append({key: str(value) for key, value in row.items()})
    rows.sort(key=lambda item: float(item["screening_rank_score"]), reverse=True)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  -> Updated ranked screening summary: {csv_path}")


def collect_exsp(mol_dir: Path, base_dir: Path) -> None:
    log_path = stage_file(mol_dir, "exsp", "log")
    if not gaussian_completed(log_path):
        raise RuntimeError(f"Gaussian excited-state single point did not terminate normally: {log_path}")
    states = parse_excited_states(log_path)
    if not states:
        raise RuntimeError(f"No excited states found in {log_path}")
    row = write_exsp_summary(mol_dir, states)
    plot_exsp(mol_dir, states)
    update_screening_summary(base_dir, row)
    print(f"  -> Wrote excited-state summary for {mol_dir.name}")


def molecule_dirs(base_dir: Path, path: str | None) -> list[Path]:
    if path:
        target = Path(path).resolve()
        if not target.is_dir():
            raise FileNotFoundError(f"Target molecule directory not found: {target}")
        return [target]
    return sorted(item for item in base_dir.iterdir() if item.is_dir() and not item.name.startswith(".") and (item / "unopt.xyz").exists())


def deploy_gsopt(mol_dir: Path, args: argparse.Namespace) -> bool:
    mol_name = mol_dir.name
    print(f"\n[GS-OPT] {mol_name}")
    coordinates = read_sorted_unopt_xyz(mol_dir / "unopt.xyz")
    com_name = f"{mol_name}_gsopt.com"
    chk_name = f"{mol_name}_gsopt.chk"
    log_name = f"{mol_name}_gsopt.log"
    err_name = f"{mol_name}_gsopt.err"
    qlog_name = f"{mol_name}_gsopt.qlog"
    sbatch_name = f"submit_{mol_name}_gsopt.sbatch"
    (mol_dir / com_name).write_text(gsopt_com_template(chk_name, mol_name, coordinates))
    (mol_dir / sbatch_name).write_text(gaussian_sbatch_template(f"{mol_name}_gsopt", com_name, log_name, err_name, qlog_name, args.mem_gb, args.cpus, args.walltime))
    if args.prepare_only:
        print("  -> Prepared Gaussian GS optimization files.")
        return True
    job_id = submit_job(sbatch_name, mol_dir)
    print(f"  -> Submitted SLURM job {job_id}")
    state = wait_for_job(job_id, mol_dir, args.poll_seconds)
    if not state.startswith("COMPLETED"):
        raise RuntimeError(f"GS optimization job {job_id} ended with state {state}")
    return collect_gsopt(mol_dir)


def deploy_exsp(mol_dir: Path, args: argparse.Namespace, base_dir: Path) -> bool:
    mol_name = mol_dir.name
    print(f"\n[EX-SP] {mol_name}")
    gs_chk = stage_file(mol_dir, "gsopt", "chk")
    if not gs_chk.exists():
        raise FileNotFoundError(f"Missing checkpoint from GS optimization: {gs_chk}")
    com_name = f"{mol_name}_exsp.com"
    chk_name = f"{mol_name}_exsp.chk"
    log_name = f"{mol_name}_exsp.log"
    err_name = f"{mol_name}_exsp.err"
    qlog_name = f"{mol_name}_exsp.qlog"
    sbatch_name = f"submit_{mol_name}_exsp.sbatch"
    (mol_dir / com_name).write_text(exsp_com_template(gs_chk.name, chk_name, mol_name, args.nstates))
    (mol_dir / sbatch_name).write_text(gaussian_sbatch_template(f"{mol_name}_exsp", com_name, log_name, err_name, qlog_name, args.mem_gb, args.cpus, args.walltime))
    if args.prepare_only:
        print("  -> Prepared Gaussian excited-state single-point files.")
        return True
    job_id = submit_job(sbatch_name, mol_dir)
    print(f"  -> Submitted SLURM job {job_id}")
    state = wait_for_job(job_id, mol_dir, args.poll_seconds)
    if not state.startswith("COMPLETED"):
        raise RuntimeError(f"Excited-state job {job_id} ended with state {state}")
    collect_exsp(mol_dir, base_dir)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Gaussian screening pipeline")
    parser.add_argument("-p", "--path", help="Molecule folder. Defaults to all folders containing unopt.xyz.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--gsopt", action="store_true", help="Submit/monitor/collect GS optimization.")
    group.add_argument("--exsp", action="store_true", help="Submit/monitor/collect excited-state single point.")
    group.add_argument("--collect-gsopt", action="store_true", help="Only parse an existing GS optimization log.")
    group.add_argument("--collect-exsp", action="store_true", help="Only parse an existing excited-state log.")
    parser.add_argument("--nstates", type=int, default=DEFAULT_EXCITED_STATES, help="Gaussian excited states to compute.")
    parser.add_argument("--mem-gb", type=int, default=100, help="SLURM and Gaussian memory in GB.")
    parser.add_argument("--cpus", type=int, default=16, help="Gaussian CPU count.")
    parser.add_argument("--walltime", default="05:00:00", help="SLURM walltime.")
    parser.add_argument("--poll-seconds", type=int, default=60, help="SLURM polling interval.")
    parser.add_argument("--prepare-only", action="store_true", help="Write inputs but do not submit.")
    args = parser.parse_args()
    base_dir = Path.cwd()
    targets = molecule_dirs(base_dir, args.path)
    if not targets:
        raise RuntimeError("No molecule folders found.")
    processed = 0
    for mol_dir in targets:
        if args.gsopt:
            processed += int(deploy_gsopt(mol_dir, args))
        elif args.exsp:
            processed += int(deploy_exsp(mol_dir, args, base_dir))
        elif args.collect_gsopt:
            processed += int(collect_gsopt(mol_dir))
        elif args.collect_exsp:
            collect_exsp(mol_dir, base_dir)
            processed += 1
    print(f"\nProcessed {processed}/{len(targets)} molecule(s).")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)


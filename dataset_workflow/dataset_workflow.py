#!/usr/bin/env python3
"""Molecule-agnostic dataset generation workflow driver.

This orchestrates the existing independently runnable stage scripts, writes
step-specific SLURM files under each molecule's slurm/ directory, and optionally
submits/monitors the jobs step-by-step.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import ast
import logging
import os
import re
import shutil
import subprocess
import shlex
import math

import numpy as np
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

STEP_ORDER = [
    "gaussian_gsopt",
    "gaussian_exsp",
    "qchem_gsopt",
    "nexmd_opt",
    "nexmd_gsdyn",
    "extract_frames",
    "prepare_frames",
    "pyseqm_batch_exsp",
    "pyseqm_opt",
    "qchem_batch_exsp",
    "qchem_collect",
    "generate_dataset",
]

STEP_FLAGS = {
    "gaussian_gsopt": "run_gaussian_gsopt",
    "gaussian_exsp": "run_gaussian_exsp",
    "qchem_gsopt": "run_qchem_gsopt",
    "nexmd_opt": "run_nexmd_opt",
    "nexmd_gsdyn": "run_nexmd_gsdyn",
    "extract_frames": "run_extract_frames",
    "prepare_frames": "run_prepare_frames",
    "pyseqm_batch_exsp": "run_pyseqm_batch_exsp",
    "pyseqm_opt": "run_pyseqm_opt",
    "qchem_batch_exsp": "run_qchem_batch_exsp",
    "qchem_collect": "run_qchem_collect",
    "generate_dataset": "run_generate_dataset",
}

DEFAULTS: Dict[str, Any] = {
    "mode": "test",
    "submit_jobs": False,
    "monitor_jobs": True,
    "continue": False,
    "overwrite": False,
    "wrapper_partition": "general",
    "default_partition": "general",
    "gpu_partition": "ml4chem",
    "gpu_gres": "gpu:4",
    "pyseqm_device": "auto",
    "pyseqm_cpus": 8,
    "pyseqm_mem": "5G",
    "pyseqm_opt_cpus": 8,
    "pyseqm_opt_mem": "5G",
    "quantum_backend": "pyseqm",
    "qchem_profile": "darwin",
    "qchem_exe": "/usr/projects/ml4chem/Programs/qchem/bin/qchem",
    "qchem_env_file": "/vast/home/akhanna2/data/software/qchem/qchem_darwin.sh",
    "qchem_scratch": "./qchem_scratch",
    "qchem_python_bin": "/usr/projects/cint/anaconda3/gpu4pyscf/bin/python",
    "qchem_partition": "shared-spr",
    "qchem_account": "y2020-bf",
    "qchem_qos": "long",
    "qchem_cpus": 32,
    "qchem_gsopt_mem": "64G",
    "qchem_gsopt_walltime": "08:00:00",
    "qchem_mem": "128G",
    "qchem_walltime": "24:00:00",
    "qchem_collect_mem": "32G",
    "qchem_collect_walltime": "02:00:00",
    "qchem_method": "CAM-B3LYP",
    "qchem_basis": "6-31G*",
    "qchem_n_roots": 5,
    "qchem_array_chunk_size": 1,
    "qchem_charge": 0,
    "qchem_multiplicity": 1,
    "qchem_mem_total_mb": 125000,
    "python_bin": "/usr/projects/ml4chem/akhanna2/softwares/envs/ml_env/bin/python",
    "pyseqm_python_bin": "/usr/projects/ml4chem/akhanna2/softwares/envs/ml_env/bin/python",
    "conda_env": "ml_env",
    "gaussian_root": "/usr/projects/cint/Gaussian/g16A03",
    "gaussian_exe": "g16",
    "gaussian_library_path": "",
    "gaussian_scratch_root": "/tmp/akhanna2/GAUSSIAN_SCR",
    "nexmd_root": "",
    "nexmd_exe": "/usr/projects/ml4chem/akhanna2/softwares/NEXMD/nexmd.exe",
    "nexmd_library_path": "",
    "pyseqm_root": "",
    "pyseqm_library_path": "",
    "cuda_path": "",
    "gaussian_mem_gb": 100,
    "gaussian_cpus": 16,
    "gaussian_walltime": "05:00:00",
    "gaussian_nstates": 10,
    "nexmd_n_class_steps_test": 10000,
    "nexmd_n_class_steps_production": 5000000,
    "nexmd_time_step": 0.1,
    "nexmd_out_coords_steps": 100,
    "frame_start_test": 1,
    "frame_stop_test": 101,
    "frame_start_production": 5000,
    "frame_stop_production": 55001,
    "frame_step": 1,
    "frame_workers": 8,
    "validation_samples": 10,
    "full_validation": False,
    "no_validation": False,
    "n_states": 11,
    "pyseqm_batch_size": 500,
    "pyseqm_method": "AM1",
    "pyseqm_scf_eps": "1e-10",
    "pyseqm_cis_tol": "1e-8",
    "gpu_ids": "0 1 2 3",
    "dataset_chop_start_test": 101,
    "dataset_chop_end_test": 102,
    "dataset_chop_start_production": 10000,
    "dataset_chop_end_production": 50001,
    "dataset_chop_step": 10000,
    "dataset_workers": 16,
    "oscillator_bright_threshold": 0.1,
    "oscillator_dark_threshold": 0.05,
    "excitation_energy_threshold_ev": 0.5,
}



QCHEM_PROFILE_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "darwin": {
        "qchem_env_file": "/vast/home/akhanna2/data/software/qchem/qchem_darwin.sh",
        "qchem_exe": "/usr/projects/ml4chem/Programs/qchem/bin/qchem",
        "qchem_python_bin": "/usr/projects/cint/anaconda3/gpu4pyscf/bin/python",
        "qchem_partition": "shared-spr",
        "qchem_account": "y2020-bf",
        "qchem_qos": "long",
        "qchem_cpus": 32,
        "qchem_gsopt_mem": "64G",
        "qchem_gsopt_walltime": "08:00:00",
        "qchem_mem": "128G",
        "qchem_walltime": "24:00:00",
    },
    "chicoma": {
        "qchem_env_file": "/usr/projects/ml4chem/envs/qchem.sh",
        "qchem_exe": "/usr/projects/ml4chem/Programs/qchem/bin/qchem",
        "qchem_python_bin": "/usr/projects/ml4chem/akhanna2/conda_envs/hiphop_env/bin/python",
        "qchem_scratch": "/users/akhanna2/scratch/qchem/tdpp",
        "qchem_partition": "standard",
        "qchem_account": "s17_cint",
        "qchem_qos": "",
        "qchem_cpus": 32,
        "qchem_gsopt_mem": "128G",
        "qchem_gsopt_walltime": "06:00:00",
        "qchem_mem": "128G",
        "qchem_walltime": "16:00:00",
    },
}
def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value!r}")


def parse_value(value: str) -> Any:
    value = value.strip()
    if not value:
        return ""
    if value.lower() in {"true", "false"}:
        return parse_bool(value)
    if "," in value and not (value.startswith("{") or value.startswith("[")):
        return [parse_value(part) for part in value.split(",")]
    try:
        return ast.literal_eval(value)
    except Exception:
        pass
    try:
        if re.match(r"^[+-]?\d+$", value):
            return int(value)
        if re.match(r"^[+-]?(\d+\.\d*|\d*\.\d+)([eE][+-]?\d+)?$", value):
            return float(value)
    except Exception:
        pass
    return value.strip('"').strip("'")


def load_config(path: Optional[Path]) -> Dict[str, Any]:
    config: Dict[str, Any] = {}
    if path is None:
        return config
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        raw = lines[i]
        line = raw.strip()
        i += 1
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().lower()
        value = value.strip()
        if value == "{":
            block: Dict[str, str] = {}
            while i < len(lines):
                inner = lines[i].strip()
                i += 1
                if not inner or inner.startswith("#"):
                    continue
                if inner == "}":
                    break
                if ":" not in inner:
                    continue
                name, mol_path = inner.split(":", 1)
                block[name.strip().strip('"').strip("'")] = mol_path.strip().rstrip(",").strip('"').strip("'")
            config[key] = block
        else:
            config[key] = parse_value(value)
    return config


def coalesce(config: Dict[str, Any], key: str, default: Any = None) -> Any:
    return config.get(key.lower(), DEFAULTS.get(key.lower(), default))


def as_path(value: Any, base: Path) -> Path:
    p = Path(str(value)).expanduser()
    return p if p.is_absolute() else (base / p).resolve()


@dataclass
class WorkflowConfig:
    project_root: Path
    molecules: Dict[str, Path]
    values: Dict[str, Any]

    @property
    def mode(self) -> str:
        return str(coalesce(self.values, "mode", "test")).lower()

    @property
    def submit_jobs(self) -> bool:
        return parse_bool(coalesce(self.values, "submit_jobs", False))

    @property
    def monitor_jobs(self) -> bool:
        return parse_bool(coalesce(self.values, "monitor_jobs", True))

    @property
    def do_continue(self) -> bool:
        return parse_bool(coalesce(self.values, "continue", False))

    @property
    def overwrite(self) -> bool:
        return parse_bool(coalesce(self.values, "overwrite", False))

    @property
    def python_bin(self) -> str:
        return str(coalesce(self.values, "python_bin"))

    @property
    def default_partition(self) -> str:
        return str(coalesce(self.values, "default_partition", "general"))

    @property
    def wrapper_partition(self) -> str:
        return str(coalesce(self.values, "wrapper_partition", DEFAULTS["wrapper_partition"]))

    @property
    def gpu_partition(self) -> str:
        return str(coalesce(self.values, "gpu_partition", "ml4chem"))

    @property
    def pyseqm_python_bin(self) -> str:
        return str(coalesce(self.values, "pyseqm_python_bin", DEFAULTS["pyseqm_python_bin"]))

    @property
    def script_root(self) -> Path:
        return self.project_root / "scripts"

    def runtime_value(self, key: str) -> str:
        return str(coalesce(self.values, key, DEFAULTS.get(key, ""))).strip()


@dataclass
class MoleculeContext:
    name: str
    source_dir: Path
    work_dir: Path
    prefix: str
    slurm_dir: Path
    gaussian_dir: Path
    nexmd_dir: Path
    pyseqm_dir: Path
    qchem_dir: Path
    dataset_dir: Path
    frame_dir: Path
    prepared_dir: Path
    pyseqm_props_dir: Path
    pyseqm_ref_dir: Path
    qchem_props_dir: Path
    qchem_ref_dir: Path


@dataclass
class StepResult:
    name: str
    sbatch: Path
    job_id: Optional[str] = None
    skipped: bool = False


class Workflow:
    def __init__(self, config: WorkflowConfig) -> None:
        self.cfg = config
        self.log = logging.getLogger("dataset_workflow")

    def build_context(self, name: str, mol_dir: Path) -> MoleculeContext:
        work_dir = mol_dir.resolve()
        prefix = str(self.cfg.values.get(f"{name.lower()}_prefix", self.cfg.values.get("prefix", name.lower())))
        slurm_dir = work_dir / "slurm"
        gaussian_dir = work_dir / "gaussian" / name
        nexmd_dir = work_dir / "nexmd"
        pyseqm_dir = work_dir / "pyseqm"
        qchem_dir = work_dir / "qchem"
        dataset_dir = as_path(self.cfg.values.get("dataset_output_dir", work_dir / "dataset"), self.cfg.project_root)
        if "dataset_output_dir" not in self.cfg.values:
            dataset_dir = work_dir / "dataset"
        frame_dir = nexmd_dir / "gs_dyn" / self.frame_output_name()
        prepared_dir = pyseqm_dir / "prepared_frames"
        pyseqm_props_dir = pyseqm_dir / "batch_exsp"
        pyseqm_ref_dir = pyseqm_dir / "gsopt_reference"
        qchem_props_dir = qchem_dir / "batch_exsp"
        qchem_ref_dir = qchem_dir / "gsopt_reference"
        return MoleculeContext(name, mol_dir, work_dir, prefix, slurm_dir, gaussian_dir, nexmd_dir, pyseqm_dir, qchem_dir, dataset_dir, frame_dir, prepared_dir, pyseqm_props_dir, pyseqm_ref_dir, qchem_props_dir, qchem_ref_dir)

    def frame_output_name(self) -> str:
        stop = self.mode_value("frame_stop")
        start = self.mode_value("frame_start")
        count = int(stop) - int(start) + 1
        return f"{count}_10fs_frames"

    def mode_value(self, stem: str) -> Any:
        explicit = self.cfg.values.get(stem)
        if explicit is not None:
            return explicit
        key = f"{stem}_{self.cfg.mode}"
        return coalesce(self.cfg.values, key)

    def quantum_backend(self) -> str:
        backend = str(coalesce(self.cfg.values, "quantum_backend", "pyseqm")).strip().lower()
        if backend not in {"pyseqm", "qchem"}:
            raise ValueError("QUANTUM_BACKEND must be pyseqm or qchem")
        return backend

    def selected_steps(self) -> List[str]:
        selected = [step for step, flag in STEP_FLAGS.items() if parse_bool(self.cfg.values.get(flag, False))]
        if parse_bool(self.cfg.values.get("run_pre_pyseqm", False)) and "prepare_frames" not in selected:
            selected.append("prepare_frames")
        if not selected:
            selected = ["qchem_gsopt"] if self.quantum_backend() == "qchem" else ["gaussian_gsopt", "gaussian_exsp"]
        if self.cfg.do_continue and selected:
            start = STEP_ORDER.index(selected[0])
            steps = STEP_ORDER[start:]
            if self.quantum_backend() == "qchem":
                return [s for s in steps if s not in {"gaussian_gsopt", "gaussian_exsp", "pyseqm_batch_exsp", "pyseqm_opt"}]
            return [s for s in steps if s not in {"qchem_gsopt", "qchem_batch_exsp", "qchem_collect"}]
        return selected

    def ensure_layout(self, ctx: MoleculeContext) -> None:
        for d in [ctx.slurm_dir, ctx.gaussian_dir, ctx.nexmd_dir, ctx.pyseqm_dir, ctx.qchem_dir, ctx.dataset_dir, ctx.pyseqm_ref_dir, ctx.qchem_ref_dir, ctx.qchem_props_dir]:
            d.mkdir(parents=True, exist_ok=True)
        src_unopt = ctx.source_dir / "unopt.xyz"
        dst_unopt = ctx.gaussian_dir / "unopt.xyz"
        if src_unopt.exists() and (self.cfg.overwrite or not dst_unopt.exists()):
            shutil.copy2(src_unopt, dst_unopt)

    def completed(self, ctx: MoleculeContext, step: str) -> bool:
        checks = {
            "gaussian_gsopt": ctx.gaussian_dir / "mol_freq.txt",
            "gaussian_exsp": ctx.gaussian_dir / "mol_exsp.txt",
            "qchem_gsopt": ctx.qchem_ref_dir / f"{ctx.prefix}_optimized_reference_forces.npy",
            "nexmd_opt": ctx.nexmd_dir / "coords.xyz",
            "nexmd_gsdyn": ctx.nexmd_dir / "gs_dyn" / "velocity.out",
            "extract_frames": ctx.frame_dir,
            "prepare_frames": ctx.prepared_dir / f"{ctx.prefix}_R.npy",
            "pyseqm_batch_exsp": ctx.pyseqm_props_dir / "energies_all_states.npy",
            "pyseqm_opt": ctx.pyseqm_ref_dir / f"{ctx.prefix}_optimized_reference_forces.npy",
            "qchem_batch_exsp": ctx.qchem_props_dir / "shards",
            "qchem_collect": ctx.qchem_props_dir / "energies_all_states.npy",
            "generate_dataset": ctx.dataset_dir / f"{ctx.prefix}_dataset.log",
        }
        target = checks[step]
        return target.exists()

    def require(self, path: Path, step: str) -> None:
        if not path.exists():
            raise FileNotFoundError(
                f"Required input for {step} is missing: {path}. "
                "Run prior steps with --continue true or provide the required path in dataset_workflow.in."
            )

    def sbatch_header(self, job_name: str, partition: str, cpus: int = 8, mem: str = "32G", walltime: str = "04:00:00", extra_directives: str = "") -> str:
        extra = f"{extra_directives.rstrip()}\n" if extra_directives.strip() else ""
        return f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --time={walltime}
#SBATCH --partition={partition}
#SBATCH --qos=long
{extra}

set -euo pipefail

"""

    def write_sbatch(self, ctx: MoleculeContext, step: str, body: str, partition: Optional[str] = None, cpus: int = 8, mem: str = "32G", walltime: str = "04:00:00", extra_directives: str = "") -> Path:
        ctx.slurm_dir.mkdir(parents=True, exist_ok=True)
        sbatch = ctx.slurm_dir / f"submit_{ctx.prefix}_{step}.sbatch"
        content = self.sbatch_header(f"{ctx.prefix}_{step}", partition or self.cfg.default_partition, cpus, mem, walltime, extra_directives) + body
        sbatch.write_text(content, encoding="utf-8", newline="\n")
        return sbatch

    def python_cmd(self, script: str, *args: str) -> str:
        quoted = " ".join(shlex.quote(str(a)) for a in args if a != "")
        if script == "06_qchem_backend.py":
            python_bin = str(coalesce(self.cfg.values, "qchem_python_bin", DEFAULTS["qchem_python_bin"]))
        else:
            python_bin = self.cfg.pyseqm_python_bin if script in {"04_run_pyseqm_properties.py", "run_pyseqm_electronic_properties_calcs.py", "pyseqm_optimized_geom.py"} else self.cfg.python_bin
        return f'"{python_bin}" "{self.cfg.script_root / script}" {quoted}'.strip()

    def optional_cli_args(self, *keys: str) -> List[str]:
        args: List[str] = []
        for key in keys:
            value = self.cfg.runtime_value(key)
            if value:
                args.extend([f"--{key.replace('_', '-')}", value])
        return args

    def bash_exports(self, *keys: str) -> str:
        lines: List[str] = []
        for key in keys:
            value = self.cfg.runtime_value(key)
            if value:
                lines.append(f'export {key.upper()}="{value}"')
        if self.cfg.runtime_value("pyseqm_root"):
            lines.append('export PYTHONPATH="${PYSEQM_ROOT}:${PYTHONPATH:-}"')
        if self.cfg.runtime_value("pyseqm_library_path"):
            lines.append('export LD_LIBRARY_PATH="${PYSEQM_LIBRARY_PATH}:${LD_LIBRARY_PATH:-}"')
        if self.cfg.runtime_value("cuda_path"):
            lines.append('export PATH="${CUDA_PATH}/bin:${PATH}"')
            lines.append('export LD_LIBRARY_PATH="${CUDA_PATH}/lib64:${LD_LIBRARY_PATH:-}"')
        return "\n".join(lines) + ("\n" if lines else "")

    def gpu_directive(self) -> str:
        gpu_gres = str(coalesce(self.cfg.values, "gpu_gres", DEFAULTS["gpu_gres"])).strip()
        if not gpu_gres or gpu_gres.lower() in {"none", "false", "off", "no"}:
            return ""
        return f"#SBATCH --gres={gpu_gres}"

    def pyseqm_device(self) -> str:
        device = str(coalesce(self.cfg.values, "pyseqm_device", DEFAULTS["pyseqm_device"])).strip().lower()
        if device not in {"auto", "cuda", "cpu"}:
            raise ValueError("PYSEQM_DEVICE must be one of: auto, cuda, cpu")
        return device

    def pyseqm_uses_gpu_scheduler(self) -> bool:
        device = self.pyseqm_device()
        if device == "cpu":
            return False
        if device == "cuda":
            return True
        return bool(self.gpu_directive())

    def pyseqm_partition(self) -> str:
        return self.cfg.gpu_partition if self.pyseqm_uses_gpu_scheduler() else self.cfg.default_partition

    def pyseqm_extra_directives(self) -> str:
        return self.gpu_directive() if self.pyseqm_uses_gpu_scheduler() else ""

    def qchem_extra_directives(self, array_tasks: Optional[int] = None) -> str:
        lines: List[str] = []
        account = str(coalesce(self.cfg.values, "qchem_account", "")).strip()
        if account:
            lines.append(f"#SBATCH --account={account}")
        if array_tasks is not None and array_tasks > 0:
            lines.append(f"#SBATCH --array=0-{array_tasks - 1}")
        return "\n".join(lines)

    def qchem_cli_args(self) -> List[str]:
        return [
            "--qchem-exe", str(coalesce(self.cfg.values, "qchem_exe", DEFAULTS["qchem_exe"])),
            "--qchem-env-file", str(coalesce(self.cfg.values, "qchem_env_file", DEFAULTS["qchem_env_file"])),
            "--qchem-scratch", str(coalesce(self.cfg.values, "qchem_scratch", DEFAULTS["qchem_scratch"])),
            "--method", str(coalesce(self.cfg.values, "qchem_method", DEFAULTS["qchem_method"])),
            "--basis", str(coalesce(self.cfg.values, "qchem_basis", DEFAULTS["qchem_basis"])),
            "--n-roots", str(coalesce(self.cfg.values, "qchem_n_roots", DEFAULTS["qchem_n_roots"])),
            "--charge", str(coalesce(self.cfg.values, "qchem_charge", DEFAULTS["qchem_charge"])),
            "--multiplicity", str(coalesce(self.cfg.values, "qchem_multiplicity", DEFAULTS["qchem_multiplicity"])),
            "--ncpu", str(coalesce(self.cfg.values, "qchem_cpus", DEFAULTS["qchem_cpus"])),
            "--qchem-mem-total-mb", str(coalesce(self.cfg.values, "qchem_mem_total_mb", DEFAULTS["qchem_mem_total_mb"])),
        ]

    def qchem_props_dir(self, ctx: MoleculeContext) -> Path:
        return ctx.qchem_props_dir if self.quantum_backend() == "qchem" else ctx.pyseqm_props_dir

    def quantum_ref_dir(self, ctx: MoleculeContext) -> Path:
        return ctx.qchem_ref_dir if self.quantum_backend() == "qchem" else ctx.pyseqm_ref_dir

    def prepare_step(self, ctx: MoleculeContext, step: str) -> Path:
        pr = self.cfg.project_root
        if step == "gaussian_gsopt":
            self.require(ctx.gaussian_dir / "unopt.xyz", step)
            body = f"cd \"{pr}\"\n{self.python_cmd('00_gaussian_screening.py', '--path', str(ctx.gaussian_dir), '--gsopt', '--poll-seconds', str(coalesce(self.cfg.values, 'poll_seconds', 60)), '--partition', self.cfg.default_partition, *self.optional_cli_args('gaussian_root', 'gaussian_exe', 'gaussian_library_path', 'gaussian_scratch_root'))}\n"
            if not self.cfg.submit_jobs:
                body = body.replace(" --gsopt ", " --gsopt --prepare-only ")
            return self.write_sbatch(ctx, step, body, self.cfg.wrapper_partition, int(coalesce(self.cfg.values, "gaussian_cpus", 16)), f"{coalesce(self.cfg.values, 'gaussian_mem_gb', 100)}G", str(coalesce(self.cfg.values, "gaussian_walltime", "05:00:00")))
        if step == "gaussian_exsp":
            if self.cfg.submit_jobs:
                self.require(ctx.gaussian_dir / f"{ctx.gaussian_dir.name}_gsopt.chk", step)
            body = f"cd \"{pr}\"\n{self.python_cmd('00_gaussian_screening.py', '--path', str(ctx.gaussian_dir), '--exsp', '--poll-seconds', str(coalesce(self.cfg.values, 'poll_seconds', 60)), '--nstates', str(coalesce(self.cfg.values, 'gaussian_nstates', 10)), '--partition', self.cfg.default_partition, *self.optional_cli_args('gaussian_root', 'gaussian_exe', 'gaussian_library_path', 'gaussian_scratch_root'))}\n"
            if not self.cfg.submit_jobs:
                body = body.replace(" --exsp ", " --exsp --prepare-only ")
            return self.write_sbatch(ctx, step, body, self.cfg.wrapper_partition, int(coalesce(self.cfg.values, "gaussian_cpus", 16)), f"{coalesce(self.cfg.values, 'gaussian_mem_gb', 100)}G", str(coalesce(self.cfg.values, "gaussian_walltime", "05:00:00")))
        if step == "qchem_gsopt":
            xyz = ctx.source_dir / "unopt.xyz"
            self.require(xyz, step)
            body = f"cd \"{pr}\"\nmkdir -p \"{ctx.qchem_ref_dir}\"\n{self.python_cmd('06_qchem_backend.py', '--gsopt', '--xyz', str(xyz), '--output-dir', str(ctx.qchem_ref_dir), '--prefix', ctx.prefix, *self.qchem_cli_args())}\n"
            if not self.cfg.submit_jobs:
                body = body.rstrip() + " --prepare-only\n"
            return self.write_sbatch(ctx, step, body, str(coalesce(self.cfg.values, "qchem_partition", DEFAULTS["qchem_partition"])), int(coalesce(self.cfg.values, "qchem_cpus", 32)), str(coalesce(self.cfg.values, "qchem_gsopt_mem", "64G")), str(coalesce(self.cfg.values, "qchem_gsopt_walltime", "08:00:00")), self.qchem_extra_directives())
        if step == "nexmd_opt":
            if self.quantum_backend() == "qchem":
                qchem_xyz = ctx.qchem_ref_dir / f"{ctx.prefix}_qchem_optimized_gs_geometry.xyz"
                self.require(qchem_xyz, step)
                path_arg = ctx.work_dir
                extra_xyz = ["--xyz", str(qchem_xyz)]
            else:
                self.require(ctx.gaussian_dir / "mol_freq.txt", step)
                path_arg = ctx.gaussian_dir
                extra_xyz = []
            body = f"cd \"{pr}\"\n{self.python_cmd('01_nexmd_ground_state.py', '--path', str(path_arg), '--gs_opt', '--out_dir', str(ctx.nexmd_dir), '--partition', self.cfg.default_partition, *extra_xyz, *self.optional_cli_args('nexmd_root', 'nexmd_exe', 'nexmd_library_path'))}\n"
            if not self.cfg.submit_jobs:
                body = body.replace(" --out_dir ", " --prepare-only --out_dir ")
            return self.write_sbatch(ctx, step, body, self.cfg.wrapper_partition, 8, "32G", "02:00:00")
        if step == "nexmd_gsdyn":
            self.require(ctx.nexmd_dir / "coords.xyz", step)
            body = f"cd \"{pr}\"\n{self.python_cmd('01_nexmd_ground_state.py', '--path', str(ctx.work_dir), '--gs_dyn', '--out_dir', str(ctx.nexmd_dir / 'gs_dyn'), '--partition', self.cfg.default_partition, *self.optional_cli_args('nexmd_root', 'nexmd_exe', 'nexmd_library_path'))}\n"
            if not self.cfg.submit_jobs:
                body = body.replace(" --out_dir ", " --prepare-only --out_dir ")
            return self.write_sbatch(ctx, step, body, self.cfg.wrapper_partition, 8, "32G", "12:00:00")
        if step == "extract_frames":
            dyn = ctx.nexmd_dir / "gs_dyn"
            self.require(dyn / "coords.xyz", step)
            self.require(dyn / "velocity.out", step)
            validation_args = ["--no-validation"] if parse_bool(coalesce(self.cfg.values, "no_validation", False)) else ["--validation-samples", str(coalesce(self.cfg.values, "validation_samples", 10))]
            if parse_bool(coalesce(self.cfg.values, "full_validation", False)):
                validation_args = ["--full-validation"]
            body = (
                f"cd \"{dyn}\"\n"
                f"{self.python_cmd('02_extract_md_frames.py', '-xyz', 'coords.xyz', '-vel', 'velocity.out', '--start', str(self.mode_value('frame_start')), '--stop', str(self.mode_value('frame_stop')), '--step', str(coalesce(self.cfg.values, 'frame_step', 1)), '--workers', str(coalesce(self.cfg.values, 'frame_workers', 8)), '--output', str(ctx.frame_dir), *validation_args)}\n"
            )
            return self.write_sbatch(ctx, step, body, self.cfg.default_partition, int(coalesce(self.cfg.values, "frame_workers", 8)), "32G", "02:00:00")
        if step == "prepare_frames":
            self.require(ctx.frame_dir, step)
            body = (
                f"cd \"{ctx.pyseqm_dir}\"\n"
                f"mkdir -p \"{ctx.prepared_dir}\"\n"
                f"{self.python_cmd('03_prepare_frame_inputs.py', '--base_path', str(ctx.frame_dir), '--start_frame', '1', '--end_frame', str(int(self.mode_value('frame_stop')) - int(self.mode_value('frame_start')) + 2), '--prefix', ctx.prefix, '--sort_ZRV')}\n"
                f"latest=$(ls -td 20*-* | head -1)\n"
                f"cp \"${{latest}}/{ctx.prefix}_R.npy\" \"{ctx.prepared_dir}/\"\n"
                f"cp \"${{latest}}/{ctx.prefix}_Z.npy\" \"{ctx.prepared_dir}/\"\n"
                f"if [ -f \"${{latest}}/{ctx.prefix}_V.npy\" ]; then cp \"${{latest}}/{ctx.prefix}_V.npy\" \"{ctx.prepared_dir}/\"; fi\n"
            )
            return self.write_sbatch(ctx, step, body, self.cfg.default_partition, 8, "32G", "01:00:00")
        if step == "pyseqm_batch_exsp":
            r = ctx.prepared_dir / f"{ctx.prefix}_R.npy"
            z = ctx.prepared_dir / f"{ctx.prefix}_Z.npy"
            self.require(r, step); self.require(z, step)
            n_states = int(coalesce(self.cfg.values, "n_states", 11))
            device = self.pyseqm_device()
            gpu_args = [] if device == "cpu" else ["--gpus", str(coalesce(self.cfg.values, "gpu_ids", "0 1 2 3"))]
            body = f"cd \"{pr}\"\n{self.bash_exports('pyseqm_root', 'pyseqm_library_path', 'cuda_path') }mkdir -p \"{ctx.pyseqm_props_dir}\"\n{self.python_cmd('04_run_pyseqm_properties.py', '--coords', str(r), '--species', str(z), '--device', device, '--cpu-workers', str(coalesce(self.cfg.values, 'pyseqm_cpus', 8)), '--n_states', str(n_states), '--batch', str(coalesce(self.cfg.values, 'pyseqm_batch_size', 500)), '--method', str(coalesce(self.cfg.values, 'pyseqm_method', 'AM1')), '--scf_eps', str(coalesce(self.cfg.values, 'pyseqm_scf_eps', '1e-10')), '--cis_tol', str(coalesce(self.cfg.values, 'pyseqm_cis_tol', '1e-8')), *gpu_args, '--out_e', str(ctx.pyseqm_props_dir / 'energies_all_states.npy'), '--out_f', str(ctx.pyseqm_props_dir / 'forces_all_states.npy'), '--out_gs_D', str(ctx.pyseqm_props_dir / 'gs_dipoles.npy'), '--out_ex_D', str(ctx.pyseqm_props_dir / 'ex_dipoles_net.npy'), '--out_td', str(ctx.pyseqm_props_dir / 'transition_dipoles.npy'), '--out_etot', str(ctx.pyseqm_props_dir / 'total_energy.npy'), '--out_nacr', str(ctx.pyseqm_props_dir / 'nacrs_all_states.npy'))}\n"
            return self.write_sbatch(ctx, step, body, self.pyseqm_partition(), int(coalesce(self.cfg.values, "pyseqm_cpus", 8)), str(coalesce(self.cfg.values, "pyseqm_mem", "5G")), "10:00:00", self.pyseqm_extra_directives())
        if step == "pyseqm_opt":
            self.require(ctx.nexmd_dir / "coords.xyz", step)
            body = f"cd \"{pr}\"\n{self.bash_exports('pyseqm_root', 'pyseqm_library_path', 'cuda_path') }mkdir -p \"{ctx.pyseqm_ref_dir}\"\n{self.python_cmd('pyseqm_optimized_geom.py', '--xyz', str(ctx.nexmd_dir / 'coords.xyz'), '--output-dir', str(ctx.pyseqm_ref_dir), '--prefix', ctx.prefix, '--method', str(coalesce(self.cfg.values, 'pyseqm_method', 'AM1')), '--device', self.pyseqm_device(), '--sort-z')}\n"
            return self.write_sbatch(ctx, step, body, self.pyseqm_partition(), int(coalesce(self.cfg.values, "pyseqm_opt_cpus", coalesce(self.cfg.values, "pyseqm_cpus", 8))), str(coalesce(self.cfg.values, "pyseqm_opt_mem", coalesce(self.cfg.values, "pyseqm_mem", "5G"))), "02:00:00", self.pyseqm_extra_directives())
        if step == "qchem_batch_exsp":
            r = ctx.prepared_dir / f"{ctx.prefix}_R.npy"
            z = ctx.prepared_dir / f"{ctx.prefix}_Z.npy"
            self.require(r, step); self.require(z, step)
            n_frames = int(np.load(r, mmap_mode="r").shape[0])
            chunk_size = int(coalesce(self.cfg.values, "qchem_array_chunk_size", 1))
            array_tasks = int(math.ceil(n_frames / max(chunk_size, 1)))
            shard_dir = ctx.qchem_props_dir / "shards"
            body = f"cd \"{pr}\"\nmkdir -p \"{ctx.qchem_props_dir}\" \"{shard_dir}\"\n{self.python_cmd('06_qchem_backend.py', '--frame', '--coords', str(r), '--species', str(z), '--output-dir', str(ctx.qchem_props_dir), '--shard-dir', str(shard_dir), '--prefix', ctx.prefix, '--chunk-size', str(chunk_size), *self.qchem_cli_args())}\n"
            if not self.cfg.submit_jobs:
                body = body.rstrip() + " --prepare-only\n"
            return self.write_sbatch(ctx, step, body, str(coalesce(self.cfg.values, "qchem_partition", DEFAULTS["qchem_partition"])), int(coalesce(self.cfg.values, "qchem_cpus", 32)), str(coalesce(self.cfg.values, "qchem_mem", "128G")), str(coalesce(self.cfg.values, "qchem_walltime", "24:00:00")), self.qchem_extra_directives(array_tasks))
        if step == "qchem_collect":
            r = ctx.prepared_dir / f"{ctx.prefix}_R.npy"
            z = ctx.prepared_dir / f"{ctx.prefix}_Z.npy"
            self.require(r, step); self.require(z, step)
            if self.cfg.submit_jobs:
                self.require(ctx.qchem_props_dir / "shards", step)
            body = f"cd \"{pr}\"\nmkdir -p \"{ctx.qchem_props_dir}\"\n{self.python_cmd('06_qchem_backend.py', '--collect', '--coords', str(r), '--species', str(z), '--output-dir', str(ctx.qchem_props_dir), '--shard-dir', str(ctx.qchem_props_dir / 'shards'), '--prefix', ctx.prefix, *self.qchem_cli_args())}\n"
            return self.write_sbatch(ctx, step, body, self.cfg.default_partition, int(coalesce(self.cfg.values, "dataset_workers", 16)), str(coalesce(self.cfg.values, "qchem_collect_mem", "32G")), str(coalesce(self.cfg.values, "qchem_collect_walltime", "02:00:00")), self.qchem_extra_directives())
        if step == "generate_dataset":
            props_dir = self.qchem_props_dir(ctx)
            ref_dir = self.quantum_ref_dir(ctx)
            self.require(props_dir / "energies_all_states.npy", step)
            r = ctx.prepared_dir / f"{ctx.prefix}_R.npy"
            z = ctx.prepared_dir / f"{ctx.prefix}_Z.npy"
            self.require(r, step); self.require(z, step)
            ref_e = Path(str(self.cfg.values.get("reference_energy_file", ref_dir / f"{ctx.prefix}_optimized_reference_energy.txt")))
            ref_f = Path(str(self.cfg.values.get("reference_forces_file", ref_dir / f"{ctx.prefix}_optimized_reference_forces.npy")))
            zero_ref_prelude = ""
            if ref_e.exists() and ref_f.exists():
                pass
            else:
                self.log.warning(
                    "Reference files missing for %s; workflow will generate explicit zero reference files for dataset generation.",
                    ctx.name,
                )
                ref_e = ctx.dataset_dir / f"{ctx.prefix}_zero_reference_energy.txt"
                ref_f = ctx.dataset_dir / f"{ctx.prefix}_zero_reference_forces.npy"
                zero_ref_prelude = (
                    f"\"{self.cfg.python_bin}\" - <<'PY'\n"
                    f"from pathlib import Path\nimport numpy as np\n"
                    f"z = np.load(r'{z}', mmap_mode='r')\n"
                    f"n_atoms = int(z.shape[1])\n"
                    f"Path(r'{ctx.dataset_dir}').mkdir(parents=True, exist_ok=True)\n"
                    f"np.savetxt(r'{ref_e}', np.asarray([0.0]), fmt='%.16f')\n"
                    f"np.save(r'{ref_f}', np.zeros((n_atoms, 3), dtype=np.float64))\n"
                    f"print('WARNING: optimized reference files missing; wrote zero reference energy/forces for dataset generation')\n"
                    f"PY\n"
                )
            ref_args = f" --reference-energy \"{ref_e}\" --reference-forces \"{ref_f}\""
            chop_start = self.mode_value("dataset_chop_start")
            chop_end = self.mode_value("dataset_chop_end")
            n_excited = int(coalesce(self.cfg.values, "qchem_n_roots", DEFAULTS["qchem_n_roots"])) if self.quantum_backend() == "qchem" else int(coalesce(self.cfg.values, "n_states", 11)) - 1
            body = f"cd \"{pr}\"\nmkdir -p \"{ctx.dataset_dir}\"\n{zero_ref_prelude}{self.python_cmd('05_generate_dataset.py', '--input-dir', str(props_dir), '--output-dir', str(ctx.dataset_dir), '--n-states', str(n_excited), '--n-workers', str(coalesce(self.cfg.values, 'dataset_workers', 16)), '--log-file', str(ctx.dataset_dir / f'{ctx.prefix}_dataset.log'), '--chop-start', str(chop_start), '--chop-end', str(chop_end), '--chop-step', str(coalesce(self.cfg.values, 'dataset_chop_step', 10000)), '--coords-r', str(r), '--coords-z', str(z), '--prefix', ctx.prefix, '--split-nacr')} {ref_args}\n"
            return self.write_sbatch(ctx, step, body, self.cfg.default_partition, int(coalesce(self.cfg.values, "dataset_workers", 16)), "64G", "10:00:00")
        raise ValueError(f"Unknown step: {step}")

    def submit(self, sbatch: Path) -> str:
        out = subprocess.check_output(["sbatch", str(sbatch)], text=True, cwd=sbatch.parent)
        match = re.search(r"Submitted batch job (\d+)", out)
        if not match:
            raise RuntimeError(f"Could not parse sbatch output: {out}")
        return match.group(1)

    def job_state(self, job_id: str) -> str:
        cmd = ["sacct", "-j", job_id, "--format=State", "-n", "-P"]
        out = subprocess.check_output(cmd, text=True).strip().splitlines()
        states = [line.split("|")[0] for line in out if line.strip()]
        return states[0] if states else "UNKNOWN"

    def wait(self, job_id: str, poll_seconds: int = 60) -> str:
        terminal = ("COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY")
        while True:
            state = self.job_state(job_id)
            self.log.info("Job %s state: %s", job_id, state)
            if state.startswith(terminal):
                return state
            time.sleep(poll_seconds)

    def run_molecule(self, ctx: MoleculeContext, steps: List[str]) -> List[StepResult]:
        self.ensure_layout(ctx)
        results: List[StepResult] = []
        self.log.info("Molecule %s -> prefix %s", ctx.name, ctx.prefix)
        for step in steps:
            if self.completed(ctx, step) and not self.cfg.overwrite:
                self.log.info("Skipping completed step %s for %s", step, ctx.name)
                results.append(StepResult(step, ctx.slurm_dir / f"submit_{ctx.prefix}_{step}.sbatch", skipped=True))
                continue
            sbatch = self.prepare_step(ctx, step)
            self.log.info("Wrote %s", sbatch)
            result = StepResult(step, sbatch)
            if self.cfg.submit_jobs:
                result.job_id = self.submit(sbatch)
                self.log.info("Submitted %s as job %s", step, result.job_id)
                if self.cfg.monitor_jobs:
                    state = self.wait(result.job_id, int(coalesce(self.cfg.values, "poll_seconds", 60)))
                    if not state.startswith("COMPLETED"):
                        raise RuntimeError(f"Step {step} job {result.job_id} ended with {state}")
            results.append(result)
        return results


def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Molecule-agnostic dataset workflow driver")
    parser.add_argument("--config", type=Path, default=None, help="dataset_workflow.in file")
    parser.add_argument("--molecule", type=Path, default=None, help="Single molecule directory")
    parser.add_argument("--prefix", default=None, help="Override output prefix for single molecule")
    parser.add_argument("--submit_jobs", default=None, help="true/false; submit generated sbatch files")
    parser.add_argument("--monitor_jobs", default=None, help="true/false; monitor submitted jobs")
    parser.add_argument("--continue", dest="do_continue", default=None, help="true/false; continue from selected step")
    parser.add_argument("--overwrite", default=None, help="true/false; overwrite completed step outputs")
    parser.add_argument("--gaussian-root", default=None, help="Gaussian root directory, for example /path/to/g16A03")
    parser.add_argument("--gaussian-exe", default=None, help="Gaussian executable or absolute path, for example g16")
    parser.add_argument("--gaussian-library-path", default=None, help="Optional Gaussian library path for LD_LIBRARY_PATH")
    parser.add_argument("--gaussian-scratch-root", default=None, help="Root directory for Gaussian scratch")
    parser.add_argument("--nexmd-root", default=None, help="Optional NEXMD installation root")
    parser.add_argument("--nexmd-exe", default=None, help="NEXMD executable path")
    parser.add_argument("--nexmd-library-path", default=None, help="Optional NEXMD library path for LD_LIBRARY_PATH")
    parser.add_argument("--pyseqm-root", default=None, help="Optional PySEQM source/package root to prepend to PYTHONPATH")
    parser.add_argument("--python-bin", default=None, help="Python executable for non-PySEQM helper stages")
    parser.add_argument("--pyseqm-python-bin", default=None, help="Python executable for PySEQM stages")
    parser.add_argument("--pyseqm-library-path", default=None, help="Optional PySEQM library path for LD_LIBRARY_PATH")
    parser.add_argument("--cuda-path", default=None, help="CUDA installation root")
    parser.add_argument("--pyseqm-device", default=None, choices=("auto", "cuda", "cpu"), help="PySEQM execution device policy")
    parser.add_argument("--pyseqm-cpus", default=None, type=int, help="CPUs for PySEQM batch jobs and CPU worker count")
    parser.add_argument("--pyseqm-mem", default=None, help="SLURM memory for PySEQM batch jobs, e.g. 5G")
    parser.add_argument("--pyseqm-opt-cpus", default=None, type=int, help="CPUs for PySEQM reference optimization")
    parser.add_argument("--pyseqm-opt-mem", default=None, help="SLURM memory for PySEQM reference optimization, e.g. 5G")
    parser.add_argument("--quantum-backend", default=None, choices=("pyseqm", "qchem"), help="Electronic-structure backend")
    parser.add_argument("--qchem-profile", default=None, choices=("darwin", "chicoma"), help="Q-Chem cluster profile label")
    parser.add_argument("--qchem-exe", default=None, help="Q-Chem executable")
    parser.add_argument("--qchem-env-file", default=None, help="Q-Chem environment file to source")
    parser.add_argument("--qchem-scratch", default=None, help="Q-Chem scratch directory")
    parser.add_argument("--qchem-python-bin", default=None, help="Python executable for Q-Chem helper stages")
    parser.add_argument("--qchem-partition", default=None, help="SLURM partition for Q-Chem jobs")
    parser.add_argument("--qchem-account", default=None, help="SLURM account for Q-Chem jobs")
    parser.add_argument("--qchem-cpus", default=None, type=int, help="CPUs per Q-Chem job")
    parser.add_argument("--qchem-array-chunk-size", default=None, type=int, help="Frames per Q-Chem SLURM array task")
    for flag in STEP_FLAGS.values():
        parser.add_argument(f"--{flag.replace('run_', '')}", default=None, help="true/false")
    parser.add_argument("--pre_pyseqm", default=None, help="Deprecated alias for --prepare_frames true/false")
    return parser.parse_args()


def merge_cli(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    merged = {k.lower(): v for k, v in config.items()}
    for key in ["submit_jobs", "monitor_jobs", "overwrite"]:
        value = getattr(args, key)
        if value is not None:
            merged[key] = parse_bool(value)
    if args.do_continue is not None:
        merged["continue"] = parse_bool(args.do_continue)
    if args.prefix:
        merged["prefix"] = args.prefix
    for key in [
        "gaussian_root",
        "gaussian_exe",
        "gaussian_library_path",
        "gaussian_scratch_root",
        "nexmd_root",
        "nexmd_exe",
        "nexmd_library_path",
        "pyseqm_root",
        "python_bin",
        "pyseqm_python_bin",
        "pyseqm_library_path",
        "cuda_path",
        "pyseqm_device",
        "pyseqm_cpus",
        "pyseqm_mem",
        "pyseqm_opt_cpus",
        "pyseqm_opt_mem",
        "quantum_backend",
        "qchem_profile",
        "qchem_exe",
        "qchem_env_file",
        "qchem_scratch",
        "qchem_python_bin",
        "qchem_partition",
        "qchem_account",
        "qchem_cpus",
        "qchem_array_chunk_size",
    ]:
        value = getattr(args, key)
        if value is not None:
            merged[key] = value
    for step, flag in STEP_FLAGS.items():
        cli_name = flag.replace("run_", "")
        value = getattr(args, cli_name)
        if value is not None:
            merged[flag] = parse_bool(value)
    if getattr(args, "pre_pyseqm", None) is not None:
        merged["run_prepare_frames"] = parse_bool(args.pre_pyseqm)
    return merged



def apply_qchem_profile_defaults(values: Dict[str, Any]) -> Dict[str, Any]:
    profile = str(values.get("qchem_profile", DEFAULTS["qchem_profile"])).strip().lower()
    profile_defaults = QCHEM_PROFILE_DEFAULTS.get(profile, {})
    for key, value in profile_defaults.items():
        values.setdefault(key, value)
    return values
def resolve_molecules(config: Dict[str, Any], args: argparse.Namespace, project_root: Path) -> Dict[str, Path]:
    if args.molecule:
        p = args.molecule.resolve()
        return {p.name: p}
    molecules = config.get("molecule") or config.get("molecules")
    if isinstance(molecules, dict):
        return {str(name): as_path(path, project_root) for name, path in molecules.items()}
    mol_dir = config.get("mol_dir") or config.get("molecule_dir")
    if mol_dir:
        p = as_path(mol_dir, project_root)
        return {p.name: p}
    raise ValueError("No molecule provided. Use --molecule MOL_DIR or Molecule = { Name: path } in dataset_workflow.in")


def configure_logging(project_root: Path) -> None:
    log_dir = project_root / "logs"
    log_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_dir / "dataset_workflow.log", mode="a")],
    )


def main() -> int:
    args = parse_cli()
    project_root = Path(__file__).resolve().parent
    configure_logging(project_root)
    raw_config = load_config(args.config)
    values = apply_qchem_profile_defaults(merge_cli(raw_config, args))
    molecules = resolve_molecules(values, args, project_root)
    cfg = WorkflowConfig(project_root=project_root, molecules=molecules, values=values)
    workflow = Workflow(cfg)
    steps = workflow.selected_steps()
    logging.info("%d molecule(s) found: %s", len(molecules), ", ".join(molecules))
    logging.info("Selected steps: %s", ", ".join(steps))
    logging.info("submit_jobs=%s monitor_jobs=%s continue=%s overwrite=%s", cfg.submit_jobs, cfg.monitor_jobs, cfg.do_continue, cfg.overwrite)
    def run_one(item: Tuple[str, Path]) -> None:
        name, mol_dir = item
        ctx = workflow.build_context(name, mol_dir)
        workflow.run_molecule(ctx, steps)

    max_parallel = int(coalesce(values, "max_parallel_molecules", len(molecules)))
    if len(molecules) > 1 and max_parallel > 1:
        logging.info("Running molecule workflows in parallel with max_parallel_molecules=%d", max_parallel)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel) as executor:
            futures = [executor.submit(run_one, item) for item in molecules.items()]
            for future in concurrent.futures.as_completed(futures):
                future.result()
    else:
        for item in molecules.items():
            run_one(item)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

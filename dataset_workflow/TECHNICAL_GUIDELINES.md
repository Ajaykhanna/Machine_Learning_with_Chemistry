# Technical Guidelines: Automated Dataset Generation Workflow

## Goal

This document defines the updated technical procedure for automating molecular dataset generation for building machine-learning interatomic potentials (MLIPs) for multistate, multitask excited-state prediction.

The workflow starts from an unoptimized molecular geometry and produces molecule-specific NumPy datasets containing geometries, species, state energies, shifted energies, forces, shifted forces, dipoles, transition dipoles, oscillator-related quantities, and nonadiabatic coupling vectors. The workflow is designed to be molecule-agnostic, SLURM-aware, and reproducible on Darwin-style cluster systems.

The production automation is implemented as an independent package:

```text
dataset_workflow/
  dataset_workflow.py
  scripts/
  configs/
  templates/
  example/
  logs/
  README.md
  TECHNICAL_GUIDELINES.md
```

The master driver orchestrates the numbered stage scripts while keeping each individual script independently runnable for debugging, manual reruns, or cluster-specific adaptation.

## Software Stack

Electronic structure and molecular simulation:

```text
Gaussian     Ground-state DFT optimization and excited-state screening
NEXMD        AM1 ground-state optimization and ground-state dynamics
PySEQM       CIS/AM1 excited-state single-point calculations and reference optimization
```

Data generation, extraction, and analysis:

```text
Python
NumPy
SciPy
Pandas
Matplotlib
Plotly
SLURM
```

Default Darwin runtime paths:

```text
PYTHON_BIN = /usr/projects/ml4chem/akhanna2/softwares/envs/ml_env/bin/python
PYSEQM_PYTHON_BIN = /usr/projects/ml4chem/akhanna2/softwares/envs/ml_env/bin/python
GAUSSIAN_ROOT = /usr/projects/cint/Gaussian/g16A03
GAUSSIAN_EXE = g16
GAUSSIAN_LIBRARY_PATH =
GAUSSIAN_SCRATCH_ROOT = /tmp/akhanna2/GAUSSIAN_SCR
NEXMD_EXE = /usr/projects/ml4chem/akhanna2/softwares/NEXMD/nexmd.exe
```

The workflow should not install, remove, or modify packages inside managed cluster environments.

Runtime executable and library paths can be supplied in `configs/dataset_workflow.inp` or from the command line. Command-line values override config values.

```text
GAUSSIAN_ROOT
GAUSSIAN_EXE
GAUSSIAN_LIBRARY_PATH
GAUSSIAN_SCRATCH_ROOT
NEXMD_ROOT
NEXMD_EXE
NEXMD_LIBRARY_PATH
PYSEQM_ROOT
PYSEQM_PYTHON_BIN
PYSEQM_LIBRARY_PATH
CUDA_PATH
```

These paths are written into generated SLURM files only when relevant to the selected step.

## High-Level Dataset Generation Route

The complete route is:

```text
unopt.xyz
  -> Gaussian ground-state geometry optimization
  -> Gaussian excited-state single-point screening
  -> ranked screening_summary.csv
  -> manual molecule approval
  -> NEXMD ground-state AM1 reoptimization
  -> NEXMD ground-state AM1 molecular dynamics
  -> uncorrelated frame extraction
  -> molecule-specific PySEQM R/Z/V arrays
  -> PySEQM CIS/AM1 batch excited-state properties
  -> PySEQM ground-state reference optimization
  -> shifted-energy and shifted-force dataset generation
  -> final molecule-specific dataset arrays
```

The workflow is intentionally conservative. Gaussian screening is the default entry point. Production continuation after screening remains a manual scientific decision unless the user explicitly selects downstream steps.

## Package Layout

Canonical package layout:

```text
dataset_workflow/
  dataset_workflow.py
  README.md
  TECHNICAL_GUIDELINES.md

  scripts/
    00_gaussian_screening.py
    0_gaussian_automation.py
    01_nexmd_ground_state.py
    1_run_nexmd_dyn.py
    02_extract_md_frames.py
    claude_read_xyz_vel_traj.py
    03_prepare_frame_inputs.py
    extract_frames_for_pyseqm.py
    04_run_pyseqm_properties.py
    run_pyseqm_electronic_properties_calcs.py
    05_generate_dataset.py
    pyseqm_optimized_geom.py
    geometry_parser.py
    pipeline_config.py

  configs/
    dataset_workflow.inp
    individual_scripts.inp
    master_slurm.inp

  templates/
    gsopt_input.ceon
    gsdyn_input.ceon
    standard_nexmd_md.sbatch
    submit_*.sbatch

  example/
    Nilered/unopt.xyz
    Prodan/unopt.xyz

  logs/
```

Generated data remain inside each molecule directory:

```text
<mol_dir>/gaussian/
<mol_dir>/nexmd/
<mol_dir>/pyseqm/
<mol_dir>/dataset/
<mol_dir>/slurm/
```

## Molecule Naming And Prefix Rules

Each molecule directory must contain:

```text
<mol_dir>/unopt.xyz
```

The output prefix defaults to the lowercase molecule key or folder name:

```text
Nilered -> nilered
Prodan  -> prodan
```

All generated files should use molecule-specific names. Generic names such as `molecule_R.npy`, `molecule_gsopt.out`, or old project-specific prefixes are not production naming conventions.

Examples:

```text
nilered_R.npy
nilered_Z.npy
nilered_optimized_gs_geometry.xyz
nilered_optimized_reference_energy.txt
nilered_optimized_reference_forces.npy
nilered_sE.npy
nilered_F.npy
```

## Configuration File

The main configuration file is:

```text
configs/dataset_workflow.inp
```

Formatting rules:

```text
KEY = VALUE
```

Rules:

```text
Lines beginning with # are ignored.
Keys are case-insensitive.
Normalized snake_case or uppercase SNAKE_CASE keys are preferred.
Boolean values are true or false.
Comma-separated values are parsed as lists unless quoted.
Multiple molecules can be provided in a Molecule dictionary block.
```

Example molecule block:

```text
Molecule = {
    Nilered: "example/Nilered"
    Prodan: "example/Prodan"
}
```

The workflow prints the number of molecules found and prepares an independent molecule-local workflow for each molecule.

## SLURM Policy

The workflow separates controller jobs from compute jobs.

Controller or wrapper jobs:

```text
WRAPPER_PARTITION = general
```

CPU compute stages:

```text
DEFAULT_PARTITION = general
```

GPU PySEQM stages:

```text
GPU_PARTITION = ml4chem
GPU_GRES = gpu:4
```

Darwin partition guidance:

```text
general          Controller/wrapper jobs and general CPU jobs
shared-spr       High-memory CPU work
shared-spr-hbm   High-memory CPU work
ml4chem          GPU work
shared-redstone  GPU work, but typically one active job at a time
```

Wrapper jobs should not occupy scarce GPU or restricted partitions while waiting for child jobs. If a workflow wrapper submits child SLURM jobs, the wrapper should run on `general` by default.

Every stage receives its own generated SLURM file:

```text
<mol_dir>/slurm/submit_<prefix>_<step>.sbatch
```

Generated SLURM files are intentionally transparent and editable. Node-specific constraints are not included by default.

## Workflow Driver Behavior

The master workflow script is:

```text
dataset_workflow.py
```

Default behavior:

```text
If no RUN_* flags are enabled, prepare Gaussian GS optimization and Gaussian EXSP screening.
If SUBMIT_JOBS = false, only generate scripts and input files.
If SUBMIT_JOBS = true, submit selected jobs.
If MONITOR_JOBS = true, monitor submitted jobs and parse outputs after completion.
```

Step flags:

```text
RUN_GAUSSIAN_GSOPT
RUN_GAUSSIAN_EXSP
RUN_NEXMD_OPT
RUN_NEXMD_GSDYN
RUN_EXTRACT_FRAMES
RUN_PREPARE_FRAMES
RUN_PYSEQM_BATCH_EXSP
RUN_PYSEQM_OPT
RUN_GENERATE_DATASET
```

Command-line equivalents:

```text
--gaussian_gsopt true
--gaussian_exsp true
--nexmd_opt true
--nexmd_gsdyn true
--extract_frames true
--prepare_frames true
--pyseqm_batch_exsp true
--pyseqm_opt true
--generate_dataset true
```

Continuation behavior:

```text
--continue true
```

If a start step is selected with continuation enabled, the workflow runs from that step through dataset generation. For example:

```text
--nexmd_gsdyn true --continue true
```

runs:

```text
nexmd_gsdyn
  -> extract_frames
  -> prepare_frames
  -> pyseqm_batch_exsp
  -> pyseqm_opt
  -> generate_dataset
```

Skip and overwrite behavior:

```text
Existing completed outputs are skipped by default.
Use --overwrite true or OVERWRITE = true to regenerate outputs.
```

Missing dependency behavior:

```text
If a required input file is missing, the workflow stops with a clear error.
The error should identify the missing file and suggest rerunning with --continue true if earlier stages need to be generated.
```

## Stage 0: Gaussian Ground-State Optimization

Purpose:

```text
Optimize the ground-state molecular geometry using DFT.
```

Default level of theory:

```text
B3LYP/6-31G(d)
```

Canonical script:

```text
scripts/00_gaussian_screening.py --gsopt
```

Compatibility script:

```text
scripts/0_gaussian_automation.py --gs_opt
```

Input:

```text
<mol_dir>/unopt.xyz
```

Workflow staging:

```text
<mol_dir>/gaussian/<MoleculeName>/unopt.xyz
```

Required behavior:

```text
Read unopt.xyz.
Sort atoms from highest atomic number to lowest when writing Gaussian input.
Write molecule-specific Gaussian input and SLURM files.
Submit and monitor if requested.
Parse final frequencies after completion.
Save frequencies in mol_freq.txt.
Stop downstream execution if any frequency is negative.
```

Frequency validation rule:

```text
Any negative frequency fails the molecule.
No tolerance is applied.
```

Expected outputs:

```text
<mol_dir>/gaussian/<MoleculeName>/mol_freq.txt
<mol_dir>/gaussian/<MoleculeName>/<prefix>_gsopt.log
<mol_dir>/slurm/submit_<prefix>_gaussian_gsopt.sbatch
```

## Stage 1: Gaussian Excited-State Single-Point Screening

Purpose:

```text
Compute excited-state properties on the Gaussian-optimized ground-state geometry.
Identify molecules with promising low-lying excited-state structure.
```

Default level of theory:

```text
TDA/CAM-B3LYP/6-31G(d)
```

Default number of excited states:

```text
GAUSSIAN_NSTATES = 10
```

Canonical script:

```text
scripts/00_gaussian_screening.py --exsp
```

Required behavior:

```text
Extract optimized geometry from Gaussian GS optimization.
Create molecule-specific Gaussian EXSP input.
Submit and monitor if requested.
Parse transition energies from "Excited State".
Parse transition dipole moments from the Gaussian transition electric dipole section.
Save grepped and parsed outputs to mol_exsp.txt.
Generate screening_summary.csv.
Rank molecules using configured criteria.
Leave final molecule approval to the user.
```

Screening criteria:

```text
MINIMUM_STATES = 5
EXCITATION_ENERGY_THRESHOLD_EV = 0.50
OSCILLATOR_BRIGHT_THRESHOLD = 0.10
OSCILLATOR_DARK_THRESHOLD = 0.05
```

Interpretation:

```text
f > 0.10        Bright state
f < 0.05        Dark state
0.05 <= f <= 0.10 Partially allowed or mixed dark/bright character
```

Molecules are ranked higher when low-lying states show useful features such as:

```text
Closely spaced excited states
Near degeneracy
Bright and dark state coexistence
Oscillator strength redistribution
Transition dipole sharing
Push-pull excitation character
Potential surface-hopping relevance
Potential conical-intersection relevance
```

Expected outputs:

```text
<mol_dir>/gaussian/<MoleculeName>/mol_exsp.txt
<mol_dir>/gaussian/<MoleculeName>/screening_summary.csv
<mol_dir>/slurm/submit_<prefix>_gaussian_exsp.sbatch
```

Manual gate:

```text
The user inspects screening_summary.csv and plots before approving the molecule for production.
```

## Stage 2: NEXMD Ground-State Optimization

Purpose:

```text
Reoptimize the Gaussian ground-state geometry with NEXMD at the AM1 level.
```

Default method:

```text
NEXMD_METHOD = AM1
```

Canonical script:

```text
scripts/01_nexmd_ground_state.py --gs_opt
```

Template:

```text
templates/gsopt_input.ceon
```

Input:

```text
Gaussian optimized ground-state geometry
```

Authoritative output geometry:

```text
<mol_dir>/nexmd/coords.xyz
```

Expected outputs:

```text
<mol_dir>/nexmd/coords.xyz
<mol_dir>/nexmd/<prefix>_gsopt.out
<mol_dir>/slurm/submit_<prefix>_nexmd_opt.sbatch
```

## Stage 3: NEXMD Ground-State Dynamics

Purpose:

```text
Generate ground-state molecular dynamics configurations for later excited-state single-point calculations.
```

Canonical script:

```text
scripts/01_nexmd_ground_state.py --gs_dyn
```

Template:

```text
templates/gsdyn_input.ceon
```

Input:

```text
<mol_dir>/nexmd/coords.xyz
```

Initial velocity default:

```text
INITIAL_VELOCITIES = zero
```

Modes:

```text
MODE = Test
NEXMD_N_CLASS_STEPS_TEST = 10000

MODE = Production
NEXMD_N_CLASS_STEPS_PRODUCTION = 5000000
```

Dynamics defaults:

```text
NEXMD_TIME_STEP = 0.1
NEXMD_OUT_COORDS_STEPS = 100
```

Saved frame spacing:

```text
0.1 fs * 100 = 10 fs
```

Expected outputs:

```text
<mol_dir>/nexmd/gs_dyn/coords.xyz
<mol_dir>/nexmd/gs_dyn/velocity.out
<mol_dir>/nexmd/gs_dyn/<prefix>_gsdyn.out
<mol_dir>/slurm/submit_<prefix>_nexmd_gsdyn.sbatch
```

## Stage 4: Frame Extraction

Purpose:

```text
Extract uncorrelated molecular configurations and velocities from NEXMD ground-state dynamics.
```

Canonical script:

```text
scripts/02_extract_md_frames.py
```

Compatibility script:

```text
scripts/claude_read_xyz_vel_traj.py
```

Inputs:

```text
<mol_dir>/nexmd/gs_dyn/coords.xyz
<mol_dir>/nexmd/gs_dyn/velocity.out
```

Configurable extraction arguments:

```text
FRAME_START_TEST
FRAME_STOP_TEST
FRAME_START_PRODUCTION
FRAME_STOP_PRODUCTION
FRAME_STEP
FRAME_WORKERS
FULL_VALIDATION
VALIDATION_SAMPLES
NO_VALIDATION
```

Test defaults:

```text
FRAME_START_TEST = 1
FRAME_STOP_TEST = 101
FRAME_STEP = 1
```

Production defaults:

```text
FRAME_START_PRODUCTION = 5000
FRAME_STOP_PRODUCTION = 55001
FRAME_STEP = 1
```

Because NEXMD saved frames are already separated by 10 fs, `FRAME_STEP = 1` preserves 10 fs spacing.

Expected output:

```text
<mol_dir>/nexmd/gs_dyn/<N>_10fs_frames/frame_000001/frame_000001.xyz
<mol_dir>/nexmd/gs_dyn/<N>_10fs_frames/frame_000001/frame_000001.vel
```

## Stage 5: Prepare PySEQM Inputs

Purpose:

```text
Convert extracted XYZ and velocity frames into molecule-specific NumPy arrays for PySEQM.
```

Canonical script:

```text
scripts/03_prepare_frame_inputs.py
```

Compatibility script:

```text
scripts/extract_frames_for_pyseqm.py
```

Required behavior:

```text
Read extracted frame directories.
Sort arrays consistently with atom/species conventions.
Write molecule-specific R, Z, and V arrays.
```

Expected outputs:

```text
<mol_dir>/pyseqm/prepared_frames/<prefix>_R.npy
<mol_dir>/pyseqm/prepared_frames/<prefix>_Z.npy
<mol_dir>/pyseqm/prepared_frames/<prefix>_V.npy
```

## Stage 6: PySEQM Batch Excited-State Single-Point Calculations

Purpose:

```text
Compute CIS/AM1 electronic properties for each extracted molecular configuration.
```

Canonical script:

```text
scripts/04_run_pyseqm_properties.py
```

Compatibility script:

```text
scripts/run_pyseqm_electronic_properties_calcs.py
```

Default method and state count:

```text
PYSEQM_METHOD = AM1
N_STATES = 11
```

State convention:

```text
N_STATES = 11 means 1 ground state + 10 excited states.
```

Default numerical settings:

```text
PYSEQM_BATCH_SIZE = 500
PYSEQM_SCF_EPS = 1e-10
PYSEQM_CIS_TOL = 1e-8
GPU_IDS = "0 1 2 3"
```

Inputs:

```text
<mol_dir>/pyseqm/prepared_frames/<prefix>_R.npy
<mol_dir>/pyseqm/prepared_frames/<prefix>_Z.npy
```

Expected outputs:

```text
<mol_dir>/pyseqm/batch_exsp/energies_all_states.npy
<mol_dir>/pyseqm/batch_exsp/forces_all_states.npy
<mol_dir>/pyseqm/batch_exsp/gs_dipoles.npy
<mol_dir>/pyseqm/batch_exsp/ex_dipoles_net.npy
<mol_dir>/pyseqm/batch_exsp/transition_dipoles.npy
<mol_dir>/pyseqm/batch_exsp/total_energy.npy
<mol_dir>/pyseqm/batch_exsp/nacrs_all_states.npy
```

GPU scheduling:

```text
Use GPU_PARTITION for this stage.
Use GPU_GRES unless the selected partition requires GPU_GRES = none.
```

## Stage 7: PySEQM Ground-State Reference Optimization

Purpose:

```text
Reoptimize the NEXMD optimized geometry with PySEQM and compute the reference ground-state minimum energy and forces.
```

Canonical script:

```text
scripts/pyseqm_optimized_geom.py
```

Input:

```text
<mol_dir>/nexmd/coords.xyz
```

The script prints:

```text
Total Energy (eV)
Ground State Forces (eV/Angstrom)
```

These values are stored and passed to dataset generation as the reference values used to construct shifted energies and shifted forces.

Expected molecule-specific outputs:

```text
<mol_dir>/pyseqm/gsopt_reference/<prefix>_optimized_gs_geometry.xyz
<mol_dir>/pyseqm/gsopt_reference/<prefix>_optimized_reference_energy.txt
<mol_dir>/pyseqm/gsopt_reference/<prefix>_optimized_reference_forces.npy
```

Naming rule:

```text
The optimized geometry must be molecule-specific, for example prodan_optimized_gs_geometry.xyz.
```

## Stage 8: Final Dataset Generation

Purpose:

```text
Assemble the final ML-ready dataset arrays from PySEQM properties, coordinates, species, velocities, NACRs, and reference ground-state values.
```

Canonical script:

```text
scripts/05_generate_dataset.py
```

Reference inputs:

```text
<prefix>_optimized_reference_energy.txt
<prefix>_optimized_reference_forces.npy
```

If reference files are missing:

```text
The workflow writes explicit zero reference values.
The workflow prints/logs a prominent warning.
Dataset generation continues for manual testing.
```

Production dataset output path:

```text
Controlled by DATASET_OUTPUT_DIR when provided.
Otherwise defaults to <mol_dir>/dataset.
```

Expected production shape family for 50000 frames, 15 atoms, and 11 states:

```text
<prefix>_D*.npy       (50000, 3)
<prefix>_dE*.npy      (50000,)
<prefix>_dENACR*.npy  (50000, 45)
<prefix>_dENACR.npy   (50000, 45, 45)
<prefix>_dE.npy       (50000, 11)
<prefix>_D.npy        (50000, 11, 3)
<prefix>_F*.npy       (50000, 15, 3)
<prefix>_F.npy        (50000, 11, 15, 3)
<prefix>_NACR*.npy    (50000, 15, 3)
<prefix>_NACR.npy     (50000, 45, 15, 3)
<prefix>_S*.npy       (50000,)
<prefix>_sE*.npy      (50000,)
<prefix>_sE.npy       (50000, 11)
<prefix>_sF*.npy      (50000, 15, 3)
<prefix>_sF.npy       (50000, 11, 15, 3)
<prefix>_S.npy        (50000, 11)
<prefix>_V.npy        (50000, 15, 3)
<prefix>_Z.npy        (50000, 15)
```

For molecules with a different atom count, atom-dependent dimensions change accordingly.

## Scientific Defaults

Gaussian:

```text
Ground-state optimization: B3LYP/6-31G(d)
Excited-state screening:  TDA/CAM-B3LYP/6-31G(d)
Excited states:           10
```

NEXMD:

```text
Ground-state optimization: AM1
Ground-state dynamics:     AM1
Initial velocities:        zero
```

PySEQM:

```text
Electronic properties: CIS/AM1
Total states:          11
State convention:      1 ground + 10 excited
Device policy:         PYSEQM_DEVICE = auto|cuda|cpu
Default CPU resources: PYSEQM_CPUS = 8, PYSEQM_MEM = 5G
```

PySEQM device routing:

```text
auto   Prefer CUDA when visible to the job; otherwise run on CPU.
cuda   Require CUDA and fail clearly if CUDA is not visible.
cpu    Run on CPU and generate SLURM jobs without GPU directives.
```

Dataset:

```text
Use molecule-specific prefixes.
Use PySEQM reference optimization values when available.
Use zero reference values only as an explicit fallback for testing.
```

## Manual Approval And Quality Gates

The workflow contains several stopping or review points.

Hard stop:

```text
Any negative Gaussian frequency stops downstream execution for that molecule.
```

Manual review:

```text
Gaussian screening_summary.csv must be inspected before production continuation.
```

Required downstream files:

```text
NEXMD optimization requires Gaussian optimized geometry.
NEXMD dynamics requires <mol_dir>/nexmd/coords.xyz.
Frame extraction requires NEXMD coords.xyz and velocity.out from dynamics.
PySEQM batch calculations require molecule-specific R and Z arrays.
Dataset generation should use PySEQM reference energy and forces.
```

Recommended validation:

```text
Check SLURM completion states.
Check final log tails for normal termination.
Check array shapes after PySEQM batch calculations.
Check final dataset .npy count and key array dimensions.
```

## Tested Smoke-Run Behavior

The updated package has been tested on Darwin through complete smoke-test dataset generation for multiple molecules.

Representative smoke-test settings:

```text
MODE = Test
NEXMD_N_CLASS_STEPS_TEST = 10000
FRAME_START_TEST = 1
FRAME_STOP_TEST = 101
N_STATES = 11
```

Representative successful final outputs:

```text
<mol_dir>/dataset/0K
178 .npy files
<prefix>_R.npy
<prefix>_Z.npy
<prefix>_F.npy
<prefix>_dENACR.npy
```

Example verified shapes:

```text
dnbp_R.npy                    (101, 32, 3)
dnbp_Z.npy                    (101, 32)
dnbp_F.npy                    (101, 11, 32, 3)
dnbp_dENACR.npy               (101, 45, 96)
dnbp_optimized_reference_forces.npy (32, 3)

dmabn_R.npy                   (101, 21, 3)
dmabn_F.npy                   (101, 11, 21, 3)
dmabn_dENACR.npy              (101, 45, 63)
dmabn_optimized_reference_forces.npy (21, 3)
```

## Recommended Darwin Usage

Prepare Gaussian screening inputs only:

```bash
cd /vast/home/akhanna2/scratch/summer_project/dataset_workflow
/usr/projects/ml4chem/akhanna2/softwares/envs/ml_env/bin/python dataset_workflow.py \
  --molecule /vast/home/akhanna2/scratch/summer_project/my_molecule \
  --submit_jobs false
```

Submit and monitor Gaussian screening:

```bash
/usr/projects/ml4chem/akhanna2/softwares/envs/ml_env/bin/python dataset_workflow.py \
  --molecule /vast/home/akhanna2/scratch/summer_project/my_molecule \
  --submit_jobs true \
  --monitor_jobs true
```

After approval, continue from NEXMD optimization to final dataset:

```bash
/usr/projects/ml4chem/akhanna2/softwares/envs/ml_env/bin/python dataset_workflow.py \
  --molecule /vast/home/akhanna2/scratch/summer_project/my_molecule \
  --nexmd_opt true \
  --continue true \
  --submit_jobs true \
  --monitor_jobs true
```

Run a downstream smoke test after NEXMD dynamics exists:

```bash
/usr/projects/ml4chem/akhanna2/softwares/envs/ml_env/bin/python dataset_workflow.py \
  --molecule /vast/home/akhanna2/scratch/summer_project/my_molecule \
  --extract_frames true \
  --continue true \
  --submit_jobs true \
  --monitor_jobs true
```

## Production Notes

Use `MODE = Test` for short validation runs and `MODE = Production` for large-scale dataset generation.

Before a production run:

```text
Confirm molecule approval from Gaussian screening.
Confirm partition choices.
Confirm frame extraction range.
Confirm target frame count.
Confirm N_STATES = 11 unless intentionally changed.
Confirm PySEQM batch size and GPU availability.
Confirm dataset output path.
Confirm reference optimization completed successfully.
```

For multi-molecule runs:

```text
Use the Molecule dictionary in dataset_workflow.inp.
Set MAX_PARALLEL_MOLECULES according to cluster policy.
Avoid launching multiple wrapper jobs on restricted one-job partitions.
Keep WRAPPER_PARTITION = general by default.
```

## Current Limitations And Future Improvements

Current limitations:

```text
Molecule approval after Gaussian screening remains manual.
Screening thresholds rank molecules but do not fully determine scientific suitability.
Cluster modules and Gaussian/NEXMD launch commands may still require site-specific edits.
Large production runs should be audited for storage, walltime, and partition policy before submission.
```

Recommended future improvements:

```text
Add richer screening plots and dashboard-style summaries.
Add automated array-shape validation reports after each PySEQM run.
Add molecule-level provenance metadata in the final dataset directory.
Add optional dependency-based SLURM chain generation for users who prefer pure scheduler control.
Add retry/wait guards after filesystem-heavy stages to reduce race conditions on parallel filesystems.
```

## Q-Chem Backend Technical Addendum

The workflow now supports `QUANTUM_BACKEND = qchem` as a full production backend. Q-Chem performs CAM-B3LYP/6-31G* ground-state optimization and TDA excited-state frame calculations for the first 5 excited states, while NEXMD remains responsible for AM1 ground-state optimization and dynamics.

Q-Chem production frame jobs run as SLURM arrays. Each array task processes `QCHEM_ARRAY_CHUNK_SIZE` frames and writes shard `.npz` files under `<mol>/qchem/batch_exsp/shards`. The `qchem_collect` step validates all shards and assembles canonical arrays for the existing dataset generator. With the default `QCHEM_N_ROOTS = 5`, Q-Chem NACs cover the 10 unique pairs among excited states `1 2 3 4 5`.

Unit conventions are fixed at the Q-Chem parser boundary: Hartree energies are converted to eV, gradients are converted to physical forces with `-gradient * Hartree_to_eV / Bohr_to_Angstrom`, and ETF derivative couplings are converted from Bohr^-1 to Angstrom^-1. The final dataset generator therefore remains backend-independent.

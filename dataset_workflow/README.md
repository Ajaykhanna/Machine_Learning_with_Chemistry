# Dataset Workflow

A molecule-agnostic, SLURM-aware workflow for generating machine-learning interatomic potential datasets for multistate, multitask excited-state prediction. The package starts from an unoptimized molecular geometry and coordinates Gaussian screening, NEXMD ground-state sampling, PySEQM excited-state single-point calculations, PySEQM reference optimization, and final dataset assembly.

The workflow was designed for Darwin-style cluster execution, but the package layout keeps every stage script independently runnable so it can be adapted to other clusters with minimal edits to generated SLURM files.

## What This Package Does

The end-to-end route is:

```text
unopt.xyz
  -> Gaussian B3LYP/6-31G(d) ground-state optimization
  -> Gaussian TDA/CAM-B3LYP/6-31G(d) excited-state screening, 10 excited states
  -> ranked screening_summary.csv for manual molecule approval
  -> NEXMD AM1 ground-state reoptimization
  -> NEXMD AM1 ground-state dynamics
  -> frame extraction from equilibrated dynamics
  -> molecule-specific PySEQM input arrays, for example prodan_R.npy and prodan_Z.npy
  -> PySEQM CIS/AM1 single-point excited-state properties, 1 ground + 10 excited states
  -> PySEQM ground-state reference optimization for shifted energies and forces
  -> final dataset arrays with molecule-specific prefixes
```

Key defaults:

- Gaussian screening uses 10 excited states.
- PySEQM and dataset generation use `N_STATES = 11`, meaning 1 ground state plus 10 excited states.
- Test mode uses `NEXMD_N_CLASS_STEPS_TEST = 10000` and extracts 101 frames by default.
- Production mode uses `NEXMD_N_CLASS_STEPS_PRODUCTION = 5000000` and targets 50000 extracted configurations by default.
- CPU stages default to the `general` partition.
- PySEQM GPU stages default to the `ml4chem` partition.
- PySEQM stages default to `/usr/projects/ml4chem/akhanna2/softwares/envs/ml_env/bin/python`.
- Other Python helper stages default to `/usr/projects/ml4chem/akhanna2/softwares/envs/ml_env/bin/python`.
- PySEQM device routing defaults to `PYSEQM_DEVICE = auto`, with `PYSEQM_CPUS = 8` and `PYSEQM_MEM = 5G`.
- Q-Chem backend defaults to Darwin profile settings: `shared-spr`, account `y2020-bf`, `32` CPUs, `CAM-B3LYP/6-31G*`, and Q-Chem `S1-S5` TDA/NAC calculations by default.

## Package Layout

```text
dataset_workflow/
  dataset_workflow.py              # Master workflow driver
  README.md
  scripts/
    00_gaussian_screening.py       # Gaussian GS opt and EXSP screening automation
    0_gaussian_automation.py       # Compatibility alias for Gaussian automation
    01_nexmd_ground_state.py       # NEXMD AM1 GS optimization and GS dynamics setup
    1_run_nexmd_dyn.py             # Compatibility alias for NEXMD setup
    02_extract_md_frames.py        # Extract XYZ/velocity frames from NEXMD trajectories
    claude_read_xyz_vel_traj.py    # Compatibility alias for frame extraction
    03_prepare_frame_inputs.py     # Build molecule-specific R/Z/V numpy arrays
    extract_frames_for_pyseqm.py   # Compatibility alias for PySEQM input prep
    04_run_pyseqm_properties.py    # PySEQM CIS/AM1 batch electronic properties
    run_pyseqm_electronic_properties_calcs.py
    05_generate_dataset.py         # Final dataset generation
    pyseqm_optimized_geom.py       # PySEQM ground-state reference optimization
    geometry_parser.py
    pipeline_config.py
  configs/
    dataset_workflow.inp           # Main workflow config example
    individual_scripts.inp         # Manual stage-by-stage reference config
    master_slurm.inp               # Scheduler policy reference config
  example/
    Nilered/unopt.xyz              # Lightweight example input only
    Prodan/unopt.xyz               # Lightweight example input only
  templates/
    gsopt_input.ceon
    gsdyn_input.ceon
    standard_nexmd_md.sbatch
    submit_dataset_extract.sbatch and other submit_*.sbatch templates
  logs/
    .gitkeep
```

Large generated outputs are intentionally not included in `example/`. The workflow creates molecule-local `gaussian/`, `nexmd/`, `pyseqm/`, `dataset/`, and `slurm/` directories when it runs.

## Requirements

Cluster-side software:

- SLURM with `sbatch`, `sacct`, and `squeue` available.
- Gaussian for DFT geometry optimization and excited-state screening.
- NEXMD for AM1 ground-state optimization and dynamics.
- PySEQM and its GPU dependencies for CIS/AM1 electronic properties.
- Python with `numpy`, `scipy`, `pandas`, `matplotlib`, and workflow-specific dependencies available in the configured environments.

Darwin defaults used by this package:

```text
PYTHON_BIN = /usr/projects/ml4chem/akhanna2/softwares/envs/ml_env/bin/python
PYSEQM_PYTHON_BIN = /usr/projects/ml4chem/akhanna2/softwares/envs/ml_env/bin/python
GAUSSIAN_ROOT = /usr/projects/cint/Gaussian/g16A03
GAUSSIAN_EXE = g16
GAUSSIAN_LIBRARY_PATH =
GAUSSIAN_SCRATCH_ROOT = /tmp/akhanna2/GAUSSIAN_SCR
NEXMD_EXE = /usr/projects/ml4chem/akhanna2/softwares/NEXMD/nexmd.exe
WRAPPER_PARTITION = general
DEFAULT_PARTITION = general
GPU_PARTITION = ml4chem
GPU_GRES = gpu:4
PYSEQM_DEVICE = auto
PYSEQM_CPUS = 8
PYSEQM_MEM = 5G
PYSEQM_OPT_CPUS = 8
PYSEQM_OPT_MEM = 5G
```

Do not install or modify packages inside these environments from the workflow. Treat them as externally managed runtime environments.

## Executable And Library Paths

Gaussian, NEXMD, PySEQM, and CUDA runtime paths can be set in `configs/dataset_workflow.inp` or overridden on the command line. CLI values take precedence over config values.

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

Equivalent CLI examples:

```bash
python dataset_workflow.py \
  --molecule example/Prodan \
  --submit_jobs false \
  --gaussian-root /usr/projects/cint/Gaussian/g16A03 \
  --gaussian-exe g16 \
  --gaussian-library-path /optional/gaussian/lib \
  --nexmd-exe /usr/projects/ml4chem/akhanna2/softwares/NEXMD/nexmd.exe \
  --python-bin /usr/projects/ml4chem/akhanna2/softwares/envs/ml_env/bin/python \
  --pyseqm-python-bin /usr/projects/ml4chem/akhanna2/softwares/envs/ml_env/bin/python \
  --cuda-path /usr/local/cuda
```

The workflow writes these values into generated SLURM files when relevant. Empty optional values are omitted, so cluster defaults remain usable.


## Q-Chem Backend

Set `QUANTUM_BACKEND = qchem` or pass `--quantum-backend qchem` to use Q-Chem for DFT/TDA quantum calculations while keeping NEXMD for AM1 optimization and ground-state dynamics.

```text
unopt.xyz
  -> Q-Chem CAM-B3LYP/6-31G* ground-state optimization and reference force
  -> NEXMD AM1 ground-state optimization
  -> NEXMD AM1 ground-state dynamics
  -> frame extraction and molecule-specific R/Z arrays
  -> Q-Chem TDA frame-property SLURM array
  -> Q-Chem shard collection into canonical dataset inputs
  -> final dataset generation
```

Q-Chem steps are independently selectable:

```text
--qchem_gsopt true
--qchem_batch_exsp true
--qchem_collect true
```

A full Q-Chem route can be launched from Q-Chem GS optimization with:

```bash
python dataset_workflow.py \
  --molecule /path/to/molecule \
  --quantum-backend qchem \
  --qchem_gsopt true \
  --continue true \
  --submit_jobs true \
  --monitor_jobs true
```

The collector writes the same files consumed by `05_generate_dataset.py`: `energies_all_states.npy`, `total_energy.npy`, `forces_all_states.npy`, `gs_dipoles.npy`, `transition_dipoles.npy`, and `nacrs_all_states.npy`. Energies are eV, forces are eV/Angstrom, and NACRs are Angstrom^-1. By default, Q-Chem frame calculations use `QCHEM_N_ROOTS = 5`, so TDA, excited-state forces, and NAC calculations cover only `S1-S5`; NACs are generated for the 10 unique excited-state pairs among `1 2 3 4 5`.

Darwin defaults are:

```text
QCHEM_PROFILE = darwin
QCHEM_EXE = /usr/projects/ml4chem/Programs/qchem/bin/qchem
QCHEM_ENV_FILE = /vast/home/akhanna2/data/software/qchem/qchem_darwin.sh
QCHEM_PYTHON_BIN = /usr/projects/cint/anaconda3/gpu4pyscf/bin/python
QCHEM_PARTITION = shared-spr
QCHEM_ACCOUNT = y2020-bf
QCHEM_CPUS = 32
QCHEM_METHOD = CAM-B3LYP
QCHEM_BASIS = 6-31G*
QCHEM_N_ROOTS = 5
QCHEM_ARRAY_CHUNK_SIZE = 1
```

Use `QCHEM_PROFILE = chicoma` for the Chicoma example defaults, or override any `QCHEM_*` key in `configs/dataset_workflow.inp`.

For complete Darwin and Chicoma run instructions, see [QCHEM_BACKEND_WORKFLOW.md](QCHEM_BACKEND_WORKFLOW.md).
## PySEQM CPU/GPU Routing

PySEQM geometry optimization and batch excited-state calculations can run on CPUs or GPUs.

```text
PYSEQM_DEVICE = auto   # Prefer CUDA when scheduled with GPU resources; fall back to CPU in the script.
PYSEQM_DEVICE = cuda   # Require CUDA. The PySEQM script fails clearly if CUDA is not visible.
PYSEQM_DEVICE = cpu    # Use CPU partition settings and do not request GPUs.
```

CPU and memory are configurable:

```text
PYSEQM_CPUS = 8
PYSEQM_MEM = 5G
PYSEQM_OPT_CPUS = 8
PYSEQM_OPT_MEM = 5G
```

Equivalent CLI example:

```bash
python dataset_workflow.py \
  --molecule /path/to/dmabn \
  --pyseqm_batch_exsp true \
  --submit_jobs true \
  --pyseqm-device cpu \
  --pyseqm-cpus 8 \
  --pyseqm-mem 5G
```

## Quick Start

From the package root:

```bash
cd dataset_workflow
```

Prepare the default Gaussian screening SLURM files for the bundled example molecules without submitting jobs:

```bash
/usr/projects/ml4chem/akhanna2/softwares/envs/ml_env/bin/python dataset_workflow.py \
  --config configs/dataset_workflow.inp \
  --submit_jobs false
```

Run a single molecule in dry-run mode:

```bash
/usr/projects/ml4chem/akhanna2/softwares/envs/ml_env/bin/python dataset_workflow.py \
  --molecule example/Prodan \
  --submit_jobs false
```

Submit and monitor the default Gaussian screening steps:

```bash
/usr/projects/ml4chem/akhanna2/softwares/envs/ml_env/bin/python dataset_workflow.py \
  --molecule example/Prodan \
  --submit_jobs true \
  --monitor_jobs true
```

The default behavior is intentionally conservative. If no `RUN_*` flags are true, only Gaussian GS optimization and Gaussian EXSP screening are prepared or run.

## Configuration File Format

The main config is `configs/dataset_workflow.inp`.

Rules:

- Lines starting with `#` are ignored.
- Keys are case-insensitive.
- Use normalized `snake_case` or uppercase `SNAKE_CASE` keys.
- Values use `KEY = VALUE` syntax.
- Boolean values should be `true` or `false`.
- Lists use comma-separated values.
- Quote strings that contain spaces and should not be parsed as lists, for example `GPU_IDS = "0 1 2 3"`.
- Molecules can be supplied as a dictionary block.

Example molecule block:

```text
Molecule = {
    Nilered: "example/Nilered"
    Prodan: "example/Prodan"
}
```

The workflow prints how many molecules were found and runs each molecule in an independent molecule-local layout. Multi-molecule workflows can be launched in parallel with `MAX_PARALLEL_MOLECULES`.

## Command-Line Interface

Main options:

```bash
python dataset_workflow.py --help
```

Common options:

```text
--config configs/dataset_workflow.inp
--molecule /path/to/molecule_dir
--prefix prodan
--submit_jobs true|false
--monitor_jobs true|false
--continue true|false
--overwrite true|false
```

Step flags:

```text
--gaussian_gsopt true
--gaussian_exsp true
--nexmd_opt true
--nexmd_gsdyn true
--extract_frames true
--prepare_frames true
--pyseqm_batch_exsp true
--pyseqm_opt true
--qchem_gsopt true
--qchem_batch_exsp true
--qchem_collect true
--generate_dataset true
```

If a single step is selected, only that step is prepared or run. Required input files from earlier steps must already exist. If a required file is missing, the workflow stops with a clear error and suggests rerunning with `--continue true` or providing the missing path.

If a start step is selected with `--continue true`, the workflow runs from that step through the final dataset stage. For example:

```bash
python dataset_workflow.py \
  --molecule example/Prodan \
  --nexmd_gsdyn true \
  --continue true \
  --submit_jobs true
```

This runs:

```text
nexmd_gsdyn -> extract_frames -> prepare_frames -> pyseqm_batch_exsp -> pyseqm_opt -> generate_dataset
```

Completed steps are skipped by default when their expected output exists. Use `--overwrite true` to regenerate step outputs.

## Workflow Stages

### 1. Gaussian Ground-State Optimization

Script:

```text
scripts/00_gaussian_screening.py --gsopt
```

Input:

```text
<mol_dir>/unopt.xyz
```

The workflow copies `unopt.xyz` into:

```text
<mol_dir>/gaussian/<MoleculeName>/unopt.xyz
```

Outputs include molecule-specific Gaussian files and:

```text
<mol_dir>/gaussian/<MoleculeName>/mol_freq.txt
```

Any negative frequency stops the molecule before downstream steps.

### 2. Gaussian Excited-State Screening

Script:

```text
scripts/00_gaussian_screening.py --exsp --nstates 10
```

Outputs:

```text
<mol_dir>/gaussian/<MoleculeName>/mol_exsp.txt
screening_summary.csv
screening plots when enabled by the stage script
```

Ranking logic uses the configured thresholds:

```text
OSCILLATOR_BRIGHT_THRESHOLD = 0.10
OSCILLATOR_DARK_THRESHOLD = 0.05
EXCITATION_ENERGY_THRESHOLD_EV = 0.50
MINIMUM_STATES = 5
```

The workflow ranks candidates but does not automatically approve a molecule for production.

### 3. NEXMD Ground-State Optimization

Script:

```text
scripts/01_nexmd_ground_state.py --gs_opt
```

Input:

```text
Gaussian optimized geometry from the GS optimization log
```

Templates:

```text
templates/gsopt_input.ceon
templates/standard_nexmd_md.sbatch
```

Output:

```text
<mol_dir>/nexmd/coords.xyz
```

This `coords.xyz` is treated as the authoritative NEXMD optimized geometry.

### 4. NEXMD Ground-State Dynamics

Script:

```text
scripts/01_nexmd_ground_state.py --gs_dyn
```

Input:

```text
<mol_dir>/nexmd/coords.xyz
```

Template:

```text
templates/gsdyn_input.ceon
```

Default initial velocities are zero. Test mode uses `n_class_steps = 10000`; production mode uses `n_class_steps = 5000000`, unless overridden.

Outputs:

```text
<mol_dir>/nexmd/gs_dyn/coords.xyz
<mol_dir>/nexmd/gs_dyn/velocity.out
```

### 5. Frame Extraction

Script:

```text
scripts/02_extract_md_frames.py
```

Default test extraction:

```text
FRAME_START_TEST = 1
FRAME_STOP_TEST = 101
FRAME_STEP = 1
```

Default production extraction:

```text
FRAME_START_PRODUCTION = 5000
FRAME_STOP_PRODUCTION = 55001
FRAME_STEP = 1
```

Because NEXMD writes saved frames every 10 fs in the configured dynamics setup, `FRAME_STEP = 1` keeps 10 fs spacing.

Output:

```text
<mol_dir>/nexmd/gs_dyn/<N>_10fs_frames/frame_*/frame_*.xyz
<mol_dir>/nexmd/gs_dyn/<N>_10fs_frames/frame_*/frame_*.vel
```

### 6. Prepare PySEQM Inputs

Script:

```text
scripts/03_prepare_frame_inputs.py --sort_ZRV
```

Outputs are molecule-specific:

```text
<mol_dir>/pyseqm/prepared_frames/<prefix>_R.npy
<mol_dir>/pyseqm/prepared_frames/<prefix>_Z.npy
<mol_dir>/pyseqm/prepared_frames/<prefix>_V.npy
```

The prefix defaults to the lowercase molecule key, for example `prodan` or `nilered`.

### 7. PySEQM Batch Excited-State Properties

Script:

```text
scripts/04_run_pyseqm_properties.py
```

Default method:

```text
CIS/AM1
N_STATES = 11
```

Outputs:

```text
<mol_dir>/pyseqm/batch_exsp/energies_all_states.npy
<mol_dir>/pyseqm/batch_exsp/forces_all_states.npy
<mol_dir>/pyseqm/batch_exsp/gs_dipoles.npy
<mol_dir>/pyseqm/batch_exsp/ex_dipoles_net.npy
<mol_dir>/pyseqm/batch_exsp/transition_dipoles.npy
<mol_dir>/pyseqm/batch_exsp/total_energy.npy
<mol_dir>/pyseqm/batch_exsp/nacrs_all_states.npy
```

### 8. PySEQM Reference Optimization

Script:

```text
scripts/pyseqm_optimized_geom.py
```

Input:

```text
<mol_dir>/nexmd/coords.xyz
```

Outputs are molecule-specific:

```text
<mol_dir>/pyseqm/gsopt_reference/<prefix>_optimized_gs_geometry.xyz
<mol_dir>/pyseqm/gsopt_reference/<prefix>_optimized_reference_energy.txt
<mol_dir>/pyseqm/gsopt_reference/<prefix>_optimized_reference_forces.npy
```

These reference values are passed into dataset generation as `OPTIMIZED_ENERGY` and `OPTIMIZED_FORCES` equivalents for shifted energy and shifted force datasets.

### 9. Dataset Generation

Script:

```text
scripts/05_generate_dataset.py
```

The workflow passes:

```text
--reference-energy <prefix>_optimized_reference_energy.txt
--reference-forces <prefix>_optimized_reference_forces.npy
```

If reference files are missing, the workflow writes explicit zero reference files and logs a warning. This keeps manual testing possible while making the missing reference state visible.

Final arrays are written to `DATASET_OUTPUT_DIR` if configured. Otherwise they go to:

```text
<mol_dir>/dataset
```

Expected production shape family for the default PySEQM route with 50000 frames, 15 atoms, and 11 total states:

```text
<prefix>_V.npy       (50000, 15, 3)
<prefix>_Z.npy       (50000, 15)
<prefix>_S.npy       (50000, 11)
<prefix>_sE.npy      (50000, 11)
<prefix>_sF.npy      (50000, 11, 15, 3)
<prefix>_D.npy       (50000, 11, 3)
<prefix>_F.npy       (50000, 11, 15, 3)
<prefix>_NACR.npy    (50000, 45, 15, 3)
<prefix>_dE.npy      (50000, 11)
<prefix>_dENACR.npy  (50000, 45, 45)
```

For the default Q-Chem route with `QCHEM_N_ROOTS = 5`, the corresponding state-dependent outputs use 6 total states and 10 excited-state NAC pairs:

```text
<prefix>_S.npy       (50000, 6)
<prefix>_sE.npy      (50000, 6)
<prefix>_sF.npy      (50000, 6, 15, 3)
<prefix>_D.npy       (50000, 6, 3)
<prefix>_F.npy       (50000, 6, 15, 3)
<prefix>_NACR.npy    (50000, 10, 15, 3)
<prefix>_dE.npy      (50000, 6)
<prefix>_dENACR.npy  (50000, 10, 45)
```

Additional split files such as `<prefix>_S0.npy`, `<prefix>_sE1.npy`, `<prefix>_F2.npy`, or `<prefix>_NACR3.npy` may also be generated depending on `05_generate_dataset.py` options.

## Generated SLURM Files

Every step gets its own generated SLURM file:

```text
<mol_dir>/slurm/submit_<prefix>_<step>.sbatch
```

Examples:

```text
example/Prodan/slurm/submit_prodan_gaussian_gsopt.sbatch
example/Prodan/slurm/submit_prodan_pyseqm_batch_exsp.sbatch
example/Prodan/slurm/submit_prodan_generate_dataset.sbatch
```

The generated files are deliberately editable. If a cluster requires a temporary node constraint, account, reservation, or module load, inspect and edit the relevant step sbatch before manual submission.

By default, the workflow does not include node-specific constraints.

## Manual Stage Usage

All stage scripts remain independently runnable. Examples:

```bash
python scripts/00_gaussian_screening.py \
  --path example/Prodan/gaussian/Prodan \
  --gsopt \
  --prepare-only
```

```bash
python scripts/01_nexmd_ground_state.py \
  --path example/Prodan/gaussian/Prodan \
  --gs_opt \
  --out_dir example/Prodan/nexmd \
  --prepare-only
```

```bash
python scripts/02_extract_md_frames.py \
  -xyz example/Prodan/nexmd/gs_dyn/coords.xyz \
  -vel example/Prodan/nexmd/gs_dyn/velocity.out \
  --start 1 \
  --stop 101 \
  --step 1 \
  --workers 8 \
  --output example/Prodan/nexmd/gs_dyn/101_10fs_frames
```

Use `configs/individual_scripts.inp` as the canonical mapping between workflow variables and individual script CLI arguments.

## Recommended Darwin Workflow

Copy or clone this package onto Darwin, then run from the package root:

```bash
cd /vast/home/akhanna2/scratch/summer_project/dataset_workflow
```

Prepare Gaussian screening files for a new molecule:

```bash
mkdir -p molecules/MyMolecule
cp /path/to/unopt.xyz molecules/MyMolecule/unopt.xyz
/usr/projects/ml4chem/akhanna2/softwares/envs/ml_env/bin/python dataset_workflow.py \
  --molecule molecules/MyMolecule \
  --submit_jobs false
```

Submit and monitor Gaussian screening:

```bash
/usr/projects/ml4chem/akhanna2/softwares/envs/ml_env/bin/python dataset_workflow.py \
  --molecule molecules/MyMolecule \
  --submit_jobs true \
  --monitor_jobs true
```

After inspecting the ranked Gaussian screening output and approving the molecule, continue with NEXMD optimization:

```bash
/usr/projects/ml4chem/akhanna2/softwares/envs/ml_env/bin/python dataset_workflow.py \
  --molecule molecules/MyMolecule \
  --nexmd_opt true \
  --continue true \
  --submit_jobs true \
  --monitor_jobs true
```

For a smoke test that starts from completed NEXMD dynamics and continues through dataset generation:

```bash
/usr/projects/ml4chem/akhanna2/softwares/envs/ml_env/bin/python dataset_workflow.py \
  --molecule molecules/MyMolecule \
  --extract_frames true \
  --continue true \
  --submit_jobs true \
  --monitor_jobs true
```

## Troubleshooting

`Required input is missing`

The selected step depends on an earlier output that does not exist. Either run earlier steps first, rerun from the desired step with `--continue true`, or provide/copy the expected file into the standard molecule-local location.

`Reference files missing; using zero reference files`

The PySEQM reference optimization has not been run, or its outputs are not in the expected location. Run `--pyseqm_opt true` before `--generate_dataset true`, or provide `REFERENCE_ENERGY_FILE` and `REFERENCE_FORCES_FILE` in the config.

`PySEQM runs in the wrong environment`

Set `PYSEQM_PYTHON_BIN` in `configs/dataset_workflow.inp` or pass an edited config. The default is `/usr/projects/ml4chem/akhanna2/softwares/envs/ml_env/bin/python`.

`Gaussian or NEXMD submits to the wrong partition`

Set `WRAPPER_PARTITION = general` for controller/wrapper jobs. Set `DEFAULT_PARTITION` for CPU compute jobs such as Gaussian, NEXMD, frame extraction, and dataset generation. Set `GPU_PARTITION` for PySEQM jobs and request GPUs through `GPU_GRES`; use `GPU_GRES = none` on partitions where GPUs are available without a `--gres` request.

`I need to rerun a step that already has outputs`

Use:

```bash
python dataset_workflow.py --molecule <mol_dir> --<step> true --overwrite true
```

## Development Notes

This package keeps compatibility filenames from the original workflow so existing command snippets remain valid. The canonical production names are the numbered scripts:

```text
00_gaussian_screening.py
01_nexmd_ground_state.py
02_extract_md_frames.py
03_prepare_frame_inputs.py
04_run_pyseqm_properties.py
05_generate_dataset.py
```

The workflow avoids node-specific defaults. Use generated SLURM files as transparent, inspectable artifacts rather than hidden scheduler state.

## License And Citation

Add project license, citation, and dataset provenance requirements here before public release.

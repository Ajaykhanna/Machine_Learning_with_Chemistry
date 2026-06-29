# Q-Chem Backend Workflow: Darwin and Chicoma

This guide describes how to run the Q-Chem backend of `dataset_workflow` for both smoke-test and production datasets.

The Q-Chem route is:

```text
unopt.xyz
  -> Q-Chem CAM-B3LYP/6-31G* ground-state optimization and reference force
  -> NEXMD AM1 ground-state optimization
  -> NEXMD AM1 ground-state dynamics
  -> frame extraction from NEXMD coordinates and velocities
  -> prepare molecule-specific R/Z/V numpy arrays
  -> Q-Chem TDA frame properties, excited-state forces, and NACs
  -> Q-Chem shard collection into canonical .npy files
  -> final dataset generation
```

## State Convention

The Q-Chem backend intentionally uses only the first 5 excited states by default:

```text
QCHEM_N_ROOTS = 5
```

This means:

```text
Total states in Q-Chem dataset = 6  # S0 + S1-S5
NAC pairs = 10                     # all unique pairs among 1 2 3 4 5
```

The canonical Q-Chem collector outputs have these shapes for `N` frames and `A` atoms:

```text
energies_all_states.npy    (N, 6)
total_energy.npy           (N, 1)
forces_all_states.npy      (N, 6, A, 3)
gs_dipoles.npy             (N, 3)
transition_dipoles.npy     (N, 5, 3)
nacrs_all_states.npy       (10, N, A, 3)
```

The final dataset generator then writes state-dependent arrays with 6 total states and 10 NAC pairs.

## Common Setup

Run from the package root on the target cluster:

```bash
cd /vast/home/akhanna2/scratch/summer_project/dataset_workflow
```

For Chicoma, replace the package path with the location where `dataset_workflow` has been copied.

Each molecule directory must contain:

```text
<mol_dir>/unopt.xyz
```

The recommended launch pattern is to select only the first Q-Chem step and let `CONTINUE = true` or `--continue true` run the remaining dependent stages.

## Darwin: 101-Frame Test Dataset

Create a Darwin test config such as `configs/qchem_darwin_101.inp`:

```text
Molecule = {
  dmabn_test: "/vast/home/akhanna2/scratch/summer_project/dmabn_qchem_test"
}

MODE = Test
QUANTUM_BACKEND = qchem
SUBMIT_JOBS = true
MONITOR_JOBS = true
CONTINUE = true
OVERWRITE = false

RUN_QCHEM_GSOPT = true
RUN_NEXMD_OPT = false
RUN_NEXMD_GSDYN = false
RUN_EXTRACT_FRAMES = false
RUN_PREPARE_FRAMES = false
RUN_QCHEM_BATCH_EXSP = false
RUN_QCHEM_COLLECT = false
RUN_GENERATE_DATASET = false

WRAPPER_PARTITION = general
DEFAULT_PARTITION = general
QCHEM_PROFILE = darwin
QCHEM_EXE = /usr/projects/ml4chem/Programs/qchem/bin/qchem
QCHEM_ENV_FILE = /vast/home/akhanna2/data/software/qchem/qchem_darwin.sh
QCHEM_PYTHON_BIN = /usr/projects/cint/anaconda3/gpu4pyscf/bin/python
QCHEM_SCRATCH = /vast/home/akhanna2/scratch/summer_project/qchem_scratch
QCHEM_PARTITION = shared-spr
QCHEM_ACCOUNT = y2020-bf
QCHEM_QOS = long
QCHEM_CPUS = 32
QCHEM_MEM = 128G
QCHEM_WALLTIME = 24:00:00
QCHEM_METHOD = CAM-B3LYP
QCHEM_BASIS = 6-31G*
QCHEM_N_ROOTS = 5
QCHEM_ARRAY_CHUNK_SIZE = 1
QCHEM_MEM_TOTAL_MB = 125000

NEXMD_N_CLASS_STEPS_TEST = 10000
FRAME_START_TEST = 1
FRAME_STOP_TEST = 101
FRAME_STEP = 1
FRAME_WORKERS = 8

DATASET_CHOP_START_TEST = 1
DATASET_CHOP_END_TEST = 102
DATASET_CHOP_STEP = 10000
DATASET_WORKERS = 16
```

Submit:

```bash
/usr/projects/ml4chem/akhanna2/softwares/envs/ml_env/bin/python dataset_workflow.py \
  --config configs/qchem_darwin_101.inp
```

Expected Q-Chem collector shapes for a 21-atom molecule are:

```text
energies_all_states.npy    (101, 6)
forces_all_states.npy      (101, 6, 21, 3)
nacrs_all_states.npy       (10, 101, 21, 3)
transition_dipoles.npy     (101, 5, 3)
```

## Darwin: 50K-Frame Production Dataset

Create a production config such as `configs/qchem_darwin_50k.inp`. The important production-specific differences are:

```text
MODE = Production
QUANTUM_BACKEND = qchem
SUBMIT_JOBS = true
MONITOR_JOBS = true
CONTINUE = true
OVERWRITE = false
RUN_QCHEM_GSOPT = true

NEXMD_N_CLASS_STEPS_PRODUCTION = 5000000

# Inclusive frame range: 5000 through 54999 gives exactly 50000 frames.
FRAME_START_PRODUCTION = 5000
FRAME_STOP_PRODUCTION = 54999
FRAME_STEP = 1
FRAME_WORKERS = 16

# Dataset slicing uses Python-style half-open [start, end).
DATASET_CHOP_START_PRODUCTION = 1
DATASET_CHOP_END_PRODUCTION = 50001
DATASET_CHOP_STEP = 10000
DATASET_WORKERS = 16

QCHEM_PROFILE = darwin
QCHEM_PARTITION = shared-spr
QCHEM_ACCOUNT = y2020-bf
QCHEM_QOS = long
QCHEM_CPUS = 32
QCHEM_MEM = 128G
QCHEM_WALLTIME = 24:00:00
QCHEM_N_ROOTS = 5

# Use 1 for maximum fault isolation. Larger chunks reduce array size but make reruns coarser.
QCHEM_ARRAY_CHUNK_SIZE = 1
```

Submit:

```bash
/usr/projects/ml4chem/akhanna2/softwares/envs/ml_env/bin/python dataset_workflow.py \
  --config configs/qchem_darwin_50k.inp
```

Expected production Q-Chem dataset shapes for `A` atoms are:

```text
<prefix>_S.npy       (50000, 6)
<prefix>_sE.npy      (50000, 6)
<prefix>_F.npy       (50000, 6, A, 3)
<prefix>_D.npy       (50000, 6, 3)
<prefix>_NACR.npy    (50000, 10, A, 3)
<prefix>_dE.npy      (50000, 6)
<prefix>_dENACR.npy  (50000, 10, 3*A)
```

## Chicoma: 101-Frame Test Dataset

Create a Chicoma test config such as `configs/qchem_chicoma_101.inp`:

```text
Molecule = {
  dmabn_test: "/path/to/dmabn_qchem_test"
}

MODE = Test
QUANTUM_BACKEND = qchem
SUBMIT_JOBS = true
MONITOR_JOBS = true
CONTINUE = true
OVERWRITE = false
RUN_QCHEM_GSOPT = true

WRAPPER_PARTITION = standard
DEFAULT_PARTITION = standard
QCHEM_PROFILE = chicoma
QCHEM_EXE = /usr/projects/ml4chem/Programs/qchem/bin/qchem
QCHEM_ENV_FILE = /usr/projects/ml4chem/envs/qchem.sh
QCHEM_PYTHON_BIN = /usr/projects/ml4chem/akhanna2/conda_envs/hiphop_env/bin/python
QCHEM_SCRATCH = /users/akhanna2/scratch/qchem/tdpp
QCHEM_PARTITION = standard
QCHEM_ACCOUNT = s17_cint
QCHEM_CPUS = 32
QCHEM_MEM = 128G
QCHEM_WALLTIME = 16:00:00
QCHEM_METHOD = CAM-B3LYP
QCHEM_BASIS = 6-31G*
QCHEM_N_ROOTS = 5
QCHEM_ARRAY_CHUNK_SIZE = 1
QCHEM_MEM_TOTAL_MB = 125000

NEXMD_N_CLASS_STEPS_TEST = 10000
FRAME_START_TEST = 1
FRAME_STOP_TEST = 101
FRAME_STEP = 1
FRAME_WORKERS = 8
DATASET_CHOP_START_TEST = 1
DATASET_CHOP_END_TEST = 102
DATASET_CHOP_STEP = 10000
DATASET_WORKERS = 16
```

Submit with the Python executable available on Chicoma. For example:

```bash
/usr/projects/ml4chem/akhanna2/conda_envs/hiphop_env/bin/python dataset_workflow.py \
  --config configs/qchem_chicoma_101.inp
```

If Chicoma uses a different workflow/helper Python, set `PYTHON_BIN` and `QCHEM_PYTHON_BIN` in the config or pass `--python-bin` and `--qchem-python-bin` on the command line.

## Chicoma: 50K-Frame Production Dataset

Use the Chicoma test config as the base and change the production controls:

```text
MODE = Production
RUN_QCHEM_GSOPT = true
CONTINUE = true
SUBMIT_JOBS = true
MONITOR_JOBS = true

NEXMD_N_CLASS_STEPS_PRODUCTION = 5000000
FRAME_START_PRODUCTION = 5000
FRAME_STOP_PRODUCTION = 54999
FRAME_STEP = 1
FRAME_WORKERS = 16
DATASET_CHOP_START_PRODUCTION = 1
DATASET_CHOP_END_PRODUCTION = 50001
DATASET_CHOP_STEP = 10000
DATASET_WORKERS = 16

QCHEM_PROFILE = chicoma
QCHEM_PARTITION = standard
QCHEM_ACCOUNT = s17_cint
QCHEM_CPUS = 32
QCHEM_MEM = 128G
QCHEM_WALLTIME = 16:00:00
QCHEM_N_ROOTS = 5
QCHEM_ARRAY_CHUNK_SIZE = 1
```

Submit:

```bash
/usr/projects/ml4chem/akhanna2/conda_envs/hiphop_env/bin/python dataset_workflow.py \
  --config configs/qchem_chicoma_50k.inp
```

## Verification

After a run completes, verify the Q-Chem canonical arrays:

```bash
python - <<'PY'
from pathlib import Path
import numpy as np

mol = Path('/path/to/molecule')
props = mol / 'qchem' / 'batch_exsp'
for name in [
    'energies_all_states.npy',
    'forces_all_states.npy',
    'transition_dipoles.npy',
    'nacrs_all_states.npy',
]:
    arr = np.load(props / name)
    print(name, arr.shape, np.isfinite(arr).all())
PY
```

For the final dataset, inspect:

```text
<mol_dir>/dataset/<chunk>/<prefix>_S.npy
<mol_dir>/dataset/<chunk>/<prefix>_F.npy
<mol_dir>/dataset/<chunk>/<prefix>_NACR.npy
<mol_dir>/dataset/<prefix>_dataset.log
```

A successful 101-frame Q-Chem test should report `101` samples and Q-Chem state dimensions based on 6 total states and 10 NAC pairs.

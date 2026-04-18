# Please write a configuration input file for a Python script named [Script Name]

## The file should follow these formatting rules

1. Syntax: Use a KEY = VALUE format.
2. Comments: Use # for comments. Include a header explaining that lines starting with # are ignored and that keys are case-insensitive.
3. Styling: Group related parameters together with descriptive comments above each section.
4. Content: Include the following parameters with realistic example values:
    a) Paths: Define a DATA DIR, an OUTPUT DIR, and a REFERENCE FILE path.
    b) Identifiers: A PREFIX for naming files.
    c) Logic Flags: A boolean USE BUILTIN REFERENCE (commented out by default).
    d) Indices: Integer values for ANCHOR STATE and REFERENCE FRAME.
    e) Lists: Comma-separated atom indices for AXIS ATOMS and CORE ATOMS.
    f) Thresholds: Floating point values for PENALTY THRESHOLD and NOISE THRESHOLD.
    g) Custom Pairs: A string list for CORRECTION PAIRS (e.g., 12, 23).

Ensure the tone of the comments is professional and technical, similar to documentation for a scientific computing or chemistry-based trajectory analysis script.

Example:

```ini
# Unified PySEQM MD input file
# Lines starting with # are ignored.
# Keys are case-insensitive and must use KEY = VALUE syntax.

# ---------------------------------------------------------------------------
# Path controls
# ---------------------------------------------------------------------------
# DATA_DIR should point to the directory containing the NumPy coordinate and
# species arrays. OUTPUT_DIR is the destination for trajectory data, resource
# reports, and optional GPU monitoring logs.
DATA_DIR = ./data
OUTPUT_DIR = ./output

# PREFIX defines the output filename stem written into each config_<idx>
# directory. COORDS and SPECIES are provided explicitly here because the sample
# data files do not follow the default PREFIX_R.npy / PREFIX_Z.npy convention.
PREFIX = enol
COORDS = ./coords_R.npy
SPECIES = ./coords_Z.npy

# USE_BUILTIN_REFERENCE is a template-only placeholder in v1.
# USE_BUILTIN_REFERENCE = FALSE

# ---------------------------------------------------------------------------
# Configuration selection and hardware
# ---------------------------------------------------------------------------
START_CONFIG = 0
END_CONFIG = 1
BATCH = 50
GPUS = 1
RESTART = FALSE

# ---------------------------------------------------------------------------
# Electronic-structure controls
# ---------------------------------------------------------------------------
METHOD = AM1
SCF_EPS = 1.0e-8
SCF_CONVERGER = 1
ACTIVE_STATE = 0
N_STATES = 0

# ---------------------------------------------------------------------------
# Molecular-dynamics controls
# ---------------------------------------------------------------------------
# THERM_FRICTION follows the NEXMD-style Langevin input convention in 1/ps and
# is converted internally to the PySEQM damping time in fs.
THERM_TYPE = 1
TEMP = 300.0
TIMESTEP = 0.1
STEPS = 5000000
THERM_FRICTION = 20.0

# ---------------------------------------------------------------------------
# Output cadence
# ---------------------------------------------------------------------------
PRINT_EVERY = 100
OUT_DATA_STEPS = 100
OUT_COORDS_STEPS = 100
CHECKPOINT_EVERY = 1000

# ---------------------------------------------------------------------------
# Monitoring and reporting
# ---------------------------------------------------------------------------
ENABLE_GPU_MONITORING = FALSE
GPU_MONITOR_INTERVAL = 30
```

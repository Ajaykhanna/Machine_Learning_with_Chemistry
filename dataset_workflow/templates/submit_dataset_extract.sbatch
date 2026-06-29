#!/bin/bash
#SBATCH --job-name=dataset_extract
#SBATCH --output=dataset_extract_%j.out
#SBATCH --error=dataset_extract_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --partition=general
#SBATCH --qos=long

# =============================================================================
# Molecule-Agnostic Dataset Generation - SLURM Batch Script
# =============================================================================
#
# This template runs the final dataset generation stage from PySEQM output arrays.
# It is intentionally molecule-agnostic. Set PREFIX to control filenames such as
# molecule_R.npy, molecule_Z.npy, and molecule_sE.npy.
#
# Typical usage:
#   PREFIX=molecule INPUT_DIR=/path/to/pyseqm/batch_exsp OUTPUT_DIR=/path/to/dataset \
#   COORDS_R=/path/to/molecule_R.npy COORDS_Z=/path/to/molecule_Z.npy \
#   sbatch submit_dataset_extract.sbatch
#
# Optional reference files:
#   REFERENCE_ENERGY=/path/to/molecule_optimized_reference_energy.txt
#   REFERENCE_FORCES=/path/to/molecule_optimized_reference_forces.npy
#
# =============================================================================

set -euo pipefail

echo "=============================================="
echo "Molecule-Agnostic Dataset Generation"
echo "=============================================="
echo "Job ID:        ${SLURM_JOB_ID}"
echo "Job Name:      ${SLURM_JOB_NAME}"
echo "Node:          ${SLURM_NODELIST}"
echo "CPUs:          ${SLURM_CPUS_PER_TASK}"
echo "Partition:     ${SLURM_JOB_PARTITION}"
echo "Submit Dir:    ${SLURM_SUBMIT_DIR}"
echo ""

# Runtime environment. Override PYTHON_BIN when running outside Darwin defaults.
PYTHON_BIN="${PYTHON_BIN:-/usr/projects/ml4chem/akhanna2/softwares/envs/ml_env/bin/python}"
SCRIPT_DIR="${SCRIPT_DIR:-${SLURM_SUBMIT_DIR}}"
SCRIPT="${SCRIPT:-${SCRIPT_DIR}/05_generate_dataset.py}"

PREFIX="${PREFIX:-molecule}"
INPUT_DIR="${INPUT_DIR:-${SLURM_SUBMIT_DIR}}"
OUTPUT_DIR="${OUTPUT_DIR:-${SLURM_SUBMIT_DIR}/dataset}"
N_STATES="${N_STATES:-10}"
N_WORKERS="${N_WORKERS:-${SLURM_CPUS_PER_TASK}}"
CHOP_START="${CHOP_START:-10000}"
CHOP_END="${CHOP_END:-50001}"
CHOP_STEP="${CHOP_STEP:-10000}"
COORDS_R="${COORDS_R:-${PREFIX}_R.npy}"
COORDS_Z="${COORDS_Z:-${PREFIX}_Z.npy}"
SORTED_Z="${SORTED_Z:-}"
REFERENCE_ENERGY="${REFERENCE_ENERGY:-}"
REFERENCE_FORCES="${REFERENCE_FORCES:-}"
LOG_FILE="${LOG_FILE:-${OUTPUT_DIR}/${PREFIX}_dataset_${SLURM_JOB_ID}.log}"

echo "Configuration:"
echo "  Python:           ${PYTHON_BIN}"
echo "  Script:           ${SCRIPT}"
echo "  Prefix:           ${PREFIX}"
echo "  Input Dir:        ${INPUT_DIR}"
echo "  Output Dir:       ${OUTPUT_DIR}"
echo "  Excited States:   ${N_STATES}"
echo "  Workers:          ${N_WORKERS}"
echo "  Chop Start:       ${CHOP_START}"
echo "  Chop End:         ${CHOP_END}"
echo "  Chop Step:        ${CHOP_STEP}"
echo "  Coords R:         ${COORDS_R}"
echo "  Coords Z:         ${COORDS_Z}"
echo "  Sorted Z:         ${SORTED_Z:-disabled}"
echo "  Reference Energy: ${REFERENCE_ENERGY:-not provided}"
echo "  Reference Forces: ${REFERENCE_FORCES:-not provided}"
echo "  Log File:         ${LOG_FILE}"
echo ""

if [ ! -f "$SCRIPT" ]; then
    echo "ERROR: Dataset script not found: $SCRIPT"
    exit 1
fi

if [ ! -d "$INPUT_DIR" ]; then
    echo "ERROR: Input directory not found: $INPUT_DIR"
    exit 1
fi

REQUIRED_FILES=(
    "energies_all_states.npy"
    "total_energy.npy"
    "transition_dipoles.npy"
    "gs_dipoles.npy"
    "forces_all_states.npy"
    "nacrs_all_states.npy"
)

for f in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "${INPUT_DIR}/${f}" ]; then
        echo "ERROR: Missing required PySEQM output: ${INPUT_DIR}/${f}"
        exit 1
    fi
done

if [ ! -f "$COORDS_R" ]; then
    echo "ERROR: Coordinate R file not found: $COORDS_R"
    exit 1
fi

if [ ! -f "$COORDS_Z" ]; then
    echo "ERROR: Coordinate Z file not found: $COORDS_Z"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

CMD_ARGS=(
    --input-dir "$INPUT_DIR"
    --output-dir "$OUTPUT_DIR"
    --n-states "$N_STATES"
    --n-workers "$N_WORKERS"
    --log-file "$LOG_FILE"
    --chop-start "$CHOP_START"
    --chop-end "$CHOP_END"
    --chop-step "$CHOP_STEP"
    --coords-r "$COORDS_R"
    --coords-z "$COORDS_Z"
    --prefix "$PREFIX"
    --split-nacr
    --log-level INFO
)

if [ -n "$SORTED_Z" ]; then
    CMD_ARGS+=(--sorted-z "$SORTED_Z")
fi

if [ -n "$REFERENCE_ENERGY" ]; then
    CMD_ARGS+=(--reference-energy "$REFERENCE_ENERGY")
fi

if [ -n "$REFERENCE_FORCES" ]; then
    CMD_ARGS+=(--reference-forces "$REFERENCE_FORCES")
fi

echo "Starting dataset generation: $(date)"
"$PYTHON_BIN" "$SCRIPT" "${CMD_ARGS[@]}"
echo "Finished dataset generation: $(date)"

echo "Output directory: ${OUTPUT_DIR}"
du -sh "$OUTPUT_DIR" || true


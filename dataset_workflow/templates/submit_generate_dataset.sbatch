#!/bin/bash
#SBATCH --job-name=dataset_generation
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --partition=general
#SBATCH --qos=long

# =============================================================================
# Molecule Dataset Generation - SLURM Batch Script
# =============================================================================
#
# This script runs the molecule-prefixed dataset generation pipeline on a SLURM cluster.
#
# Usage:
#   sbatch submit_generate_dataset.sbatch
#
# Or with custom parameters:
#   sbatch --cpus-per-task=32 --mem=128G submit_generate_dataset.sbatch
#
# Environment variables (optional):
#   INPUT_DIR  - Input directory (default: current directory)
#   OUTPUT_DIR - Output directory (default: current directory)
#   N_STATES   - Number of excited states (default: 10)
#   N_WORKERS  - Number of parallel workers (default: $SLURM_CPUS_PER_TASK)
#   COORDS_R   - Atomic coordinates file (default: ${PREFIX}_R.npy)
#   COORDS_Z   - Atomic numbers file (default: ${PREFIX}_Z.npy)
#   SORTED_Z   - Sorted Z indices file for atomic reordering (optional)
#   REF_ENERGY - Optimized reference energy text file (optional)
#   REF_FORCES - Optimized reference forces .npy file (optional)
#
# =============================================================================

echo "=============================================="
echo "Molecule Dataset Generation Pipeline"
echo "=============================================="
echo ""
echo "Job Information:"
echo "  Job ID:        ${SLURM_JOB_ID}"
echo "  Job Name:      ${SLURM_JOB_NAME}"
echo "  Node:          ${SLURM_NODELIST}"
echo "  CPUs:          ${SLURM_CPUS_PER_TASK}"
echo "  Memory:        ${SLURM_MEM_PER_NODE}MB"
echo "  Partition:     ${SLURM_JOB_PARTITION}"
echo "  Submit Dir:    ${SLURM_SUBMIT_DIR}"
echo ""

# =============================================================================
# Environment Setup
# =============================================================================

# Load required modules (adjust for your cluster)
# module purge
# module load python/3.10
# module load gcc/11.2.0

# Use explicit Python from the known-working ml_env. Avoid relying on shell
# activation because non-interactive SLURM shells may pick up old system conda.
PYTHON_BIN="${PYTHON_BIN:-/usr/projects/ml4chem/akhanna2/softwares/envs/ml_env/bin/python}"

if [ ! -x "$PYTHON_BIN" ]; then
    echo "ERROR: Python executable not found: $PYTHON_BIN"
    exit 1
fi

# Print Python info
echo ""
echo "Python Environment:"
echo "$PYTHON_BIN"
"$PYTHON_BIN" --version
echo ""

# =============================================================================
# Configuration
# =============================================================================

# Script location (adjust if needed)
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(dirname $(realpath $0))}"
SCRIPT="${SCRIPT_DIR}/05_generate_dataset.py"

# Input/Output directories (use environment variables or defaults)
INPUT_DIR="${INPUT_DIR:-${SLURM_SUBMIT_DIR}}"
OUTPUT_DIR="${OUTPUT_DIR:-${SLURM_SUBMIT_DIR}/processed_data}"

# Number of excited states (default: 10 for full 11-state processing)
N_STATES="${N_STATES:-10}"

# Number of workers (default: use all allocated CPUs)
N_WORKERS="${N_WORKERS:-${SLURM_CPUS_PER_TASK}}"

# Chop Start, End, Step Frames (optional)
CHOP_START="${CHOP_START:-1000}"
CHOP_END="${CHOP_END:-5000}"
CHOP_STEP="${CHOP_STEP:-1000}"

# Coordinate input files (configurable)
PREFIX="${PREFIX:-molecule}"
COORDS_R="${COORDS_R:-${PREFIX}_R.npy}"
COORDS_Z="${COORDS_Z:-${PREFIX}_Z.npy}"

# Optional sorted Z indices file for atomic reordering
# Leave empty or unset to disable atomic reordering
SORTED_Z="${SORTED_Z:-}"

# Molecule-specific optimized reference files for shifted energies/forces
REF_DIR="${REF_DIR:-$(dirname "$(dirname "$INPUT_DIR")")/gsopt_reference}"
REF_ENERGY="${REF_ENERGY:-${REF_DIR}/${PREFIX}_optimized_reference_energy.txt}"
REF_FORCES="${REF_FORCES:-${REF_DIR}/${PREFIX}_optimized_reference_forces.npy}"

# Log file
LOG_FILE="${OUTPUT_DIR}/${PREFIX}_dataset_${SLURM_JOB_ID}.log"

echo "Configuration:"
echo "  Script:        ${SCRIPT}"
echo "  Input Dir:     ${INPUT_DIR}"
echo "  Output Dir:    ${OUTPUT_DIR}"
echo "  N States:      ${N_STATES}"
echo "  N Workers:     ${N_WORKERS}"
echo "  Chop Start:    ${CHOP_START:-default}"
echo "  Chop End:      ${CHOP_END:-default}"
echo "  Chop Step:     ${CHOP_STEP:-default}"
echo "  Coords R:      ${COORDS_R}"
echo "  Coords Z:      ${COORDS_Z}"
echo "  Sorted Z:      ${SORTED_Z:-disabled}"
echo "  Prefix:        ${PREFIX}"
echo "  Ref Energy:    ${REF_ENERGY}"
echo "  Ref Forces:    ${REF_FORCES}"
echo "  Log File:      ${LOG_FILE}"
echo ""

# =============================================================================
# Pre-flight Checks
# =============================================================================

# Check if script exists
if [ ! -f "$SCRIPT" ]; then
    echo "ERROR: Script not found: $SCRIPT"
    exit 1
fi

# Check if input directory exists
if [ ! -d "$INPUT_DIR" ]; then
    echo "ERROR: Input directory not found: $INPUT_DIR"
    exit 1
fi

# Check for required input files
REQUIRED_FILES=(
    "energies_all_states.npy"
    "total_energy.npy"
    "transition_dipoles.npy"
    "gs_dipoles.npy"
    "forces_all_states.npy"
    "nacrs_all_states.npy"
    "${COORDS_R}"
    "${COORDS_Z}"
    "${REF_ENERGY}"
    "${REF_FORCES}"
)

# Add sorted_Z file to required list if specified
if [ -n "$SORTED_Z" ]; then
    REQUIRED_FILES+=("${SORTED_Z}")
fi

echo "Checking input files..."
MISSING_FILES=0
for f in "${REQUIRED_FILES[@]}"; do
    if [ -f "${INPUT_DIR}/${f}" ]; then
        echo "  ??? Found: ${f}"
    else
        echo "  ??? Missing: ${f}"
        MISSING_FILES=$((MISSING_FILES + 1))
    fi
done

if [ $MISSING_FILES -gt 0 ]; then
    echo ""
    echo "ERROR: Missing ${MISSING_FILES} required input file(s)"
    exit 1
fi
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# =============================================================================
# Run Processing
# =============================================================================

echo "Starting data extraction..."
echo "Start time: $(date)"
echo ""

# Set OpenMP/MKL threads to avoid oversubscription
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Build command arguments
CMD_ARGS=(
    --input-dir "$INPUT_DIR"
    --output-dir "$OUTPUT_DIR"
    --n-states "$N_STATES"
    --n-workers "$N_WORKERS"
    --log-file "$LOG_FILE"
    --chop-start "${CHOP_START}"
    --chop-end "${CHOP_END}"
    --chop-step "${CHOP_STEP}"
    --coords-r "${COORDS_R}"
    --coords-z "${COORDS_Z}"
    --prefix "${PREFIX}"
    --reference-energy "${REF_ENERGY}"
    --reference-forces "${REF_FORCES}"
    --split-nacr
    --log-level INFO
)

# Add sorted-z argument if specified
if [ -n "$SORTED_Z" ]; then
    CMD_ARGS+=(--sorted-z "${SORTED_Z}")
fi

# Run the parallel processing script
"$PYTHON_BIN" "$SCRIPT" "${CMD_ARGS[@]}"

EXIT_CODE=$?

echo ""
echo "End time: $(date)"

# =============================================================================
# Post-processing
# =============================================================================

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=============================================="
    echo "Processing completed successfully!"
    echo "=============================================="
    echo ""
    echo "Output directory: ${OUTPUT_DIR}"
    echo ""
    
    # List output directories
    echo "Created chunk directories:"
    ls -la "$OUTPUT_DIR" | grep "^d"
    
    # Show disk usage
    echo ""
    echo "Disk usage:"
    du -sh "$OUTPUT_DIR"
    
else
    echo ""
    echo "=============================================="
    echo "ERROR: Processing failed with exit code $EXIT_CODE"
    echo "=============================================="
    echo ""
    echo "Check log file for details: ${LOG_FILE}"
    echo ""
    
    # Show last 50 lines of log
    if [ -f "$LOG_FILE" ]; then
        echo "Last 50 lines of log:"
        tail -50 "$LOG_FILE"
    fi
fi

exit $EXIT_CODE





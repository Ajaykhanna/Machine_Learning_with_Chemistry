#!/bin/bash
#===============================================================================
# SEQM Electronic Structure Calculation - SLURM Batch Script
#===============================================================================
#
# Description:
#   Multi-GPU SEQM calculation for electronic structure properties including
#   energies, forces, dipoles, transition dipoles, and non-adiabatic couplings.
#
# Computed Properties:
#   - All state energies (ground + excited) in eV
#   - All state forces/gradients in eV/Angstrom
#   - Ground state dipole moments in Debye
#   - Excited state dipole moments (net) in Debye
#   - Transition dipole moments (S0â†’Sn) in Debye
#   - Non-adiabatic coupling vectors (NACRs)
#
# Usage:
#   # Basic usage with defaults
#   sbatch submit_pyseqm_job.sbatch
#
#   # Override input files
#   sbatch --export=COORDS=path/to/coords.npy,SPECIES=path/to/species.npy submit_pyseqm_job.sbatch
#
#   # Override multiple parameters
#   sbatch --export=ALL,N_STATES=11,BATCH_SIZE=250,GPU_IDS="0 1" submit_pyseqm_job.sbatch
#
#   # Custom output directory
#   sbatch --export=ALL,OUTPUT_DIR=/path/to/output submit_pyseqm_job.sbatch
#
# Environment Variables:
#   COORDS      - Path to coordinates .npy file (N, A, 3)
#   SPECIES     - Path to species .npy file (N, A)
#   N_STATES    - Total electronic states (default: 11)
#   BATCH_SIZE  - Batch size per GPU (default: 500)
#   METHOD      - Semi-empirical method: AM1, PM3, MNDO (default: AM1)
#   SCF_EPS     - SCF convergence threshold (default: 1e-10)
#   CIS_TOL     - CIS convergence tolerance (default: 1e-8)
#   GPU_IDS     - Space-separated GPU IDs (default: "0 1 2 3")
#   OUTPUT_DIR  - Output directory (default: current directory)
#
# Output Files:
#   energies_all_states.npy  - (N, n_states) state energies
#   forces_all_states.npy    - (N, n_states, A, 3) gradients
#   gs_dipoles.npy           - (N, 3) ground state dipoles
#   ex_dipoles_net.npy       - (N, n_states-1, 3) excited state dipoles
#   transition_dipoles.npy   - (N, n_states-1, 3) transition dipoles
#   total_energy.npy         - (N, 1) SCF total energies
#   nacrs_all_states.npy     - (n_couplings, N, A, 3) NAC vectors
#
# Author: NEXMD/SEQM Workflow
# Version: 2.0.0
# Date: November 2025
#===============================================================================

#-------------------------------------------------------------------------------
# SLURM Configuration
#-------------------------------------------------------------------------------
#SBATCH --job-name=seqm_calc              # Job name
#SBATCH --output=seqm_%j.out              # Standard output log (%j = job ID)
#SBATCH --error=seqm_%j.err               # Standard error log
#SBATCH --ntasks=1                        # Number of tasks (1 main process)
#SBATCH --cpus-per-task=8                 # CPU cores for data loading/processing
#SBATCH --mem=64G                         # Memory request (adjust based on dataset)
#SBATCH --time=00-10:00:00                # Time limit (DD-HH:MM:SS)
#SBATCH --partition=shared-redstone     # Partition/queue name
#SBATCH --gres=gpu:4                      # Request 4 GPUs
#SBATCH --qos=long                        # Quality of service
#SBATCH --mail-type=END,FAIL              # Email notifications
#SBATCH --mail-user=                      # Email address (fill in if needed)

#-------------------------------------------------------------------------------
# Configuration Variables (can be overridden via --export)
#-------------------------------------------------------------------------------
# Input files (relative to script directory or absolute paths)
PREFIX="${PREFIX:-molecule}"
COORDS="${COORDS:-./${PREFIX}_R.npy}"
SPECIES="${SPECIES:-./${PREFIX}_Z.npy}"

# Calculation parameters
N_STATES="${N_STATES:-11}"                # Total states: 1 ground + (N-1) excited
BATCH_SIZE="${BATCH_SIZE:-500}"           # Molecules per batch (reduce if OOM)
METHOD="${METHOD:-AM1}"                   # AM1, PM3, or MNDO
SCF_EPS="${SCF_EPS:-1e-10}"               # SCF convergence threshold
CIS_TOL="${CIS_TOL:-1e-8}"                # CIS convergence tolerance

# GPU configuration (space-separated list matching SLURM --gres)
GPU_IDS="${GPU_IDS:-0 1 2 3}"

# Output directory (relative to script location or absolute)
OUTPUT_DIR="${OUTPUT_DIR:-.}"

# Python environment configuration
CONDA_ENV_NAME="ml_env"
PYTHON_ENV="/usr/projects/ml4chem/akhanna2/softwares/envs/${CONDA_ENV_NAME}"
PYTHON_BIN="${PYTHON_ENV}/bin/python"

# Script location - use absolute path (SLURM copies scripts to temp dirs)
SCRIPT_DIR="${SCRIPT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
SCRIPT_NAME="${SCRIPT_NAME:-04_run_pyseqm_properties.py}"

# Logging configuration (uses SCRIPT_DIR)
LOG_DIR="${SCRIPT_DIR}/logs"
TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"

#-------------------------------------------------------------------------------
# Setup and Validation
#-------------------------------------------------------------------------------
set -eo pipefail  # Exit on error, pipe failures (removed -u to allow bashrc sourcing)

# Create logs directory in the script location (not tmp)
mkdir -p "${LOG_DIR}"

# Function to print formatted messages
log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO:  $*"
}

log_warn() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] WARN:  $*" >&2
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $*" >&2
}

log_section() {
    echo ""
    echo "========================================================================"
    echo " $*"
    echo "========================================================================"
}

# Function to format bytes to human readable
format_bytes() {
    local bytes=$1
    if [[ ${bytes} -ge 1073741824 ]]; then
        echo "$(echo "scale=2; ${bytes}/1073741824" | bc) GB"
    elif [[ ${bytes} -ge 1048576 ]]; then
        echo "$(echo "scale=2; ${bytes}/1048576" | bc) MB"
    else
        echo "${bytes} bytes"
    fi
}

# Trap for cleanup on exit
cleanup() {
    local exit_code=$?
    log_info "Cleaning up..."
    
    # Kill GPU monitoring if running
    if [[ -n "${GPU_MONITOR_PID:-}" ]]; then
        kill "${GPU_MONITOR_PID}" 2>/dev/null || true
        wait "${GPU_MONITOR_PID}" 2>/dev/null || true
    fi
    
    # Deactivate conda if active
    conda deactivate 2>/dev/null || true
    
    if [[ ${exit_code} -eq 0 ]]; then
        log_info "Job completed successfully"
    else
        log_error "Job failed with exit code ${exit_code}"
        
        # Save diagnostic info on failure
        if [[ -n "${SLURM_JOB_ID:-}" ]]; then
            log_info "Saving diagnostic information..."
            {
                echo "=== Job Failure Diagnostic ==="
                echo "Timestamp: $(date)"
                echo "Exit Code: ${exit_code}"
                echo ""
                echo "=== Environment ==="
                env | sort
                echo ""
                echo "=== GPU State ==="
                nvidia-smi 2>/dev/null || echo "nvidia-smi not available"
                echo ""
                echo "=== Memory ==="
                free -h
                echo ""
                echo "=== Disk Space ==="
                df -h "${SCRIPT_DIR}"
            } > "${LOG_DIR}/diagnostic_${SLURM_JOB_ID}.txt" 2>&1
        fi
    fi
    
    exit ${exit_code}
}
trap cleanup EXIT

#-------------------------------------------------------------------------------
# Environment Activation
#-------------------------------------------------------------------------------
log_section "ENVIRONMENT SETUP"

# Initialize conda properly for non-interactive shell
log_info "Initializing conda..."
CONDA_BASE="/projects/opt/centos8/x86_64/miniconda3/py311_23.11.0"
if [[ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
else
    log_error "Conda initialization script not found: ${CONDA_BASE}/etc/profile.d/conda.sh"
    exit 1
fi

# Activate conda environment
log_info "Activating conda environment: ${CONDA_ENV_NAME}"
if ! conda activate "${CONDA_ENV_NAME}" 2>&1; then
    log_error "Failed to activate conda environment '${CONDA_ENV_NAME}'"
    log_error "Available environments:"
    conda env list
    exit 1
fi
log_info "Conda environment activated successfully"

# Verify Python executable
if [[ ! -x "${PYTHON_BIN}" ]]; then
    log_error "Python executable not found or not executable: ${PYTHON_BIN}"
    exit 1
fi
log_info "Python binary:    ${PYTHON_BIN}"
log_info "Python version:   $(${PYTHON_BIN} --version 2>&1)"

# Verify required packages
log_info "Verifying required packages..."
if ! ${PYTHON_BIN} -c "import torch; import numpy; from seqm.Molecule import Molecule" 2>/dev/null; then
    log_error "Required Python packages not found (torch, numpy, seqm)"
    exit 1
fi
log_info "PyTorch version:  $(${PYTHON_BIN} -c 'import torch; print(torch.__version__)')"
log_info "NumPy version:    $(${PYTHON_BIN} -c 'import numpy; print(numpy.__version__)')"
log_info "CUDA available:   $(${PYTHON_BIN} -c 'import torch; print(torch.cuda.is_available())')"

#-------------------------------------------------------------------------------
# Job Information
#-------------------------------------------------------------------------------
log_section "JOB INFORMATION"

log_info "SLURM Job ID:     ${SLURM_JOB_ID:-N/A}"
log_info "SLURM Job Name:   ${SLURM_JOB_NAME:-N/A}"
log_info "SLURM Node:       ${SLURM_NODELIST:-N/A}"
log_info "SLURM Partition:  ${SLURM_JOB_PARTITION:-N/A}"
log_info "SLURM CPUs:       ${SLURM_CPUS_PER_TASK:-1}"
log_info "SLURM Memory:     ${SLURM_MEM_PER_NODE:-unknown}"
log_info "SLURM GPUs:       ${SLURM_GPUS:-${SLURM_JOB_GPUS:-unknown}}"
log_info "Working Dir:      ${SCRIPT_DIR}"
log_info "Hostname:         $(hostname)"
log_info "Start Time:       $(date '+%Y-%m-%d %H:%M:%S %Z')"

# System information
log_info "Kernel:           $(uname -r)"
log_info "Available RAM:    $(free -h | awk '/^Mem:/{print $2}')"
log_info "Available Disk:   $(df -h "${SCRIPT_DIR}" | awk 'NR==2{print $4}')"

#-------------------------------------------------------------------------------
# Input Validation
#-------------------------------------------------------------------------------
log_section "INPUT VALIDATION"

# Change to script directory
cd "${SCRIPT_DIR}"

# Validate input files exist and are readable
if [[ ! -f "${COORDS}" ]]; then
    log_error "Coordinates file not found: ${COORDS}"
    exit 1
fi
if [[ ! -r "${COORDS}" ]]; then
    log_error "Coordinates file not readable: ${COORDS}"
    exit 1
fi
COORDS_SIZE=$(du -h "${COORDS}" | cut -f1)
COORDS_ABS=$(realpath "${COORDS}")
log_info "Coordinates:      ${COORDS_ABS} (${COORDS_SIZE})"

if [[ ! -f "${SPECIES}" ]]; then
    log_error "Species file not found: ${SPECIES}"
    exit 1
fi
if [[ ! -r "${SPECIES}" ]]; then
    log_error "Species file not readable: ${SPECIES}"
    exit 1
fi
SPECIES_SIZE=$(du -h "${SPECIES}" | cut -f1)
SPECIES_ABS=$(realpath "${SPECIES}")
log_info "Species:          ${SPECIES_ABS} (${SPECIES_SIZE})"

# Get input array dimensions using Python
log_info "Reading input dimensions..."
INPUT_INFO=$(${PYTHON_BIN} -c "
import numpy as np
coords = np.load('${COORDS}', mmap_mode='r')
species = np.load('${SPECIES}', mmap_mode='r')
n_configs, n_atoms, _ = coords.shape
print(f'{n_configs},{n_atoms}')
" 2>/dev/null) || {
    log_error "Failed to read input array dimensions"
    exit 1
}
N_CONFIGS=$(echo "${INPUT_INFO}" | cut -d',' -f1)
N_ATOMS=$(echo "${INPUT_INFO}" | cut -d',' -f2)
log_info "Configurations:   ${N_CONFIGS}"
log_info "Atoms/molecule:   ${N_ATOMS}"

# Validate script exists
if [[ ! -f "${SCRIPT_NAME}" ]]; then
    log_error "Python script not found: ${SCRIPT_NAME}"
    exit 1
fi
log_info "Script:           ${SCRIPT_DIR}/${SCRIPT_NAME}"

# Create output directory
mkdir -p "${OUTPUT_DIR}"
OUTPUT_DIR_ABS=$(realpath "${OUTPUT_DIR}")
log_info "Output Dir:       ${OUTPUT_DIR_ABS}"

# Check output directory is writable
if [[ ! -w "${OUTPUT_DIR}" ]]; then
    log_error "Output directory not writable: ${OUTPUT_DIR}"
    exit 1
fi

#-------------------------------------------------------------------------------
# GPU Information
#-------------------------------------------------------------------------------
log_section "GPU INFORMATION"

# Check CUDA availability
if ! command -v nvidia-smi &> /dev/null; then
    log_error "nvidia-smi not found - no GPU support detected"
    exit 1
fi

# Count available GPUs
GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
if [[ ${GPU_COUNT} -eq 0 ]]; then
    log_error "No GPUs detected on this node"
    exit 1
fi
log_info "Available GPUs:   ${GPU_COUNT}"

# Display detailed GPU info
log_info "GPU Details:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free,driver_version,compute_cap --format=csv,noheader | \
    while IFS=, read -r idx name mem_total mem_free driver compute; do
        log_info "  GPU ${idx}: ${name// /} | ${mem_total// /} total, ${mem_free// /} free | Driver ${driver// /} | CC ${compute// /}"
    done

# Verify requested GPUs are available
IFS=' ' read -ra GPU_ARRAY <<< "${GPU_IDS}"
for gpu_id in "${GPU_ARRAY[@]}"; do
    if [[ ${gpu_id} -ge ${GPU_COUNT} ]]; then
        log_error "Requested GPU ${gpu_id} not available (only ${GPU_COUNT} GPUs on node)"
        exit 1
    fi
done
log_info "GPUs to use:      ${GPU_IDS} (${#GPU_ARRAY[@]} GPUs)"

# Check CUDA toolkit
CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
log_info "CUDA Driver:      ${CUDA_VERSION}"

#-------------------------------------------------------------------------------
# Calculation Parameters
#-------------------------------------------------------------------------------
log_section "CALCULATION PARAMETERS"

# Compute derived parameters
N_EXCITED=$((N_STATES - 1))
N_COUPLINGS=$(( (N_STATES - 1) * (N_STATES - 2) / 2 ))  # NAC couplings between excited states

log_info "Method:           ${METHOD}"
log_info "N States:         ${N_STATES} (1 ground + ${N_EXCITED} excited)"
log_info "NAC Couplings:    ${N_COUPLINGS} (excited state pairs)"
log_info "Batch Size:       ${BATCH_SIZE}"
log_info "SCF Epsilon:      ${SCF_EPS}"
log_info "CIS Tolerance:    ${CIS_TOL}"

# Estimate memory requirements
ESTIMATED_MEM=$(${PYTHON_BIN} -c "
import sys
n_configs = ${N_CONFIGS}
n_atoms = ${N_ATOMS}
n_states = ${N_STATES}
n_couplings = (n_states - 1) * (n_states - 2) // 2

# Float64 = 8 bytes
bytes_per_float = 8

# Estimate sizes
energies = n_configs * n_states * bytes_per_float
forces = n_configs * n_states * n_atoms * 3 * bytes_per_float
gs_dipoles = n_configs * 3 * bytes_per_float
ex_dipoles = n_configs * (n_states - 1) * 3 * bytes_per_float
trans_dipoles = n_configs * (n_states - 1) * 3 * bytes_per_float
nacrs = n_couplings * n_configs * n_atoms * 3 * bytes_per_float

total = energies + forces + gs_dipoles + ex_dipoles + trans_dipoles + nacrs
print(f'{total / 1e9:.2f}')
" 2>/dev/null)
log_info "Est. Output Size: ${ESTIMATED_MEM:-unknown} GB"

#-------------------------------------------------------------------------------
# GPU Monitoring
#-------------------------------------------------------------------------------
log_section "STARTING GPU MONITORING"

mkdir -p "${LOG_DIR}"
GPU_LOG_FILE="${LOG_DIR}/gpu_monitor_${SLURM_JOB_ID}.csv"
log_info "GPU log file:     ${GPU_LOG_FILE}"

# Start GPU monitoring in background (every 30 seconds for more granular data)
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.free,temperature.gpu,power.draw \
    --format=csv \
    -l 30 \
    > "${GPU_LOG_FILE}" 2>&1 &
GPU_MONITOR_PID=$!
log_info "GPU monitor PID:  ${GPU_MONITOR_PID}"

# Verify monitor started
sleep 1
if ! kill -0 "${GPU_MONITOR_PID}" 2>/dev/null; then
    log_warn "GPU monitoring failed to start - continuing without monitoring"
    GPU_MONITOR_PID=""
fi

#-------------------------------------------------------------------------------
# Execute Calculation
#-------------------------------------------------------------------------------
log_section "EXECUTING SEQM CALCULATION"

# Build command with all arguments
CMD="${PYTHON_BIN} -u ${SCRIPT_NAME}"  # -u for unbuffered output
CMD="${CMD} --coords ${COORDS}"
CMD="${CMD} --species ${SPECIES}"
CMD="${CMD} --n_states ${N_STATES}"
CMD="${CMD} --batch ${BATCH_SIZE}"
CMD="${CMD} --method ${METHOD}"
CMD="${CMD} --scf_eps ${SCF_EPS}"
CMD="${CMD} --cis_tol ${CIS_TOL}"
CMD="${CMD} --gpus ${GPU_IDS}"
CMD="${CMD} --out_e ${OUTPUT_DIR}/energies_all_states.npy"
CMD="${CMD} --out_f ${OUTPUT_DIR}/forces_all_states.npy"
CMD="${CMD} --out_gs_D ${OUTPUT_DIR}/gs_dipoles.npy"
CMD="${CMD} --out_ex_D ${OUTPUT_DIR}/ex_dipoles_net.npy"
CMD="${CMD} --out_td ${OUTPUT_DIR}/transition_dipoles.npy"
CMD="${CMD} --out_etot ${OUTPUT_DIR}/total_energy.npy"
CMD="${CMD} --out_nacr ${OUTPUT_DIR}/nacrs_all_states.npy"

log_info "Command:"
log_info "  ${CMD}"
echo ""

# Execute with timing
log_info "Starting calculation at $(date '+%Y-%m-%d %H:%M:%S')..."
echo ""
CALC_START=$(date +%s)

# Run with error handling
set +e  # Don't exit on error - we want to capture the exit code
${CMD}
CALC_EXIT_CODE=$?
set -e

CALC_END=$(date +%s)
CALC_DURATION=$((CALC_END - CALC_START))
CALC_HOURS=$((CALC_DURATION / 3600))
CALC_MINS=$(((CALC_DURATION % 3600) / 60))
CALC_SECS=$((CALC_DURATION % 60))

echo ""
log_info "Calculation finished at $(date '+%Y-%m-%d %H:%M:%S')"
log_info "Calculation time: ${CALC_HOURS}h ${CALC_MINS}m ${CALC_SECS}s (${CALC_DURATION} seconds)"

#-------------------------------------------------------------------------------
# Results Summary
#-------------------------------------------------------------------------------
log_section "RESULTS SUMMARY"

if [[ ${CALC_EXIT_CODE} -eq 0 ]]; then
    log_info "âœ“ Calculation completed successfully!"
    log_info "Duration: ${CALC_HOURS}h ${CALC_MINS}m ${CALC_SECS}s"
    
    echo ""
    log_info "Output files:"
    TOTAL_SIZE=0
    for f in "${OUTPUT_DIR}"/*.npy; do
        if [[ -f "$f" ]]; then
            SIZE_BYTES=$(stat --printf="%s" "$f")
            SIZE_MB=$(echo "scale=2; ${SIZE_BYTES} / 1048576" | bc)
            TOTAL_SIZE=$((TOTAL_SIZE + SIZE_BYTES))
            
            # Get array shape using Python
            SHAPE=$(${PYTHON_BIN} -c "import numpy as np; print(np.load('${f}', mmap_mode='r').shape)" 2>/dev/null)
            log_info "  $(basename "$f"): ${SIZE_MB} MB | Shape: ${SHAPE}"
        fi
    done
    TOTAL_SIZE_GB=$(echo "scale=2; ${TOTAL_SIZE} / 1073741824" | bc)
    log_info "Total output size: ${TOTAL_SIZE_GB} GB"
    
    # Verify output integrity
    echo ""
    log_info "Verifying output integrity..."
    VERIFY_OK=true
    for f in "${OUTPUT_DIR}"/*.npy; do
        if [[ -f "$f" ]]; then
            if ! ${PYTHON_BIN} -c "import numpy as np; np.load('${f}')" 2>/dev/null; then
                log_error "  FAILED: $(basename "$f") - file may be corrupted"
                VERIFY_OK=false
            fi
        fi
    done
    if [[ "${VERIFY_OK}" == "true" ]]; then
        log_info "âœ“ All output files verified successfully"
    fi
else
    log_error "âœ— Calculation failed with exit code: ${CALC_EXIT_CODE}"
    
    # Show last few lines of output for debugging
    echo ""
    log_error "Check the job output file for details: slurm-${SLURM_JOB_ID}.out"
    
    exit ${CALC_EXIT_CODE}
fi

#-------------------------------------------------------------------------------
# Cleanup and Final Status
#-------------------------------------------------------------------------------
log_section "JOB COMPLETED"

log_info "End Time:         $(date '+%Y-%m-%d %H:%M:%S %Z')"
log_info "Total Duration:   ${CALC_HOURS}h ${CALC_MINS}m ${CALC_SECS}s"
log_info "GPU Log:          ${GPU_LOG_FILE}"
log_info "Output Dir:       ${OUTPUT_DIR_ABS}"

# GPU usage summary from log file
if [[ -f "${GPU_LOG_FILE}" ]]; then
    echo ""
    log_info "GPU Usage Summary:"
    AVG_UTIL=$(tail -n +2 "${GPU_LOG_FILE}" | awk -F',' '{sum+=$4; n++} END {if(n>0) printf "%.1f", sum/n; else print "N/A"}')
    MAX_MEM=$(tail -n +2 "${GPU_LOG_FILE}" | awk -F',' '{gsub(/ MiB/, "", $6); if($6>max) max=$6} END {print max " MiB"}')
    MAX_TEMP=$(tail -n +2 "${GPU_LOG_FILE}" | awk -F',' '{if($8>max) max=$8} END {print max "Â°C"}')
    log_info "  Avg GPU Util:   ${AVG_UTIL}%"
    log_info "  Max Memory:     ${MAX_MEM}"
    log_info "  Max Temp:       ${MAX_TEMP}"
fi

# Stop GPU monitoring
if [[ -n "${GPU_MONITOR_PID:-}" ]]; then
    kill "${GPU_MONITOR_PID}" 2>/dev/null || true
fi

echo ""
log_info "=========================================="
log_info "  Job ${SLURM_JOB_ID} completed successfully"
log_info "=========================================="
echo ""

# Brief pause before exit
sleep 2


#!/bin/bash
#SBATCH --job-name=pyseqm_gsopt
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --partition=ml4chem
#SBATCH --qos=long
#SBATCH --nodelist=cn4075

set -euo pipefail

echo "=============================================="
echo "PySEQM Ground-State Reference Optimization"
echo "=============================================="
echo "Job ID:     ${SLURM_JOB_ID}"
echo "Node:       ${SLURM_NODELIST}"
echo "Submit Dir: ${SLURM_SUBMIT_DIR}"
echo "Start:      $(date)"
echo ""

PYTHON_BIN="${PYTHON_BIN:-/usr/projects/ml4chem/akhanna2/softwares/envs/ml_env/bin/python}"
SCRIPT="${SCRIPT:-${SLURM_SUBMIT_DIR}/pyseqm_optimized_geom.py}"
PREFIX="${PREFIX:-molecule}"
XYZ="${XYZ:-${SLURM_SUBMIT_DIR}/${PREFIX}/nexmd/coords.xyz}"
OUTPUT_DIR="${OUTPUT_DIR:-${SLURM_SUBMIT_DIR}/${PREFIX}/pyseqm/gsopt_reference}"
METHOD="${METHOD:-AM1}"
SCF_EPS="${SCF_EPS:-1e-10}"
MAX_ITER="${MAX_ITER:-1000}"
TOL="${TOL:-1e-6}"

if [ ! -x "$PYTHON_BIN" ]; then
    echo "ERROR: Python executable not found: $PYTHON_BIN"
    exit 1
fi
if [ ! -f "$SCRIPT" ]; then
    echo "ERROR: Script not found: $SCRIPT"
    exit 1
fi
if [ ! -f "$XYZ" ]; then
    echo "ERROR: XYZ file not found: $XYZ"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "Python:    $PYTHON_BIN"
"$PYTHON_BIN" --version
echo "Script:    $SCRIPT"
echo "XYZ:       $XYZ"
echo "Output:    $OUTPUT_DIR"
echo "Prefix:    $PREFIX"
echo "Method:    $METHOD"
echo ""

"$PYTHON_BIN" "$SCRIPT" \
    --xyz "$XYZ" \
    --output-dir "$OUTPUT_DIR" \
    --prefix "$PREFIX" \
    --method "$METHOD" \
    --scf-eps "$SCF_EPS" \
    --max-iter "$MAX_ITER" \
    --tol "$TOL" \
    --sort-z

echo ""
echo "End: $(date)"
echo "PySEQM reference optimization completed."



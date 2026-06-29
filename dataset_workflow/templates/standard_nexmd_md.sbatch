#!/bin/bash
#SBATCH --job-name=nexmd_ground_state
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --partition=general
#SBATCH --qos=long

set -euo pipefail

PREFIX="${PREFIX:-molecule}"
NEXMD_ROOT="${NEXMD_ROOT:-}"
NEXMD_LIBRARY_PATH="${NEXMD_LIBRARY_PATH:-}"
NEXMD_EXE="${NEXMD_EXE:-/usr/projects/ml4chem/akhanna2/softwares/NEXMD/nexmd.exe}"
NEXMD_OUTPUT="${NEXMD_OUTPUT:-${PREFIX}_nexmd.out}"

if [ -n "$NEXMD_ROOT" ]; then
    export PATH="${NEXMD_ROOT}:${NEXMD_ROOT}/bin:${PATH}"
fi

if [ -n "$NEXMD_LIBRARY_PATH" ]; then
    export LD_LIBRARY_PATH="${NEXMD_LIBRARY_PATH}:${LD_LIBRARY_PATH:-}"
fi

"$NEXMD_EXE" < input.ceon > "$NEXMD_OUTPUT"

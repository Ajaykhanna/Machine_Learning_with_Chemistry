#!/bin/bash
#SBATCH --job-name=gaussian_gsopt
#SBATCH --error=%x_%j.err
#SBATCH --output=%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100GB
#SBATCH --time=05:00:00
#SBATCH --partition=general
#SBATCH --qos=long

set -euo pipefail

PREFIX="${PREFIX:-molecule}"
MEMORY="${MEMORY:-100GB}"
N_CPUS="${N_CPUS:-16}"

export GAUSS_SCRDIR="${GAUSS_SCRDIR:-${TMPDIR:-/tmp}/${USER:-user}_${SLURM_JOB_ID:-manual}}"
mkdir -p "$GAUSS_SCRDIR"

echo "Job started at $(date)"
echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Working directory: ${PWD}"
echo "Scratch directory: ${GAUSS_SCRDIR}"
echo "Prefix: ${PREFIX}"

g16 -m="$MEMORY" -p="$N_CPUS" < "${PREFIX}_gsopt.com" > "${PREFIX}_gsopt.log"

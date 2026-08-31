#!/bin/bash
#SBATCH --job-name=mml_v2_eval
#SBATCH --output=logs/mml_v2_eval_%j.stdout
#SBATCH --error=logs/mml_v2_eval_%j.stderr
#SBATCH -N 1
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpuq
#SBATCH --account=hpcusers
#SBATCH --mem=32G

set -e

PROJECT_DIR="${PROJECT_DIR:-$PWD}"

module add CUDA
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate env_gm

cd "${PROJECT_DIR}"
mkdir -p logs

echo "Starting on: $(hostname)"
echo "Started: $(date)"

python -u evaluate_mml_v2_per_disease.py

echo
echo "Finished: $(date)"

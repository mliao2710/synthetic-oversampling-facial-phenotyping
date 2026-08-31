#!/bin/bash
#SBATCH --job-name=mml_v2_base
#SBATCH --output=logs/mml_v2_base_%j.stdout
#SBATCH --error=logs/mml_v2_base_%j.stderr
#SBATCH -N 1
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks-per-node=1
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpuq
#SBATCH --account=hpcusers
#SBATCH --mem=32G

set -e

PROJECT_DIR="${PROJECT_DIR:-$PWD}"

echo "Starting job on: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "User: $(whoami)"
echo "Start time: $(date)"

module add CUDA

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate env_gm

cd "${PROJECT_DIR}"
mkdir -p logs

echo
echo "=== GPU CHECK ==="

nvidia-smi

python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA devices:", torch.cuda.device_count())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA unavailable.")

print("GPU:", torch.cuda.get_device_name(0))
PY

echo
echo "=== GESTALTMML V2 BASELINE ==="

python -u train_gestaltmml_10d.py \
  --dataset v1.1.0_10d_70_30 \
  --epochs 10 \
  --batch-size 64 \
  --eval-batch-size 64 \
  --learning-rate 5e-5 \
  --seed 11 \
  --max-length 40 \
  --real-augmentation matched_star \
  --synthetic-text star

echo
echo "Finished at: $(date)"

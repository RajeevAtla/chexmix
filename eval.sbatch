#!/usr/bin/env bash
#SBATCH --job-name=jax_chess_eval
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --output=runs/slurm_%j.out
#SBATCH --error=runs/slurm_%j.err

set -euo pipefail

# Optional: cluster-specific module loads here
# module load cuda/13.x

export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"
export JAX_PLATFORM_NAME="gpu"
export PYTHONPATH="${PWD}/src"

uv sync --group cuda --group dev

uv run python -m cli eval --config config/default.toml


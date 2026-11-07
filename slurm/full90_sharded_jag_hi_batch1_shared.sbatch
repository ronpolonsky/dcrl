#!/bin/bash
#SBATCH --account=bgqe-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --array=9-18%10
#SBATCH --gres=gpu:nvidia_a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/work/hdd/bgqe/rpolonsky/dcrl/slurm/full90-sc-%A_%a.out
#SBATCH --error=/work/hdd/bgqe/rpolonsky/dcrl/slurm/full90-sc-%A_%a.err
set -euo pipefail
export TASK_INDEX="$SLURM_ARRAY_TASK_ID"
exec /work/hdd/bgqe/rpolonsky/dcrl/slurm/full90_one_iris_generic_shared.sbatch

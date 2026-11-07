#!/usr/bin/env bash
# Submit one-node, two-GPU isolated EXPO run.
# The trainer launcher itself lives beside this file under sharder/training.
#SBATCH --job-name=expo100k-final
#SBATCH --partition=gpuA100x4
#SBATCH --account=bgqe-delta-gpu
#SBATCH --qos=bgqe-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=180G
#SBATCH --time=2-00:00:00
#SBATCH --output=/u/rpolonsky/dcrlcheckpoints/expo_rl_task2_batch4_utd20_final100k_2gpu/slurm-%j.out
#SBATCH --open-mode=append

set -euo pipefail

PROJECT=/projects/bgqe/rpolonsky/dcrl/sharder/training
LAUNCHER="$PROJECT/scripts/train_expo_rl_100k_task2_finalonly_2gpu.sh"
OUTPUT_ROOT=/u/rpolonsky/dcrlcheckpoints/expo_rl_task2_batch4_utd20_final100k_2gpu
RUN_NAME=official_pi05_task2_batch4_utd20_final100k_2gpu
COORDINATOR="$(hostname -s):29517"

mkdir -p "$OUTPUT_ROOT/logs"
echo "job=$SLURM_JOB_ID host=$(hostname) coordinator=$COORDINATOR"

srun --exclusive --ntasks=1 --gpus-per-task=1 \
  env JAX_PROCESS_ID=0 JAX_NUM_PROCESSES=2 \
      JAX_COORDINATOR_ADDRESS="$COORDINATOR" \
      OUTPUT_ROOT="$OUTPUT_ROOT" RUN_NAME="$RUN_NAME" \
      "$LAUNCHER" >"$OUTPUT_ROOT/logs/proc0.log" 2>&1 &
p0=$!

srun --exclusive --ntasks=1 --gpus-per-task=1 \
  env JAX_PROCESS_ID=1 JAX_NUM_PROCESSES=2 \
      JAX_COORDINATOR_ADDRESS="$COORDINATOR" \
      OUTPUT_ROOT="$OUTPUT_ROOT" RUN_NAME="$RUN_NAME" \
      "$LAUNCHER" >"$OUTPUT_ROOT/logs/proc1.log" 2>&1 &
p1=$!

status=0
wait "$p0" || status=$?
wait "$p1" || status=$?
exit "$status"

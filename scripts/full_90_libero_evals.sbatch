#!/bin/bash
# Evaluate one checkpoint across LIBERO-90 using seven concurrent GPU shards.
#
# Example (full Pi EXPO checkpoint):
#   sbatch --export=ALL,MODE=full,CHECKPOINT=/path/to/checkpoints,OUTPUT_DIR=/path/to/eval \
#     scripts/full_90_libero_evals.sbatch
#
# Example (Pi0.5 SFT checkpoint):
#   sbatch --export=ALL,MODE=actor,CHECKPOINT=/path/to/sft_run,OUTPUT_DIR=/path/to/eval \
#     scripts/full_90_libero_evals.sbatch
#
# Each array element evaluates 13 (or 12) tasks sequentially while reusing the
# loaded model. The seven array jobs run concurrently when seven GPUs are
# available; Slurm queues the remainder if fewer are available.
#SBATCH --account=nlp
#SBATCH --partition=sphinx
#SBATCH --job-name=libero90-eval
#SBATCH --array=0-6%7
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --exclude=sphinx[1-3,8]
#SBATCH --output=/nlp/scr/ronpo/projects/dcrl/slurm/%x-%A_%a.out
#SBATCH --error=/nlp/scr/ronpo/projects/dcrl/slurm/%x-%A_%a.err

set -euo pipefail

: "${MODE:?Set MODE=actor for SFT, compact for SmallExpo, or full for a Pi EXPO checkpoint}"
: "${CHECKPOINT:?Set CHECKPOINT to the SFT, compact, or full-agent checkpoint path}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR for the per-task JSON results}"

PROJECT=/nlp/scr2/ronpo/projects/dcrl
EXPO="$PROJECT/third_party/expo-ft"
NUM_SHARDS=7
CACHE_ROOT="$OUTPUT_DIR/cache"
RUNTIME_ROOT="$OUTPUT_DIR/runtime"

export LIBERO_CONFIG_PATH="$PROJECT/.deps/libero_config"
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="$PROJECT:$EXPO/expo_ft/agents/vla/openpi/src:$EXPO/expo_ft/agents/vla/openpi/packages/openpi-client/src:$PROJECT/.deps/LIBERO${PYTHONPATH:+:$PYTHONPATH}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/nlp/scr2/ronpo/dcrlcheckpoints/openpi_cache}"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"
export XDG_CONFIG_HOME="$CACHE_ROOT/xdg_config"
export TMPDIR="$RUNTIME_ROOT/tmp"

mkdir -p "$OUTPUT_DIR" "$CACHE_ROOT/xdg" "$CACHE_ROOT/xdg_config" "$RUNTIME_ROOT/tmp" "$PROJECT/slurm"
cd "$PROJECT"

exec "$EXPO/.venv/bin/python" "$PROJECT/full_90_libero_evals.py" \
  --mode "$MODE" \
  --checkpoint "$CHECKPOINT" \
  --suite libero_90 \
  --shard-index "$SLURM_ARRAY_TASK_ID" \
  --num-shards "$NUM_SHARDS" \
  --episodes "${EPISODES:-20}" \
  --eval-seed "${EVAL_SEED:-7}" \
  --seed "${SEED:-42}" \
  --image-size 224 \
  --action-horizon 10 \
  --replan-steps 5 \
  --output-dir "$OUTPUT_DIR"

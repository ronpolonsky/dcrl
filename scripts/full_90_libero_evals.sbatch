#!/bin/bash
# Evaluate one checkpoint across LIBERO-90 using five concurrent GPU shards.
#
# Example (full Pi EXPO checkpoint):
#   sbatch --export=ALL,MODE=direct_rl_pi,CHECKPOINT=/path/to/checkpoints,OUTPUT_DIR=/path/to/eval \
#     scripts/full_90_libero_evals.sbatch
#
# Example (MyEXPOLearner checkpoint from train_pi_sft_agnostic.py, actor only):
#   sbatch --export=ALL,MODE=sft_pi,CHECKPOINT=/path/to/sft_run,OUTPUT_DIR=/path/to/eval \
#     scripts/full_90_libero_evals.sbatch
#
# Example (SFT-trained Pi0.5 actor plus Small RL EXPO components):
#   sbatch --export=ALL,MODE=sft_plus_editor,CHECKPOINT=/path/to/sft_run,EDITOR_CHECKPOINT=/path/to/small_rl/task_28,OUTPUT_DIR=/path/to/eval \
#     scripts/full_90_libero_evals.sbatch
#
# Example (official Pi0.5 LIBERO actor; checkpoint is staged locally):
#   sbatch --export=ALL,MODE=official_pi05,CHECKPOINT=/projects/bgqe/rpolonsky/dcrlcheckpoints/openpi_cache/openpi-assets/checkpoints/pi05_libero,OUTPUT_DIR=/projects/bgqe/rpolonsky/dcrl/results/official_pi0.5_libero_chkpt \
#     scripts/full_90_libero_evals.sbatch
#
# Each array element evaluates 18 tasks sequentially while reusing the loaded
# model. Five array jobs cover all 90 tasks concurrently.
#SBATCH --account=bgqe-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --job-name=libero90-eval
#SBATCH --array=0-4%5
#SBATCH --gres=gpu:nvidia_a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/projects/bgqe/rpolonsky/dcrl_slurm/%x-%A_%a.out
#SBATCH --error=/projects/bgqe/rpolonsky/dcrl_slurm/%x-%A_%a.err

set -euo pipefail

: "${MODE:?Set MODE=official_pi05, sft_pi, sft_plus_editor, compact, or direct_rl_pi}"
: "${CHECKPOINT:?Set CHECKPOINT to a local checkpoint path}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR for the per-task JSON results}"
if [[ "$MODE" == "sft_plus_editor" && -z "${EDITOR_CHECKPOINT:-}" ]]; then
  echo 'EDITOR_CHECKPOINT is required when MODE=sft_plus_editor' >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="${PROJECT_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
PROJECT_PARENT="$(cd -- "$PROJECT/.." && pwd)"
EXPO="$PROJECT/third_party/expo-ft"
LIBERO_ROOT="$PROJECT/.deps/LIBERO/libero/libero"
NUM_SHARDS=5
SHARD_INDEX="${SHARD_INDEX:-${SLURM_ARRAY_TASK_ID:-}}"
if [[ -z "$SHARD_INDEX" ]]; then
  echo 'SHARD_INDEX or SLURM_ARRAY_TASK_ID is required' >&2
  exit 2
fi
CACHE_ROOT="$OUTPUT_DIR/cache"
RUNTIME_ROOT="$OUTPUT_DIR/runtime"

export LIBERO_CONFIG_PATH="$PROJECT/.deps/libero_config"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-$MUJOCO_GL}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENPI_SKIP_INIT_JIT=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTHONPATH="$PROJECT:$EXPO/expo_ft/agents/vla/openpi/src:$EXPO/expo_ft/agents/vla/openpi/packages/openpi-client/src:$PROJECT/.deps/LIBERO${PYTHONPATH:+:$PYTHONPATH}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$PROJECT_PARENT/dcrlcheckpoints/openpi_cache}"
if [[ "$MODE" == "official_pi05" ]]; then
  export PI05_ASSETS_DIR="${PI05_ASSETS_DIR:-$CHECKPOINT/assets}"
else
  export PI05_ASSETS_DIR="${PI05_ASSETS_DIR:-$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_libero/assets}"
fi
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"
export XDG_CONFIG_HOME="$CACHE_ROOT/xdg_config"
export TMPDIR="$RUNTIME_ROOT/tmp"

mkdir -p "$OUTPUT_DIR" "$CACHE_ROOT/xdg" "$CACHE_ROOT/xdg_config" "$RUNTIME_ROOT/tmp" \
  "$PROJECT/slurm" "$PROJECT/.deps/libero_config"
cat > "$PROJECT/.deps/libero_config/config.yaml" <<CFG
benchmark_root: $LIBERO_ROOT
bddl_files: $LIBERO_ROOT/bddl_files
init_states: $LIBERO_ROOT/init_files
datasets: $PROJECT/datasets
assets: $LIBERO_ROOT/assets
CFG
cd "$PROJECT"

EVAL_ARGS=( \
  --mode "$MODE" \
  --checkpoint "$CHECKPOINT" \
  --suite libero_90 \
  --shard-index "$SHARD_INDEX" \
  --num-shards "$NUM_SHARDS" \
  --episodes "${EPISODES:-20}" \
  --eval-seed "${EVAL_SEED:-7}" \
  --seed "${SEED:-42}" \
  --image-size 224 \
  --action-horizon 10 \
  --replan-steps 5 \
  --output-dir "$OUTPUT_DIR" )
if [[ "$MODE" == "sft_plus_editor" ]]; then
  EVAL_ARGS+=(--editor-checkpoint "$EDITOR_CHECKPOINT")
  if [[ -n "${EDITOR_STEP:-}" ]]; then
    EVAL_ARGS+=(--editor-step "$EDITOR_STEP")
  fi
  if [[ -n "${EDITOR_NORM_STATS_DIR:-}" ]]; then
    EVAL_ARGS+=(--editor-norm-stats-dir "$EDITOR_NORM_STATS_DIR")
  fi
fi
if [[ -n "${SFT_ASSETS_DIR:-}" && ( "$MODE" == "sft_pi" || "$MODE" == "sft_plus_editor" ) ]]; then
  EVAL_ARGS+=(--assets-dir "$SFT_ASSETS_DIR")
fi
if [[ "$MODE" == "official_pi05" || "$MODE" == "direct_rl_pi" ]]; then
  EVAL_ARGS+=(--assets-dir "$PI05_ASSETS_DIR")
fi
PYTHON="${PYTHON:-$EXPO/.venv/bin/python}"
exec "$PYTHON" "$PROJECT/full_90_libero_evals.py" "${EVAL_ARGS[@]}"

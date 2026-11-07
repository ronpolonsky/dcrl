#!/usr/bin/env bash
#SBATCH --partition=gpuA100x4
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
# Run the compact learner on the Delta A100 GPU nodes.
#SBATCH --gres=gpu:nvidia_a100:1
#SBATCH --account=bgqe-delta-gpu
# The GPU type can be overridden at submission time with --partition/--gres.
#SBATCH --job-name=small-expo-3p
# Keep array-task logs separate, and store them on the non-full /projects filesystem.
#SBATCH --output=/projects/bgqe/rpolonsky/dcrl_slurm/%x-%A_%a.out
#SBATCH --error=/projects/bgqe/rpolonsky/dcrl_slurm/%x-%A_%a.err

set -euo pipefail

TASK_ID="${TASK_ID:-${SLURM_ARRAY_TASK_ID:-}}"
: "${TASK_ID:?submit with --export=ALL,TASK_ID=<id> or --array=0-89}"
export TASK_ID
: "${BATCH_SIZE:=64}"
: "${RESUME:=0}"
: "${INIT_DISTILL_STEPS:=0}"
: "${ONLINE_RL_CHECKPOINT_INTERVAL:=25000}"
: "${REPLAY_CAPACITY:=20000}"
: "${UTD_RATIO:=10}"
: "${NUM_UPDATES:=2}"
: "${OFFLINE_RATIO:=0.5}"
: "${NORM_STATS_DIR:=/scratch/bgqe/rpolonsky/dcrlcheckpoints/openpi_cache/openpi-assets/checkpoints/pi05_libero/assets/physical-intelligence/libero}"
: "${WANDB_PROJECT_NAME:=small_learner}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="${PROJECT_ROOT:-/u/rpolonsky/dcrl_src}"
PROJECT_PARENT="$(cd -- "$PROJECT/.." && pwd)"
EXPO="$PROJECT/third_party/expo-ft"
PYTHON="${PYTHON:-/projects/bgqe/rpolonsky/dcrl/third_party/expo-ft/.venv/bin/python}"
CODE_ROOT="${CODE_ROOT:-/scratch/bgqe/rpolonsky/tuned_src}"
LIBERO_ROOT="$PROJECT/.deps/LIBERO/libero/libero"
RUNTIME_ROOT="${RUNTIME_ROOT:-/tmp/dcrl_runtime_${SLURM_JOB_ID:-$$}}"
LOG_ROOT="${LOG_ROOT:-$RUNTIME_ROOT/logs}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/scratch/bgqe/rpolonsky/dcrlcheckpoints/tuned_small_rl_10_5_offline50k_online500k}"
DEMO_ROOT="${DEMO_ROOT:-/u/rpolonsky/libero_demos}"
export PYTHONPATH="$CODE_ROOT:$PROJECT:$EXPO/expo_ft/agents/vla/openpi/src:$EXPO/expo_ft/agents/vla/openpi/packages/openpi-client/src:$PROJECT/.deps/LIBERO${PYTHONPATH:+:$PYTHONPATH}"
# Forge serves the W&B API under /api/wandb.  The /wandb UI path returns
# HTTP 405 for the SDK's GraphQL POST and leaves online initialization hanging.
export WANDB_BASE_URL=https://forge.coreweave.com/api/wandb
export WANDB_PROJECT="$WANDB_PROJECT_NAME"
export WANDB_ENTITY=distill_rl
export WANDB_DIR="$OUTPUT_ROOT/wandb"
export WANDB_CONFIG_DIR="$RUNTIME_ROOT/wandb_config"
export WANDB_CACHE_DIR="$RUNTIME_ROOT/wandb_cache"
export WANDB_DATA_DIR="$RUNTIME_ROOT/wandb_data"
export NETRC="$RUNTIME_ROOT/netrc"
# Delta defaults to EGL; set MUJOCO_GL/PYOPENGL_PLATFORM to osmesa when needed.
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-$MUJOCO_GL}"
RENDER_BACKEND="$MUJOCO_GL"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
# LIBERO init-state files are trusted pickle data; PyTorch 2.6 otherwise
# rejects them under its new default weights_only=True behavior.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export XDG_CACHE_HOME="$RUNTIME_ROOT/cache"
export JAX_COMPILATION_CACHE_DIR="$RUNTIME_ROOT/jax_cache"

if [[ -f /work/hdd/bgqe/rpolonsky/.env ]]; then
  set -a
  source /work/hdd/bgqe/rpolonsky/.env
  set +a
fi
: "${WANDB_API_KEY:?WANDB_API_KEY must be provided in /work/hdd/bgqe/rpolonsky/.env}"

case "$TASK_ID" in
  2|02) ACTOR_LR=1e-4; CRITIC_LR=3e-4; TEMP_LR=1e-5; EDIT_SCALE=0.10; TASK_ID=2 ;;
  28)  ACTOR_LR=1e-4; CRITIC_LR=3e-4; TEMP_LR=1e-5; EDIT_SCALE=0.05; TASK_ID=28 ;;
  29)  ACTOR_LR=1e-4; CRITIC_LR=3e-4; TEMP_LR=1e-5; EDIT_SCALE=0.10; TASK_ID=29 ;;
  60)  ACTOR_LR=5e-5; CRITIC_LR=2e-4; TEMP_LR=1e-5; EDIT_SCALE=0.15; TASK_ID=60 ;;
  79)  ACTOR_LR=5e-5; CRITIC_LR=2e-4; TEMP_LR=1e-5; EDIT_SCALE=0.10; TASK_ID=79 ;;
  *) echo "No tuned recipe configured for TASK_ID=$TASK_ID" >&2; exit 2 ;;
esac

mkdir -p "$LOG_ROOT" "$RUNTIME_ROOT/libero_config" "$RUNTIME_ROOT/jax_cache" \
  "$WANDB_CONFIG_DIR" "$WANDB_CACHE_DIR" "$WANDB_DATA_DIR" "$OUTPUT_ROOT" "$WANDB_DIR"
exec > >(tee -a "$OUTPUT_ROOT/launcher_task_${TASK_ID}.log") 2>&1
cat > "$RUNTIME_ROOT/libero_config/config.yaml" <<CFG
benchmark_root: $LIBERO_ROOT
bddl_files: $LIBERO_ROOT/bddl_files
init_states: $LIBERO_ROOT/init_files
datasets: $PROJECT/datasets
assets: $LIBERO_ROOT/assets
CFG
export LIBERO_CONFIG_PATH="$RUNTIME_ROOT/libero_config"

echo "job=$SLURM_JOB_ID host=$(hostname) node=$SLURMD_NODENAME render_backend=$RENDER_BACKEND task=$TASK_ID"
timeout 30s nvidia-smi -L
"$PYTHON" - <<'PY'
import os
import jax
from libero.libero import benchmark

devices = jax.devices()
if not any(device.platform == 'gpu' for device in devices):
    raise RuntimeError(f'JAX did not find a GPU: {devices}')
suite = benchmark.get_benchmark_dict()['libero_90']()
task_id = int(os.environ['TASK_ID'])
print('JAX devices:', devices)
print('LIBERO task:', task_id, suite.get_task(task_id).language)
print('LIBERO initial states:', len(suite.get_task_init_states(task_id)))
PY

cd "$CODE_ROOT"
RESUME_ARGS=()
if [[ "$RESUME" == "1" ]]; then
  RESUME_ARGS+=(--resume)
fi
"$PYTHON" -m dcrl.train_libero "${RESUME_ARGS[@]}" \
  --suite libero_90 \
  --task-id "$TASK_ID" \
  --init-distill-steps "$INIT_DISTILL_STEPS" \
  --offline-rl-steps 50000 \
  --online-rl-steps 500000 \
  --pi-distill-eval-interval 50000 \
  --pi-distill-chckpt-interval 50000 \
  --offline-rl-eval-interval 25000 \
  --offline-rl-checkpoint-interval 25000 \
  --online-rl-eval-interval 25000 \
  --online-rl-checkpot-interval "$ONLINE_RL_CHECKPOINT_INTERVAL" \
  --demos-root "$DEMO_ROOT" \
  --update-type episode \
  --num-updates "$NUM_UPDATES" \
  --batch-size "$BATCH_SIZE" \
  --utd-ratio "$UTD_RATIO" \
  --replan-steps 5 \
  --action-horizon 10 \
  --offline-ratio "$OFFLINE_RATIO" \
  --replay-capacity "$REPLAY_CAPACITY" \
  --norm-stats-dir "$NORM_STATS_DIR" \
  --actor-lr "$ACTOR_LR" \
  --critic-lr "$CRITIC_LR" \
  --temp-lr "$TEMP_LR" \
  --edit-scale "$EDIT_SCALE" \
  --adjust-target-entropy \
  --eval-episodes 20 \
  --output-dir "$OUTPUT_ROOT" \
  --wandb-entity distill_rl \
  --wandb-project "$WANDB_PROJECT_NAME" \
  --wandb-name "tuned_small_rl_libero90_task${TASK_ID}_seed42" \
  --wandb-mode "${WANDB_MODE:-online}" \
  --seed 42

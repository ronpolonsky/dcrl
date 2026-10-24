#!/usr/bin/env bash
#SBATCH --partition=iris-hi
#SBATCH --time=21-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
# Run the compact learner on the regular iris-hi GPU tiers.
#SBATCH --gres=gpu:1
#SBATCH --account=iris
# Leave H100/H200 capacity for large-model jobs; use A40/A6000 nodes.
#SBATCH --exclude=iris[1-4,8-10],iris-hp-z8
#SBATCH --job-name=small-expo-3p
#SBATCH --output=/nlp/scr/ronpo/projects/dcrl/slurm/%x-%j.out
#SBATCH --error=/nlp/scr/ronpo/projects/dcrl/slurm/%x-%j.err

set -euo pipefail

: "${TASK_ID:?submit with --export=ALL,TASK_ID=<2|28|29|60|79>}"
: "${BATCH_SIZE:=32}"
: "${RESUME:=0}"
: "${INIT_DISTILL_STEPS:=300000}"

PROJECT=/nlp/scr/ronpo/projects/dcrl
EXPO="$PROJECT/third_party/expo-ft"
PYTHON="$EXPO/.venv/bin/python"
LIBERO_ROOT="$PROJECT/.deps/LIBERO/libero/libero"
OUTPUT_ROOT=/nlp/scr/ronpo/dcrlcheckpoints/small_expo_3phase_10_5_sharedstats
export PYTHONPATH="$PROJECT:$EXPO/expo_ft/agents/vla/openpi/src:$EXPO/expo_ft/agents/vla/openpi/packages/openpi-client/src:$PROJECT/.deps/LIBERO${PYTHONPATH:+:$PYTHONPATH}"
# Forge serves the W&B API under /api/wandb.  The /wandb UI path returns
# HTTP 405 for the SDK's GraphQL POST and leaves online initialization hanging.
export WANDB_BASE_URL=https://forge.coreweave.com/api/wandb
export WANDB_PROJECT=dcrl
export WANDB_ENTITY=distill_rl
export WANDB_DIR="$OUTPUT_ROOT/wandb"
# The HGX EGL stack has reproducibly aborted in MuJoCo read_pixels().
# Use OSMesa only on those hosts; healthy nodes retain the faster EGL path.
if [[ "${SLURMD_NODENAME:-}" == iris-hgx-* ]]; then
  export MUJOCO_GL=osmesa
  export PYOPENGL_PLATFORM=osmesa
  RENDER_BACKEND=osmesa
else
  export MUJOCO_GL=egl
  export PYOPENGL_PLATFORM=egl
  RENDER_BACKEND=egl
fi
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

mkdir -p "$PROJECT/slurm" "$PROJECT/.deps/libero_config" "$OUTPUT_ROOT" "$WANDB_DIR"
cat > "$PROJECT/.deps/libero_config/config.yaml" <<CFG
benchmark_root: $LIBERO_ROOT
bddl_files: $LIBERO_ROOT/bddl_files
init_states: $LIBERO_ROOT/init_files
datasets: $PROJECT/datasets
assets: $LIBERO_ROOT/assets
CFG
export LIBERO_CONFIG_PATH="$PROJECT/.deps/libero_config"

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

cd "$PROJECT"
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
  --offline-rl-eval-interval 50000 \
  --offline-rl-checkpoint-interval 50000 \
  --online-rl-eval-interval 25000 \
  --online-rl-checkpot-interval 25000 \
  --distill-data-root /nlp/scr/ronpo/dcrlcheckpoints/pi_base_data/fixed125 \
  --demos-root datasets \
  --update-type episode \
  --num-updates 3 \
  --batch-size "$BATCH_SIZE" \
  --utd-ratio 20 \
  --replan-steps 5 \
  --action-horizon 10 \
  --offline-ratio 0.5 \
  --eval-episodes 20 \
  --output-dir "$OUTPUT_ROOT" \
  --wandb-entity distill_rl \
  --wandb-project dcrl \
  --wandb-name "small_expo_3phase_libero90_task${TASK_ID}_seed42" \
  --wandb-mode online \
  --seed 42

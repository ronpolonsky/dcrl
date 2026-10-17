#!/usr/bin/env bash
#SBATCH --partition=iris-hi
#SBATCH --time=21-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --gres=gpu:h100:1
#SBATCH --account=iris
#SBATCH --exclude=iris[1-4],iris9,iris10,iris-hp-z8
#SBATCH --job-name=small-expo-l60
#SBATCH --output=/nlp/scr/ronpo/projects/dcrl/slurm/%x-%j.out
#SBATCH --error=/nlp/scr/ronpo/projects/dcrl/slurm/%x-%j.err

set -euo pipefail

PROJECT=/nlp/scr/ronpo/projects/dcrl
EXPO=/iris/u/ronpo/projects/expo-ft
PYTHON="$EXPO/.venv/bin/python"
LIBERO_ROOT="$PROJECT/.deps/LIBERO/libero/libero"
export PYTHONPATH="$PROJECT:$EXPO/expo_ft/agents/vla/openpi/src:$EXPO/expo_ft/agents/vla/openpi/packages/openpi-client/src:$PROJECT/.deps/LIBERO${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_BASE_URL=https://forge.coreweave.com/wandb
export WANDB_PROJECT=dcrl
export WANDB_ENTITY=distill_rl
export WANDB_DIR=/nlp/scr/ronpo/dcrlcheckpoints/small_expo_learner/wandb
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TOKENIZERS_PARALLELISM=false

mkdir -p "$PROJECT/slurm" "$PROJECT/.deps/libero_config"
mkdir -p /nlp/scr/ronpo/dcrlcheckpoints/small_expo_learner "$WANDB_DIR"

cat > "$PROJECT/.deps/libero_config/config.yaml" <<EOF
benchmark_root: $LIBERO_ROOT
bddl_files: $LIBERO_ROOT/bddl_files
init_states: $LIBERO_ROOT/init_files
datasets: $PROJECT/datasets
assets: $LIBERO_ROOT/assets
EOF
export LIBERO_CONFIG_PATH="$PROJECT/.deps/libero_config"

echo "job=$SLURM_JOB_ID host=$(hostname) node=$SLURMD_NODENAME"
timeout 30s nvidia-smi -L
"$PYTHON" - <<'PY'
import jax
from libero.libero import benchmark
from dcrl.libero_env import LiberoTaskEnv

devices = jax.devices()
if not any(device.platform == 'gpu' for device in devices):
    raise RuntimeError(f'JAX did not find a GPU: {devices}')
suite = benchmark.get_benchmark_dict()['libero_90']()
task = suite.get_task(60)
print('JAX devices:', devices)
print('LIBERO task:', task.language)
print('LIBERO initial states:', len(suite.get_task_init_states(60)))
PY

cd "$PROJECT"
RESUME_ARGS=()
if [[ "${RESUME:-0}" == "1" ]]; then
  RESUME_ARGS+=(--resume)
fi
"$PYTHON" -m dcrl.train_libero "${RESUME_ARGS[@]}" \
  --suite libero_90 \
  --task-id 60 \
  --num-steps 1000000 \
  --num-offline-steps 10000 \
  --demos-root datasets \
  --update-type episode \
  --num-updates 3 \
  --batch-size 64 \
  --utd-ratio 20 \
  --replan-steps 8 \
  --action-horizon 16 \
  --eval-every 25000 \
  --offline-eval-interval 25000 \
  --offline-ratio 0.5 \
  --eval-episodes 20 \
  --checkpoint-every 25000 \
  --output-dir /nlp/scr/ronpo/dcrlcheckpoints/small_expo_learner \
  --wandb-entity distill_rl \
  --wandb-project dcrl \
  --wandb-name small_expo_libero90_task60_seed42 \
  --wandb-mode offline \
  --seed 42

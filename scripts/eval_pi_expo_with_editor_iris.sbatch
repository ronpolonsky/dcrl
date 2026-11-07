#!/usr/bin/env bash
# One-shot hybrid evaluation: large Pi0.5 actor + compact EXPO editor/critic.
#SBATCH --partition=gpuA100x4
#SBATCH --account=bgqe-delta-gpu
#SBATCH --gres=gpu:nvidia_a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=pi-expo-editor-eval
#SBATCH --output=/work/hdd/bgqe/rpolonsky/dcrl/slurm/%x-%j.out
#SBATCH --error=/work/hdd/bgqe/rpolonsky/dcrl/slurm/%x-%j.err
set -euo pipefail

PROJECT=/work/hdd/bgqe/rpolonsky/dcrl
EXPO="$PROJECT/third_party/expo-ft"
TEACHER_DIR=/work/hdd/bgqe/rpolonsky/dcrlcheckpoints/small_expo_learner/libero_90/task_29
TEACHER_STEP=335076
OUTPUT=/work/hdd/bgqe/rpolonsky/dcrlcheckpoints/pi_expo_editor_eval/task29_teacher335076.json

export PYTHONPATH="$PROJECT:$EXPO/expo_ft/agents/vla/openpi/src:$EXPO/expo_ft/agents/vla/openpi/packages/openpi-client/src:$PROJECT/.deps/LIBERO${PYTHONPATH:+:$PYTHONPATH}"
export LIBERO_CONFIG_PATH="$PROJECT/.deps/libero_config"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENPI_SKIP_INIT_JIT=1
export JAX_COMPILATION_CACHE_DIR=${JAX_COMPILATION_CACHE_DIR:-/work/hdd/bgqe/rpolonsky/dcrlcheckpoints/jax_cache}
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENPI_DATA_HOME=${OPENPI_DATA_HOME:-/work/hdd/bgqe/rpolonsky/dcrlcheckpoints/openpi_cache}
mkdir -p "$PROJECT/slurm" "$(dirname "$OUTPUT")" "$JAX_COMPILATION_CACHE_DIR"
cd "$PROJECT"

echo "job=$SLURM_JOB_ID host=$(hostname) node=$SLURMD_NODENAME"
timeout 30s nvidia-smi -L
"$EXPO/.venv/bin/python" - <<'PYPROBE'
import jax
from libero.libero import benchmark
if not any(d.platform == 'gpu' for d in jax.devices()):
    raise RuntimeError(f'JAX did not find a GPU: {jax.devices()}')
suite=benchmark.get_benchmark_dict()['libero_90']()
print('JAX devices:', jax.devices())
print('LIBERO task 29:', suite.get_task(29).language)
print('LIBERO initial states:', len(suite.get_task_init_states(29)))
PYPROBE

"$EXPO/.venv/bin/python" -m dcrl.eval_pi_expo_with_editor \
  --editor-checkpoint-dir "$TEACHER_DIR" \
  --editor-step "$TEACHER_STEP" \
  --editor-norm-stats-dir "$TEACHER_DIR" \
  --suite libero_90 \
  --task-id 29 \
  --episodes 20 \
  --image-size 224 \
  --action-horizon 10 \
  --replan-steps 5 \
  --eval-seed 7 \
  --seed 42 \
  --output "$OUTPUT"

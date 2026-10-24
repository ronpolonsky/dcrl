#!/usr/bin/env bash
# Collect frozen official Pi0.5 base-actor chunks for one LIBERO task.
# Override TASK_ID, EPISODES, and OUTPUT when submitting.
#SBATCH --partition=iris-hi
#SBATCH --account=iris
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --exclude=iris[1-4],iris8,iris10,iris-hp-z8
#SBATCH --job-name=pi05-base-collect
#SBATCH --output=/nlp/scr/ronpo/projects/dcrl/slurm/%x-%A_%a.out
#SBATCH --error=/nlp/scr/ronpo/projects/dcrl/slurm/%x-%A_%a.err
set -euo pipefail

PROJECT=/nlp/scr/ronpo/projects/dcrl
EXPO="$PROJECT/third_party/expo-ft"
export PYTHONPATH="$PROJECT:$EXPO/expo_ft/agents/vla/openpi/src:$EXPO/expo_ft/agents/vla/openpi/packages/openpi-client/src:$PROJECT/.deps/LIBERO"
export LIBERO_CONFIG_PATH="$PROJECT/.deps/libero_config"
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR=${JAX_COMPILATION_CACHE_DIR:-/nlp/scr/ronpo/dcrlcheckpoints/jax_cache}
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
mkdir -p "$JAX_COMPILATION_CACHE_DIR"
export OPENPI_SKIP_INIT_JIT=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export OPENPI_DATA_HOME=${OPENPI_DATA_HOME:-/nlp/scr/ronpo/dcrlcheckpoints/openpi_cache}
cd "$PROJECT"

if [[ -z "${TASK_ID:-}" ]]; then
  case "${SLURM_ARRAY_TASK_ID:-}" in
    0) TASK_ID=60 ;; 1) TASK_ID=79 ;; 2) TASK_ID=29 ;; 3) TASK_ID=28 ;; 4) TASK_ID=2 ;;
    *) echo 'set TASK_ID or submit array indices 0..4' >&2; exit 2 ;;
  esac
fi
OUTPUT=${OUTPUT:-/nlp/scr/ronpo/dcrlcheckpoints/pi_base_data/pi05_task_${TASK_ID}.hdf5}

ARGS=(--checkpoint "${PI05_CHECKPOINT:-gs://openpi-assets/checkpoints/pi05_base}" \
  --norm-stats "${PI05_NORM_STATS:-gs://openpi-assets/checkpoints/pi05_libero/assets/physical-intelligence/libero/norm_stats.json}" \
  --suite libero_90 --task-id "$TASK_ID" --seed 7 --output "$OUTPUT")
if [[ -n "${EPISODES:-}" ]]; then
  ARGS+=(--episodes "$EPISODES" --successes 1 --failures 1)
else
  ARGS+=(--successes "${SUCCESSES:-100}" --failures "${FAILURES:-25}")
fi
"$EXPO/.venv/bin/python" -m dcrl.collect_pi05_libero "${ARGS[@]}"

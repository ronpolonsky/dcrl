#!/usr/bin/env bash
#SBATCH --partition=iris-hi
#SBATCH --account=iris
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --exclude=iris[1-4],iris10
#SBATCH --job-name=full-chunk-sft
#SBATCH --output=/nlp/scr/ronpo/projects/dcrl/slurm/%x-%j.out
#SBATCH --error=/nlp/scr/ronpo/projects/dcrl/slurm/%x-%j.err
set -euo pipefail
PROJECT=/nlp/scr/ronpo/projects/dcrl
EXPO=/iris/u/ronpo/projects/expo-ft
export PYTHONPATH="$PROJECT:$EXPO/expo_ft/agents/vla/openpi/src:$EXPO/expo_ft/agents/vla/openpi/packages/openpi-client/src:$PROJECT/.deps/LIBERO"
export LIBERO_CONFIG_PATH="$PROJECT/.deps/libero_config"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl XLA_PYTHON_CLIENT_PREALLOCATE=false
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export OPENPI_DATA_HOME=/nlp/scr/ronpo/dcrlcheckpoints/openpi_cache
cd "$PROJECT"
timeout 30s nvidia-smi -L
"$EXPO/.venv/bin/python" -c 'import jax; assert any(d.platform == "gpu" for d in jax.devices()), jax.devices(); print(jax.devices())'
OUT=${SFT_OUTPUT_DIR:-/nlp/scr/ronpo/dcrlcheckpoints/full_chunk_sft/smoke_task28}
TEACHER_DIR=${TEACHER_DIR:-/nlp/scr/ronpo/dcrlcheckpoints/small_expo_learner/libero_90/task_28}
TEACHER_STEP=${TEACHER_STEP:-110016}
COMMON=(--teacher-dir "$TEACHER_DIR" --teacher-step "$TEACHER_STEP" --output-dir "$OUT" --collection-episodes "${COLLECTION_EPISODES:-2}" --batch-size "${BATCH_SIZE:-1}" --eval-episodes "${EVAL_EPISODES:-1}" --eval-every "${EVAL_EVERY:-2}" --checkpoint-every "${CHECKPOINT_EVERY:-2}")
RESUME_ARGS=()
if [[ "${RESUME_SFT:-0}" == 1 ]]; then
  RESUME_ARGS+=(--resume)
fi
"$EXPO/.venv/bin/python" -m dcrl.train_full_chunk_sft "${COMMON[@]}" --num-sft-steps "${SFT_STEPS:-2}" "${RESUME_ARGS[@]}"
if [[ "${SMOKE_RESUME:-1}" == 1 && "${RESUME_SFT:-0}" != 1 ]]; then
  "$EXPO/.venv/bin/python" -m dcrl.train_full_chunk_sft "${COMMON[@]}" --phase train --resume --num-sft-steps 3
fi

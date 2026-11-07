#!/usr/bin/env bash
#SBATCH --partition=gpuA100x4
#SBATCH --account=bgqe-delta-gpu
#SBATCH --gres=gpu:nvidia_a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=pi05-bc-distill
#SBATCH --output=/work/hdd/bgqe/rpolonsky/dcrl/slurm/%x-%j.out
#SBATCH --error=/work/hdd/bgqe/rpolonsky/dcrl/slurm/%x-%j.err
set -euo pipefail

PROJECT=/work/hdd/bgqe/rpolonsky/dcrl
EXPO="$PROJECT/third_party/expo-ft"
export PYTHONPATH="$PROJECT:$EXPO/expo_ft/agents/vla/openpi/src:$EXPO/expo_ft/agents/vla/openpi/packages/openpi-client/src:$PROJECT/.deps/LIBERO"
export LIBERO_CONFIG_PATH="$PROJECT/.deps/libero_config"
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR=${JAX_COMPILATION_CACHE_DIR:-/work/hdd/bgqe/rpolonsky/dcrlcheckpoints/jax_cache}
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
mkdir -p "$JAX_COMPILATION_CACHE_DIR"
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export OPENPI_DATA_HOME=${OPENPI_DATA_HOME:-/work/hdd/bgqe/rpolonsky/dcrlcheckpoints/openpi_cache}
cd "$PROJECT"
timeout 30s nvidia-smi -L
"$EXPO/.venv/bin/python" -c 'import jax; assert any(d.platform == "gpu" for d in jax.devices()), jax.devices(); print(jax.devices())'

OUT=${SFT_OUTPUT_DIR:-/work/hdd/bgqe/rpolonsky/dcrlcheckpoints/pi_base_sft/task29}
COMMON=(--collection-source small_expo \
  --teacher-dir "${TEACHER_DIR:-/work/hdd/bgqe/rpolonsky/dcrlcheckpoints/small_expo_learner/libero_90/task_29}" \
  --teacher-step "${TEACHER_STEP:-335076}" \
  --source-checkpoint "${PI05_CHECKPOINT:-gs://openpi-assets/checkpoints/pi05_base}" \
  --student-checkpoint "${PI05_CHECKPOINT:-gs://openpi-assets/checkpoints/pi05_base}" \
  --norm-stats "${PI05_NORM_STATS:-gs://openpi-assets/checkpoints/pi05_libero/assets/physical-intelligence/libero/norm_stats.json}" \
  --finetune-mode "${FINETUNE_MODE:-lora}" \
  --source-suite libero_90 --source-task-id "${TASK_ID:-29}" --output-dir "$OUT")

if [[ "${COLLECT_ONLY:-0}" == 1 ]]; then
  # Compact-teacher collection fits workstation GPUs. Keep it separate from
  # Pi0.5 student training, which requires an 80GB-class GPU in this setup.
  "${EXPO}/.venv/bin/python" -m dcrl.train_bc_distill \
    "${COMMON[@]}" --phase collect \
    --collection-episodes "${COLLECTION_EPISODES:-200}" \
    --collection-successes "${COLLECTION_SUCCESSES:-100}" \
    --collection-failures "${COLLECTION_FAILURES:-25}"
  exit 0
fi

if [[ "${TRAIN_ONLY_FULL:-0}" == 1 ]]; then
  "${EXPO}/.venv/bin/python" -m dcrl.train_bc_distill \
    "${COMMON[@]}" --phase train \
    --num-sft-steps "${SFT_STEPS:-500000}" \
    --batch-size "${BATCH_SIZE:-16}" \
    --eval-episodes "${EVAL_EPISODES:-20}" \
    --eval-every "${EVAL_EVERY:-25000}" \
    --checkpoint-every "${CHECKPOINT_EVERY:-50000}" \
    --warmup-steps "${WARMUP_STEPS:-10000}" \
    --lr-decay-steps "${LR_DECAY_STEPS:-1000000}" \
    ${MAX_TO_KEEP:+--max-to-keep "$MAX_TO_KEEP"}
  exit 0
fi

if [[ "${TRAIN_ONLY:-0}" == 1 ]]; then
  "${EXPO}/.venv/bin/python" -m dcrl.train_bc_distill \
    "${COMMON[@]}" --phase train --num-sft-steps 1 \
    --batch-size "${BATCH_SIZE:-1}" --eval-episodes "${EVAL_EPISODES:-1}" \
    --eval-every "${EVAL_EVERY:-1}" --checkpoint-every "${CHECKPOINT_EVERY:-1}" \
    --warmup-steps "${WARMUP_STEPS:-1}" --lr-decay-steps "${LR_DECAY_STEPS:-10}" \
    ${MAX_TO_KEEP:+--max-to-keep "$MAX_TO_KEEP"}
  "${EXPO}/.venv/bin/python" -m dcrl.train_bc_distill \
    "${COMMON[@]}" --phase train --resume --num-sft-steps 2 \
    --batch-size "${BATCH_SIZE:-1}" --eval-episodes "${EVAL_EPISODES:-1}" \
    --eval-every "${EVAL_EVERY:-1}" --checkpoint-every "${CHECKPOINT_EVERY:-1}" \
    --warmup-steps "${WARMUP_STEPS:-1}" --lr-decay-steps "${LR_DECAY_STEPS:-10}" \
    ${MAX_TO_KEEP:+--max-to-keep "$MAX_TO_KEEP"}
  exit 0
fi

if [[ "${SMOKE:-0}" == 1 ]]; then
  "${EXPO}/.venv/bin/python" -m dcrl.train_bc_distill \
    "${COMMON[@]}" --phase all --collection-episodes 2 --num-sft-steps 2 \
    --batch-size 1 --eval-episodes 1 --eval-every 1 --checkpoint-every 1 \
    --warmup-steps 1 --lr-decay-steps 10
  "${EXPO}/.venv/bin/python" -m dcrl.train_bc_distill \
    "${COMMON[@]}" --phase train --resume --num-sft-steps 3 \
    --batch-size 1 --eval-episodes 1 --eval-every 1 --checkpoint-every 1 \
    --warmup-steps 1 --lr-decay-steps 10
else
  "${EXPO}/.venv/bin/python" -m dcrl.train_bc_distill \
  "${COMMON[@]}" --phase all \
    --collection-episodes "${COLLECTION_EPISODES:-200}" \
    --collection-successes "${COLLECTION_SUCCESSES:-100}" \
    --collection-failures "${COLLECTION_FAILURES:-25}" \
    --num-sft-steps "${SFT_STEPS:-1000000}" \
    --batch-size "${BATCH_SIZE:-256}" \
    --eval-episodes "${EVAL_EPISODES:-20}" \
    --eval-every "${EVAL_EVERY:-25000}" \
    --checkpoint-every "${CHECKPOINT_EVERY:-25000}" \
    ${MAX_TO_KEEP:+--max-to-keep "$MAX_TO_KEEP"}
fi

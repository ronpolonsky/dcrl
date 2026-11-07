#!/usr/bin/env bash
set -euo pipefail

ROOT="/tmp/usertmp/rpolonsky/task79_sft_screen_19_30_79_20261107"
OUT="${ROOT}/sft_output"
TMP="${ROOT}/runtime/tmp"
PROJECT="/projects/bgqe/rpolonsky/dcrl"
EXPO="${PROJECT}/third_party/expo-ft"
AUDIT="/projects/bgqe/rpolonsky/dcrl/sharder/training/sft_analysis_20261103"
TEACHER="/work/hdd/bgqe/rpolonsky/dcrlcheckpoints/small_expo_rl_10_5_offline50k_online500k_resume_work/libero_90/task_79"

mkdir -p "${OUT}" "${TMP}"
cd "${PROJECT}"
exec env \
  CUDA_VISIBLE_DEVICES=0 \
  PYTHONPATH="${AUDIT}:${PROJECT}:${EXPO}/expo_ft/agents/vla/openpi/src:${EXPO}/expo_ft/agents/vla/openpi/packages/openpi-client/src:${PROJECT}/.deps/LIBERO:/projects/bgqe/rpolonsky/dcrl/sharder/training" \
  LIBERO_CONFIG_PATH="${PROJECT}/sharder/training/runtime/libero_config" \
  OPENPI_DATA_HOME=/work/hdd/bgqe/rpolonsky/dcrlcheckpoints/openpi_cache \
  TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  XLA_PYTHON_CLIENT_PREALLOCATE=false TOKENIZERS_PARALLELISM=false \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 TMPDIR="${TMP}" \
  "${EXPO}/.venv/bin/python" "${AUDIT}/train_sft_retention.py" \
  --phase all --trainable-scope existing --target-mode full_proposal \
  --retention-tasks 19 30 79 --collection-source small_expo \
  --teacher-dir "${TEACHER}" --teacher-step 425121 \
  --source-suite libero_90 --source-task-id 79 \
  --collection-successes 150 --collection-failures 0 --collection-episodes 200 \
  --output-dir "${OUT}" \
  --student-checkpoint gs://openpi-assets/checkpoints/pi05_libero \
  --norm-stats gs://openpi-assets/checkpoints/pi05_libero/assets/physical-intelligence/libero/norm_stats.json \
  --action-horizon 10 --replan-steps 5 --image-size 224 \
  --num-sft-steps 1500 --batch-size 16 --official-demo-ratio 0 \
  --learning-rate 5e-5 --warmup-steps 10000 --lr-decay-steps 1000000 \
  --finetune-mode lora --zero-lora-b \
  --eval-every 500 --eval-episodes 20 --checkpoint-every 500 \
  --seed 42 --eval-seed 7 --wandb-mode disabled

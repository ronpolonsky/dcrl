#!/usr/bin/env bash
set -euo pipefail

SESSION="full90_task28_actionbounds_raw_step1500_plus_editor"
SOCKET="/tmp/tmux_task28_raw1500_plus.sock"
ROOT="/tmp/usertmp/rpolonsky/task28_actionbounds_raw_step1500_plus_editor_full90_20261107"
OUT="${ROOT}/full90_output"
TMP="${ROOT}/runtime/tmp"
PROJECT="/projects/bgqe/rpolonsky/dcrl"
EXPO="${PROJECT}/third_party/expo-ft"
AUDIT="/projects/bgqe/rpolonsky/dcrl/sharder/training/sft_analysis_20261103"
SFT="/work/hdd/bgqe/rpolonsky/dcrlcheckpoints/sft_task28_actionbounds_20261103/raw"
EDITOR="/u/rpolonsky/dcrlcheckpoints/small_expo_rl_10_5_offline50k_online500k_resume_newgpu/libero_90/task_28"

mkdir -p "${OUT}" "${TMP}"
if tmux -S "${SOCKET}" has-session -t "${SESSION}" 2>/dev/null; then
  echo "session already exists: ${SESSION}" >&2
  exit 2
fi

tmux -S "${SOCKET}" new-session -d -s "${SESSION}" bash -lc "exec env \
  CUDA_VISIBLE_DEVICES=0 \
  PYTHONPATH=${AUDIT}:${PROJECT}:${EXPO}/expo_ft/agents/vla/openpi/src:${EXPO}/expo_ft/agents/vla/openpi/packages/openpi-client/src:${PROJECT}/.deps/LIBERO:/projects/bgqe/rpolonsky/dcrl/sharder/training \
  LIBERO_CONFIG_PATH=${PROJECT}/sharder/training/runtime/libero_config \
  OPENPI_DATA_HOME=/work/hdd/bgqe/rpolonsky/dcrlcheckpoints/openpi_cache \
  TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  XLA_PYTHON_CLIENT_PREALLOCATE=false TOKENIZERS_PARALLELISM=false \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 TMPDIR=${TMP} \
  ${EXPO}/.venv/bin/python \
  ${AUDIT}/full90_audit_eval.py \
  --mode sft_plus_editor \
  --checkpoint ${SFT}/latest_checkpoints \
  --assets-dir ${SFT}/student_assets \
  --editor-checkpoint ${EDITOR}/checkpoints/275001 \
  --editor-step 275001 \
  --editor-norm-stats-dir ${EDITOR} \
  --suite libero_90 --task-ids all --episodes 20 --eval-seed 7 --seed 42 \
  --image-size 224 --action-horizon 10 --replan-steps 5 --fsdp-devices 1 \
  --output-dir ${OUT} > ${OUT}/launcher.log 2>&1"

echo "started ${SESSION}"

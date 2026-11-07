#!/usr/bin/env bash
set -euo pipefail

SESSION="full90_task60_sft_plus_editor_diag"
OUT="/work/hdd/bgqe/rpolonsky/dcrlcheckpoints/sft_full90_diagnosis_20261106/task60_seed42_step200_plus_editor"
TMP="${OUT}/runtime/tmp"
SOCKET="/work/hdd/bgqe/rpolonsky/tmux_task60_plus.sock"

mkdir -p "${TMP}"
if tmux -S "${SOCKET}" has-session -t "${SESSION}" 2>/dev/null; then
  echo "session already exists: ${SESSION}" >&2
  exit 2
fi

tmux -S "${SOCKET}" new-session -d -s "${SESSION}" bash -lc "exec env \
  CUDA_VISIBLE_DEVICES=0 \
  PYTHONPATH=/projects/bgqe/rpolonsky/dcrl/sharder/training:/projects/bgqe/rpolonsky/dcrl/sharder/training/sft_analysis_20261103:/projects/bgqe/rpolonsky/dcrl:/projects/bgqe/rpolonsky/dcrl/third_party/expo-ft/expo_ft/agents/vla/openpi/src:/projects/bgqe/rpolonsky/dcrl/third_party/expo-ft/expo_ft/agents/vla/openpi/packages/openpi-client/src:/projects/bgqe/rpolonsky/dcrl/.deps/LIBERO \
  LIBERO_CONFIG_PATH=/projects/bgqe/rpolonsky/dcrl/sharder/training/runtime/libero_config \
  OPENPI_DATA_HOME=/work/hdd/bgqe/rpolonsky/dcrlcheckpoints/openpi_cache \
  TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  XLA_PYTHON_CLIENT_PREALLOCATE=false TOKENIZERS_PARALLELISM=false \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 TMPDIR='${TMP}' \
  /projects/bgqe/rpolonsky/dcrl/third_party/expo-ft/.venv/bin/python \
  /projects/bgqe/rpolonsky/dcrl/sharder/training/sft_analysis_20261103/full90_audit_eval.py \
  --mode sft_plus_editor \
  --checkpoint /work/hdd/bgqe/rpolonsky/dcrlcheckpoints/sft_retention_diagnosis_20261104/task_60_full_proposal/checkpoints/200 \
  --assets-dir /work/hdd/bgqe/rpolonsky/dcrlcheckpoints/sft_retention_diagnosis_20261104/task_60_full_proposal/student_assets \
  --editor-checkpoint /work/hdd/bgqe/rpolonsky/dcrlcheckpoints/small_expo_rl_10_5_offline50k_online500k_resume_work/libero_90/task_60 \
  --editor-step 450223 \
  --editor-norm-stats-dir /work/hdd/bgqe/rpolonsky/dcrlcheckpoints/small_expo_rl_10_5_offline50k_online500k_resume_work/libero_90/task_60 \
  --suite libero_90 --task-ids all --episodes 20 --eval-seed 7 --seed 42 \
  --image-size 224 --action-horizon 10 --replan-steps 5 --fsdp-devices 1 \
  --output-dir '${OUT}' > '${OUT}/launcher.log' 2>&1"

echo "started ${SESSION}"

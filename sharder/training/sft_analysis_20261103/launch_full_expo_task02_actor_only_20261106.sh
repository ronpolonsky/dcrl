#!/usr/bin/env bash
set -euo pipefail

SESSION="full90_expo_task02_actor_only"
SOCKET="/tmp/tmux_expo02.sock"
OUT="/tmp/usertmp/rpolonsky/expo02_actor_only_full90_20261106"
TMP="${OUT}/runtime/tmp"
PROJECT="/projects/bgqe/rpolonsky/dcrl"
EXPO="${PROJECT}/third_party/expo-ft"

mkdir -p "${OUT}" "${TMP}"
if tmux -S "${SOCKET}" has-session -t "${SESSION}" 2>/dev/null; then
  echo "session already exists: ${SESSION}" >&2
  exit 2
fi

tmux -S "${SOCKET}" new-session -d -s "${SESSION}" bash -lc "exec env \
  CUDA_VISIBLE_DEVICES=0 \
  PYTHONPATH=${PROJECT}:${EXPO}/expo_ft/agents/vla/openpi/src:${EXPO}/expo_ft/agents/vla/openpi/packages/openpi-client/src:${PROJECT}/.deps/LIBERO:/projects/bgqe/rpolonsky/dcrl/sharder/training \
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl XLA_PYTHON_CLIENT_PREALLOCATE=false \
  OPENPI_SKIP_INIT_JIT=1 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
  TOKENIZERS_PARALLELISM=false OPENPI_DATA_HOME=/work/hdd/bgqe/rpolonsky/dcrlcheckpoints/openpi_cache \
  XDG_CACHE_HOME=${OUT}/xdg XDG_CONFIG_HOME=${OUT}/xdg_config TMPDIR=${TMP} \
  LIBERO_CONFIG_PATH=${PROJECT}/sharder/training/runtime/libero_config \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 \
  ${EXPO}/.venv/bin/python \
  /projects/bgqe/rpolonsky/dcrl/sharder/training/full_90_expo_actor_only_eval_20261103.py \
  --mode direct_rl_actor_only \
  --checkpoint /work/hdd/bgqe/rpolonsky/dcrlcheckpoints/pi_robo_libero_unsharded_h200_b16_200k/pi_robo_libero_unsharded_h200_b16_task2/checkpoints/60244 \
  --suite libero_90 --task-ids all --episodes 20 --eval-seed 7 --seed 42 \
  --image-size 224 --action-horizon 10 --replan-steps 5 --fsdp-devices 1 \
  --output-dir ${OUT} > ${OUT}/launcher.log 2>&1"

echo "started ${SESSION}"

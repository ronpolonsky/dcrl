#!/usr/bin/env bash
set -euo pipefail

SHARD="${1:?usage: launch_full_expo_task29_actor_only_20261106.sh 0|1}"
[[ "${SHARD}" =~ ^[01]$ ]] || { echo "invalid shard: ${SHARD}" >&2; exit 2; }

PROJECT=/projects/bgqe/rpolonsky/dcrl
EXPO="${PROJECT}/third_party/expo-ft"
EVALUATOR=/projects/bgqe/rpolonsky/dcrl/sharder/training/full_90_expo_actor_only_eval_20261103.py
CHECKPOINT=/work/hdd/bgqe/rpolonsky/dcrlcheckpoints/pi_robo_libero_unsharded_h200_b16_200k/pi_robo_libero_unsharded_h200_b16_task29/checkpoints/60486
OUT=/work/hdd/bgqe/rpolonsky/dcrlcheckpoints/full_expo_task29_actor_only_full90_20261106/shard_${SHARD}
RUNTIME=/tmp/usertmp/rpolonsky/expo29_actor_only90_shard${SHARD}
SESSION=full90_expo_task29_actor_only_shard${SHARD}
SOCKET=/work/hdd/bgqe/rpolonsky/tmux_expo29_shard${SHARD}.sock

[[ -f "${CHECKPOINT}/_CHECKPOINT_METADATA" ]] || {
  echo "checkpoint incomplete: ${CHECKPOINT}" >&2
  exit 3
}
mkdir -p "${OUT}" "${RUNTIME}/tmp" "${RUNTIME}/xdg" "${RUNTIME}/xdg_config" "${RUNTIME}/libero_config"
cat > "${RUNTIME}/libero_config/config.yaml" <<CFG
benchmark_root: ${PROJECT}/.deps/LIBERO/libero/libero
bddl_files: ${PROJECT}/.deps/LIBERO/libero/libero/bddl_files
init_states: ${PROJECT}/.deps/LIBERO/libero/libero/init_files
datasets: ${PROJECT}/.deps/LIBERO/datasets
assets: ${PROJECT}/.deps/LIBERO/libero/libero/assets
CFG

if tmux -S "${SOCKET}" has-session -t "${SESSION}" 2>/dev/null; then
  echo "session already exists: ${SESSION}" >&2
  exit 4
fi

tmux -S "${SOCKET}" new-session -d -s "${SESSION}" bash -lc "exec env \
  CUDA_VISIBLE_DEVICES=0 \
  PYTHONPATH=${PROJECT}:${EXPO}/expo_ft/agents/vla/openpi/src:${EXPO}/expo_ft/agents/vla/openpi/packages/openpi-client/src:${PROJECT}/.deps/LIBERO \
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl XLA_PYTHON_CLIENT_PREALLOCATE=false OPENPI_SKIP_INIT_JIT=1 \
  TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 TOKENIZERS_PARALLELISM=false \
  OPENPI_DATA_HOME=/work/hdd/bgqe/rpolonsky/dcrlcheckpoints/openpi_cache \
  XDG_CACHE_HOME=${RUNTIME}/xdg XDG_CONFIG_HOME=${RUNTIME}/xdg_config TMPDIR=${RUNTIME}/tmp \
  LIBERO_CONFIG_PATH=${RUNTIME}/libero_config OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 \
  ${EXPO}/.venv/bin/python ${EVALUATOR} --mode direct_rl_actor_only \
  --checkpoint ${CHECKPOINT} --suite libero_90 --shard-index ${SHARD} --num-shards 2 \
  --episodes 20 --eval-seed 7 --seed 42 --image-size 224 --action-horizon 10 \
  --replan-steps 5 --fsdp-devices 1 --output-dir ${OUT} > ${OUT}/launcher.log 2>&1"

echo "started ${SESSION}"

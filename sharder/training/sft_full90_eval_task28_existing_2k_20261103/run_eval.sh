#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0

SHARD=${1:?usage: run_eval.sh SHARD_INDEX}
[[ "$SHARD" =~ ^[0-2]$ ]] || { echo "invalid shard: $SHARD" >&2; exit 2; }

PROJECT=/projects/bgqe/rpolonsky/dcrl
EXPO="$PROJECT/third_party/expo-ft"
RUN=/projects/bgqe/rpolonsky/dcrl/sharder/training/sft_full90_eval_task28_existing_2k_20261103
OUT=/work/hdd/bgqe/rpolonsky/dcrlcheckpoints/sft_full90_eval_task28_existing_2k_20261103
EVALUATOR=/projects/bgqe/rpolonsky/dcrl/sharder/training/sft_full90_eval_task28_2k_20261103/evaluate_retention_latest.py
CHECKPOINT=/work/hdd/bgqe/rpolonsky/dcrlcheckpoints/sft_retention_scope_20261103/task_28_existing/latest_checkpoints/2000
ASSETS=/work/hdd/bgqe/rpolonsky/dcrlcheckpoints/sft_retention_scope_20261103/task_28_existing/student_assets
RUNTIME="/tmp/rpolonsky-sft90-task28-existing-${SHARD}"

export PYTHONPATH="$PROJECT:$EXPO/expo_ft/agents/vla/openpi/src:$EXPO/expo_ft/agents/vla/openpi/packages/openpi-client/src:$PROJECT/.deps/LIBERO${PYTHONPATH:+:$PYTHONPATH}"
export LIBERO_CONFIG_PATH="$PROJECT/sharder/training/runtime/libero_config"
export OPENPI_DATA_HOME=/work/hdd/bgqe/rpolonsky/dcrlcheckpoints/openpi_cache
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
export JAX_COMPILATION_CACHE_DIR="$RUNTIME/jax_cache" TMPDIR="$RUNTIME/tmp"
mkdir -p "$RUNTIME/jax_cache" "$RUNTIME/tmp" "$OUT/results" "$RUN/logs"

cd "$PROJECT"
exec >>"$RUN/logs/shard_${SHARD}.log" 2>&1
echo "START task=28 checkpoint=2000 shard=$SHARD/3 at $(date -Is)"
"$EXPO/.venv/bin/python" "$EVALUATOR" \
  --mode sft_pi --checkpoint "$CHECKPOINT" --assets-dir "$ASSETS" \
  --suite libero_90 --shard-index "$SHARD" --num-shards 3 \
  --episodes 20 --eval-seed 7 --seed 42 --image-size 224 \
  --action-horizon 10 --replan-steps 5 --fsdp-devices 1 \
  --output-dir "$OUT/results"
echo "DONE task=28 checkpoint=2000 shard=$SHARD/3 at $(date -Is)"

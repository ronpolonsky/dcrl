#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0

ARM=${1:?usage: run_eval.sh raw|clipped}
case "$ARM" in
  raw|clipped) ;;
  *) echo "unknown arm: $ARM" >&2; exit 2 ;;
esac

PROJECT=/projects/bgqe/rpolonsky/dcrl
RUN=/projects/bgqe/rpolonsky/dcrl/sharder/training/sft_full90_eval_task28_actionbounds_20261103
AUDIT=/projects/bgqe/rpolonsky/dcrl/sharder/training/sft_analysis_20261103
EVALUATOR=/projects/bgqe/rpolonsky/dcrl/sharder/training/sft_full90_eval_task28_2k_20261103/evaluate_retention_latest.py
CHECKPOINT=/work/hdd/bgqe/rpolonsky/dcrlcheckpoints/sft_task28_actionbounds_20261103/${ARM}/latest_checkpoints/1500
ASSETS=/work/hdd/bgqe/rpolonsky/dcrlcheckpoints/sft_task28_actionbounds_20261103/${ARM}/student_assets
OUT=/work/hdd/bgqe/rpolonsky/dcrlcheckpoints/sft_full90_eval_task28_actionbounds_20261103/${ARM}/results
RUNTIME=/tmp/rpolonsky-sft90-actionbounds-${ARM}

export PYTHONPATH="$RUN:$AUDIT:$PROJECT:$PROJECT/third_party/expo-ft/expo_ft/agents/vla/openpi/src:$PROJECT/third_party/expo-ft/expo_ft/agents/vla/openpi/packages/openpi-client/src:$PROJECT/.deps/LIBERO${PYTHONPATH:+:$PYTHONPATH}"
export LIBERO_CONFIG_PATH="$PROJECT/sharder/training/runtime/libero_config"
export OPENPI_DATA_HOME=/work/hdd/bgqe/rpolonsky/dcrlcheckpoints/openpi_cache
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
export JAX_COMPILATION_CACHE_DIR="$RUNTIME/jax_cache" TMPDIR="$RUNTIME/tmp"
mkdir -p "$RUNTIME/jax_cache" "$RUNTIME/tmp" "$OUT" "$RUN/logs"

cd "$PROJECT"
exec >"$RUN/logs/${ARM}.log" 2>&1
echo "START arm=$ARM checkpoint=$CHECKPOINT at $(date -Is)"
"$PROJECT/third_party/expo-ft/.venv/bin/python" "$EVALUATOR" \
  --mode sft_pi \
  --checkpoint "$CHECKPOINT" \
  --assets-dir "$ASSETS" \
  --suite libero_90 --task-ids all \
  --episodes 20 --eval-seed 7 --seed 42 --image-size 224 \
  --action-horizon 10 --replan-steps 5 --fsdp-devices 1 \
  --output-dir "$OUT"
echo "DONE arm=$ARM at $(date -Is)"

#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0

# Evaluation-only continuation of the checkpointed Task-29 reproduction.
# This intentionally does not rerun SFT; it restores the verified step-200
# checkpoint produced by run_task29_demo0625_step200_checkpointed_full90_20261106.sh.
PROJECT=/projects/bgqe/rpolonsky/dcrl
AUDIT=/projects/bgqe/rpolonsky/dcrl/sharder/training/sft_analysis_20261103
OUT=/tmp/usertmp/rpolonsky/sft_retention_diagnosis_20261106/task_29_demo0625_step200_checkpointed_validation
FULL90_OUT="$OUT/full90"
RUNTIME=/tmp/usertmp/rpolonsky/sft_runtime_task29_demo0625_step200_checkpointed_validation

export PYTHONPATH="$AUDIT:$AUDIT/..:$PROJECT:$PROJECT/third_party/expo-ft/expo_ft/agents/vla/openpi/src:$PROJECT/third_party/expo-ft/expo_ft/agents/vla/openpi/packages/openpi-client/src:$PROJECT/.deps/LIBERO"
export LIBERO_CONFIG_PATH="$PROJECT/sharder/training/runtime/libero_config"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false TOKENIZERS_PARALLELISM=false
export OPENPI_DATA_HOME=/work/hdd/bgqe/rpolonsky/dcrlcheckpoints/openpi_cache
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
export JAX_COMPILATION_CACHE_DIR="$RUNTIME/jax_cache" TMPDIR="$RUNTIME/tmp"
mkdir -p "$FULL90_OUT" "$TMPDIR" "$JAX_COMPILATION_CACHE_DIR"

cd "$PROJECT"
echo "START FULL90_EVAL_ONLY task29_demo0625_step200 $(date --iso-8601=seconds)"
"$PROJECT/third_party/expo-ft/.venv/bin/python" "$AUDIT/full90_audit_eval.py" \
  --mode sft_pi --checkpoint "$OUT/latest_checkpoints" \
  --assets-dir "$OUT/student_assets" --suite libero_90 --task-ids all \
  --shard-index 0 --num-shards 1 --episodes 20 --eval-seed 7 --seed 42 \
  --image-size 224 --action-horizon 10 --replan-steps 5 --fsdp-devices 1 \
  --output-dir "$FULL90_OUT"
echo "FULL90_DONE $(date --iso-8601=seconds)"

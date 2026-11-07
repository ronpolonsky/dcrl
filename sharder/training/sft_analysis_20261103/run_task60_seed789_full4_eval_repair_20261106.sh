#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0

PROJECT=/projects/bgqe/rpolonsky/dcrl
AUDIT=/projects/bgqe/rpolonsky/dcrl/sharder/training/sft_analysis_20261103
ROOT=/tmp/usertmp/rpolonsky/sft_retention_diagnosis_20261106/task_60_pure_seed789_update150
EVAL_OUT=/tmp/usertmp/rpolonsky/sft_retention_diagnosis_20261106/task_60_pure_seed789_update150_eval_repaired
RUNTIME=/tmp/usertmp/rpolonsky/sft_runtime_task60_seed789_eval_repaired

export PYTHONPATH="/projects/bgqe/rpolonsky/dcrl/sharder/training:$AUDIT:$PROJECT:$PROJECT/third_party/expo-ft/expo_ft/agents/vla/openpi/src:$PROJECT/third_party/expo-ft/expo_ft/agents/vla/openpi/packages/openpi-client/src:$PROJECT/.deps/LIBERO"
export LIBERO_CONFIG_PATH="$PROJECT/sharder/training/runtime/libero_config"
export OPENPI_DATA_HOME=/work/hdd/bgqe/rpolonsky/dcrlcheckpoints/openpi_cache
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
export JAX_COMPILATION_CACHE_DIR="$RUNTIME/jax_cache" TMPDIR="$RUNTIME/tmp"
mkdir -p "$RUNTIME" "$JAX_COMPILATION_CACHE_DIR" "$TMPDIR" "$EVAL_OUT"
test -d "$ROOT/checkpoints/150"
cd "$PROJECT"
PY="$PROJECT/third_party/expo-ft/.venv/bin/python"
exec "$PY" "$AUDIT/full90_audit_eval.py" \
  --mode sft_pi --checkpoint "$ROOT/checkpoints" \
  --assets-dir "$ROOT/student_assets" --suite libero_90 --task-ids 2,28,29,60 \
  --episodes 20 --eval-seed 7 --seed 42 --image-size 224 \
  --action-horizon 10 --replan-steps 5 --fsdp-devices 1 --output-dir "$EVAL_OUT"

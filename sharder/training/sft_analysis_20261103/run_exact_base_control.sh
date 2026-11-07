#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0

# Isolated evaluator-only control.  This does not train, alter checkpoints, or
# use the production evaluator.  Run only after a reserved experimental GPU is
# free; the current scope arms must remain untouched until then.
PROJECT=/projects/bgqe/rpolonsky/dcrl
AUDIT=/projects/bgqe/rpolonsky/dcrl/sharder/training/sft_analysis_20261103
BASE=/projects/bgqe/rpolonsky/dcrlcheckpoints/openpi_cache/openpi-assets/checkpoints/pi05_libero
OUTPUT=/work/hdd/bgqe/rpolonsky/dcrlcheckpoints/sft_exact_base_control_20261103
RUNTIME=/tmp/rpolonsky-sft-exact-base

export PYTHONPATH="$AUDIT:$PROJECT:$PROJECT/third_party/expo-ft/expo_ft/agents/vla/openpi/src:$PROJECT/third_party/expo-ft/expo_ft/agents/vla/openpi/packages/openpi-client/src:$PROJECT/.deps/LIBERO"
export LIBERO_CONFIG_PATH="$PROJECT/sharder/training/runtime/libero_config"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false TOKENIZERS_PARALLELISM=false
export OPENPI_DATA_HOME=/work/hdd/bgqe/rpolonsky/dcrlcheckpoints/openpi_cache
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
export JAX_COMPILATION_CACHE_DIR="$RUNTIME/jax_cache" TMPDIR="$RUNTIME/tmp"
mkdir -p "$OUTPUT" "$TMPDIR" "$JAX_COMPILATION_CACHE_DIR"

cd "$PROJECT"
exec "$PROJECT/third_party/expo-ft/.venv/bin/python" "$AUDIT/evaluate_retention.py" \
  --mode official_pi05 --checkpoint "$BASE" \
  --assets-dir "$BASE/assets" --suite libero_90 \
  --task-ids 2,28,29,60 --episodes 20 --eval-seed 7 --seed 42 \
  --image-size 224 --action-horizon 10 --replan-steps 5 \
  --output-dir "$OUTPUT"

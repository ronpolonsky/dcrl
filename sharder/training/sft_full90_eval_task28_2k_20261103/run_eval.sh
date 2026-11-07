#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0

KIND=${1:?usage: run_eval.sh probe | shard INDEX}
PROJECT=/projects/bgqe/rpolonsky/dcrl
AUDIT=/projects/bgqe/rpolonsky/dcrl/sharder/training/sft_analysis_20261103
RUN=/projects/bgqe/rpolonsky/dcrl/sharder/training/sft_full90_eval_task28_2k_20261103
BUNDLE="$RUN/checkpoint_bundle"
CHECKPOINT=/work/hdd/bgqe/rpolonsky/dcrlcheckpoints/sft_retention_scope_20261103/task_28_lora_only/latest_checkpoints/2000
EVALUATOR="$RUN/evaluate_retention_latest.py"
OUTPUT="$RUN/results"
if [[ "$KIND" == probe || ( "$KIND" == shard && "${2:-}" == 2 ) ]]; then
  # Reuse the probe's node-local JAX executable cache for shard 2, which will
  # run on the same A100 after the one-task restore/evaluation probe exits.
  RUNTIME=/tmp/rpolonsky-sft90-task28-probe-probe
else
  RUNTIME="/tmp/rpolonsky-sft90-task28-${KIND}-${2:-probe}"
fi

export PYTHONPATH="$AUDIT:$PROJECT:$PROJECT/third_party/expo-ft/expo_ft/agents/vla/openpi/src:$PROJECT/third_party/expo-ft/expo_ft/agents/vla/openpi/packages/openpi-client/src:$PROJECT/.deps/LIBERO"
export LIBERO_CONFIG_PATH="$PROJECT/sharder/training/runtime/libero_config"
export OPENPI_DATA_HOME=/work/hdd/bgqe/rpolonsky/dcrlcheckpoints/openpi_cache
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
export JAX_COMPILATION_CACHE_DIR="$RUNTIME/jax_cache" TMPDIR="$RUNTIME/tmp"
mkdir -p "$JAX_COMPILATION_CACHE_DIR" "$TMPDIR" "$OUTPUT"
cd "$PROJECT"

COMMON=(
  --mode sft_pi
  --checkpoint "$CHECKPOINT"
  --assets-dir "$BUNDLE/student_assets"
  --suite libero_90
  --episodes 20
  --eval-seed 7
  --seed 42
  --fsdp-devices 1
)

if [[ "$KIND" == probe ]]; then
  exec "$PROJECT/third_party/expo-ft/.venv/bin/python" "$EVALUATOR" \
    "${COMMON[@]}" --task-id 28 --output-dir "$RUN/probe"
elif [[ "$KIND" == shard ]]; then
  SHARD=${2:?shard index required}
  exec "$PROJECT/third_party/expo-ft/.venv/bin/python" "$EVALUATOR" \
    "${COMMON[@]}" --shard-index "$SHARD" --num-shards 3 --output-dir "$OUTPUT"
else
  echo "unknown eval kind: $KIND" >&2
  exit 2
fi

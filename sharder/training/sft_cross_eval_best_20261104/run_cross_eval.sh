#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0

SOURCE_TASK=${1:?usage: run_cross_eval.sh source_task checkpoint_step allocation_label task_ids}
CHECKPOINT_STEP=${2:?}
LABEL=${3:?}
TASK_IDS=${4:?}
case "$SOURCE_TASK:$CHECKPOINT_STEP:$LABEL" in
  2:3000:task2) ;;
  60:1000:task60) ;;
  *) echo "unexpected checkpoint selection" >&2; exit 2 ;;
esac

PROJECT=/projects/bgqe/rpolonsky/dcrl
RUN=/projects/bgqe/rpolonsky/dcrl/sharder/training/sft_cross_eval_best_20261104
printf -v SOURCE_TASK_PADDED '%02d' "$SOURCE_TASK"
ROOT=/work/hdd/bgqe/rpolonsky/dcrlcheckpoints/sft_task02_60_bestcheckpoint_20261103/task_${SOURCE_TASK_PADDED}_small_expo_only
CHECKPOINT=$ROOT/checkpoints/$CHECKPOINT_STEP
ASSETS=$ROOT/student_assets
OUT=$ROOT/cross_task_evals/best_step_${CHECKPOINT_STEP}
RUNTIME=/tmp/rpolonsky-sft-cross-eval-${LABEL}
EVALUATOR=$RUN/evaluate_retention.py

export PYTHONPATH="$RUN:/projects/bgqe/rpolonsky/dcrl/sharder/training/sft_task02_60_earlystop_20261103:$PROJECT:$PROJECT/third_party/expo-ft/expo_ft/agents/vla/openpi/src:$PROJECT/third_party/expo-ft/expo_ft/agents/vla/openpi/packages/openpi-client/src:$PROJECT/.deps/LIBERO${PYTHONPATH:+:$PYTHONPATH}"
export LIBERO_CONFIG_PATH="$PROJECT/sharder/training/runtime/libero_config"
export OPENPI_DATA_HOME=/work/hdd/bgqe/rpolonsky/dcrlcheckpoints/openpi_cache
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
export JAX_COMPILATION_CACHE_DIR="$RUNTIME/jax_cache" TMPDIR="$RUNTIME/tmp"
mkdir -p "$RUNTIME/jax_cache" "$RUNTIME/tmp" "$OUT" "$RUN/logs"

cd "$PROJECT"
exec >>"$RUN/logs/${LABEL}.log" 2>&1
echo "START label=$LABEL source_task=$SOURCE_TASK checkpoint_step=$CHECKPOINT_STEP tasks=$TASK_IDS at $(date -Is)"
"$PROJECT/third_party/expo-ft/.venv/bin/python" "$EVALUATOR" \
  --mode sft_pi \
  --checkpoint "$CHECKPOINT" \
  --assets-dir "$ASSETS" \
  --suite libero_90 \
  --task-ids "$TASK_IDS" \
  --episodes 20 \
  --eval-seed 7 \
  --seed 42 \
  --image-size 224 \
  --action-horizon 10 \
  --replan-steps 5 \
  --fsdp-devices 1 \
  --output-dir "$OUT"
echo "DONE label=$LABEL at $(date -Is)"

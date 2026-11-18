#!/usr/bin/env bash
set -euo pipefail

# Relaunch only the difficult task-28/task-60 SFT runs.  The output path is
# intentionally the existing task-specific directory: the staged checkpoint
# writer replaces only that directory's checkpoint tree after a complete new
# checkpoint has been written.
VARIANT=${1:?usage: $0 small|small_dist 28|60}
TASK=${2:?usage: $0 small|small_dist 28|60}

case "$VARIANT" in
  small)
    TEACHER_ROOT=/work/hdd/bgqe/rpolonsky/dcrlcheckpoints/tuned_small_rl_10_5_offline50k_online500k/libero_90
    WANDB_PROJECT=sft_from_hyb_small_20k
    ;;
  small_dist)
    TEACHER_ROOT=/work/hdd/bgqe/rpolonsky/dcrlcheckpoints/small_learner_with_dist/libero_90
    WANDB_PROJECT=sft_from_hyb_small_dist_20k
    ;;
  *)
    echo "This relaunch script only accepts small or small_dist" >&2
    exit 2
    ;;
esac

case "$TASK" in
  28) TASK_PAD=28 ;;
  60) TASK_PAD=60 ;;
  *) echo "This relaunch script only accepts task 28 or task 60" >&2; exit 2 ;;
esac

PROJECT=/projects/bgqe/rpolonsky/dcrl
EXPO="$PROJECT/third_party/expo-ft"
BASE=/projects/bgqe/rpolonsky/dcrlcheckpoints/openpi_cache/openpi-assets/checkpoints/pi05_libero
OUTPUT_ROOT=/work/hdd/bgqe/rpolonsky/dcrlcheckpoints/sft_from_hybrid_tuned_20k_20261116
OUTPUT="$OUTPUT_ROOT/$VARIANT/task_$TASK_PAD"
TEACHER_DIR="$TEACHER_ROOT/task_$TASK_PAD"
TEACHER_STEP=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["step"])' "$TEACHER_DIR/best_record.json")
SMALL_CHECKPOINT="$TEACHER_DIR/checkpoints_best/$TEACHER_STEP"
RUNTIME="/tmp/rpolonsky-hybrid-sft-relaunch-${VARIANT}-task${TASK_PAD}"

[[ -d "$OUTPUT" ]] || { echo "existing task output is missing: $OUTPUT" >&2; exit 3; }
[[ -f "$OUTPUT/pi_base_data/success/hybrid_teacher_chunks.hdf5" ]] || {
  echo "missing saved successful hybrid targets: $OUTPUT/pi_base_data/success/hybrid_teacher_chunks.hdf5" >&2
  exit 4
}
[[ -d "$SMALL_CHECKPOINT/agent" && -d "$SMALL_CHECKPOINT/params" ]] || {
  echo "missing teacher checkpoint: $SMALL_CHECKPOINT" >&2
  exit 5
}
[[ -f "$BASE/params/manifest.ocdbt" && -f "$BASE/assets/physical-intelligence/libero/norm_stats.json" ]] || {
  echo "missing official Pi0.5 LIBERO assets: $BASE" >&2
  exit 6
}
[[ -f /work/hdd/bgqe/rpolonsky/.env ]] || { echo "missing W&B environment file" >&2; exit 7; }
[[ -d "$OUTPUT/checkpoints" ]] || { echo "existing checkpoint root is missing: $OUTPUT/checkpoints" >&2; exit 8; }

# Preserve old metrics/evals/manifests for comparison.  The checkpoint tree,
# pi_base_data, and student assets are deliberately not moved here.
ARCHIVE="$OUTPUT/archive_previous_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$ARCHIVE"
for item in metrics.jsonl evals sft_config.json launcher.log wandb_run_id; do
  if [[ -e "$OUTPUT/$item" || -L "$OUTPUT/$item" ]]; then
    mv "$OUTPUT/$item" "$ARCHIVE/$item"
  fi
done

mkdir -p "$OUTPUT_ROOT" "$RUNTIME/tmp" "$RUNTIME/jax_cache" "$RUNTIME/wandb-cache" "$RUNTIME/wandb-data" "$OUTPUT/wandb"

export PYTHONPATH="$PROJECT:$EXPO/expo_ft/agents/vla/openpi/src:$EXPO/expo_ft/agents/vla/openpi/packages/openpi-client/src:$PROJECT/.deps/LIBERO${PYTHONPATH:+:$PYTHONPATH}"
export LIBERO_CONFIG_PATH="$PROJECT/sharder/training/runtime/libero_config"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false TOKENIZERS_PARALLELISM=false
export OPENPI_DATA_HOME=/work/hdd/bgqe/rpolonsky/dcrlcheckpoints/openpi_cache
export WANDB_BASE_URL=https://forge.coreweave.com/api/wandb
export SFT_STAGE_ROOT=/projects/bgqe/rpolonsky/dcrlcheckpoints/sft_hybrid_20k_checkpoint_stage
export SFT_REPLACE_OLD_CHECKPOINT_DIR="$OUTPUT/checkpoints"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
export JAX_COMPILATION_CACHE_DIR="$RUNTIME/jax_cache" TMPDIR="$RUNTIME/tmp"
export WANDB_DIR="$OUTPUT/wandb" WANDB_CACHE_DIR="$RUNTIME/wandb-cache" WANDB_DATA_DIR="$RUNTIME/wandb-data"
set -a
source /work/hdd/bgqe/rpolonsky/.env
set +a
: "${WANDB_API_KEY:?Missing WANDB_API_KEY}"

exec > >(tee -a "$OUTPUT/launcher.log") 2>&1
echo "START variant=$VARIANT task=$TASK_PAD teacher=$SMALL_CHECKPOINT targets=$OUTPUT/pi_base_data/success/hybrid_teacher_chunks.hdf5 steps=20000 eval_every=250 batch=16 demo_ratio=0.5 lr=5e-6 decay_lr=1e-6 warmup=2000 decay_steps=10000"
echo "CHECKPOINT_REPLACEMENT_SCOPE=$OUTPUT/checkpoints"
cd "$PROJECT"
exec "$EXPO/.venv/bin/python" -m dcrl.train_pi_sft_relaunch_hard_tasks \
  --phase train --collection-source pi05_small_expo_editor \
  --small-expo-checkpoint "$SMALL_CHECKPOINT" --pi05-checkpoint "$BASE" \
  --student-checkpoint "$BASE" --source-checkpoint "$BASE" \
  --norm-stats "$BASE/assets/physical-intelligence/libero/norm_stats.json" \
  --source-suite libero_90 --source-task-id "$TASK_PAD" \
  --output-dir "$OUTPUT" \
  --collection-episodes 400 --collection-successes 100 --collection-failures 0 \
  --num-sft-steps 20000 --batch-size 16 --finetune-mode lora \
  --trainable-scope lora_only --zero-lora-b \
  --official-demo-ratio 0.5 --official-demo-root "$PROJECT/datasets" \
  --learning-rate 5e-6 --decay-lr 1e-6 --warmup-steps 2000 --lr-decay-steps 10000 \
  --learner-config configs.sft.my_expo_libero_relaunch \
  --eval-every 250 --eval-episodes 20 --checkpoint-every 250 \
  --seed 42 --eval-seed 7 \
  --wandb-project "$WANDB_PROJECT" --wandb-entity distill_rl \
  --wandb-name "${VARIANT}_task${TASK_PAD}_hybrid_sft_relaunch_20k_lr5e-6" \
  --wandb-mode online --wandb-log-every 10

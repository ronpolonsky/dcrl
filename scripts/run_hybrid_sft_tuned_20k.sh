#!/usr/bin/env bash
set -euo pipefail

# One task/teacher per invocation. Reuse the completed 100-success dataset.
# Each run stages its full checkpoint on /projects before replacing its old
# checkpoint on /work. All ten writers share one stage lock.
VARIANT=${1:?usage: OUTPUT_ROOT=/persistent/path $0 small|small_dist TASK_ID}
TASK=${2:?usage: OUTPUT_ROOT=/persistent/path $0 small|small_dist TASK_ID}
OUTPUT_ROOT=${OUTPUT_ROOT:-/work/hdd/bgqe/rpolonsky/dcrlcheckpoints/sft_from_hybrid_tuned_20k_20261116}
EVAL_EVERY=${EVAL_EVERY:-250}

PROJECT=/projects/bgqe/rpolonsky/dcrl
EXPO="$PROJECT/third_party/expo-ft"
BASE=/projects/bgqe/rpolonsky/dcrlcheckpoints/openpi_cache/openpi-assets/checkpoints/pi05_libero
OLD_ROOT=/work/hdd/bgqe/rpolonsky/dcrlcheckpoints/sft_from_hybrid_10_20261115

case "$VARIANT" in
  small) TEACHER_ROOT=/work/hdd/bgqe/rpolonsky/dcrlcheckpoints/tuned_small_rl_10_5_offline50k_online500k/libero_90
         WANDB_PROJECT=sft_from_hyb_small_20k ;;
  small_dist) TEACHER_ROOT=/work/hdd/bgqe/rpolonsky/dcrlcheckpoints/small_learner_with_dist/libero_90
              WANDB_PROJECT=sft_from_hyb_small_dist_20k ;;
  *) echo "unknown variant: $VARIANT" >&2; exit 2 ;;
esac
case "$TASK" in
  2|02) TASK_PAD=02; PEAK_LR=7.5e-6 ;;
  28) TASK_PAD=28; PEAK_LR=1e-5 ;;
  29) TASK_PAD=29; PEAK_LR=7.5e-6 ;;
  60) TASK_PAD=60; PEAK_LR=1e-5 ;;
  79) TASK_PAD=79; PEAK_LR=1e-5 ;;
  *) echo "unknown task: $TASK" >&2; exit 2 ;;
esac

OLD="$OLD_ROOT/$VARIANT/task_$TASK_PAD"
OUTPUT="$OUTPUT_ROOT/$VARIANT/task_$TASK_PAD"
TEACHER_DIR="$TEACHER_ROOT/task_$TASK_PAD"
TEACHER_STEP=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["step"])' "$TEACHER_DIR/best_record.json")
SMALL_CHECKPOINT="$TEACHER_DIR/checkpoints_best/$TEACHER_STEP"
RUNTIME="/tmp/rpolonsky-hybrid-sft-20k-${VARIANT}-task${TASK_PAD}"

[[ -f "$OLD/pi_base_data/success/hybrid_teacher_chunks.hdf5" ]] || { echo "missing saved successful trajectories" >&2; exit 3; }
[[ -d "$SMALL_CHECKPOINT/agent" && -d "$SMALL_CHECKPOINT/params" ]] || { echo "missing teacher checkpoint" >&2; exit 4; }
[[ -f "$BASE/params/manifest.ocdbt" && -f "$BASE/assets/physical-intelligence/libero/norm_stats.json" ]] || { echo "missing official Pi0.5 LIBERO assets" >&2; exit 5; }
[[ -f /work/hdd/bgqe/rpolonsky/.env ]] || { echo "missing W&B environment file" >&2; exit 6; }
[[ ! -e "$OUTPUT" ]] || { echo "output already exists: $OUTPUT" >&2; exit 7; }
[[ "$EVAL_EVERY" =~ ^[1-9][0-9]*$ ]] || { echo "EVAL_EVERY must be positive" >&2; exit 8; }
[[ -d "$OLD/checkpoints" ]] || { echo "old checkpoint directory missing: $OLD/checkpoints" >&2; exit 9; }
mkdir -p "$OUTPUT_ROOT"
mkdir -p "$OUTPUT" "$RUNTIME/tmp" "$RUNTIME/jax_cache" "$RUNTIME/wandb-cache" "$RUNTIME/wandb-data"
ln -s "$OLD/pi_base_data" "$OUTPUT/pi_base_data"

export PYTHONPATH="$PROJECT:$EXPO/expo_ft/agents/vla/openpi/src:$EXPO/expo_ft/agents/vla/openpi/packages/openpi-client/src:$PROJECT/.deps/LIBERO${PYTHONPATH:+:$PYTHONPATH}"
export LIBERO_CONFIG_PATH="$PROJECT/sharder/training/runtime/libero_config"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false TOKENIZERS_PARALLELISM=false
export OPENPI_DATA_HOME=/work/hdd/bgqe/rpolonsky/dcrlcheckpoints/openpi_cache
export WANDB_BASE_URL=https://forge.coreweave.com/api/wandb
export SFT_STAGE_ROOT=/projects/bgqe/rpolonsky/dcrlcheckpoints/sft_hybrid_20k_checkpoint_stage
export SFT_REPLACE_OLD_CHECKPOINT_DIR="$OLD/checkpoints"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
export JAX_COMPILATION_CACHE_DIR="$RUNTIME/jax_cache" TMPDIR="$RUNTIME/tmp"
export WANDB_DIR="$OUTPUT/wandb" WANDB_CACHE_DIR="$RUNTIME/wandb-cache" WANDB_DATA_DIR="$RUNTIME/wandb-data"
mkdir -p "$WANDB_DIR"
set -a
source /work/hdd/bgqe/rpolonsky/.env
set +a
: "${WANDB_API_KEY:?Missing WANDB_API_KEY}"

exec > >(tee -a "$OUTPUT/launcher.log") 2>&1
echo "START variant=$VARIANT task=$TASK_PAD teacher=$SMALL_CHECKPOINT steps=20000 eval_every=$EVAL_EVERY lr=$PEAK_LR"
cd "$PROJECT"
exec "$EXPO/.venv/bin/python" -m dcrl.train_pi_sft_tuned_20k \
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
  --learning-rate "$PEAK_LR" --warmup-steps 1000 --lr-decay-steps 20000 \
  --eval-every "$EVAL_EVERY" --eval-episodes 20 --checkpoint-every "$EVAL_EVERY" \
  --seed 42 --eval-seed 7 \
  --wandb-project "$WANDB_PROJECT" --wandb-entity distill_rl \
  --wandb-name "${VARIANT}_task${TASK_PAD}_hybrid_sft_20k" \
  --wandb-mode online --wandb-log-every 10

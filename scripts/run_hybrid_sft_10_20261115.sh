#!/usr/bin/env bash
set -euo pipefail

# Launch one isolated hybrid SFT run.  The compact checkpoint is used only as
# the editor/critic during target collection; the student is official Pi0.5.

VARIANT=${1:?usage: run_hybrid_sft_10_20261115.sh small|small_dist TASK_ID}
TASK=${2:?usage: run_hybrid_sft_10_20261115.sh small|small_dist TASK_ID}

PROJECT=/projects/bgqe/rpolonsky/dcrl
EXPO="$PROJECT/third_party/expo-ft"
BASE=/projects/bgqe/rpolonsky/dcrlcheckpoints/openpi_cache/openpi-assets/checkpoints/pi05_libero
OUTPUT_ROOT=/work/hdd/bgqe/rpolonsky/dcrlcheckpoints/sft_from_hybrid_10_20261115

case "$VARIANT" in
  small)
    TEACHER_ROOT=/work/hdd/bgqe/rpolonsky/dcrlcheckpoints/tuned_small_rl_10_5_offline50k_online500k/libero_90
    WANDB_PROJECT=sft_from_hyb_small
    VARIANT_DIR=small
    ;;
  small_dist)
    TEACHER_ROOT=/work/hdd/bgqe/rpolonsky/dcrlcheckpoints/small_learner_with_dist/libero_90
    WANDB_PROJECT='sft_from_hyb_small+dist'
    VARIANT_DIR=small_dist
    ;;
  *)
    echo "unknown variant: $VARIANT (expected small or small_dist)" >&2
    exit 2
    ;;
esac

case "$TASK" in
  2|02)
    TASK_PAD=02
    NUM_STEPS=3000
    # Controlled task-2 change: mix official demonstrations because the prior
    # pure-target task-2 SFT arm underperformed on retention.
    OFFICIAL_DEMO_RATIO=0.5
    ;;
  28)
    TASK_PAD=28
    NUM_STEPS=2000
    OFFICIAL_DEMO_RATIO=0.0
    ;;
  29)
    TASK_PAD=29
    NUM_STEPS=1500
    OFFICIAL_DEMO_RATIO=0.5
    ;;
  60)
    TASK_PAD=60
    NUM_STEPS=1000
    OFFICIAL_DEMO_RATIO=0.0
    ;;
  79)
    TASK_PAD=79
    NUM_STEPS=1500
    OFFICIAL_DEMO_RATIO=0.0
    ;;
  *)
    echo "unknown task: $TASK (expected 2, 28, 29, 60, or 79)" >&2
    exit 2
    ;;
esac

TEACHER_DIR="$TEACHER_ROOT/task_$TASK_PAD"
TEACHER_STEP=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["step"])' "$TEACHER_DIR/best_record.json")
SMALL_CHECKPOINT="$TEACHER_DIR/checkpoints_best/$TEACHER_STEP"
OUTPUT="$OUTPUT_ROOT/$VARIANT_DIR/task_$TASK_PAD"
RUNTIME="/tmp/rpolonsky-hybrid-sft-${VARIANT_DIR}-task${TASK_PAD}-20261115"

[[ -d "$SMALL_CHECKPOINT/agent" && -d "$SMALL_CHECKPOINT/params" ]] || {
  echo "missing compact checkpoint: $SMALL_CHECKPOINT" >&2
  exit 3
}
[[ -f "$BASE/params/manifest.ocdbt" ]] || {
  echo "missing official Pi0.5 params: $BASE/params" >&2
  exit 4
}
[[ -f "$BASE/assets/physical-intelligence/libero/norm_stats.json" ]] || {
  echo "missing official Pi0.5 LIBERO norm stats" >&2
  exit 5
}
if [[ -e "$OUTPUT/sft_config.json" || -e "$OUTPUT/pi_base_data/hybrid_teacher_chunks.hdf5" ]]; then
  echo "refusing to overwrite existing run output: $OUTPUT" >&2
  exit 6
fi

export PYTHONPATH="$PROJECT:$EXPO/expo_ft/agents/vla/openpi/src:$EXPO/expo_ft/agents/vla/openpi/packages/openpi-client/src:$PROJECT/.deps/LIBERO${PYTHONPATH:+:$PYTHONPATH}"
export LIBERO_CONFIG_PATH="$PROJECT/sharder/training/runtime/libero_config"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TOKENIZERS_PARALLELISM=false
export OPENPI_DATA_HOME=/work/hdd/bgqe/rpolonsky/dcrlcheckpoints/openpi_cache
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
export JAX_COMPILATION_CACHE_DIR="$RUNTIME/jax_cache"
export TMPDIR="$RUNTIME/tmp"
export WANDB_DIR="$OUTPUT/wandb"
export WANDB_CACHE_DIR="$RUNTIME/wandb-cache"
export WANDB_DATA_DIR="$RUNTIME/wandb-data"
mkdir -p "$OUTPUT" "$RUNTIME" "$TMPDIR" "$JAX_COMPILATION_CACHE_DIR" \
  "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_DATA_DIR"

set -a
source /work/hdd/bgqe/rpolonsky/.env
set +a
: "${WANDB_API_KEY:?WANDB_API_KEY missing from /work/hdd/bgqe/rpolonsky/.env}"

if [[ "$OFFICIAL_DEMO_RATIO" != 0.0 ]]; then
  DEMO_ARGS=(--official-demo-root "$PROJECT/datasets")
else
  DEMO_ARGS=()
fi

exec > >(tee -a "$OUTPUT/launcher.log") 2>&1
echo "START variant=$VARIANT task=$TASK_PAD teacher_step=$TEACHER_STEP"
echo "teacher=$SMALL_CHECKPOINT"
echo "output=$OUTPUT"
echo "collection=100 successful trajectories; eval_every=250; steps=$NUM_STEPS"
echo "wandb_project=$WANDB_PROJECT"

cd "$PROJECT"
exec "$EXPO/.venv/bin/python" -m dcrl.train_pi_sft_agnostic \
  --phase all \
  --collection-source pi05_small_expo_editor \
  --small-expo-checkpoint "$SMALL_CHECKPOINT" \
  --pi05-checkpoint "$BASE" \
  --student-checkpoint "$BASE" \
  --source-checkpoint "$BASE" \
  --norm-stats "$BASE/assets/physical-intelligence/libero/norm_stats.json" \
  --source-suite libero_90 --source-task-id "$TASK_PAD" \
  --output-dir "$OUTPUT" \
  --collection-episodes 400 --collection-successes 100 --collection-failures 0 \
  --num-sft-steps "$NUM_STEPS" --batch-size 16 --finetune-mode lora \
  --trainable-scope lora_only --zero-lora-b \
  --official-demo-ratio "$OFFICIAL_DEMO_RATIO" "${DEMO_ARGS[@]}" \
  --learning-rate 5e-5 --warmup-steps 10000 --lr-decay-steps 1000000 \
  --eval-every 250 --eval-episodes 20 --checkpoint-every 250 \
  --seed 42 --eval-seed 7 \
  --wandb-project "$WANDB_PROJECT" --wandb-entity distill_rl \
  --wandb-name "${VARIANT_DIR}_task${TASK_PAD}_hybrid_sft_100success_20261115" \
  --wandb-mode online --wandb-log-every 10

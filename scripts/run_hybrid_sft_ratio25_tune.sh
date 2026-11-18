#!/usr/bin/env bash
set -euo pipefail

VARIANT=${1:?small or small_dist required}
TASK=${2:?task 28 or 60 required}
case "$VARIANT:$TASK" in
  small:28|small:60|small_dist:60) ;;
  *) echo "Only the three unsolved 200-success runs are supported" >&2; exit 2 ;;
esac

PROJECT=/projects/bgqe/rpolonsky/dcrl
EXPO="$PROJECT/third_party/expo-ft"
BASE=/projects/bgqe/rpolonsky/dcrlcheckpoints/openpi_cache/openpi-assets/checkpoints/pi05_libero
PREVIOUS=/work/hdd/bgqe/rpolonsky/dcrlcheckpoints/sft_hybrid_more_data_20261116/$VARIANT/task_$TASK
OUTPUT=/work/hdd/bgqe/rpolonsky/dcrlcheckpoints/sft_hybrid_ratio25_20261117/$VARIANT/task_$TASK
RUNTIME=/tmp/rpolonsky-sft-ratio25-${VARIANT}-task${TASK}
case "$VARIANT" in
  small) TEACHER_ROOT=/work/hdd/bgqe/rpolonsky/dcrlcheckpoints/tuned_small_rl_10_5_offline50k_online500k/libero_90 ;;
  small_dist) TEACHER_ROOT=/work/hdd/bgqe/rpolonsky/dcrlcheckpoints/small_learner_with_dist/libero_90 ;;
esac
TEACHER_DIR="$TEACHER_ROOT/task_$TASK"
TEACHER_STEP=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["step"])' "$TEACHER_DIR/best_record.json")
TEACHER="$TEACHER_DIR/checkpoints_best/$TEACHER_STEP"
[[ -d "$TEACHER/agent" && -d "$TEACHER/params" ]] || { echo "teacher checkpoint missing" >&2; exit 3; }
[[ -f "$PREVIOUS/pi_base_data/success/hybrid_teacher_chunks.hdf5" ]] || { echo "200-success data missing" >&2; exit 4; }
[[ ! -e "$OUTPUT" ]] || { echo "output already exists: $OUTPUT" >&2; exit 5; }
mkdir -p "$OUTPUT" "$RUNTIME/tmp" "$RUNTIME/jax_cache" "$RUNTIME/wandb-cache" "$RUNTIME/wandb-data"
ln -s "$PREVIOUS/pi_base_data" "$OUTPUT/pi_base_data"

export PYTHONPATH="$PROJECT:$EXPO/expo_ft/agents/vla/openpi/src:$EXPO/expo_ft/agents/vla/openpi/packages/openpi-client/src:$PROJECT/.deps/LIBERO${PYTHONPATH:+:$PYTHONPATH}"
export LIBERO_CONFIG_PATH="$PROJECT/sharder/training/runtime/libero_config"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false TOKENIZERS_PARALLELISM=false
export OPENPI_DATA_HOME=/work/hdd/bgqe/rpolonsky/dcrlcheckpoints/openpi_cache
export WANDB_BASE_URL=https://forge.coreweave.com/api/wandb
export SFT_STAGE_ROOT=/projects/bgqe/rpolonsky/dcrlcheckpoints/sft_hybrid_20k_checkpoint_stage
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
export JAX_COMPILATION_CACHE_DIR="$RUNTIME/jax_cache" TMPDIR="$RUNTIME/tmp"
export WANDB_DIR="$OUTPUT/wandb" WANDB_CACHE_DIR="$RUNTIME/wandb-cache" WANDB_DATA_DIR="$RUNTIME/wandb-data"
mkdir -p "$WANDB_DIR"
set -a
source /work/hdd/bgqe/rpolonsky/.env
set +a
: "${WANDB_API_KEY:?Missing WANDB_API_KEY}"

cd "$PROJECT"
exec "$EXPO/.venv/bin/python" -m dcrl.train_pi_sft_relaunch_hard_tasks \
  --phase train --collection-source pi05_small_expo_editor \
  --small-expo-checkpoint "$TEACHER" --pi05-checkpoint "$BASE" \
  --student-checkpoint "$BASE" --source-checkpoint "$BASE" \
  --norm-stats "$BASE/assets/physical-intelligence/libero/norm_stats.json" \
  --source-suite libero_90 --source-task-id "$TASK" --output-dir "$OUTPUT" \
  --collection-episodes 800 --collection-successes 200 --collection-failures 0 \
  --num-sft-steps 4000 --batch-size 16 --finetune-mode lora \
  --trainable-scope lora_only --zero-lora-b \
  --official-demo-ratio 0.25 --official-demo-root "$PROJECT/datasets" \
  --learning-rate 5e-6 --decay-lr 1e-6 --warmup-steps 2000 --lr-decay-steps 10000 \
  --learner-config configs.sft.my_expo_libero_relaunch \
  --eval-every 250 --eval-episodes 20 --checkpoint-every 250 \
  --seed 42 --eval-seed 7 \
  --wandb-project sft_hybrid_ratio25_200success --wandb-entity distill_rl \
  --wandb-name "${VARIANT}_task${TASK}_200success_ratio25_4k" \
  --wandb-mode online --wandb-log-every 10

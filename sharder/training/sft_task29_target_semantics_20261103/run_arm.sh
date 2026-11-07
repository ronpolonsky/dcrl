#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0

ARM=${1:?usage: run_arm.sh full_proposal|executed_backfill}
case "$ARM" in
  full_proposal) TARGET_MODE=full_proposal ;;
  executed_backfill) TARGET_MODE=executed_backfill ;;
  *) echo "unknown arm: $ARM" >&2; exit 2 ;;
esac

PROJECT=/projects/bgqe/rpolonsky/dcrl
EXPO="$PROJECT/third_party/expo-ft"
RUN=/projects/bgqe/rpolonsky/dcrl/sharder/training/sft_task29_target_semantics_20261103
SOURCE=/work/hdd/bgqe/rpolonsky/dcrlcheckpoints/sft_agnostic_small_expo/task_29_teacher_375380_success150_sft200k
TEACHER_DIR=/u/rpolonsky/dcrlcheckpoints/small_expo_rl_10_5_offline50k_online500k_resume_newgpu/libero_90/task_29
TEACHER_STEP=375380
BASE=/projects/bgqe/rpolonsky/dcrlcheckpoints/openpi_cache/openpi-assets/checkpoints/pi05_libero
OUTPUT=/work/hdd/bgqe/rpolonsky/dcrlcheckpoints/sft_task29_target_semantics_20261103/$ARM
RUNTIME=/tmp/rpolonsky-sft-task29-target-${ARM}

[[ -f "$SOURCE/pi_base_data/success/teacher_chunks.hdf5" ]] || { echo "Missing Task 29 Small EXPO chunks" >&2; exit 2; }
[[ -f "$BASE/params/manifest.ocdbt" ]] || { echo "Missing official Pi0.5 LIBERO params" >&2; exit 2; }
[[ -f "$BASE/assets/physical-intelligence/libero/norm_stats.json" ]] || { echo "Missing Pi0.5 LIBERO norm stats" >&2; exit 2; }
[[ ! -e "$OUTPUT/sft_config.json" ]] || { echo "Output already initialized: $OUTPUT" >&2; exit 2; }

export PYTHONPATH="$RUN:$PROJECT:$EXPO/expo_ft/agents/vla/openpi/src:$EXPO/expo_ft/agents/vla/openpi/packages/openpi-client/src:$PROJECT/.deps/LIBERO"
export LIBERO_CONFIG_PATH="$PROJECT/sharder/training/runtime/libero_config"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false TOKENIZERS_PARALLELISM=false
export JAX_COMPILATION_CACHE_DIR="$RUNTIME/jax_cache" TMPDIR="$RUNTIME/tmp"
export OPENPI_DATA_HOME=/work/hdd/bgqe/rpolonsky/dcrlcheckpoints/openpi_cache
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
export WANDB_DIR="$OUTPUT/wandb" WANDB_CACHE_DIR="$RUNTIME/wandb-cache" WANDB_DATA_DIR="$RUNTIME/wandb-data"
mkdir -p "$OUTPUT" "$TMPDIR" "$JAX_COMPILATION_CACHE_DIR" "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_DATA_DIR"
if [[ ! -e "$OUTPUT/pi_base_data" ]]; then ln -s "$SOURCE/pi_base_data" "$OUTPUT/pi_base_data"; fi
exec > >(tee -a "$OUTPUT/launcher.log") 2>&1

set -a
source /work/hdd/bgqe/rpolonsky/.env
set +a
: "${WANDB_API_KEY:?WANDB_API_KEY missing from /work/hdd/bgqe/rpolonsky/.env}"

echo "START arm=$ARM target_mode=$TARGET_MODE task=29 gpu_visible=${CUDA_VISIBLE_DEVICES:-unset} at $(date -Is)"
cd "$PROJECT"
exec "$EXPO/.venv/bin/python" "$RUN/train_sft_retention.py" \
  --phase train --collection-source small_expo \
  --teacher-dir "$TEACHER_DIR" --teacher-step "$TEACHER_STEP" \
  --source-checkpoint "$BASE" --student-checkpoint "$BASE" \
  --norm-stats "$BASE/assets/physical-intelligence/libero/norm_stats.json" \
  --source-suite libero_90 --source-task-id 29 --output-dir "$OUTPUT" \
  --collection-episodes 200 --collection-successes 150 --collection-failures 0 \
  --learner-config audit_sft_config --trainable-scope lora_only --zero-lora-b \
  --target-mode "$TARGET_MODE" --retention-tasks 2 28 29 60 \
  --num-sft-steps 1500 --batch-size 16 --finetune-mode lora \
  --official-demo-ratio 0.5 --official-demo-root "$PROJECT/datasets" \
  --learning-rate 5e-5 --warmup-steps 10000 --lr-decay-steps 1000000 \
  --eval-every 500 --eval-episodes 20 --checkpoint-every 500 \
  --seed 42 --eval-seed 7 \
  --wandb-project dcrl --wandb-entity distill_rl \
  --wandb-name "sft_task29_demo50_${ARM}_1500_seed42" \
  --wandb-mode online --wandb-log-every 10

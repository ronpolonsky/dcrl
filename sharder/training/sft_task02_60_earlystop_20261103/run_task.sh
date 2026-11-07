#!/usr/bin/env bash
set -euo pipefail
TASK=${1:?usage: run_task.sh 2|60}
case "$TASK" in
  2)
    TASK_PAD=02
    TEACHER_STEP=375398
    SOURCE=/work/hdd/bgqe/rpolonsky/dcrlcheckpoints/sft_agnostic_small_expo/task_02_teacher_375398_success150_sft200k
    TEACHER_DIR=/work/hdd/bgqe/rpolonsky/dcrlcheckpoints/small_expo_rl_10_5_offline50k_online500k_resume_work/libero_90/task_02
    ;;
  60)
    TASK_PAD=60
    TEACHER_STEP=450223
    SOURCE=/work/hdd/bgqe/rpolonsky/dcrlcheckpoints/sft_agnostic_small_expo/task_60_teacher_450223_success150_sft200k
    TEACHER_DIR=/work/hdd/bgqe/rpolonsky/dcrlcheckpoints/small_expo_rl_10_5_offline50k_online500k_resume_work/libero_90/task_60
    ;;
  *) echo "task must be 2 or 60" >&2; exit 2 ;;
esac

PROJECT=/projects/bgqe/rpolonsky/dcrl
EXPO="$PROJECT/third_party/expo-ft"
RUN=/projects/bgqe/rpolonsky/dcrl/sharder/training/sft_task02_60_earlystop_20261103
BASE=/projects/bgqe/rpolonsky/dcrlcheckpoints/openpi_cache/openpi-assets/checkpoints/pi05_libero
OUTPUT=/work/hdd/bgqe/rpolonsky/dcrlcheckpoints/sft_task02_60_bestcheckpoint_20261103/task_${TASK_PAD}_small_expo_only
RUNTIME="/tmp/rpolonsky-sft-task${TASK_PAD}-bestcheckpoint"

[[ -f "$SOURCE/pi_base_data/success/teacher_chunks.hdf5" ]] || { echo "missing task-specific Small EXPO targets: $SOURCE" >&2; exit 3; }
[[ -d "$TEACHER_DIR/checkpoints/$TEACHER_STEP" ]] || { echo "missing matching Small EXPO teacher checkpoint" >&2; exit 4; }
[[ -f "$BASE/params/manifest.ocdbt" ]] || { echo "missing official Pi0.5 LIBERO parameters" >&2; exit 5; }
export PYTHONPATH="$RUN:$PROJECT:$EXPO/expo_ft/agents/vla/openpi/src:$EXPO/expo_ft/agents/vla/openpi/packages/openpi-client/src:$PROJECT/.deps/LIBERO${PYTHONPATH:+:$PYTHONPATH}"
export LIBERO_CONFIG_PATH="$PROJECT/sharder/training/runtime/libero_config"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false TOKENIZERS_PARALLELISM=false
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 OPENPI_SKIP_INIT_JIT=1
export OPENPI_DATA_HOME=/work/hdd/bgqe/rpolonsky/dcrlcheckpoints/openpi_cache
export XDG_CACHE_HOME="$RUNTIME/xdg" XDG_CONFIG_HOME="$RUNTIME/xdg_config" TMPDIR="$RUNTIME/tmp"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
export WANDB_DIR="$OUTPUT/wandb" WANDB_CACHE_DIR="$RUNTIME/wandb-cache" WANDB_DATA_DIR="$RUNTIME/wandb-data"
mkdir -p "$OUTPUT" "$TMPDIR" "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_DATA_DIR"
if [[ ! -e "$OUTPUT/pi_base_data" ]]; then
  ln -s "$SOURCE/pi_base_data" "$OUTPUT/pi_base_data"
fi
exec > >(tee -a "$OUTPUT/launcher.log") 2>&1

set -a
source /work/hdd/bgqe/rpolonsky/.env
set +a
: "${WANDB_API_KEY:?WANDB_API_KEY missing from /work/hdd/bgqe/rpolonsky/.env}"

echo "START task=$TASK_PAD teacher_step=$TEACHER_STEP source=Small-EXPO-only max_updates=5000 eval_every=500 save_best_on_improvement stop_at_first_target_100 at $(date -Is)"
cd "$PROJECT"
exec "$EXPO/.venv/bin/python" "$RUN/train_sft_retention.py" \
  --phase train --collection-source small_expo \
  --teacher-dir "$TEACHER_DIR" --teacher-step "$TEACHER_STEP" \
  --source-checkpoint "$BASE" --student-checkpoint "$BASE" \
  --norm-stats "$BASE/assets/physical-intelligence/libero/norm_stats.json" \
  --source-suite libero_90 --source-task-id "$TASK" --output-dir "$OUTPUT" \
  --collection-episodes 200 --collection-successes 150 --collection-failures 0 \
  --learner-config audit_sft_config --trainable-scope lora_only --zero-lora-b \
  --target-mode full_proposal --retention-tasks 2 28 29 60 \
  --num-sft-steps 5000 --batch-size 16 --finetune-mode lora \
  --official-demo-ratio 0 \
  --learning-rate 5e-5 --warmup-steps 10000 --lr-decay-steps 1000000 \
  --eval-every 500 --eval-episodes 20 --checkpoint-every 500 \
  --stop-on-target-success \
  --seed 42 --eval-seed 7 \
  --wandb-project dcrl --wandb-entity distill_rl \
  --wandb-name "sft_task${TASK_PAD}_small_expo_only_bestcheckpoint5k_seed42" \
  --wandb-mode online --wandb-log-every 10

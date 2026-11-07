#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0

# Isolated diagnosis arm.  This does not modify any existing trainer or
# checkpoint tree: all learner state is node-local and the arm is task-pure.
PROJECT=/projects/bgqe/rpolonsky/dcrl
AUDIT=/projects/bgqe/rpolonsky/dcrl/sharder/training/sft_analysis_20261103
SOURCE=/work/hdd/bgqe/rpolonsky/dcrlcheckpoints/sft_agnostic_small_expo/task_60_teacher_450223_success150_sft200k
SEED="${SFT_SEED:-789}"
TAG="${SFT_TAG:-seed789_update150}"
OUTPUT=/tmp/usertmp/rpolonsky/sft_retention_diagnosis_20261106/task_60_pure_${TAG}
BASE=/projects/bgqe/rpolonsky/dcrlcheckpoints/openpi_cache/openpi-assets/checkpoints/pi05_libero
RUNTIME=/tmp/usertmp/rpolonsky/sft_runtime_task60_pure_${TAG}

export PYTHONPATH="/projects/bgqe/rpolonsky/dcrl/sharder/training:$AUDIT:$PROJECT:$PROJECT/third_party/expo-ft/expo_ft/agents/vla/openpi/src:$PROJECT/third_party/expo-ft/expo_ft/agents/vla/openpi/packages/openpi-client/src:$PROJECT/.deps/LIBERO"
export LIBERO_CONFIG_PATH="$PROJECT/sharder/training/runtime/libero_config"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false TOKENIZERS_PARALLELISM=false
export OPENPI_DATA_HOME=/work/hdd/bgqe/rpolonsky/dcrlcheckpoints/openpi_cache
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
export JAX_COMPILATION_CACHE_DIR="$RUNTIME/jax_cache" TMPDIR="$RUNTIME/tmp"
export WANDB_DIR="$OUTPUT/wandb" WANDB_CACHE_DIR="$RUNTIME/wandb-cache" WANDB_DATA_DIR="$RUNTIME/wandb-data"
mkdir -p "$OUTPUT" "$TMPDIR" "$JAX_COMPILATION_CACHE_DIR" "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_DATA_DIR"
if [[ ! -e "$OUTPUT/pi_base_data" ]]; then ln -s "$SOURCE/pi_base_data" "$OUTPUT/pi_base_data"; fi
set -a
source /work/hdd/bgqe/rpolonsky/.env
set +a
: "${WANDB_API_KEY:?Missing W&B key}"

cd "$PROJECT"
PY="$PROJECT/third_party/expo-ft/.venv/bin/python"
"$PY" "$AUDIT/train_sft_retention.py" \
  --phase train --collection-source small_expo \
  --teacher-dir /work/hdd/bgqe/rpolonsky/dcrlcheckpoints/small_expo_rl_10_5_offline50k_online500k_resume_work/libero_90/task_60 \
  --teacher-step 450223 --source-checkpoint "$BASE" --student-checkpoint "$BASE" \
  --norm-stats "$BASE/assets/physical-intelligence/libero/norm_stats.json" \
  --source-suite libero_90 --source-task-id 60 --output-dir "$OUTPUT" \
  --collection-episodes 200 --collection-successes 150 --collection-failures 0 \
  --learner-config audit_sft_config --trainable-scope lora_only --zero-lora-b --target-mode full_proposal \
  --retention-tasks 60 --num-sft-steps 150 --batch-size 16 --finetune-mode lora \
  --official-demo-ratio 0 --learning-rate 5e-5 --warmup-steps 10000 --lr-decay-steps 1000000 \
  --eval-every 150 --eval-episodes 20 --checkpoint-every 150 \
  --seed "$SEED" --eval-seed 7 \
  --wandb-project dcrl --wandb-entity distill_rl \
  --wandb-name "sft_diag_task60_pure_${TAG}_20261106" --wandb-mode online --wandb-log-every 10

if [[ ! -e "$OUTPUT/checkpoints/150" ]]; then
  ln -s "$OUTPUT/latest_checkpoints/150" "$OUTPUT/checkpoints/150"
fi

# Only after the source-task gate is available, evaluate the four-task panel.
EVAL_OUT=/tmp/usertmp/rpolonsky/sft_retention_diagnosis_20261106/task_60_pure_${TAG}_eval
mkdir -p "$EVAL_OUT"
exec "$PY" "$AUDIT/full90_audit_eval.py" \
  --mode sft_pi --checkpoint "$OUTPUT/checkpoints" \
  --assets-dir "$OUTPUT/student_assets" --suite libero_90 --task-ids 2,28,29,60 \
  --episodes 20 --eval-seed 7 --seed 42 --image-size 224 \
  --action-horizon 10 --replan-steps 5 --fsdp-devices 1 --output-dir "$EVAL_OUT"

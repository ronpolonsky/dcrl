#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0

PROJECT=/projects/bgqe/rpolonsky/dcrl
EXPO="$PROJECT/third_party/expo-ft"
TEACHER_DIR=/work/hdd/bgqe/rpolonsky/dcrlcheckpoints/small_expo_rl_10_5_offline50k_online500k_resume_work/libero_90/task_02
TEACHER_STEP=375398
PI05_LOCAL_CHECKPOINT=/projects/bgqe/rpolonsky/dcrlcheckpoints/openpi_cache/openpi-assets/checkpoints/pi05_libero
OUTPUT=/work/hdd/bgqe/rpolonsky/dcrlcheckpoints/sft_agnostic_small_expo/task_02_teacher_375398_success150_sft200k
RUN_TMP=/tmp/rpolonsky-sft-agnostic-task02
OFFICIAL_DEMO_RATIO="${OFFICIAL_DEMO_RATIO:-0}"
OFFICIAL_DEMO_ROOT="${OFFICIAL_DEMO_ROOT:-$PROJECT/datasets}"

export PYTHONPATH="$PROJECT:$EXPO/expo_ft/agents/vla/openpi/src:$EXPO/expo_ft/agents/vla/openpi/packages/openpi-client/src:$PROJECT/.deps/LIBERO"
export LIBERO_CONFIG_PATH="$PROJECT/sharder/training/runtime/libero_config"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR="$RUN_TMP/jax_cache"
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
export OPENPI_DATA_HOME=/work/hdd/bgqe/rpolonsky/dcrlcheckpoints/openpi_cache
export WANDB_MODE=online
export WANDB_DIR="$OUTPUT/wandb"
export WANDB_CACHE_DIR="$RUN_TMP/wandb-cache"
export WANDB_DATA_DIR="$RUN_TMP/wandb-data"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMPDIR="$RUN_TMP/tmp"
mkdir -p "$OUTPUT" "$RUN_TMP" "$TMPDIR" "$JAX_COMPILATION_CACHE_DIR" "$OPENPI_DATA_HOME" "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_DATA_DIR"
set -a
source /work/hdd/bgqe/rpolonsky/.env
set +a
: "${WANDB_API_KEY:?WANDB_API_KEY missing from /work/hdd/bgqe/rpolonsky/.env}"
cd "$PROJECT"

[[ -d "$PI05_LOCAL_CHECKPOINT/params" ]] || { echo "Missing local Pi0.5 params: $PI05_LOCAL_CHECKPOINT/params" >&2; exit 2; }
[[ -f "$PI05_LOCAL_CHECKPOINT/assets/physical-intelligence/libero/norm_stats.json" ]] || { echo "Missing local Pi0.5 LIBERO norm stats" >&2; exit 2; }

"$EXPO/.venv/bin/python" -m dcrl.train_pi_sft_agnostic \
  --phase all \
  --collection-source small_expo \
  --teacher-dir "$TEACHER_DIR" \
  --teacher-step "$TEACHER_STEP" \
  --source-checkpoint "$PI05_LOCAL_CHECKPOINT" \
  --student-checkpoint "$PI05_LOCAL_CHECKPOINT" \
  --norm-stats "$PI05_LOCAL_CHECKPOINT/assets/physical-intelligence/libero/norm_stats.json" \
  --source-suite libero_90 --source-task-id 2 \
  --output-dir "$OUTPUT" \
  --collection-episodes 200 --collection-successes 150 --collection-failures 0 \
  --num-sft-steps 200000 --batch-size 16 --finetune-mode lora \
  --official-demo-ratio "$OFFICIAL_DEMO_RATIO" --official-demo-root "$OFFICIAL_DEMO_ROOT" \
  --eval-every 20000 --eval-episodes 20 --checkpoint-every 20000 \
  --warmup-steps 10000 --lr-decay-steps 1000000 \
  --seed 42 --eval-seed 7 \
  --wandb-project dcrl --wandb-entity distill_rl \
  --wandb-name sft_agnostic_small_expo_libero90_task02_teacher375398_lora_200k_seed42 \
  --wandb-mode online --wandb-log-every 10

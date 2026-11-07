#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0

PROJECT=/projects/bgqe/rpolonsky/dcrl
EXPO="$PROJECT/third_party/expo-ft"
OUTPUT=/u/rpolonsky/dcrlcheckpoints/small_expo_rl_10_5_offline50k_online500k_resume_newgpu
RUN_TMP=/tmp/rpolonsky-small-rl-task02-a100

export PYTHONPATH="$PROJECT:$EXPO/expo_ft/agents/vla/openpi/src:$EXPO/expo_ft/agents/vla/openpi/packages/openpi-client/src:$PROJECT/.deps/LIBERO"
export LIBERO_CONFIG_PATH="$PROJECT/sharder/training/runtime/libero_config"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export WANDB_MODE=offline WANDB_DIR="$RUN_TMP/wandb"
export TMPDIR="$RUN_TMP/tmp"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$WANDB_DIR" "$TMPDIR"
cd "$PROJECT"

exec "$EXPO/.venv/bin/python" -m dcrl.train_libero \
  --resume \
  --suite libero_90 --task-id 2 \
  --init-distill-steps 0 \
  --offline-rl-steps 50000 --online-rl-steps 500000 \
  --pi-distill-eval-interval 50000 --pi-distill-chckpt-interval 50000 \
  --offline-rl-eval-interval 25000 --offline-rl-checkpoint-interval 25000 \
  --online-rl-eval-interval 25000 --online-rl-checkpot-interval 25000 \
  --distill-data-root /u/rpolonsky/dcrlcheckpoints/pi_base_data/fixed125 \
  --demos-root /u/rpolonsky/libero_demos \
  --update-type episode --num-updates 3 --batch-size 32 --utd-ratio 20 \
  --replan-steps 5 --action-horizon 10 --offline-ratio 0.5 \
  --eval-episodes 20 \
  --output-dir "$OUTPUT" \
  --wandb-entity distill_rl --wandb-project dcrl \
  --wandb-name small_expo_rl_from_50k_libero90_task2_seed42 \
  --wandb-mode offline --seed 42

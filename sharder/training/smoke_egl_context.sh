#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
TASK_ID="${1:-29}"
PROJECT=/projects/bgqe/rpolonsky/dcrl
EXPO="$PROJECT/third_party/expo-ft"
SMOKE_ROOT="${SMOKE_ROOT:-/tmp/rpolonsky-egl-context-task${TASK_ID}}"
SMOKE_RUN_NAME="${SMOKE_RUN_NAME:-egl_context_task${TASK_ID}}"
SMOKE_MAX_STEPS="${SMOKE_MAX_STEPS:-4500}"
mkdir -p "$SMOKE_ROOT" "$SMOKE_ROOT/wandb" "$SMOKE_ROOT/tmp"
export PYTHONUNBUFFERED=1
export PYTHONPATH="/projects/bgqe/rpolonsky/dcrl/sharder/training:$PROJECT:$EXPO/expo_ft/agents/vla/openpi/src:$EXPO/expo_ft/agents/vla/openpi/packages/openpi-client/src:$PROJECT/.deps/LIBERO"
export LIBERO_CONFIG_PATH="$PROJECT/sharder/training/runtime/libero_config"
export OPENPI_DATA_HOME=/projects/bgqe/rpolonsky/dcrlcheckpoints/openpi_cache
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export WANDB_MODE=offline WANDB_DIR="$SMOKE_ROOT/wandb"
export TMPDIR="$SMOKE_ROOT/tmp"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$PROJECT"
SMOKE_TRAINER="${SMOKE_TRAINER:-/projects/bgqe/rpolonsky/dcrl/sharder/training/train_pi_robo_libero_egl_context.py}"
SMOKE_PREFIX=()
if [[ "${RUN_GDB:-0}" == 1 ]]; then
  SMOKE_PREFIX=(gdb -batch -ex 'set pagination off' -ex 'set print thread-events off' -ex run -ex 'bt 30' --args)
elif [[ "${RUN_STRACE:-0}" == 1 ]]; then
  SMOKE_PREFIX=(strace -f -qq -Z -e trace=%memory,ioctl,openat,memfd_create,ftruncate -o "${TRACE_LOG:-$SMOKE_ROOT/failed_syscalls.log}")
fi
exec "${SMOKE_PREFIX[@]}" "$EXPO/.venv/bin/python" "$SMOKE_TRAINER" \
  --overwrite --config=configs/model/expo_ft_pi_libero_config.py \
  --project_name=dcrl-expo-smoke --run_name="$SMOKE_RUN_NAME" \
  --suite=libero_90 --task_id="$TASK_ID" \
  --demos_root=/projects/bgqe/rpolonsky/libero_demos \
  --output_dir="$SMOKE_ROOT/output" --batch_size=16 --max_steps="$SMOKE_MAX_STEPS" \
  --max_episode_steps=400 --num_data=0 --offline_ratio=0.5 \
  --update_type=episode --num_updates=15 --utd_ratio=20 --replan_steps=5 \
  --eval_interval=400 --eval_episodes=1 --eval_seed=7 --eval_max_steps=400 \
  --checkpoint_model=false --checkpoint_buffer=false --checkpoint_interval=0 \
  --checkpoint_max_to_keep=1 --fsdp_devices=1 --seed=42 --tqdm=true

#!/usr/bin/env bash
set -euo pipefail

TASK_ID="${1:?usage: $0 TASK_ID}"

PROJECT=/projects/bgqe/rpolonsky/dcrl
EXPO="$PROJECT/third_party/expo-ft"
TRAINER=/projects/bgqe/rpolonsky/dcrl/sharder/training/train_pi_robo_libero_logged.py
BATCH_SIZE="${BATCH_SIZE:-16}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/work/hdd/bgqe/rpolonsky/dcrlcheckpoints/pi_robo_libero_unsharded_h200_b${BATCH_SIZE}_200k}"
RUN_NAME="pi_robo_libero_unsharded_h200_b${BATCH_SIZE}_task${TASK_ID}"
RUNTIME_ROOT="$OUTPUT_ROOT/runtime/task${TASK_ID}"
CACHE_ROOT="$OUTPUT_ROOT/cache/task${TASK_ID}"

case "$TASK_ID" in
  2) DEMO_PATH="$PROJECT/datasets/libero_90/KITCHEN_SCENE10_put_the_black_bowl_in_the_top_drawer_of_the_cabinet_demo.hdf5" ;;
  28) DEMO_PATH="$PROJECT/datasets/libero_90/KITCHEN_SCENE5_close_the_top_drawer_of_the_cabinet_demo.hdf5" ;;
  *) echo "Unsupported TASK_ID=$TASK_ID (expected 2 or 28)" >&2; exit 2 ;;
esac

export LIBERO_CONFIG_PATH="$OUTPUT_ROOT/libero_config"
export WANDB_PROJECT=dcrl
export WANDB_ENTITY=distill_rl
export WANDB_BASE_URL=https://forge.coreweave.com/api/wandb
WANDB_ENV_FILE="${WANDB_ENV_FILE:-/work/hdd/bgqe/rpolonsky/.env}"
if [[ -f "$WANDB_ENV_FILE" ]]; then
  set -a
  source "$WANDB_ENV_FILE"
  set +a
fi
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  export WANDB_MODE=online
else
  export WANDB_MODE=offline
fi
export WANDB_DIR="$OUTPUT_ROOT/wandb/task${TASK_ID}"
export OPENPI_DATA_HOME=/projects/bgqe/rpolonsky/dcrlcheckpoints/openpi_cache
export PI05_ASSETS_DIR="$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_libero/assets"
export DCRL_JAX_CACHE_ROOT="$CACHE_ROOT/jax"
export WANDB_CACHE_DIR="$CACHE_ROOT/wandb"
export WANDB_CONFIG_DIR="$CACHE_ROOT/wandb/config"
export WANDB_DATA_DIR="$CACHE_ROOT/wandb/data"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"
export XDG_CONFIG_HOME="$CACHE_ROOT/xdg_config"
export TMPDIR="$RUNTIME_ROOT/tmp"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# LIBERO's trusted init-state files are legacy PyTorch pickles.  PyTorch 2.6
# otherwise defaults torch.load(..., weights_only=True) and rejects them.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONPATH="$PROJECT:$EXPO/expo_ft/agents/vla/openpi/src:$EXPO/expo_ft/agents/vla/openpi/packages/openpi-client/src:$PROJECT/.deps/LIBERO${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$OUTPUT_ROOT/wandb/task${TASK_ID}" "$CACHE_ROOT/jax" \
  "$CACHE_ROOT/wandb/config" "$CACHE_ROOT/wandb/data" "$CACHE_ROOT/xdg" \
  "$CACHE_ROOT/xdg_config" "$RUNTIME_ROOT/tmp"
cd "$PROJECT"

exec "$EXPO/.venv/bin/python" "$TRAINER" \
  --config=configs/model/expo_ft_pi_libero_config.py \
  --config.pi05_assets_dir="$PI05_ASSETS_DIR" \
  --project_name=dcrl \
  --run_name="$RUN_NAME" \
  --output_dir="$OUTPUT_ROOT" \
  --suite=libero_90 \
  --task_id="$TASK_ID" \
  --demo_path="$DEMO_PATH" \
  --num_data=0 \
  --offline_ratio=0.5 \
  --max_steps=200000 \
  --max_episode_steps=400 \
  --batch_size="$BATCH_SIZE" \
  --fsdp_devices=1 \
  --utd_ratio=20 \
  --update_type=episode \
  --num_updates=15 \
  --replan_steps=5 \
  --eval_interval=20000 \
  --eval_episodes=20 \
  --eval_seed=7 \
  --eval_max_steps=0 \
  --checkpoint_model=true \
  --checkpoint_buffer=true \
  --checkpoint_interval=20000 \
  --checkpoint_keep_without_metrics=true \
  --checkpoint_max_to_keep=1 \
  --tqdm=true \
  --overwrite=true \
  --resume=false

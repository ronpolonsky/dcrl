#!/usr/bin/env bash
set -euo pipefail

# Original full-EXPO settings; run unchanged LIBERO on a separate allocation.
# The original seven trainers and their launchers are not modified.
ulimit -c 0
TASK_ID="${1:?usage: $0 TASK_ID}"
: "${RENDER_JOB_ID:?Set RENDER_JOB_ID to the idle renderer allocation}"
: "${RENDER_RPC_HOST:?Set RENDER_RPC_HOST to the learner node hostname}"

PROJECT=/projects/bgqe/rpolonsky/dcrl
EXPO="$PROJECT/third_party/expo-ft"
TRAINER=/projects/bgqe/rpolonsky/dcrl/sharder/training/train_pi_robo_libero_render_process.py
BATCH_SIZE="${BATCH_SIZE:-16}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/work/hdd/bgqe/rpolonsky/dcrlcheckpoints/pi_robo_libero_unsharded_h200_b${BATCH_SIZE}_200k}"
RUN_NAME="pi_robo_libero_unsharded_h200_b${BATCH_SIZE}_task${TASK_ID}"
RUNTIME_ROOT="$OUTPUT_ROOT/runtime/task${TASK_ID}"
CACHE_ROOT="$OUTPUT_ROOT/cache/task${TASK_ID}"

# The fresh-run trainer uses --overwrite. Never apply that to a saved run.
CHECKPOINT_DIR="$OUTPUT_ROOT/$RUN_NAME/checkpoints"
if [[ -d "$CHECKPOINT_DIR" ]] && find "$CHECKPOINT_DIR" -mindepth 1 -maxdepth 1 -type d -name '[0-9]*' -print -quit | read -r _; then
  echo "Refusing fresh launch: saved checkpoint exists in $CHECKPOINT_DIR" >&2
  exit 2
fi

case "$TASK_ID" in
  29) DEMO_PATH="$PROJECT/datasets/libero_90/KITCHEN_SCENE5_put_the_black_bowl_in_the_top_drawer_of_the_cabinet_demo.hdf5" ;;
  60) DEMO_PATH="$PROJECT/datasets/libero_90/LIVING_ROOM_SCENE4_pick_up_the_black_bowl_on_the_left_and_put_it_in_the_tray_demo.hdf5" ;;
  79) DEMO_PATH="$PROJECT/datasets/libero_90/STUDY_SCENE2_pick_up_the_book_and_place_it_in_the_left_compartment_of_the_caddy_demo.hdf5" ;;
  *) echo "Unsupported TASK_ID=$TASK_ID (expected 29, 60, or 79)" >&2; exit 2 ;;
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
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
# Slurm remaps each allocated GPU to local CUDA index 0.  Robosuite validates
# this value against CUDA_VISIBLE_DEVICES, so use the job-local index rather
# than SLURM_STEP_GPUS' physical node index.
export MUJOCO_EGL_DEVICE_ID=0
export PYTHONPATH="/projects/bgqe/rpolonsky/dcrl/sharder/training:$PROJECT:$EXPO/expo_ft/agents/vla/openpi/src:$EXPO/expo_ft/agents/vla/openpi/packages/openpi-client/src:$PROJECT/.deps/LIBERO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

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

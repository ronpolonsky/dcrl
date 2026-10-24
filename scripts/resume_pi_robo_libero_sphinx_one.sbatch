#!/bin/bash
# Launch one fresh Pi0.5 EXPO-F/T LIBERO RL run from its latest finalized checkpoint.
# Submit with: sbatch --export=ALL,TASK_ID=<2|28|29|60|79> this_file
# Checkpoint retention is deliberately one and no keep_period is used.
#SBATCH --account=nlp
#SBATCH --job-name=pi-robo-libero-resume
#SBATCH --partition=sphinx
# Sphinx8 exhausted its GPU during task-60 batch prefetch (job 18009807).
# Keep it out of subsequent Pi0.5 retries.
#SBATCH --exclude=sphinx[1-3,8]
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=256G
#SBATCH --time=2-00:00:00
#SBATCH --output=/nlp/scr/ronpo/projects/dcrl/slurm/pi-robo-libero-resume-%j.out
#SBATCH --error=/nlp/scr/ronpo/projects/dcrl/slurm/pi-robo-libero-resume-%j.err

set -euo pipefail

: "${TASK_ID:?Submit with TASK_ID=2,28,29,60,or79}"

PROJECT=/nlp/scr/ronpo/projects/dcrl
EXPO="$PROJECT/third_party/expo-ft"
OUTPUT_ROOT="${OUTPUT_ROOT:-/nlp/scr/ronpo/dcrlcheckpoints/pi_robo_libero_prod_200k_sphinx}"
RUN_NAME="${RUN_NAME:-pi_robo_libero_prod_200k_sphinx_task${TASK_ID}}"
BATCH_SIZE="${BATCH_SIZE:-16}"
RUNTIME_ROOT="$OUTPUT_ROOT/runtime"
CACHE_ROOT="$OUTPUT_ROOT/cache"

case "$TASK_ID" in
  2) DEMO_PATH="$PROJECT/datasets/libero_90/KITCHEN_SCENE10_put_the_black_bowl_in_the_top_drawer_of_the_cabinet_demo.hdf5" ;;
  28) DEMO_PATH="$PROJECT/datasets/libero_90/KITCHEN_SCENE5_close_the_top_drawer_of_the_cabinet_demo.hdf5" ;;
  29) DEMO_PATH="$PROJECT/datasets/libero_90/KITCHEN_SCENE5_put_the_black_bowl_in_the_top_drawer_of_the_cabinet_demo.hdf5" ;;
  60) DEMO_PATH="$PROJECT/datasets/libero_90/LIVING_ROOM_SCENE4_pick_up_the_black_bowl_on_the_left_and_put_it_in_the_tray_demo.hdf5" ;;
  79) DEMO_PATH="$PROJECT/datasets/libero_90/STUDY_SCENE2_pick_up_the_book_and_place_it_in_the_left_compartment_of_the_caddy_demo.hdf5" ;;
  *) echo "Unsupported TASK_ID=$TASK_ID" >&2; exit 2 ;;
esac

export LIBERO_CONFIG_PATH="$PROJECT/.deps/libero_config"
export WANDB_PROJECT=dcrl
export WANDB_ENTITY=distill_rl
export WANDB_BASE_URL=https://forge.coreweave.com/api/wandb
export WANDB_DIR="$OUTPUT_ROOT/wandb"
# Route all model, JAX, W&B, XDG, and temporary caches to project storage.
# Home caches on AFS/Sailhome are quota limited and abort Pi initialization.
export OPENPI_DATA_HOME="$OUTPUT_ROOT/openpi_cache"
export DCRL_JAX_CACHE_ROOT="$CACHE_ROOT/jax"
export WANDB_CACHE_DIR="$CACHE_ROOT/wandb"
export WANDB_CONFIG_DIR="$CACHE_ROOT/wandb/config"
export WANDB_DATA_DIR="$CACHE_ROOT/wandb/data"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"
export XDG_CONFIG_HOME="$CACHE_ROOT/xdg_config"
export TMPDIR="$RUNTIME_ROOT/tmp"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="$PROJECT:$EXPO/expo_ft/agents/vla/openpi/src:$EXPO/expo_ft/agents/vla/openpi/packages/openpi-client/src:$PROJECT/.deps/LIBERO${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$OUTPUT_ROOT/wandb" "$PROJECT/slurm" "$OPENPI_DATA_HOME" \
  "$CACHE_ROOT/jax" "$CACHE_ROOT/wandb/config" \
  "$CACHE_ROOT/wandb/data" "$CACHE_ROOT/xdg" "$CACHE_ROOT/xdg_config" \
  "$RUNTIME_ROOT/tmp"
cd "$PROJECT"

exec "$EXPO/.venv/bin/python" "$PROJECT/train_pi_robo_libero.py" \
  --config=configs/model/expo_ft_pi_libero_config.py \
  --project_name=dcrl \
  --run_name="$RUN_NAME" \
  --output_dir="$OUTPUT_ROOT/task${TASK_ID}" \
  --suite=libero_90 \
  --task_id="$TASK_ID" \
  --demo_path="$DEMO_PATH" \
  --num_data=50 \
  --max_steps=200000 \
  --max_episode_steps=400 \
  --batch_size="$BATCH_SIZE" \
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

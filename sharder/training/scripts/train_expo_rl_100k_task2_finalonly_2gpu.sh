#!/usr/bin/env bash
# Isolated 2-GPU Task 2 EXPO run: final learner checkpoint only.
# This launcher does not modify the production trainer or existing launchers.
set -euo pipefail

: "${JAX_PROCESS_ID:?set JAX_PROCESS_ID=0 or 1}"
: "${JAX_COORDINATOR_ADDRESS:?set coordinator host:port}"

PROJECT=/projects/bgqe/rpolonsky/dcrl/sharder/training
BASE_PROJECT=/projects/bgqe/rpolonsky/dcrl
EXPO="$BASE_PROJECT/third_party/expo-ft"
PYTHON="$EXPO/.venv/bin/python"
LIBERO_ROOT="$BASE_PROJECT/.deps/LIBERO/libero/libero"
OUTPUT_ROOT="${OUTPUT_ROOT:-/u/rpolonsky/dcrlcheckpoints/expo_rl_task2_batch4_utd20_final100k_2gpu}"
RUN_NAME="${RUN_NAME:-official_pi05_task2_batch4_utd20_final100k_2gpu}"
RUNTIME_ROOT="$OUTPUT_ROOT/runtime"

export PYTHONPATH="$PROJECT:$EXPO/expo_ft/agents/vla/openpi/src:$EXPO/expo_ft/agents/vla/openpi/packages/openpi-client/src:$BASE_PROJECT/.deps/LIBERO${PYTHONPATH:+:$PYTHONPATH}"
export LIBERO_CONFIG_PATH="$RUNTIME_ROOT/libero_config"
export OPENPI_DATA_HOME=/projects/bgqe/rpolonsky/dcrlcheckpoints/openpi_cache
export JAX_COMPILATION_CACHE_DIR="/tmp/rpolonsky-expo-final100k/jax_cache/proc${JAX_PROCESS_ID}"
export DCRL_JAX_CACHE_ROOT="/tmp/rpolonsky-expo-final100k/jax_cache/proc${JAX_PROCESS_ID}"
export WANDB_MODE=offline
export WANDB_DIR="$OUTPUT_ROOT/wandb"
export XDG_CACHE_HOME="/tmp/rpolonsky-expo-final100k/xdg_cache/proc${JAX_PROCESS_ID}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TOKENIZERS_PARALLELISM=false
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

mkdir -p "$OUTPUT_ROOT" "$RUNTIME_ROOT/libero_config" "$WANDB_DIR" \
  "/tmp/rpolonsky-expo-final100k/jax_cache/proc${JAX_PROCESS_ID}" \
  "/tmp/rpolonsky-expo-final100k/xdg_cache/proc${JAX_PROCESS_ID}"
cat > "$LIBERO_CONFIG_PATH/config.yaml" <<CFG
benchmark_root: $LIBERO_ROOT
bddl_files: $LIBERO_ROOT/bddl_files
init_states: $LIBERO_ROOT/init_files
datasets: $PROJECT/datasets
assets: $LIBERO_ROOT/assets
CFG

if [[ "$JAX_PROCESS_ID" == 0 ]]; then
  RESUME_ARG=(--overwrite)
else
  RESUME_ARG=(--resume)
fi

echo "process=$JAX_PROCESS_ID job=${SLURM_JOB_ID:-unknown} host=$(hostname)"
echo "output=$OUTPUT_ROOT/$RUN_NAME max_steps=100000 final_checkpoint_only=true checkpoint_buffer=false"
nvidia-smi -L

cd "$PROJECT"
"$PYTHON" "$PROJECT/scripts/jax_distributed_run.py" "$PROJECT/train_pi_robo_libero.py" \
  "${RESUME_ARG[@]}" \
  --config=configs/model/expo_ft_pi_libero_config.py \
  --project_name=dcrl-expo-rl-100k-finalonly \
  --run_name="$RUN_NAME" \
  --suite=libero_90 \
  --task_id=2 \
  --demos_root=/projects/bgqe/rpolonsky/libero_demos \
  --output_dir="$OUTPUT_ROOT" \
  --batch_size=4 \
  --max_steps=100000 \
  --max_episode_steps=400 \
  --num_data=2 \
  --offline_ratio=0.5 \
  --update_type=episode \
  --num_updates=15 \
  --utd_ratio=20 \
  --replan_steps=5 \
  --eval_interval=20000 \
  --eval_episodes=20 \
  --eval_max_steps=0 \
  --eval_seed=7 \
  --checkpoint_model \
  --checkpoint_interval=0 \
  --checkpoint_max_to_keep=1 \
  --checkpoint_keep_without_metrics=true \
  --fsdp_devices=2 \
  --seed=42 \
  --tqdm=false

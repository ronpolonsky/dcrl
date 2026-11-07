#!/usr/bin/env bash
# Isolated two-host EXPO RL batch-16 run. This launcher and its output are
# separate from the earlier batch-2/4/8/16 experiments and production jobs.
set -euo pipefail

: "${JAX_PROCESS_ID:?set JAX_PROCESS_ID=0 or 1}"
: "${JAX_NUM_PROCESSES:?set JAX_NUM_PROCESSES=2}"
: "${JAX_COORDINATOR_ADDRESS:?set coordinator host:port}"

PROJECT=/projects/bgqe/rpolonsky/dcrl/sharder/training
BASE_PROJECT=/projects/bgqe/rpolonsky/dcrl
DATA_ROOT=/projects/bgqe/rpolonsky
EXPO="$BASE_PROJECT/third_party/expo-ft"
PYTHON="$EXPO/.venv/bin/python"
LIBERO_ROOT="$BASE_PROJECT/.deps/LIBERO/libero/libero"
OUTPUT_ROOT="$PROJECT/checkpoints/expo_rl_task2_batch16_defercache_200k_multihost2"
RUN_NAME=official_pi05_task2_batch16_defercache_200k_multihost2
RUNTIME_ROOT="$OUTPUT_ROOT/runtime"

export PYTHONPATH="$PROJECT:$EXPO/expo_ft/agents/vla/openpi/src:$EXPO/expo_ft/agents/vla/openpi/packages/openpi-client/src:$BASE_PROJECT/.deps/LIBERO${PYTHONPATH:+:$PYTHONPATH}"
export LIBERO_CONFIG_PATH="$RUNTIME_ROOT/libero_config"
export OPENPI_DATA_HOME=/projects/bgqe/rpolonsky/dcrlcheckpoints/openpi_cache
export JAX_COMPILATION_CACHE_DIR="$RUNTIME_ROOT/jax_cache/proc${JAX_PROCESS_ID}"
export DCRL_JAX_CACHE_ROOT="$RUNTIME_ROOT/jax_cache/proc${JAX_PROCESS_ID}"
export WANDB_MODE=offline
export WANDB_DIR="$OUTPUT_ROOT/wandb"
export XDG_CACHE_HOME="$RUNTIME_ROOT/xdg_cache/proc${JAX_PROCESS_ID}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TOKENIZERS_PARALLELISM=false
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

mkdir -p "$OUTPUT_ROOT" "$RUNTIME_ROOT/libero_config" "$RUNTIME_ROOT/jax_cache/proc${JAX_PROCESS_ID}" "$RUNTIME_ROOT/xdg_cache/proc${JAX_PROCESS_ID}" "$WANDB_DIR"
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

echo "process=$JAX_PROCESS_ID/$JAX_NUM_PROCESSES coordinator=$JAX_COORDINATOR_ADDRESS job=${SLURM_JOB_ID:-unknown} host=$(hostname) node=${SLURMD_NODENAME:-$(hostname)}"
echo "output=$OUTPUT_ROOT/$RUN_NAME batch_size=16 max_steps=200000 checkpoint/eval_interval=20000 checkpoint_max_to_keep=1"
nvidia-smi -L

cd "$PROJECT"
"$PYTHON" "$PROJECT/scripts/jax_distributed_run.py" "$PROJECT/train_pi_robo_libero.py" \
  "${RESUME_ARG[@]}" \
  --config=configs/model/expo_ft_pi_libero_config.py \
  --project_name=dcrl-expo-rl-200k-batch16 \
  --run_name="$RUN_NAME" \
  --suite=libero_90 \
  --task_id=2 \
  --demos_root="$DATA_ROOT/libero_demos" \
  --output_dir="$OUTPUT_ROOT" \
  --batch_size=16 \
  --max_steps=200000 \
  --max_episode_steps=400 \
  --num_data=2 \
  --offline_ratio=0.5 \
  --update_type=episode \
  --num_updates=15 \
  --utd_ratio=1 \
  --replan_steps=5 \
  --eval_interval=20000 \
  --eval_episodes=20 \
  --eval_max_steps=0 \
  --eval_seed=7 \
  --checkpoint_model \
  --checkpoint_buffer \
  --checkpoint_interval=20000 \
  --checkpoint_max_to_keep=1 \
  --checkpoint_keep_without_metrics=true \
  --fsdp_devices=2 \
  --seed=42 \
  --tqdm=false

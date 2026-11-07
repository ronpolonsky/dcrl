#!/usr/bin/env bash
# Two-host EXPO smoke worker. This is a launch wrapper only: the trainer and
# expo-ft FSDP implementation remain unchanged.
set -euo pipefail

: "${STAGE:?set STAGE=initial or STAGE=resume}"
: "${JAX_PROCESS_ID:?set JAX_PROCESS_ID=0 or 1}"
: "${JAX_COORDINATOR_ADDRESS:?set coordinator host:port}"

case "$STAGE" in
  initial)
    MAX_STEPS=60
    if [[ "$JAX_PROCESS_ID" == 0 ]]; then RESUME_ARG=(--overwrite); else RESUME_ARG=(--resume); fi
    ;;
  resume)  MAX_STEPS=110; RESUME_ARG=(--resume) ;;
  *) echo "invalid STAGE=$STAGE" >&2; exit 2 ;;
esac

PROJECT=/projects/bgqe/rpolonsky/dcrl/sharder/training
BASE_PROJECT=/projects/bgqe/rpolonsky/dcrl
DATA_ROOT=/projects/bgqe/rpolonsky
EXPO="$BASE_PROJECT/third_party/expo-ft"
PYTHON="$EXPO/.venv/bin/python"
LIBERO_ROOT="$BASE_PROJECT/.deps/LIBERO/libero/libero"
OUTPUT_ROOT="$PROJECT/checkpoints/smoke_expo_rl_task2_batch2_official_multihost2"
RUN_NAME=official_pi05_task2_batch2_multihost2
RUNTIME_ROOT="$OUTPUT_ROOT/runtime"

export PYTHONPATH="$PROJECT:$EXPO/expo_ft/agents/vla/openpi/src:$EXPO/expo_ft/agents/vla/openpi/packages/openpi-client/src:$BASE_PROJECT/.deps/LIBERO${PYTHONPATH:+:$PYTHONPATH}"
export LIBERO_CONFIG_PATH="$RUNTIME_ROOT/libero_config"
export OPENPI_DATA_HOME=/projects/bgqe/rpolonsky/dcrlcheckpoints/openpi_cache
export JAX_COMPILATION_CACHE_DIR="$RUNTIME_ROOT/jax_cache/proc${JAX_PROCESS_ID}"
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

echo "stage=$STAGE process=$JAX_PROCESS_ID job=${SLURM_JOB_ID:-unknown} host=$(hostname) node=${SLURMD_NODENAME:-$(hostname)}"
echo "output=$OUTPUT_ROOT/$RUN_NAME max_steps=$MAX_STEPS"
nvidia-smi -L

cd "$PROJECT"
"$PYTHON" "$PROJECT/scripts/jax_distributed_run.py" "$PROJECT/train_pi_robo_libero.py" \
  "${RESUME_ARG[@]}" \
  --config=configs/model/expo_ft_pi_libero_config.py \
  --project_name=dcrl-smoke \
  --run_name="$RUN_NAME" \
  --suite=libero_90 \
  --task_id=2 \
  --demos_root="$DATA_ROOT/libero_demos" \
  --output_dir="$OUTPUT_ROOT" \
  --batch_size=2 \
  --max_steps="$MAX_STEPS" \
  --max_episode_steps=10 \
  --num_data=2 \
  --offline_ratio=0.5 \
  --update_type=episode \
  --num_updates=1 \
  --utd_ratio=1 \
  --replan_steps=5 \
  --eval_interval=50 \
  --eval_episodes=2 \
  --eval_max_steps=10 \
  --eval_seed=7 \
  --checkpoint_model \
  --checkpoint_buffer \
  --checkpoint_interval=50 \
  --checkpoint_max_to_keep=10 \
  --checkpoint_keep_without_metrics=true \
  --fsdp_devices=2 \
  --seed=42 \
  --tqdm=false

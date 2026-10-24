#!/usr/bin/env bash
#SBATCH --partition=iris-hi
#SBATCH --time=21-00:00:00
#SBATCH --nodes=1
# A compact learner plus 10k-row replay fits the available A40 host slots.
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --account=iris
# Use the next available healthy Iris-hi tier; do not pin a host.
# iris7/8 allocations 17961871/72/70 hung in nvidia-smi; exclude them for retry.
#SBATCH --exclude=iris[1-4,7-10],iris-hp-z8
#SBATCH --job-name=small-expo-rl
#SBATCH --output=/nlp/scr/ronpo/projects/dcrl/slurm/%x-%j.out
#SBATCH --error=/nlp/scr/ronpo/projects/dcrl/slurm/%x-%j.err

set -euo pipefail
: "${TASK_ID:?submit with --export=ALL,TASK_ID=<2|28|29|60|79>}"
: "${OUTPUT_ROOT:=/nlp/scr/ronpo/dcrlcheckpoints/small_expo_3phase_dualopt_10_5_sharedstats}"
: "${BATCH_SIZE:=32}"

PROJECT=/nlp/scr/ronpo/projects/dcrl
EXPO="$PROJECT/third_party/expo-ft"
PYTHON="$EXPO/.venv/bin/python"
LIBERO_ROOT="$PROJECT/.deps/LIBERO/libero/libero"
RUNTIME_ROOT="${OUTPUT_ROOT}/runtime"
WANDB_CACHE_ROOT="${OUTPUT_ROOT}/wandb_cache"
XDG_CACHE_ROOT="${OUTPUT_ROOT}/xdg_cache"
TMP_ROOT="${OUTPUT_ROOT}/tmp"
export PYTHONPATH="$PROJECT:$EXPO/expo_ft/agents/vla/openpi/src:$EXPO/expo_ft/agents/vla/openpi/packages/openpi-client/src:$PROJECT/.deps/LIBERO${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_BASE_URL=https://forge.coreweave.com/api/wandb
export WANDB_PROJECT=dcrl
export WANDB_ENTITY=distill_rl
export WANDB_DIR="$OUTPUT_ROOT/wandb"
# Keep W&B core logs, config, cache, and temporary files on the project
# filesystem; the default AFS home cache is quota limited.
export WANDB_CACHE_DIR="$WANDB_CACHE_ROOT"
export WANDB_CONFIG_DIR="$WANDB_CACHE_ROOT/config"
export WANDB_DATA_DIR="$WANDB_CACHE_ROOT/data"
export XDG_CACHE_HOME="$XDG_CACHE_ROOT"
export TMPDIR="$TMP_ROOT"
# The HGX EGL stack has reproducibly aborted in MuJoCo read_pixels().
# Use OSMesa only on those hosts; healthy nodes retain the faster EGL path.
if [[ "${SLURMD_NODENAME:-}" == iris-hgx-* ]]; then
  export MUJOCO_GL=osmesa
  export PYOPENGL_PLATFORM=osmesa
  RENDER_BACKEND=osmesa
else
  export MUJOCO_GL=egl
  export PYOPENGL_PLATFORM=egl
  RENDER_BACKEND=egl
fi
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export JAX_PLATFORMS=cuda

mkdir -p "$PROJECT/slurm" "$PROJECT/.deps/libero_config" "$OUTPUT_ROOT" \
  "$WANDB_DIR" "$RUNTIME_ROOT" "$WANDB_CACHE_ROOT/config" \
  "$WANDB_CACHE_ROOT/data" "$XDG_CACHE_ROOT" "$TMP_ROOT"
cat > "$PROJECT/.deps/libero_config/config.yaml" <<CFG
benchmark_root: $LIBERO_ROOT
bddl_files: $LIBERO_ROOT/bddl_files
init_states: $LIBERO_ROOT/init_files
datasets: $PROJECT/datasets
assets: $LIBERO_ROOT/assets
CFG
export LIBERO_CONFIG_PATH="$PROJECT/.deps/libero_config"

echo "job=$SLURM_JOB_ID host=$(hostname) node=$SLURMD_NODENAME render_backend=$RENDER_BACKEND task=$TASK_ID output=$OUTPUT_ROOT"
nvidia-smi -L
cd "$PROJECT"
"$PYTHON" -m dcrl.train_libero --resume \
  --suite libero_90 \
  --task-id "$TASK_ID" \
  --init-distill-steps 0 \
  --offline-rl-steps 50000 \
  --online-rl-steps 500000 \
  --pi-distill-eval-interval 50000 \
  --pi-distill-chckpt-interval 50000 \
  --offline-rl-eval-interval 50000 \
  --offline-rl-checkpoint-interval 50000 \
  --online-rl-eval-interval 25000 \
  --online-rl-checkpot-interval 25000 \
  --distill-data-root /nlp/scr/ronpo/dcrlcheckpoints/pi_base_data/fixed125 \
  --demos-root datasets \
  --update-type episode \
  --num-updates 3 \
  --batch-size "$BATCH_SIZE" \
  --utd-ratio 20 \
  --replan-steps 5 \
  --action-horizon 10 \
  --offline-ratio 0.5 \
  --eval-episodes 20 \
  --output-dir "$OUTPUT_ROOT" \
  --wandb-entity distill_rl \
  --wandb-project dcrl \
  --wandb-name "small_expo_rl_from_50k_libero90_task${TASK_ID}_seed42" \
  --wandb-mode online \
  --seed 42

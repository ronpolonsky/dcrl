#!/usr/bin/env bash
#SBATCH --partition=gpuA100x4
#SBATCH --time=02:00:00
#SBATCH --nodes=1
# Compact model migration needs only a small CPU/RAM footprint.
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
# Use an available Delta A100 slot for the migration when the H100/H200 tiers are busy.
#SBATCH --gres=gpu:nvidia_a100:1
#SBATCH --account=bgqe-delta-gpu
# Use the next available healthy Delta A100 node; do not pin a host.
# H100/H200 are allowed when available, while known bad nodes are excluded.
#SBATCH --job-name=upgrade-small-expo
#SBATCH --output=/work/hdd/bgqe/rpolonsky/dcrl/slurm/%x-%j.out
#SBATCH --error=/work/hdd/bgqe/rpolonsky/dcrl/slurm/%x-%j.err

set -euo pipefail
PROJECT=/work/hdd/bgqe/rpolonsky/dcrl
EXPO="$PROJECT/third_party/expo-ft"
PYTHON="$EXPO/.venv/bin/python"
LIBERO_ROOT="$PROJECT/.deps/LIBERO/libero/libero"
SOURCE_ROOT=/work/hdd/bgqe/rpolonsky/dcrlcheckpoints/small_expo_3phase_10_5_sharedstats
TARGET_ROOT=/work/hdd/bgqe/rpolonsky/dcrlcheckpoints/small_expo_3phase_dualopt_10_5_sharedstats
export PYTHONPATH="$PROJECT:$EXPO/expo_ft/agents/vla/openpi/src:$EXPO/expo_ft/agents/vla/openpi/packages/openpi-client/src:$PROJECT/.deps/LIBERO${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export JAX_PLATFORMS=cuda

mkdir -p "$PROJECT/.deps/libero_config"
cat > "$PROJECT/.deps/libero_config/config.yaml" <<CFG
benchmark_root: $LIBERO_ROOT
bddl_files: $LIBERO_ROOT/bddl_files
init_states: $LIBERO_ROOT/init_files
datasets: $PROJECT/datasets
assets: $LIBERO_ROOT/assets
CFG
export LIBERO_CONFIG_PATH="$PROJECT/.deps/libero_config"

echo "job=$SLURM_JOB_ID host=$(hostname) node=$SLURMD_NODENAME"
nvidia-smi -L
cd "$PROJECT"
"$PYTHON" scripts/upgrade_legacy_distill_checkpoint.py \
  --source-root "$SOURCE_ROOT" \
  --target-root "$TARGET_ROOT" \
  --suite libero_90 \
  --task-ids 28 29 60 79 \
  --image-size 224 \
  --action-horizon 10 \
  --replan-steps 5

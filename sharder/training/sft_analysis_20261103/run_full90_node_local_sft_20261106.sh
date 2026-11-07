#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0

# Evaluate an isolated audit-SFT checkpoint that lives on the current compute
# node's /tmp.  The caller supplies the checkpoint root and output directory;
# this wrapper never reads or writes a production checkpoint tree.
CHECKPOINT_ROOT=${1:?usage: $0 CHECKPOINT_ROOT OUTPUT_DIR RUNTIME_DIR [SHARD_INDEX] [NUM_SHARDS]}
OUT=${2:?usage: $0 CHECKPOINT_ROOT OUTPUT_DIR RUNTIME_DIR [SHARD_INDEX] [NUM_SHARDS]}
RUNTIME=${3:?usage: $0 CHECKPOINT_ROOT OUTPUT_DIR RUNTIME_DIR [SHARD_INDEX] [NUM_SHARDS]}
SHARD_INDEX=${4:-0}
NUM_SHARDS=${5:-1}

PROJECT=/projects/bgqe/rpolonsky/dcrl
EXPO="$PROJECT/third_party/expo-ft"
AUDIT=/projects/bgqe/rpolonsky/dcrl/sharder/training/sft_analysis_20261103
EVALUATOR="$AUDIT/full90_audit_eval.py"
CHECKPOINT="$CHECKPOINT_ROOT/checkpoints"
ASSETS="$CHECKPOINT_ROOT/student_assets"

[[ -f "$CHECKPOINT_ROOT/sft_config.json" ]] || { echo "missing SFT manifest: $CHECKPOINT_ROOT/sft_config.json" >&2; exit 3; }
[[ -f "$ASSETS/physical-intelligence/libero/norm_stats.json" ]] || { echo "missing SFT assets: $ASSETS" >&2; exit 4; }
[[ "$SHARD_INDEX" =~ ^[0-9]+$ && "$NUM_SHARDS" =~ ^[1-9][0-9]*$ && "$SHARD_INDEX" -lt "$NUM_SHARDS" ]] || { echo "invalid shard arguments" >&2; exit 2; }

export PYTHONPATH="/projects/bgqe/rpolonsky/dcrl/sharder/training:$AUDIT:$PROJECT:$EXPO/expo_ft/agents/vla/openpi/src:$EXPO/expo_ft/agents/vla/openpi/packages/openpi-client/src:$PROJECT/.deps/LIBERO${PYTHONPATH:+:$PYTHONPATH}"
export LIBERO_CONFIG_PATH="$PROJECT/sharder/training/checkpoints/expo_rl_task2_batch4_defercache_200k_multihost2/runtime/libero_config"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false OPENPI_SKIP_INIT_JIT=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 TOKENIZERS_PARALLELISM=false
export OPENPI_DATA_HOME=/work/hdd/bgqe/rpolonsky/dcrlcheckpoints/openpi_cache
export XDG_CACHE_HOME="$RUNTIME/xdg" XDG_CONFIG_HOME="$RUNTIME/xdg_config" TMPDIR="$RUNTIME/tmp"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
mkdir -p "$RUNTIME" "$TMPDIR" "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$OUT"

cd "$PROJECT"
exec "$EXPO/.venv/bin/python" "$EVALUATOR" \
  --mode sft_pi --checkpoint "$CHECKPOINT" --assets-dir "$ASSETS" \
  --suite libero_90 --task-ids all --shard-index "$SHARD_INDEX" --num-shards "$NUM_SHARDS" \
  --episodes 20 --eval-seed 7 --seed 42 --image-size 224 \
  --action-horizon 10 --replan-steps 5 --fsdp-devices 1 \
  --output-dir "$OUT"

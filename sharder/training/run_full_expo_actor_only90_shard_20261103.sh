#!/usr/bin/env bash
set -euo pipefail
SHARD=${1:?shard index 0 or 1 required}
[[ "$SHARD" =~ ^[01]$ ]] || { echo "invalid shard: $SHARD" >&2; exit 2; }

PROJECT=/projects/bgqe/rpolonsky/dcrl
EXPO="$PROJECT/third_party/expo-ft"
EVALUATOR=/projects/bgqe/rpolonsky/dcrl/sharder/training/full_90_expo_actor_only_eval_20261103.py
ROOT=/work/hdd/bgqe/rpolonsky/dcrlcheckpoints/pi_robo_libero_unsharded_h200_b16_200k
CHECKPOINT="$ROOT/pi_robo_libero_unsharded_h200_b16_task29/checkpoints/60486"
OUT=/work/hdd/bgqe/rpolonsky/dcrlcheckpoints/full_expo_task29_actor_only_full90_20261103
RUNTIME="/tmp/rpolonsky-expo29-actor-only90-${SHARD}"

export PYTHONPATH="$PROJECT:$EXPO/expo_ft/agents/vla/openpi/src:$EXPO/expo_ft/agents/vla/openpi/packages/openpi-client/src:$PROJECT/.deps/LIBERO${PYTHONPATH:+:$PYTHONPATH}"
export LIBERO_CONFIG_PATH="$PROJECT/sharder/training/runtime/libero_config"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false OPENPI_SKIP_INIT_JIT=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 TOKENIZERS_PARALLELISM=false
export OPENPI_DATA_HOME=/work/hdd/bgqe/rpolonsky/dcrlcheckpoints/openpi_cache
export XDG_CACHE_HOME="$RUNTIME/xdg" XDG_CONFIG_HOME="$RUNTIME/xdg_config" TMPDIR="$RUNTIME/tmp"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
mkdir -p "$RUNTIME" "$TMPDIR" "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$OUT/shard_${SHARD}"
cat > "$LIBERO_CONFIG_PATH/config.yaml" <<CFG
benchmark_root: $PROJECT/.deps/LIBERO/libero/libero
bddl_files: $PROJECT/.deps/LIBERO/libero/libero/bddl_files
init_states: $PROJECT/.deps/LIBERO/libero/libero/init_files
datasets: $PROJECT/.deps/LIBERO/datasets
assets: $PROJECT/.deps/LIBERO/libero/libero/assets
CFG
cd "$PROJECT"
exec >"$OUT/shard_${SHARD}/launcher.log" 2>&1
echo "START actor-only Full EXPO Task29 checkpoint=60486 shard=${SHARD}/2 at $(date -Is)"
"$EXPO/.venv/bin/python" "$EVALUATOR" \
  --mode direct_rl_actor_only --checkpoint "$CHECKPOINT" --suite libero_90 \
  --shard-index "$SHARD" --num-shards 2 --episodes 20 --eval-seed 7 --seed 42 \
  --image-size 224 --action-horizon 10 --replan-steps 5 --fsdp-devices 1 \
  --output-dir "$OUT/shard_${SHARD}"
echo "DONE actor-only Full EXPO Task29 checkpoint=60486 shard=${SHARD}/2 at $(date -Is)"

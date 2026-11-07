#!/usr/bin/env bash
set -euo pipefail
SHARD=${1:?shard index 0..5 required}
[[ "$SHARD" =~ ^[0-5]$ ]] || { echo "invalid shard: $SHARD" >&2; exit 2; }
PROJECT=/projects/bgqe/rpolonsky/dcrl
EXPO="$PROJECT/third_party/expo-ft"
ROOT=/work/hdd/bgqe/rpolonsky/dcrlcheckpoints/pi_robo_libero_unsharded_h200_b16_200k
OUT="$ROOT/full90_eval_direct_rl_pi_20261102"
RUNTIME="/tmp/rpolonsky-expo90-shard-${SHARD}"
export PYTHONPATH="$PROJECT:$EXPO/expo_ft/agents/vla/openpi/src:$EXPO/expo_ft/agents/vla/openpi/packages/openpi-client/src:$PROJECT/.deps/LIBERO${PYTHONPATH:+:$PYTHONPATH}"
export LIBERO_CONFIG_PATH="$OUT/libero_config"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false OPENPI_SKIP_INIT_JIT=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 TOKENIZERS_PARALLELISM=false
export OPENPI_DATA_HOME=/work/hdd/bgqe/rpolonsky/dcrlcheckpoints/openpi_cache
export XDG_CACHE_HOME="$RUNTIME/xdg" XDG_CONFIG_HOME="$RUNTIME/xdg_config" TMPDIR="$RUNTIME/tmp"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
mkdir -p "$RUNTIME" "$TMPDIR" "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$OUT" "$LIBERO_CONFIG_PATH"
cat > "$LIBERO_CONFIG_PATH/config.yaml" <<CFG
benchmark_root: $PROJECT/.deps/LIBERO/libero/libero
bddl_files: $PROJECT/.deps/LIBERO/libero/libero/bddl_files
init_states: $PROJECT/.deps/LIBERO/libero/libero/init_files
datasets: $PROJECT/.deps/LIBERO/datasets
assets: $PROJECT/.deps/LIBERO/libero/libero/assets
CFG
cd "$PROJECT"
PY="$EXPO/.venv/bin/python"
declare -a TASKS=(2 28 29 60 79)
declare -a STEPS=(60165 40095 60486 60386 60241)
for i in "${!TASKS[@]}"; do
  task=${TASKS[$i]}; step=${STEPS[$i]}
  checkpoint="$ROOT/pi_robo_libero_unsharded_h200_b16_task${task}/checkpoints/$step"
  out="$OUT/task_${task}/shard_${SHARD}"
  mkdir -p "$out"
  echo "START task=$task checkpoint_step=$step shard=$SHARD/6 at $(date -Is)"
  "$PY" /projects/bgqe/rpolonsky/dcrl/sharder/training/full_90_libero_evals_inference.py \
    --mode direct_rl_pi --checkpoint "$checkpoint" --suite libero_90 \
    --shard-index "$SHARD" --num-shards 6 --episodes 20 --eval-seed 7 --seed 42 \
    --image-size 224 --action-horizon 10 --replan-steps 5 --fsdp-devices 1 \
    --output-dir "$out"
  echo "DONE task=$task shard=$SHARD at $(date -Is)"
done

#!/usr/bin/env bash
set -euo pipefail

SHARD=${1:?shard index 0, 1, or 2 required}
[[ "$SHARD" =~ ^[0-2]$ ]] || { echo "invalid shard: $SHARD" >&2; exit 2; }

PROJECT=/projects/bgqe/rpolonsky/dcrl
EXPO="$PROJECT/third_party/expo-ft"
EVALUATOR=/projects/bgqe/rpolonsky/dcrl/sharder/training/full_90_libero_evals_inference.py
ROOT=/work/hdd/bgqe/rpolonsky/dcrlcheckpoints/sft_agnostic_small_expo
OUT=/work/hdd/bgqe/rpolonsky/dcrlcheckpoints/sft_full90_actor_only_eval_20261103
RUNTIME="/tmp/rpolonsky-sft90-shard-${SHARD}"

export PYTHONPATH="$PROJECT:$EXPO/expo_ft/agents/vla/openpi/src:$EXPO/expo_ft/agents/vla/openpi/packages/openpi-client/src:$PROJECT/.deps/LIBERO${PYTHONPATH:+:$PYTHONPATH}"
export LIBERO_CONFIG_PATH="$OUT/libero_config"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false OPENPI_SKIP_INIT_JIT=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 TOKENIZERS_PARALLELISM=false
export OPENPI_DATA_HOME=/work/hdd/bgqe/rpolonsky/dcrlcheckpoints/openpi_cache
export XDG_CACHE_HOME="$RUNTIME/xdg" XDG_CONFIG_HOME="$RUNTIME/xdg_config" TMPDIR="$RUNTIME/tmp"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1

mkdir -p "$RUNTIME" "$TMPDIR" "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$OUT" "$LIBERO_CONFIG_PATH"
mkdir -p "$OUT/logs"
exec >>"$OUT/logs/shard_${SHARD}.log" 2>&1
cat > "$LIBERO_CONFIG_PATH/config.yaml" <<CFG
benchmark_root: $PROJECT/.deps/LIBERO/libero/libero
bddl_files: $PROJECT/.deps/LIBERO/libero/libero/bddl_files
init_states: $PROJECT/.deps/LIBERO/libero/libero/init_files
datasets: $PROJECT/.deps/LIBERO/datasets
assets: $PROJECT/.deps/LIBERO/libero/libero/assets
CFG

cd "$PROJECT"
PY="$EXPO/.venv/bin/python"
declare -a TASKS=(2 28 29)
declare -a STEPS=(20000 40000 20000)

for i in "${!TASKS[@]}"; do
  task=${TASKS[$i]}
  step=${STEPS[$i]}
  checkpoint="$ROOT/task_$(printf '%02d' "$task")_"*
  matches=( $checkpoint )
  ((${#matches[@]} == 1)) || { echo "Expected one SFT directory for task $task, found ${#matches[@]}" >&2; exit 3; }
  checkpoint="${matches[0]}/checkpoints/$step"
  [[ -d "$checkpoint" ]] || { echo "Missing SFT checkpoint: $checkpoint" >&2; exit 4; }
  output="$OUT/task_$(printf '%02d' "$task")_step_${step}/shard_${SHARD}"
  mkdir -p "$output"
  echo "START task=$task step=$step shard=$SHARD/3 at $(date -Is)"
  "$PY" "$EVALUATOR" \
    --mode sft_pi --checkpoint "$checkpoint" --suite libero_90 \
    --shard-index "$SHARD" --num-shards 3 --episodes 20 --eval-seed 7 --seed 42 \
    --image-size 224 --action-horizon 10 --replan-steps 5 --fsdp-devices 1 \
    --output-dir "$output"
  echo "DONE task=$task step=$step shard=$SHARD/3 at $(date -Is)"
done

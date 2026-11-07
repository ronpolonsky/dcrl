#!/usr/bin/env bash
set -euo pipefail

SHARD=${1:?expected shard index 0, 1, or 2}
[[ "$SHARD" =~ ^[0-2]$ ]] || { echo "invalid shard: $SHARD" >&2; exit 2; }
NAME=${2:-all}

PROJECT=/projects/bgqe/rpolonsky/dcrl
EXPO="$PROJECT/third_party/expo-ft"
AUDIT=/projects/bgqe/rpolonsky/dcrl/sharder/training/sft_analysis_20261103
EVALUATOR=/projects/bgqe/rpolonsky/dcrl/sharder/training/sft_full90_eval_task28_2k_20261103/evaluate_retention_latest.py
OUTPUT_ROOT=/work/hdd/bgqe/rpolonsky/dcrlcheckpoints/distillation_full90_eval_sharded_retry_20261104
OPENPI_DATA_HOME=/work/hdd/bgqe/rpolonsky/dcrlcheckpoints/openpi_cache

if [[ "$NAME" == all ]]; then
  NAMES=(task28_lora2000 task29_demo50_1500 task60_1000 task02_3000)
else
  NAMES=("$NAME")
fi

for NAME in "${NAMES[@]}"; do
  case "$NAME" in
    task28_lora2000)
      CHECKPOINT=/work/hdd/bgqe/rpolonsky/dcrlcheckpoints/sft_retention_scope_20261103/task_28_lora_only/latest_checkpoints
      ;;
    task29_demo50_1500)
      CHECKPOINT=/work/hdd/bgqe/rpolonsky/dcrlcheckpoints/sft_task29_replay_ablation_20261103/demo50_1500/latest_checkpoints
      ;;
    task60_1000)
      CHECKPOINT=/work/hdd/bgqe/rpolonsky/dcrlcheckpoints/sft_task02_60_bestcheckpoint_20261103/task_60_small_expo_only/checkpoints
      ;;
    task02_3000)
      CHECKPOINT=/work/hdd/bgqe/rpolonsky/dcrlcheckpoints/sft_task02_60_bestcheckpoint_20261103/task_02_small_expo_only/checkpoints
      ;;
    *)
      echo "unknown evaluation name: $NAME" >&2
      exit 2
      ;;
  esac

  RUN_OUT="$OUTPUT_ROOT/$NAME"
  OUT="$RUN_OUT/shard_$SHARD"
  RUNTIME="/tmp/rpolonsky-distill90-${NAME}-shard${SHARD}"
  LIBERO_CONFIG_PATH="$RUNTIME/libero_config"
  export PYTHONPATH="$AUDIT:$PROJECT:$EXPO/expo_ft/agents/vla/openpi/src:$EXPO/expo_ft/agents/vla/openpi/packages/openpi-client/src:$PROJECT/.deps/LIBERO${PYTHONPATH:+:$PYTHONPATH}"
  export LIBERO_CONFIG_PATH
  export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
  export XLA_PYTHON_CLIENT_PREALLOCATE=false OPENPI_SKIP_INIT_JIT=1
  export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 TOKENIZERS_PARALLELISM=false
  export OPENPI_DATA_HOME
  export XDG_CACHE_HOME="$RUNTIME/xdg" XDG_CONFIG_HOME="$RUNTIME/xdg_config" TMPDIR="$RUNTIME/tmp"
  export JAX_COMPILATION_CACHE_DIR="$RUNTIME/jax_cache"
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1

  mkdir -p "$RUNTIME" "$TMPDIR" "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$JAX_COMPILATION_CACHE_DIR" "$LIBERO_CONFIG_PATH" "$OUT"
  cat > "$LIBERO_CONFIG_PATH/config.yaml" <<CFG
benchmark_root: $PROJECT/.deps/LIBERO/libero/libero
bddl_files: $PROJECT/.deps/LIBERO/libero/libero/bddl_files
init_states: $PROJECT/.deps/LIBERO/libero/libero/init_files
datasets: $PROJECT/.deps/LIBERO/datasets
assets: $PROJECT/.deps/LIBERO/libero/libero/assets
CFG

  cd "$PROJECT"
  LOG="$OUT/launcher.log"
  echo "START name=$NAME shard=$SHARD/3 checkpoint=$CHECKPOINT at $(date -Is)" | tee -a "$LOG"
  test -d "$CHECKPOINT"
  "$EXPO/.venv/bin/python" "$EVALUATOR" \
    --mode sft_pi --checkpoint "$CHECKPOINT" --suite libero_90 \
    --shard-index "$SHARD" --num-shards 3 --episodes 20 --eval-seed 7 --seed 42 \
    --image-size 224 --action-horizon 10 --replan-steps 5 --fsdp-devices 1 \
    --output-dir "$OUT" >>"$LOG" 2>&1
  echo "DONE name=$NAME shard=$SHARD/3 at $(date -Is)" | tee -a "$LOG"
done

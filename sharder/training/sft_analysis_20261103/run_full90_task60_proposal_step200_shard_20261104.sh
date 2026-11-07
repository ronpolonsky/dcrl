#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0

SHARD=${1:?usage: run_full90_task60_proposal_step200_shard_20261104.sh SHARD_INDEX NUM_SHARDS}
NUM_SHARDS=${2:?usage: run_full90_task60_proposal_step200_shard_20261104.sh SHARD_INDEX NUM_SHARDS}
[[ "$SHARD" =~ ^[0-9]+$ && "$NUM_SHARDS" =~ ^[1-9][0-9]*$ ]] || {
  echo "SHARD_INDEX and NUM_SHARDS must be non-negative/positive integers" >&2
  exit 2
}
(( SHARD < NUM_SHARDS )) || { echo "shard index must be smaller than num shards" >&2; exit 2; }

PROJECT=/projects/bgqe/rpolonsky/dcrl
EXPO="$PROJECT/third_party/expo-ft"
AUDIT=/projects/bgqe/rpolonsky/dcrl/sharder/training/sft_analysis_20261103
EVALUATOR="$AUDIT/full90_audit_eval.py"
ROOT=/work/hdd/bgqe/rpolonsky/dcrlcheckpoints/sft_retention_diagnosis_20261104/task_60_full_proposal
CHECKPOINT="$ROOT/checkpoints/200"
ASSETS="$ROOT/student_assets"
OUT=/work/hdd/bgqe/rpolonsky/dcrlcheckpoints/sft_retention_full90_eval_20261104/task60_full_proposal_step200
RUN="$AUDIT/full90_task60_step200_sharded"
RUNTIME="/tmp/rpolonsky-sft90-task60-proposal200-shard${SHARD}"

[[ -f "$CHECKPOINT/_CHECKPOINT_METADATA" ]] || { echo "checkpoint incomplete: $CHECKPOINT" >&2; exit 3; }
[[ -f "$ASSETS/physical-intelligence/libero/norm_stats.json" ]] || { echo "missing assets: $ASSETS" >&2; exit 4; }
export PYTHONPATH="/projects/bgqe/rpolonsky/dcrl/sharder/training:$AUDIT:$PROJECT:$EXPO/expo_ft/agents/vla/openpi/src:$EXPO/expo_ft/agents/vla/openpi/packages/openpi-client/src:$PROJECT/.deps/LIBERO${PYTHONPATH:+:$PYTHONPATH}"
export LIBERO_CONFIG_PATH="$PROJECT/sharder/training/checkpoints/expo_rl_task2_batch4_defercache_200k_multihost2/runtime/libero_config"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false OPENPI_SKIP_INIT_JIT=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 TOKENIZERS_PARALLELISM=false
export OPENPI_DATA_HOME=/work/hdd/bgqe/rpolonsky/dcrlcheckpoints/openpi_cache
export XDG_CACHE_HOME="$RUNTIME/xdg" XDG_CONFIG_HOME="$RUNTIME/xdg_config" TMPDIR="$RUNTIME/tmp"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
mkdir -p "$RUNTIME" "$TMPDIR" "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$OUT" "$RUN"
cd "$PROJECT"
exec >>"$RUN/shard_${SHARD}_of_${NUM_SHARDS}.log" 2>&1
echo "START task=60 checkpoint=200 shard=$SHARD/$NUM_SHARDS at $(date -Is)"
exec "$EXPO/.venv/bin/python" "$EVALUATOR" \
  --mode sft_pi --checkpoint "$CHECKPOINT" --assets-dir "$ASSETS" \
  --suite libero_90 --shard-index "$SHARD" --num-shards "$NUM_SHARDS" \
  --episodes 20 --eval-seed 7 --seed 42 --image-size 224 \
  --action-horizon 10 --replan-steps 5 --fsdp-devices 1 \
  --output-dir "$OUT"

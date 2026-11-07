#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0

# The Task-29 trainer owns this node-local /tmp tree.  Do not evaluate until
# that trainer exits, then evaluate the already-committed update-300 point.
TRAINER_PID=2770312
CHECKPOINT_ROOT=/tmp/usertmp/rpolonsky/sft_retention_diagnosis_20261106/task_29_pure_seed123
OUT=/tmp/usertmp/rpolonsky/sft_full90_confirmation_20261106/task29_update300
RUNTIME=/tmp/usertmp/rpolonsky/sft_full90_runtime_20261106/task29_update300
PROJECT=/projects/bgqe/rpolonsky/dcrl
EXPO="$PROJECT/third_party/expo-ft"
AUDIT=/projects/bgqe/rpolonsky/dcrl/sharder/training/sft_analysis_20261103

while kill -0 "$TRAINER_PID" 2>/dev/null; do sleep 60; done

test -d "$CHECKPOINT_ROOT/checkpoints/300"
test -d "$CHECKPOINT_ROOT/student_assets"

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
exec "$EXPO/.venv/bin/python" "$AUDIT/full90_audit_eval.py" \
  --mode sft_pi --checkpoint "$CHECKPOINT_ROOT/checkpoints/300" \
  --assets-dir "$CHECKPOINT_ROOT/student_assets" --suite libero_90 \
  --task-ids all --shard-index 0 --num-shards 1 --episodes 20 \
  --eval-seed 7 --seed 42 --image-size 224 --action-horizon 10 \
  --replan-steps 5 --fsdp-devices 1 --output-dir "$OUT"

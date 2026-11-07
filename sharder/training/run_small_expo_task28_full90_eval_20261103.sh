#!/usr/bin/env bash
set -euo pipefail
PROJECT=/projects/bgqe/rpolonsky/dcrl
EXPO="$PROJECT/third_party/expo-ft"
EVALUATOR=/projects/bgqe/rpolonsky/dcrl/sharder/training/full_90_small_expo_teacher_eval_20261103.py
CHECKPOINT=/u/rpolonsky/dcrlcheckpoints/small_expo_rl_10_5_offline50k_online500k_resume_newgpu/libero_90/task_28/checkpoints/275001
OUT=/work/hdd/bgqe/rpolonsky/dcrlcheckpoints/small_expo_task28_full90_eval_20261103
RUNTIME=/tmp/rpolonsky-small-expo28-full90

export PYTHONPATH="$PROJECT:$EXPO/expo_ft/agents/vla/openpi/src:$EXPO/expo_ft/agents/vla/openpi/packages/openpi-client/src:$PROJECT/.deps/LIBERO${PYTHONPATH:+:$PYTHONPATH}"
export LIBERO_CONFIG_PATH="$PROJECT/sharder/training/runtime/libero_config"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false OPENPI_SKIP_INIT_JIT=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 TOKENIZERS_PARALLELISM=false
export OPENPI_DATA_HOME=/work/hdd/bgqe/rpolonsky/dcrlcheckpoints/openpi_cache
export XDG_CACHE_HOME="$RUNTIME/xdg" XDG_CONFIG_HOME="$RUNTIME/xdg_config" TMPDIR="$RUNTIME/tmp"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
mkdir -p "$RUNTIME" "$TMPDIR" "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$OUT"
cat > "$LIBERO_CONFIG_PATH/config.yaml" <<CFG
benchmark_root: $PROJECT/.deps/LIBERO/libero/libero
bddl_files: $PROJECT/.deps/LIBERO/libero/libero/bddl_files
init_states: $PROJECT/.deps/LIBERO/libero/libero/init_files
datasets: $PROJECT/.deps/LIBERO/datasets
assets: $PROJECT/.deps/LIBERO/libero/libero/assets
CFG
cd "$PROJECT"
exec >"$OUT/launcher.log" 2>&1
echo "START Small EXPO Task28 teacher checkpoint=275001 full LIBERO-90 at $(date -Is)"
"$EXPO/.venv/bin/python" "$EVALUATOR" \
  --mode compact --checkpoint "$CHECKPOINT" --suite libero_90 --task-ids all \
  --episodes 20 --eval-seed 7 --seed 42 --image-size 224 \
  --action-horizon 10 --replan-steps 5 --fsdp-devices 1 --output-dir "$OUT"
echo "DONE Small EXPO Task28 teacher full LIBERO-90 at $(date -Is)"

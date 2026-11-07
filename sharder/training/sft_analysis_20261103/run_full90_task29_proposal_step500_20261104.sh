#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0

PROJECT=/projects/bgqe/rpolonsky/dcrl
EXPO="$PROJECT/third_party/expo-ft"
EVALUATOR=/projects/bgqe/rpolonsky/dcrl/sharder/training/sft_analysis_20261103/full90_audit_eval.py
ROOT=/work/hdd/bgqe/rpolonsky/dcrlcheckpoints/sft_retention_diagnosis_20261104/task_29_full_proposal
CHECKPOINT="$ROOT/latest_checkpoints"
ASSETS="$ROOT/student_assets"
OUT=/work/hdd/bgqe/rpolonsky/dcrlcheckpoints/sft_retention_full90_eval_20261104/task29_full_proposal_step500
RUNTIME=/tmp/rpolonsky-sft90-task29-proposal500

[[ -f "$CHECKPOINT/500/_CHECKPOINT_METADATA" ]] || { echo "checkpoint incomplete: $CHECKPOINT/500" >&2; exit 3; }
[[ -f "$ASSETS/physical-intelligence/libero/norm_stats.json" ]] || { echo "missing assets: $ASSETS" >&2; exit 4; }
export PYTHONPATH="/projects/bgqe/rpolonsky/dcrl/sharder/training:/projects/bgqe/rpolonsky/dcrl/sharder/training/sft_analysis_20261103:$PROJECT:$EXPO/expo_ft/agents/vla/openpi/src:$EXPO/expo_ft/agents/vla/openpi/packages/openpi-client/src:$PROJECT/.deps/LIBERO${PYTHONPATH:+:$PYTHONPATH}"
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
  --suite libero_90 --task-ids all --episodes 20 --eval-seed 7 --seed 42 \
  --image-size 224 --action-horizon 10 --replan-steps 5 --fsdp-devices 1 \
  --output-dir "$OUT"

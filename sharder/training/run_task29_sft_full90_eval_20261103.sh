#!/usr/bin/env bash
set -euo pipefail
ARM=${1:?usage: run_task29_sft_full90_eval_20261103.sh no_replay|demo50}
case "$ARM" in
  no_replay) SOURCE=no_replay_1500 ;;
  demo50) SOURCE=demo50_1500 ;;
  *) echo "unknown Task 29 SFT arm: $ARM" >&2; exit 2 ;;
esac

PROJECT=/projects/bgqe/rpolonsky/dcrl
EXPO="$PROJECT/third_party/expo-ft"
EVALUATOR=/projects/bgqe/rpolonsky/dcrl/sharder/training/full_90_libero_evals_inference.py
ROOT=/work/hdd/bgqe/rpolonsky/dcrlcheckpoints/sft_task29_replay_ablation_20261103/$SOURCE
CHECKPOINT="$ROOT/latest_checkpoints/1500"
ASSETS="$ROOT/student_assets"
OUT=/work/hdd/bgqe/rpolonsky/dcrlcheckpoints/sft_task29_full90_eval_20261103/$ARM
RUNTIME="/tmp/rpolonsky-task29-sft90-${ARM}"

[[ -d "$CHECKPOINT" ]] || { echo "missing SFT checkpoint: $CHECKPOINT" >&2; exit 3; }
[[ -f "$ASSETS/physical-intelligence/libero/norm_stats.json" ]] || { echo "missing SFT norm stats: $ASSETS" >&2; exit 4; }
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
exec >"$OUT/launcher.log" 2>&1
echo "START actor-only Task29 SFT arm=$ARM checkpoint=1500 full LIBERO-90 at $(date -Is)"
"$EXPO/.venv/bin/python" "$EVALUATOR" \
  --mode sft_pi --checkpoint "$CHECKPOINT" --assets-dir "$ASSETS" \
  --suite libero_90 --task-ids all --episodes 20 --eval-seed 7 --seed 42 \
  --image-size 224 --action-horizon 10 --replan-steps 5 --fsdp-devices 1 \
  --output-dir "$OUT"
echo "DONE actor-only Task29 SFT arm=$ARM full LIBERO-90 at $(date -Is)"

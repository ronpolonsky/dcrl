#!/usr/bin/env bash
set -euo pipefail

# Pi0.5-compatible standalone OCBench SFT. This uses Pi0.5's native
# 10-action output head and executes 5 actions before replanning.

task=${1:?task id required}
steps=${2:-500000}
eval_every=${3:-50000}
eval_episodes=${4:-50}

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd -- "$script_dir/../.." && pwd)
project_root=${OCBENCH_PROJECT_ROOT:-$(dirname -- "$repo")}
run_root=${OCBENCH_RUN_ROOT:-$project_root/ocbench_pi05_sft_tuning_20261118}
runtime=${OCBENCH_RUNTIME:-$project_root/ocbench-runtime.kuYajpdO}
expo=${EXPO_FT_ROOT:-$project_root/dcrl/third_party/expo-ft}
openpi=$expo/expo_ft/agents/vla/openpi
python=${OCBENCH_PYTHON:-$expo/.venv/bin/python}
scratch=${OCBENCH_SCRATCH:-/tmp/${USER:-unknown}-ocbench-pi05-sft}
env_file=${OCBENCH_ENV_FILE:-$project_root/.env}
suite=ocbench_lite

result_root=${RESULT_ROOT:-$run_root/full_official_standalone_long_500k}
result="$result_root/task_$task"
archives="$scratch/full_official_standalone_long/task_$task"
cache="$scratch/release"

cd "$repo"
export PYTHONPATH="$repo:$runtime:$openpi/src:$openpi/packages/openpi-client/src"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.95
export XDG_CACHE_HOME="$scratch/cache"
export JAX_COMPILATION_CACHE_DIR="$scratch/jax-cache"
export TQDM_MININTERVAL=60

bash "$script_dir/stage_pi05_base.sh"
base=$(<"$scratch/base_path")
test -s "$base/params/manifest.ocdbt"
export OPENPI_DATA_HOME="${base%/openpi-assets/checkpoints/pi05_base}"

if [[ -f "$env_file" ]]; then
  set -a
  source "$env_file"
  set +a
fi
test -n "${WANDB_API_KEY:-}"
export WANDB_MODE=${WANDB_MODE:-online}

env_id=$("$python" -c 'import sys; from dcrl.ocbench_env import environment_id; print(environment_id("ocbench_lite", int(sys.argv[1])))' "$task")
mkdir -p "$result" "$cache"

echo "Preparing all eligible official OCBench episodes for $env_id on $(hostname -s)"
"$python" -u scripts/prepare_ocbench_sft_data.py \
  --suite "$suite" --task-id "$task" --episodes 0 --replan-steps 5 \
  --cache "$cache" --output-dir "$archives"

archive="$archives/$env_id.hdf5"
cp -- "${archive%.hdf5}.json" "$result/demo_selection.json"
episodes=$("$python" -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["episodes"]))' "$result/demo_selection.json")

"$python" -u -m dcrl.collect_ocbench_demos \
  --stats-from "$archive" --norm-stats-dir "$result/normalization" \
  --native-gripper-range

echo "SFT data loaded: task=$task episodes=$episodes archive=$archive"
echo "Training: steps=$steps eval_every=$eval_every eval_episodes=$eval_episodes batch=4 lr=5e-5 standard_lora horizon=10 replan=5"

"$python" -u -m dcrl.train_pi_sft_agnostic \
  --phase train --collection-source benchmark_demos \
  --source-suite "$suite" --source-task-id "$task" \
  --official-demo-root "$archives" --collection-episodes "$episodes" \
  --student-checkpoint "$base" \
  --norm-stats "$result/normalization/norm_stats.json" \
  --output-dir "$result" \
  --action-horizon 10 --replan-steps 5 \
  --num-sft-steps "$steps" --batch-size 4 \
  --warmup-steps 200 --lr-decay-steps "$steps" --learning-rate 0.00005 \
  --seed 42 --eval-seed 100000 --eval-every "$eval_every" \
  --eval-episodes "$eval_episodes" --skip-initial-eval \
  --checkpoint-policy "${CHECKPOINT_POLICY:-none}" --checkpoint-every "$eval_every" \
  --finetune-mode lora --trainable-scope lora_only \
  --wandb-project sft_checkpoint --wandb-entity distill_rl \
  --wandb-name "ocbench-lite-task-${task}-full-official-pi05-10_5" \
  --wandb-mode online --ocbench-reference-action-normalization

if [[ "${CHECKPOINT_POLICY:-none}" != none ]]; then
  test -d "$result/checkpoints"
fi
test -s "$result/metrics.jsonl"
echo "Completed task $task; output=$result"

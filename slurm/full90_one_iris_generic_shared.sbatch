#!/bin/bash
#SBATCH --account=bgqe-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --job-name=full90-one
#SBATCH --gres=gpu:nvidia_a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=/work/hdd/bgqe/rpolonsky/dcrl/slurm/%x-%j.out
#SBATCH --error=/work/hdd/bgqe/rpolonsky/dcrl/slurm/%x-%j.err
set -euo pipefail
: "${TASK_INDEX:?Set TASK_INDEX 0..62}"
PROJECT=/work/hdd/bgqe/rpolonsky/dcrl
EXPO="$PROJECT/third_party/expo-ft"
ROOT=/work/hdd/bgqe/rpolonsky/dcrlcheckpoints/full90_eval_sharded_20261023
NUM_SHARDS=7
checkpoints=(
  /work/hdd/bgqe/rpolonsky/dcrlcheckpoints/bc_distill_small_to_pi05base/task2_rlstep285184_500k/checkpoints/75000
  /work/hdd/bgqe/rpolonsky/dcrlcheckpoints/bc_distill_small_to_pi05base/task28_rlstep260029_500k/checkpoints/50000
  /work/hdd/bgqe/rpolonsky/dcrlcheckpoints/bc_distill_small_to_pi05base/task29_500k/checkpoints/100000
  /work/hdd/bgqe/rpolonsky/dcrlcheckpoints/bc_distill_small_to_pi05base/task60_rlstep235055_500k/checkpoints/50000
  /work/hdd/bgqe/rpolonsky/dcrlcheckpoints/bc_distill_small_to_pi05base/task79_rlstep410016_500k/checkpoints/75000
  /work/hdd/bgqe/rpolonsky/dcrlcheckpoints/pi_robo_libero_prod_200k_sphinx/task2/pi_robo_libero_prod_200k_sphinx_task2/checkpoints/161045
  /work/hdd/bgqe/rpolonsky/dcrlcheckpoints/pi_robo_libero_prod_200k_sphinx/task28/pi_robo_libero_prod_200k_sphinx_task28/checkpoints/60163
  /work/hdd/bgqe/rpolonsky/dcrlcheckpoints/pi_robo_libero_prod_200k_sphinx/task29/pi_robo_libero_prod_200k_sphinx_task29/checkpoints/80269
  /work/hdd/bgqe/rpolonsky/dcrlcheckpoints/pi_robo_libero_prod_200k_sphinx_retry_after_space/task60/pi_robo_libero_prod_200k_sphinx_task60_restart/checkpoints/40089
)
modes=(actor actor actor actor actor direct_rl_pi direct_rl_pi direct_rl_pi direct_rl_pi)
names=(sft_task02_75k sft_task28_50k sft_task29_100k sft_task60_50k sft_task79_75k full_task02_161045 full_task28_60163 full_task29_80269 full_task60_40089)
index=$((TASK_INDEX))
if (( index < 0 || index >= 63 )); then echo "bad TASK_INDEX=$index" >&2; exit 2; fi
ckpt_idx=$((index / NUM_SHARDS)); shard=$((index % NUM_SHARDS))
mode=${modes[$ckpt_idx]}; name=${names[$ckpt_idx]}
out="$ROOT/$mode/$name/shard_${shard}"
mkdir -p "$out/cache/xdg" "$out/cache/xdg_config" "$out/runtime/tmp" "$PROJECT/slurm"
export LIBERO_CONFIG_PATH="$PROJECT/.deps/libero_config"
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="$PROJECT:$EXPO/expo_ft/agents/vla/openpi/src:$EXPO/expo_ft/agents/vla/openpi/packages/openpi-client/src:$PROJECT/.deps/LIBERO${PYTHONPATH:+:$PYTHONPATH}"
export OPENPI_DATA_HOME=/work/hdd/bgqe/rpolonsky/dcrlcheckpoints/openpi_cache
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export XDG_CACHE_HOME="$out/cache/xdg"
export XDG_CONFIG_HOME="$out/cache/xdg_config"
export TMPDIR="$out/runtime/tmp"
cd "$PROJECT"
exec "$EXPO/.venv/bin/python" "$PROJECT/full_90_libero_evals.py" \
  --mode "$mode" --checkpoint "${checkpoints[$ckpt_idx]}" --suite libero_90 \
  --shard-index "$shard" --num-shards "$NUM_SHARDS" --episodes 20 --eval-seed 7 --seed 42 \
  --image-size 224 --action-horizon 10 --replan-steps 5 --output-dir "$out"

#!/bin/bash
#SBATCH --account=nlp
#SBATCH --partition=jag-hi
#SBATCH --array=19-20,23-28%8
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --output=/nlp/scr/ronpo/projects/dcrl/slurm/full90-jag-filtered-%A_%a.out
#SBATCH --error=/nlp/scr/ronpo/projects/dcrl/slurm/full90-jag-filtered-%A_%a.err

set -euo pipefail

TASK_INDEX="$SLURM_ARRAY_TASK_ID"
PROJECT=/nlp/scr2/ronpo/projects/dcrl
EXPO="$PROJECT/third_party/expo-ft"
ROOT=/nlp/scr2/ronpo/dcrlcheckpoints/full90_eval_sharded_20261023
NUM_SHARDS=7

checkpoints=(
  /nlp/scr2/ronpo/dcrlcheckpoints/bc_distill_small_to_pi05base/task2_rlstep285184_500k/checkpoints/75000
  /nlp/scr2/ronpo/dcrlcheckpoints/bc_distill_small_to_pi05base/task28_rlstep260029_500k/checkpoints/50000
  /nlp/scr2/ronpo/dcrlcheckpoints/bc_distill_small_to_pi05base/task29_500k/checkpoints/100000
  /nlp/scr2/ronpo/dcrlcheckpoints/bc_distill_small_to_pi05base/task60_rlstep235055_500k/checkpoints/50000
  /nlp/scr2/ronpo/dcrlcheckpoints/bc_distill_small_to_pi05base/task79_rlstep410016_500k/checkpoints/75000
  /nlp/scr2/ronpo/dcrlcheckpoints/pi_robo_libero_prod_200k_sphinx/task2/pi_robo_libero_prod_200k_sphinx_task2/checkpoints/161045
  /nlp/scr2/ronpo/dcrlcheckpoints/pi_robo_libero_prod_200k_sphinx/task28/pi_robo_libero_prod_200k_sphinx_task28/checkpoints/60163
  /nlp/scr2/ronpo/dcrlcheckpoints/pi_robo_libero_prod_200k_sphinx/task29/pi_robo_libero_prod_200k_sphinx_task29/checkpoints/80269
  /nlp/scr2/ronpo/dcrlcheckpoints/pi_robo_libero_prod_200k_sphinx_retry_after_space/task60/pi_robo_libero_prod_200k_sphinx_task60_restart/checkpoints/40089
)
modes=(actor actor actor actor actor full full full full)
names=(sft_task02_75k sft_task28_50k sft_task29_100k sft_task60_50k sft_task79_75k full_task02_161045 full_task28_60163 full_task29_80269 full_task60_40089)

if (( TASK_INDEX < 0 || TASK_INDEX >= 63 )); then
  echo "bad TASK_INDEX=$TASK_INDEX" >&2
  exit 2
fi

checkpoint_index=$((TASK_INDEX / NUM_SHARDS))
shard_index=$((TASK_INDEX % NUM_SHARDS))
mode=${modes[$checkpoint_index]}
name=${names[$checkpoint_index]}

requested=()
for ((task_id=shard_index; task_id<90; task_id+=NUM_SHARDS)); do
  requested+=("$task_id")
done
TASK_IDS=$(IFS=,; echo "${requested[*]}")

# Recheck all canonical roots at actual startup. Skip only valid, complete
# results for this exact checkpoint and task.
TASK_IDS=$("$EXPO/.venv/bin/python" - "$mode" "$name" "$TASK_IDS" <<'PY'
import json
import sys
from pathlib import Path

mode, checkpoint_name = sys.argv[1:3]
requested = [int(value) for value in sys.argv[3].split(",") if value]
roots = [
    Path("/nlp/scr2/ronpo/dcrlcheckpoints/full90_eval_sharded_20261023"),
    Path("/nlp/scr2/ronpo/dcrlcheckpoints/full90_eval_iris_hi_parallel_20261023"),
    Path("/nlp/scr2/ronpo/dcrlcheckpoints/full90_eval_twojob_20261023"),
]
complete = set()
for root in roots:
    directory = root / mode / checkpoint_name
    if not directory.is_dir():
        continue
    for path in directory.glob("**/task_*.json"):
        try:
            result = json.loads(path.read_text())
        except Exception:
            continue
        trials = result.get("trials")
        if (
            result.get("suite") == "libero_90"
            and result.get("episodes") == 20
            and isinstance(trials, list)
            and len(trials) == 20
            and [trial.get("initial_state_index") for trial in trials] == list(range(20))
            and isinstance(result.get("successes"), int)
            and result.get("action_horizon") == 10
            and result.get("replan_steps") == 5
        ):
            complete.add(result.get("task_id"))
print(",".join(str(task_id) for task_id in requested if task_id not in complete))
PY
)

if [[ -z "$TASK_IDS" ]]; then
  echo "All requested tasks already have valid 20-trial results; exiting."
  exit 0
fi

out="$ROOT/$mode/$name/shard_${shard_index}"
mkdir -p "$out/cache/xdg" "$out/cache/xdg_config" "$out/runtime/tmp" "$PROJECT/slurm"

export LIBERO_CONFIG_PATH="$PROJECT/.deps/libero_config"
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="$PROJECT:$EXPO/expo_ft/agents/vla/openpi/src:$EXPO/expo_ft/agents/vla/openpi/packages/openpi-client/src:$PROJECT/.deps/LIBERO${PYTHONPATH:+:$PYTHONPATH}"
export OPENPI_DATA_HOME=/nlp/scr2/ronpo/dcrlcheckpoints/openpi_cache
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export XDG_CACHE_HOME="$out/cache/xdg"
export XDG_CONFIG_HOME="$out/cache/xdg_config"
export TMPDIR="$out/runtime/tmp"

cd "$PROJECT"
exec "$EXPO/.venv/bin/python" "$PROJECT/full_90_libero_evals.py" \
  --mode "$mode" --checkpoint "${checkpoints[$checkpoint_index]}" --suite libero_90 \
  --task-ids "$TASK_IDS" --shard-index "$shard_index" --num-shards "$NUM_SHARDS" \
  --episodes 20 --eval-seed 7 --seed 42 --image-size 224 \
  --action-horizon 10 --replan-steps 5 --output-dir "$out"

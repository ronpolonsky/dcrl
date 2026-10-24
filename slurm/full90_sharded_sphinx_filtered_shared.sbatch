#!/bin/bash
#SBATCH --account=nlp
#SBATCH --partition=sphinx
#SBATCH --array=2,4,6,7,8,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62%4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --exclude=sphinx[1-3]
#SBATCH --output=/nlp/scr/ronpo/projects/dcrl/slurm/full90-filtered-%A_%a.out
#SBATCH --error=/nlp/scr/ronpo/projects/dcrl/slurm/full90-filtered-%A_%a.err

set -euo pipefail

TASK_INDEX="$SLURM_ARRAY_TASK_ID"
case "$TASK_INDEX" in
  2) TASK_IDS='23,30,37,44,51,58,65,72,79,86' ;;
  4) TASK_IDS='18,25,32,39,46,53,60,67,74,81,88' ;;
  6) TASK_IDS='13,20,27,34,41,48,55,62,69,76,83' ;;
  7) TASK_IDS='7,14,21,28,35,42,49,56,63,70,77,84' ;;
  8) TASK_IDS='8,15,22,29,36,43,50,57,64,71,78,85' ;;
  29) TASK_IDS='1,8,15,22,29,36,43,50,57,64,71,78,85' ;;
  30) TASK_IDS='2,9,16,23,30,37,44,51,58,65,72,79,86' ;;
  31) TASK_IDS='3,10,17,24,31,38,45,52,59,66,73,80,87' ;;
  32) TASK_IDS='4,11,18,25,32,39,46,53,60,67,74,81,88' ;;
  33) TASK_IDS='5,12,19,26,33,40,47,54,61,68,75,82,89' ;;
  34) TASK_IDS='6,13,20,27,34,41,48,55,62,69,76,83' ;;
  35) TASK_IDS='14,21,28,35,42,49,56,63,70,77,84' ;;
  36) TASK_IDS='15,22,29,36,43,50,57,64,71,78,85' ;;
  37) TASK_IDS='9,16,23,30,37,44,51,58,65,72,79,86' ;;
  38) TASK_IDS='10,17,24,31,38,45,52,59,66,73,80,87' ;;
  39) TASK_IDS='11,18,25,32,39,46,53,60,67,74,81,88' ;;
  40) TASK_IDS='12,19,26,33,40,47,54,61,68,75,82,89' ;;
  41) TASK_IDS='13,20,27,34,41,48,55,62,69,76,83' ;;
  42) TASK_IDS='0,7,14,21,28,35,42,49,56,63,70,77,84' ;;
  43) TASK_IDS='1,8,15,22,29,36,43,50,57,64,71,78,85' ;;
  44) TASK_IDS='2,9,16,23,30,37,44,51,58,65,72,79,86' ;;
  45) TASK_IDS='3,10,17,24,31,38,45,52,59,66,73,80,87' ;;
  46) TASK_IDS='4,11,18,25,32,39,46,53,60,67,74,81,88' ;;
  47) TASK_IDS='5,12,19,26,33,40,47,54,61,68,75,82,89' ;;
  48) TASK_IDS='6,13,20,27,34,41,48,55,62,69,76,83' ;;
  49) TASK_IDS='0,7,14,21,28,35,42,49,56,63,70,77,84' ;;
  50) TASK_IDS='1,8,15,22,29,36,43,50,57,64,71,78,85' ;;
  51) TASK_IDS='2,9,16,23,30,37,44,51,58,65,72,79,86' ;;
  52) TASK_IDS='3,10,17,24,31,38,45,52,59,66,73,80,87' ;;
  53) TASK_IDS='4,11,18,25,32,39,46,53,60,67,74,81,88' ;;
  54) TASK_IDS='5,12,19,26,33,40,47,54,61,68,75,82,89' ;;
  55) TASK_IDS='6,13,20,27,34,41,48,55,62,69,76,83' ;;
  56) TASK_IDS='0,7,14,21,28,35,42,49,56,63,70,77,84' ;;
  57) TASK_IDS='1,8,15,22,29,36,43,50,57,64,71,78,85' ;;
  58) TASK_IDS='2,9,16,23,30,37,44,51,58,65,72,79,86' ;;
  59) TASK_IDS='3,10,17,24,31,38,45,52,59,66,73,80,87' ;;
  60) TASK_IDS='4,11,18,25,32,39,46,53,60,67,74,81,88' ;;
  61) TASK_IDS='5,12,19,26,33,40,47,54,61,68,75,82,89' ;;
  62) TASK_IDS='6,13,20,27,34,41,48,55,62,69,76,83' ;;
  *) echo "unexpected shard index: $TASK_INDEX" >&2; exit 2 ;;
esac

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

checkpoint_index=$((TASK_INDEX / NUM_SHARDS))
shard_index=$((TASK_INDEX % NUM_SHARDS))
mode=${modes[$checkpoint_index]}
name=${names[$checkpoint_index]}

# Results can finish while this array waits in the queue. Re-filter at actual
# startup and skip only task files that contain all 20 requested trials.
TASK_IDS=$("$EXPO/.venv/bin/python" - "$name" "$TASK_IDS" <<'PY'
import json
import sys
from pathlib import Path

checkpoint_name = sys.argv[1]
requested = [int(value) for value in sys.argv[2].split(",") if value]
roots = [
    Path("/nlp/scr2/ronpo/dcrlcheckpoints/full90_eval_sharded_20261023"),
    Path("/nlp/scr2/ronpo/dcrlcheckpoints/full90_eval_iris_hi_parallel_20261023"),
    Path("/nlp/scr2/ronpo/dcrlcheckpoints/full90_eval_twojob_20261023"),
]
complete = set()
for root in roots:
    for path in root.rglob(f"{checkpoint_name}/**/task_*.json"):
        try:
            result = json.loads(path.read_text())
        except Exception:
            continue
        trials = result.get("trials")
        if (
            result.get("episodes") == 20
            and isinstance(trials, list)
            and len(trials) == 20
            and isinstance(result.get("successes"), int)
        ):
            complete.add(result.get("task_id"))
print(",".join(str(task_id) for task_id in requested if task_id not in complete))
PY
)
if [[ -z "$TASK_IDS" ]]; then
  echo "All requested tasks already have complete 20-trial results; exiting."
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

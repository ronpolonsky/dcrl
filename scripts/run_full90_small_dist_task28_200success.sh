#!/usr/bin/env bash
set -euo pipefail

PROJECT=/projects/bgqe/rpolonsky/dcrl
CHECKPOINT=/work/hdd/bgqe/rpolonsky/dcrlcheckpoints/sft_hybrid_more_data_20261116/small_dist/task_28
OUTPUT=/work/hdd/bgqe/rpolonsky/results/full90_small_dist_task28_200success_20261117
EVAL_SCRIPT=$PROJECT/scripts/full_90_libero_evals.sbatch
MERGE_SCRIPT=$PROJECT/scripts/merge_full_90_libero_evals.py
PYTHON=$PROJECT/third_party/expo-ft/.venv/bin/python
JOBS=(23832136_4 23832136_2 23832136_1 23811972_5 23811972_2)

[[ -f "$CHECKPOINT/sft_config.json" ]] || { echo "SFT manifest missing" >&2; exit 2; }
[[ -f "$CHECKPOINT/checkpoints/1500/_CHECKPOINT_METADATA" ]] || { echo "Complete best checkpoint missing" >&2; exit 2; }
[[ ! -e "$OUTPUT/summary.json" ]] || { echo "Summary already exists; refusing overwrite" >&2; exit 3; }
mkdir -p "$OUTPUT"

for shard in 0 1 2 3 4; do
  session=eval90_small_dist28_200_s${shard}
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "Session already exists: $session" >&2; exit 4
  fi
  job=${JOBS[$shard]}
  log=$OUTPUT/shard_${shard}.log
  tmux new-session -d -s "$session" \
    "srun --jobid=$job --overlap --unbuffered --ntasks=1 --cpus-per-task=12 bash -lc 'MODE=sft_pi CHECKPOINT=$CHECKPOINT OUTPUT_DIR=$OUTPUT SHARD_INDEX=$shard PROJECT_ROOT=$PROJECT bash $EVAL_SCRIPT' >$log 2>&1"
  echo "START shard=$shard job=$job session=$session"
done

while true; do
  live=0
  for shard in 0 1 2 3 4; do
    tmux has-session -t "eval90_small_dist28_200_s${shard}" 2>/dev/null && live=$((live + 1)) || true
  done
  (( live == 0 )) && break
  sleep 20
done

"$PYTHON" "$MERGE_SCRIPT" "$OUTPUT" --output "$OUTPUT/summary.json"
echo "COMPLETE $OUTPUT/summary.json"

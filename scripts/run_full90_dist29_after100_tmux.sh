#!/usr/bin/env bash
set -euo pipefail

# Queue the newly completed Small+Dist task-29 checkpoint behind the existing
# five-checkpoint full-suite controller. It reuses the same five idle A100
# allocations only after all current evaluation summaries exist.

PROJECT=/projects/bgqe/rpolonsky/dcrl
EXPO=$PROJECT/third_party/expo-ft
EVAL_SCRIPT=$PROJECT/scripts/full_90_libero_evals.sbatch
MERGE_SCRIPT=$PROJECT/scripts/merge_full_90_libero_evals.py
ROOT=/work/hdd/bgqe/rpolonsky/results/full90_100pct_sft_pi_20261116
LABEL=small_dist_task29_after100
CHECKPOINT=/work/hdd/bgqe/rpolonsky/dcrlcheckpoints/sft_from_hybrid_tuned_20k_20261116/small_dist/task_29
OUTPUT=$ROOT/$LABEL
JOBS=(23832136_4 23832136_2 23832136_1 23811972_5 23811972_2)

for current in small_task02 small_task29 small_task79 small_dist_task02 small_dist_task79; do
  while [[ ! -f "$ROOT/$current/summary.json" ]]; do
    if ! tmux has-session -t eval90_controller 2>/dev/null; then
      echo "current controller ended before $current completed" >&2
      exit 1
    fi
    sleep 30
  done
done

[[ -f "$CHECKPOINT/sft_config.json" ]] || { echo "missing checkpoint config" >&2; exit 2; }
[[ -f "$CHECKPOINT/checkpoints/2000/_CHECKPOINT_METADATA" ]] || { echo "missing complete step-2000 checkpoint" >&2; exit 2; }
[[ ! -e "$OUTPUT/summary.json" ]] || { echo "evaluation already exists: $OUTPUT" >&2; exit 2; }
mkdir -p "$OUTPUT"

for shard in 0 1 2 3 4; do
  session="eval90_${LABEL}_s${shard}"
  job=${JOBS[$shard]}
  log="$OUTPUT/shard_${shard}.log"
  tmux new-session -d -s "$session" \
    "srun --jobid=$job --overlap --unbuffered --ntasks=1 --cpus-per-task=12 bash -lc 'MODE=sft_pi CHECKPOINT=$CHECKPOINT OUTPUT_DIR=$OUTPUT SHARD_INDEX=$shard PROJECT_ROOT=$PROJECT bash $EVAL_SCRIPT' >$log 2>&1"
done

while true; do
  alive=0
  for shard in 0 1 2 3 4; do
    tmux has-session -t "eval90_${LABEL}_s${shard}" 2>/dev/null && alive=$((alive + 1)) || true
  done
  (( alive == 0 )) && break
  sleep 20
done

"$EXPO/.venv/bin/python" "$MERGE_SCRIPT" "$OUTPUT" --output "$OUTPUT/summary.json"
echo "DONE $LABEL"

#!/usr/bin/env bash
set -euo pipefail

# Run complete LIBERO-90 actor-only evaluations for the five SFT checkpoints
# that previously reached 100% on their task. Five idle A100 allocations are
# reused as modulo shards, one checkpoint at a time. Existing training jobs
# are deliberately not referenced here.

PROJECT=/projects/bgqe/rpolonsky/dcrl
EXPO=$PROJECT/third_party/expo-ft
EVAL_SCRIPT=$PROJECT/scripts/full_90_libero_evals.sbatch
MERGE_SCRIPT=$PROJECT/scripts/merge_full_90_libero_evals.py
ROOT=/work/hdd/bgqe/rpolonsky/results/full90_100pct_sft_pi_20261116

labels=(small_task02 small_task29 small_task79 small_dist_task02 small_dist_task79)
checkpoints=(
  /work/hdd/bgqe/rpolonsky/dcrlcheckpoints/sft_from_hybrid_tuned_20k_20261116/small/task_02
  /work/hdd/bgqe/rpolonsky/dcrlcheckpoints/sft_from_hybrid_tuned_20k_20261116/small/task_29
  /work/hdd/bgqe/rpolonsky/dcrlcheckpoints/sft_from_hybrid_tuned_20k_20261116/small/task_79
  /work/hdd/bgqe/rpolonsky/dcrlcheckpoints/sft_from_hybrid_tuned_20k_20261116/small_dist/task_02
  /work/hdd/bgqe/rpolonsky/dcrlcheckpoints/sft_from_hybrid_tuned_20k_20261116/small_dist/task_79
)

# These are the five allocations verified idle before launch.
jobs=(23832136_4 23832136_2 23832136_1 23811972_5 23811972_2)

mkdir -p "$ROOT"

for i in "${!labels[@]}"; do
  label=${labels[$i]}
  checkpoint=${checkpoints[$i]}
  output="$ROOT/$label"
  [[ -d "$checkpoint" ]] || { echo "missing checkpoint root: $checkpoint" >&2; exit 2; }
  [[ -f "$checkpoint/sft_config.json" ]] || { echo "missing SFT config: $checkpoint" >&2; exit 2; }
  [[ -d "$checkpoint/checkpoints" ]] || { echo "missing checkpoints: $checkpoint" >&2; exit 2; }
  [[ ! -e "$output/summary.json" ]] || { echo "evaluation already exists: $output" >&2; exit 2; }

  mkdir -p "$output"
  echo "START $label checkpoint=$checkpoint output=$output"

  for shard in 0 1 2 3 4; do
    session="eval90_${label}_s${shard}"
    if tmux has-session -t "$session" 2>/dev/null; then
      echo "tmux session already exists: $session" >&2
      exit 3
    fi
    job=${jobs[$shard]}
    log="$output/shard_${shard}.log"
    tmux new-session -d -s "$session" \
      "srun --jobid=$job --overlap --unbuffered --ntasks=1 --cpus-per-task=12 bash -lc 'MODE=sft_pi CHECKPOINT=$checkpoint OUTPUT_DIR=$output SHARD_INDEX=$shard PROJECT_ROOT=$PROJECT bash $EVAL_SCRIPT' >$log 2>&1"
  done

  while true; do
    alive=0
    for shard in 0 1 2 3 4; do
      tmux has-session -t "eval90_${label}_s${shard}" 2>/dev/null && alive=$((alive + 1)) || true
    done
    (( alive == 0 )) && break
    sleep 20
  done

  "$EXPO/.venv/bin/python" "$MERGE_SCRIPT" "$output" --output "$output/summary.json"
  echo "DONE $label"
done

echo "ALL FULL LIBERO-90 EVALUATIONS COMPLETE: $ROOT"

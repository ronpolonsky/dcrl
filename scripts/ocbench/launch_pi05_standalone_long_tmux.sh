#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd -- "$script_dir/../.." && pwd)
run_root=${OCBENCH_RUN_ROOT:-$(dirname -- "$repo")/ocbench_pi05_sft_tuning_20261118}
runner="$script_dir/run_pi05_standalone_long.sh"
result_root=${RESULT_ROOT:-$run_root/full_official_standalone_long_500k}
mkdir -p "$run_root/logs"

if [[ $# -eq 0 ]]; then
  echo "usage: $0 JOBID:task [JOBID:task ...]" >&2
  exit 2
fi

for spec in "$@"; do
  allocation_id=${spec%%:*}
  task=${spec##*:}
  [[ "$allocation_id" =~ ^[0-9]+(_[0-9]+)?$ && "$task" =~ ^[0-9]+$ ]] || {
    echo "invalid allocation:task specification: $spec" >&2
    exit 2
  }
  session="ocbench-standalone-long-task${task}-pi05-10_5"
  log="$run_root/logs/${session}.log"
  if tmux has-session -t "=$session" 2>/dev/null; then
    echo "tmux session already exists: $session" >&2
    continue
  fi
  echo "Launching long task $task on allocation $allocation_id in $session"
  tmux new-session -d -s "$session" \
    "bash -o pipefail -c 'srun --jobid=$allocation_id --overlap --unbuffered --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 --export=ALL,RESULT_ROOT=$result_root,CHECKPOINT_POLICY=none bash $runner $task 500000 50000 50 2>&1 | tee -a $log; rc=\${PIPESTATUS[0]}; echo \"[\$(date -Is)] exit=\$rc\" >> $log; exit \$rc'"
done

#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd -- "$script_dir/../.." && pwd)
run_root=${OCBENCH_RUN_ROOT:-$(dirname -- "$repo")/ocbench_pi05_sft_tuning_20261118}
launcher="$script_dir/launch_pi05_standalone_long_tmux.sh"
log="$run_root/logs/dispatch_pi05_standalone_long.log"
mkdir -p "$run_root/logs"
exec >>"$log" 2>&1

if [[ -n "${OCBENCH_TASK_QUEUE:-}" ]]; then
  read -r -a tasks <<<"$OCBENCH_TASK_QUEUE"
else
  tasks=(10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27)
fi
next=0

echo "[$(date -Is)] dispatcher started; remaining tasks: ${tasks[*]}"
while (( next < ${#tasks[@]} )); do
  launched=0
  mapfile -t jobs < <(
    squeue -r -h -u "$(id -un)" -o '%i %T %j' \
      | awk '$2 == "RUNNING" && $3 ~ /^gpu-sleep-ocbench/ {print $1}'
  )
  for array_job in "${jobs[@]}"; do
    (( next < ${#tasks[@]} )) || break
    mapfile -t active_steps < <(squeue -h --steps -j "$array_job" -o '%i' 2>/dev/null || true)
    busy=0
    for step in "${active_steps[@]}"; do
      [[ "$step" =~ \.(batch|extern)$ ]] || { busy=1; break; }
    done
    (( busy == 0 )) || continue

    concrete=$(scontrol -o show job "$array_job" | sed -n 's/^JobId=\([^ ]*\).*/\1/p')
    [[ "$concrete" =~ ^[0-9]+$ ]] || concrete="$array_job"
    task=${tasks[$next]}
    session="ocbench-standalone-long-task${task}-pi05-10_5"
    if tmux has-session -t "=$session" 2>/dev/null; then
      echo "[$(date -Is)] task $task already has session $session; marking dispatched"
      ((next += 1))
      continue
    fi
    echo "[$(date -Is)] dispatching task $task on $array_job (allocation $concrete)"
    "$launcher" "$concrete:$task"
    ((next += 1))
    launched=1
  done
  if (( next < ${#tasks[@]} )); then
    if (( launched == 0 )); then
      sleep 60
    else
      sleep 5
    fi
  fi
done
echo "[$(date -Is)] all standalone tasks dispatched"

#!/usr/bin/env bash
set -euo pipefail

# One-shot handoff from the no-checkpoint Task-28 screen to its validation.
# The caller supplies the login-node PID of the existing Task-28 srun.  This
# never kills or signals that process; it only waits for normal termination.
WAIT_PID=${1:?usage: $0 EXISTING_TASK28_SRUN_PID}
while [[ -e "/proc/${WAIT_PID}/cmdline" ]]; do
  sleep 20
done

exec srun --jobid=23433381_2 --overlap --ntasks=1 --cpus-per-task=12 --gres=gpu:1 \
  bash -lc 'exec bash /projects/bgqe/rpolonsky/dcrl/sharder/training/sft_analysis_20261103/run_task28_demo25_checkpointed_validation_20261106.sh'

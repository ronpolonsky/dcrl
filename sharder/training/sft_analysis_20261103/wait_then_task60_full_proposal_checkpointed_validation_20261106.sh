#!/usr/bin/env bash
set -euo pipefail
WAIT_PID=${1:?usage: $0 EXISTING_TASK28_VALIDATION_SRUN_PID}
while [[ -e "/proc/${WAIT_PID}/cmdline" ]]; do
  sleep 20
done
exec srun --jobid=23433381_2 --overlap --ntasks=1 --cpus-per-task=12 --gres=gpu:1 \
  bash -lc 'exec bash /projects/bgqe/rpolonsky/dcrl/sharder/training/sft_analysis_20261103/run_task60_full_proposal_checkpointed_validation_capture_20261106.sh'

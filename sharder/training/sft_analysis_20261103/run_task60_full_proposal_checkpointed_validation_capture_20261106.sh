#!/usr/bin/env bash
set -u

OUT=/tmp/usertmp/rpolonsky/sft_retention_diagnosis_20261106/task_60_full_proposal_checkpointed_validation
mkdir -p "$OUT"
export SFT_TASK60_VALIDATION_OUTPUT="$OUT"
export SFT_TASK60_VALIDATION_RUNTIME=/tmp/usertmp/rpolonsky/sft_runtime_task60_full_proposal_checkpointed_validation
export SFT_TASK60_VALIDATION_EVAL_EPISODES=50

set +e
bash /projects/bgqe/rpolonsky/dcrl/sharder/training/sft_analysis_20261103/run_task60_full_proposal_checkpointed_validation_20261106.sh \
  >"$OUT/launcher.log" 2>&1
rc=$?
printf 'EXIT_CODE=%s\n' "$rc" >>"$OUT/launcher.log"
if [[ "$rc" != 0 ]]; then sleep 300; fi
exit "$rc"

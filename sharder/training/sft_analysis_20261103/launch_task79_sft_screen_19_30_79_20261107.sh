#!/usr/bin/env bash
set -euo pipefail

# Isolated diagnosis arm.  This uses the already completed/task-pure Task-79
# Small EXPO teacher; it does not modify the live Task-79 RL allocation.
JOB_ID="23505371_2"
SESSION="sft_task79_screen_19_30_79"
SOCKET="/tmp/tmux_sft_task79_screen_19_30_79.sock"
ROOT="/tmp/usertmp/rpolonsky/task79_sft_screen_19_30_79_20261107"
OUT="${ROOT}/sft_output"
TMP="${ROOT}/runtime/tmp"
PROJECT="/projects/bgqe/rpolonsky/dcrl"
EXPO="${PROJECT}/third_party/expo-ft"
AUDIT="/projects/bgqe/rpolonsky/dcrl/sharder/training/sft_analysis_20261103"
TEACHER="/work/hdd/bgqe/rpolonsky/dcrlcheckpoints/small_expo_rl_10_5_offline50k_online500k_resume_work/libero_90/task_79"

srun --jobid="${JOB_ID}" --overlap --ntasks=1 --nodes=1 \
  env SESSION="${SESSION}" SOCKET="${SOCKET}" ROOT="${ROOT}" OUT="${OUT}" \
      TMP="${TMP}" PROJECT="${PROJECT}" EXPO="${EXPO}" AUDIT="${AUDIT}" \
      TEACHER="${TEACHER}" \
  bash -lc '
    set -euo pipefail
    mkdir -p "${OUT}" "${TMP}"
    if tmux -S "${SOCKET}" has-session -t "${SESSION}" 2>/dev/null; then
      echo "session already exists: ${SESSION}" >&2
      exit 2
    fi
    tmux -S "${SOCKET}" new-session -d -s "${SESSION}" bash -lc \
      "bash ${AUDIT}/run_task79_sft_screen_19_30_79_20261107.sh > ${OUT}/launcher.log 2>&1; rc=\$?; echo runner_rc=\$rc >> ${OUT}/launcher.log; sleep 300"
    echo "started ${SESSION} on $(hostname)"
  '

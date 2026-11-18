#!/usr/bin/env bash
set -euo pipefail

JOB_ID=${1:?job id required}
VARIANT=${2:?variant required}
TASK=${3:?task required}
ROOT=/work/hdd/bgqe/rpolonsky/dcrlcheckpoints/sft_from_hybrid_tuned_20k_20261116
LAUNCHER=/projects/bgqe/rpolonsky/dcrl/scripts/run_hybrid_sft_tuned_20k.sh

# /work's statfs reports zero after its soft quota is exceeded even while
# hard-quota space is writable. Wait for the two project-backed full saves
# to complete and release their corresponding old checkpoints instead.
while [[ -d /work/hdd/bgqe/rpolonsky/dcrlcheckpoints/sft_from_hybrid_10_20261115/small/task_60/checkpoints ||
         -d /work/hdd/bgqe/rpolonsky/dcrlcheckpoints/sft_from_hybrid_10_20261115/small_dist/task_28/checkpoints ]]; do
  echo "Waiting for the first two complete checkpoint replacements"
  sleep 30
done
exec srun --jobid="$JOB_ID" --overlap --unbuffered --ntasks=1 \
  env OUTPUT_ROOT="$ROOT" bash "$LAUNCHER" "$VARIANT" "$TASK"

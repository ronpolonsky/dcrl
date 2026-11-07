#!/usr/bin/env bash
set -euo pipefail
PROJECT=/projects/bgqe/rpolonsky/dcrl
AUDIT=/projects/bgqe/rpolonsky/dcrl/sharder/training/sft_analysis_20261103
export PYTHONPATH="$PROJECT:$PROJECT/third_party/expo-ft/expo_ft/agents/vla/openpi/src:$PROJECT/third_party/expo-ft/expo_ft/agents/vla/openpi/packages/openpi-client/src:$PROJECT/.deps/LIBERO"
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
cd "$PROJECT"
exec "$PROJECT/third_party/expo-ft/.venv/bin/python" "$AUDIT/audit_weight_drift.py"

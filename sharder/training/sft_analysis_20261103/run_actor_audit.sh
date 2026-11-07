#!/usr/bin/env bash
set -euo pipefail
PROJECT=/projects/bgqe/rpolonsky/dcrl
AUDIT=/projects/bgqe/rpolonsky/dcrl/sharder/training/sft_analysis_20261103
export PYTHONPATH="$PROJECT:$PROJECT/third_party/expo-ft/expo_ft/agents/vla/openpi/src:$PROJECT/third_party/expo-ft/expo_ft/agents/vla/openpi/packages/openpi-client/src:$PROJECT/.deps/LIBERO"
export OPENPI_DATA_HOME=/work/hdd/bgqe/rpolonsky/dcrlcheckpoints/openpi_cache
export OPENPI_SKIP_INIT_JIT=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
export JAX_COMPILATION_CACHE_DIR=/tmp/rpolonsky-sft-analysis/jax_cache
mkdir -p "$JAX_COMPILATION_CACHE_DIR"
cd "$PROJECT"
exec "$PROJECT/third_party/expo-ft/.venv/bin/python" "$AUDIT/audit_actor_outputs.py"

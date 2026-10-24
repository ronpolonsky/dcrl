#!/usr/bin/env bash
# Collect behavior chunks from one retained SmallExpo RL checkpoint, then
# distill those chunks into a Pi0.5 base actor with the official 10/5 interface.
# Submit one job per task with TASK_ID and TEACHER_STEP exported.
#SBATCH --partition=sphinx
#SBATCH --account=nlp
#SBATCH --time=14-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
# Keep this pipeline on the requested Sphinx[4-11] pool; nodes 1-3 have 40-GiB
# A100s and are too small for the Pi0.5 student.
#SBATCH --exclude=sphinx[1-3]
#SBATCH --job-name=small-expo-sft
#SBATCH --output=/nlp/scr/ronpo/projects/dcrl/slurm/%x-%j.out
#SBATCH --error=/nlp/scr/ronpo/projects/dcrl/slurm/%x-%j.err

set -euo pipefail
: "${TASK_ID:?submit with TASK_ID=<2|28|60|79>}"
: "${TEACHER_STEP:?submit with the retained SmallExpo checkpoint step}"

PROJECT=/nlp/scr/ronpo/projects/dcrl
EXPO="$PROJECT/third_party/expo-ft"
PYTHON="$EXPO/.venv/bin/python"
# The project checkout is under /nlp/scr/ronpo/projects, while RL checkpoints
# live directly under /nlp/scr/ronpo/dcrlcheckpoints.
TEACHER_DIR="/nlp/scr/ronpo/dcrlcheckpoints/small_expo_learner/libero_90/task_$(printf '%02d' "$TASK_ID")"
OUT="/nlp/scr/ronpo/dcrlcheckpoints/bc_distill_small_to_pi05base/task${TASK_ID}_rlstep${TEACHER_STEP}_500k"

export PYTHONPATH="$PROJECT:$EXPO/expo_ft/agents/vla/openpi/src:$EXPO/expo_ft/agents/vla/openpi/packages/openpi-client/src:$PROJECT/.deps/LIBERO"
export LIBERO_CONFIG_PATH="$PROJECT/.deps/libero_config"
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR=/nlp/scr/ronpo/dcrlcheckpoints/jax_cache
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENPI_DATA_HOME=/nlp/scr/ronpo/dcrlcheckpoints/openpi_cache
export WANDB_BASE_URL=https://forge.coreweave.com/api/wandb
export WANDB_ENTITY=distill_rl
export WANDB_PROJECT=dcrl

mkdir -p "$PROJECT/slurm" "$PROJECT/.deps/libero_config" "$OUT" "$JAX_COMPILATION_CACHE_DIR"
cat > "$PROJECT/.deps/libero_config/config.yaml" <<CFG
benchmark_root: $PROJECT/.deps/LIBERO/libero/libero
bddl_files: $PROJECT/.deps/LIBERO/libero/libero/bddl_files
init_states: $PROJECT/.deps/LIBERO/libero/libero/init_files
datasets: $PROJECT/datasets
assets: $PROJECT/.deps/LIBERO/libero/libero/assets
CFG

cd "$PROJECT"
echo "job=$SLURM_JOB_ID host=$(hostname) node=$SLURMD_NODENAME task=$TASK_ID teacher_step=$TEACHER_STEP"
timeout 30s nvidia-smi -L
"$PYTHON" -c 'import jax; ds=jax.devices(); assert any(d.platform == "gpu" for d in ds), ds; print("JAX devices:", ds)'

"$PYTHON" -m dcrl.train_bc_distill \
  --phase all \
  --collection-source small_expo \
  --teacher-dir "$TEACHER_DIR" \
  --teacher-step "$TEACHER_STEP" \
  --source-checkpoint gs://openpi-assets/checkpoints/pi05_base \
  --student-checkpoint gs://openpi-assets/checkpoints/pi05_base \
  --norm-stats gs://openpi-assets/checkpoints/pi05_libero/assets/physical-intelligence/libero/norm_stats.json \
  --finetune-mode lora \
  --source-suite libero_90 \
  --source-task-id "$TASK_ID" \
  --output-dir "$OUT" \
  --collection-episodes "${COLLECTION_EPISODES:-200}" \
  --collection-successes "${COLLECTION_SUCCESSES:-100}" \
  --collection-failures "${COLLECTION_FAILURES:-25}" \
  ${COLLECTION_MAX_ATTEMPTS:+--collection-max-attempts "$COLLECTION_MAX_ATTEMPTS"} \
  --num-sft-steps 500000 \
  --batch-size 16 \
  --eval-episodes 20 \
  --eval-every 25000 \
  --checkpoint-every 25000 \
  --warmup-steps 10000 \
  --lr-decay-steps 1000000 \
  --seed 42 \
  --eval-seed 7

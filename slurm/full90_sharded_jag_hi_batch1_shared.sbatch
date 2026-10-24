#!/bin/bash
#SBATCH --account=nlp
#SBATCH --partition=jag-hi
#SBATCH --array=9-18%10
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --output=/nlp/scr/ronpo/projects/dcrl/slurm/full90-sc-%A_%a.out
#SBATCH --error=/nlp/scr/ronpo/projects/dcrl/slurm/full90-sc-%A_%a.err
set -euo pipefail
export TASK_INDEX="$SLURM_ARRAY_TASK_ID"
exec /nlp/scr2/ronpo/projects/dcrl/slurm/full90_one_iris_generic_shared.sbatch

#!/usr/bin/env bash
set -euo pipefail

name=$1
script=$2
out=/tmp/usertmp/rpolonsky/sft_retention_diagnosis_20261106/$name
mkdir -p "$out"
tmux new-session -d -s "$name" "bash /projects/bgqe/rpolonsky/dcrl/sharder/training/sft_analysis_20261103/$script > $out/launcher.log 2>&1"
sleep 172700

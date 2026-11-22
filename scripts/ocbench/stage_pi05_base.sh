#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd -- "$script_dir/../.." && pwd)
project_root=${OCBENCH_PROJECT_ROOT:-$(dirname -- "$repo")}
expo=${EXPO_FT_ROOT:-$project_root/dcrl/third_party/expo-ft}
openpi=$expo/expo_ft/agents/vla/openpi
python=${OCBENCH_PYTHON:-$expo/.venv/bin/python}
scratch=${OCBENCH_SCRATCH:-/tmp/${USER:-unknown}-ocbench-pi05-sft}

umask 077
mkdir -p "$scratch"
exec 9>"$scratch/stage.lock"
flock 9

base=
if [[ -s "$scratch/base_path" ]]; then
  base=$(<"$scratch/base_path")
fi
if [[ -z "$base" || ! -s "$base/params/manifest.ocdbt" ]]; then
  export OPENPI_DATA_HOME="$scratch/openpi"
  export PYTHONPATH="$openpi/src:$openpi/packages/openpi-client/src"
  base=$("$python" -c 'from openpi.shared.download import maybe_download; print(maybe_download("gs://openpi-assets/checkpoints/pi05_base/params").parent)')
  test -s "$base/params/manifest.ocdbt"
fi

cache=${base%/openpi-assets/checkpoints/pi05_base}
tokenizer="$cache/big_vision/paligemma_tokenizer.model"
if [[ ! -s "$tokenizer" ]]; then
  mkdir -p "$(dirname -- "$tokenizer")"
  curl -fsSL --retry 3 --connect-timeout 30 \
    https://storage.googleapis.com/big_vision/paligemma_tokenizer.model \
    -o "$tokenizer.part"
  mv -- "$tokenizer.part" "$tokenizer"
fi

printf '%s\n' "$base" >"$scratch/base_path.tmp"
mv -- "$scratch/base_path.tmp" "$scratch/base_path"
printf 'Pi0.5 and tokenizer ready on %s\n' "$(hostname -s)"

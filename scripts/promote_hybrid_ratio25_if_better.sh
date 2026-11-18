#!/usr/bin/env bash
set -euo pipefail

VARIANT=${1:?variant required}
TASK=${2:?task required}
case "$VARIANT:$TASK" in
  small:28|small:60|small_dist:60) ;;
  *) echo "Unsupported promotion target" >&2; exit 2 ;;
esac

SESSION=hyb_ratio25_${VARIANT}_task${TASK}
OLD=/work/hdd/bgqe/rpolonsky/dcrlcheckpoints/sft_hybrid_more_data_20261116/$VARIANT/task_$TASK
NEW=/work/hdd/bgqe/rpolonsky/dcrlcheckpoints/sft_hybrid_ratio25_20261117/$VARIANT/task_$TASK

while tmux has-session -t "$SESSION" 2>/dev/null; do
  sleep 30
done

# Require a complete planned run and compare its saved best checkpoint with
# the previous best on the identical task evaluation. Never touch the data.
export OLD NEW
python - <<'PY'
import json
import os
import pathlib
import shutil

old = pathlib.Path(os.environ['OLD'])
new = pathlib.Path(os.environ['NEW'])
root = pathlib.Path('/work/hdd/bgqe/rpolonsky/dcrlcheckpoints')
assert old.resolve().is_relative_to(root) and new.resolve().is_relative_to(root)

metrics = new / 'metrics.jsonl'
if not metrics.is_file() or json.loads(metrics.read_text().splitlines()[-1])['step'] != 4000:
    raise SystemExit('New run did not complete 4000 updates; keeping old checkpoint')

def saved_best(path):
    entries = [p for p in (path / 'checkpoints').iterdir() if p.is_dir() and p.name.isdigit()]
    if len(entries) != 1:
        raise SystemExit(f'Expected one saved checkpoint in {path}; found {entries}')
    checkpoint = entries[0]
    if not (checkpoint / 'agent').is_dir() or not (checkpoint / 'params').is_dir():
        raise SystemExit(f'Incomplete checkpoint: {checkpoint}')
    ev = path / 'evals' / f'{int(checkpoint.name):08d}.json'
    score = float(json.loads(ev.read_text())['eval/success_rate'])
    return checkpoint, score

old_checkpoint, old_score = saved_best(old)
new_checkpoint, new_score = saved_best(new)
print(f'old={old_checkpoint} score={old_score:.2f}; new={new_checkpoint} score={new_score:.2f}', flush=True)
if new_score <= old_score:
    print('No strict improvement; keeping both checkpoints', flush=True)
    raise SystemExit(0)

# Deletion is limited to the superseded checkpoint. Keep the old run's data,
# metrics, and evaluations for provenance.
shutil.rmtree(old_checkpoint)
print(f'Replaced old checkpoint with improved checkpoint at {new_checkpoint}', flush=True)
PY

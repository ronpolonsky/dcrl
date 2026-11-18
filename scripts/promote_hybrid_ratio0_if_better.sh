#!/usr/bin/env bash
set -euo pipefail

VARIANT=${1:?variant required}
TASK=${2:?task required}
case "$VARIANT:$TASK" in
  small:28|small:60|small_dist:28|small_dist:60) ;;
  *) echo "Unsupported promotion target" >&2; exit 2 ;;
esac

SESSION=hyb_ratio0_${VARIANT}_task${TASK}
OLD=/work/hdd/bgqe/rpolonsky/dcrlcheckpoints/sft_hybrid_more_data_20261116/$VARIANT/task_$TASK
NEW=/work/hdd/bgqe/rpolonsky/dcrlcheckpoints/sft_hybrid_ratio0_20261117/$VARIANT/task_$TASK

while tmux has-session -t "$SESSION" 2>/dev/null; do sleep 30; done

export OLD NEW
python - <<'PY'
import json, os, pathlib, shutil
old, new = pathlib.Path(os.environ['OLD']), pathlib.Path(os.environ['NEW'])
root = pathlib.Path('/work/hdd/bgqe/rpolonsky/dcrlcheckpoints')
assert old.resolve().is_relative_to(root) and new.resolve().is_relative_to(root)
metrics = new / 'metrics.jsonl'
if not metrics.is_file() or json.loads(metrics.read_text().splitlines()[-1])['step'] != 4000:
    raise SystemExit('New run did not complete 4000 updates; keeping old checkpoint')
def best(path):
    cs=[p for p in (path/'checkpoints').iterdir() if p.is_dir() and p.name.isdigit()]
    if len(cs)!=1: raise SystemExit(f'Expected one checkpoint in {path}, found {cs}')
    c=cs[0]
    if not (c/'agent').is_dir() and not (c/'params').is_dir(): raise SystemExit(f'Incomplete {c}')
    score=float(json.loads((path/'evals'/f'{int(c.name):08d}.json').read_text())['eval/success_rate'])
    return c,score
oc,os=best(old); nc,ns=best(new)
print(f'old={oc} score={os:.2f}; new={nc} score={ns:.2f}',flush=True)
if ns<=os:
    print('No strict improvement; keeping both checkpoints',flush=True); raise SystemExit(0)
shutil.rmtree(oc)
print(f'Removed superseded old checkpoint; improved checkpoint remains at {nc}',flush=True)
PY

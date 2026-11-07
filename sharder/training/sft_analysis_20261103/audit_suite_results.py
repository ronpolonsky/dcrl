"""Validate full-suite comparability and tabulate paired retention outcomes."""
import json
from pathlib import Path

OUT = Path(__file__).parent
BASE = Path('/projects/bgqe/rpolonsky/dcrl/results/official_pi0.5_libero_chkpt')
EXPO = Path('/work/hdd/bgqe/rpolonsky/dcrlcheckpoints/pi_robo_libero_unsharded_h200_b16_200k/full90_eval_direct_rl_pi_20261102')
SFT = Path('/work/hdd/bgqe/rpolonsky/dcrlcheckpoints/sft_full90_actor_only_eval_20261103')

def load(paths):
    records = {}
    for path in paths:
        row = json.loads(path.read_text())
        task = row['task_id']
        assert task not in records, (task, path)
        for key, expected in dict(suite='libero_90', episodes=20, eval_seed=7,
                                  max_steps=400, action_horizon=10, replan_steps=5).items():
            assert row[key] == expected, (path, key, row.get(key))
        assert len(row['trials']) == 20
        assert {t['initial_state_index'] for t in row['trials']} == set(range(20))
        assert sum(bool(t['success']) for t in row['trials']) == row['successes']
        assert abs(row['eval/success_rate'] - row['successes']/20) < 1e-10
        records[task] = row
    return records

base = load(BASE.glob('task_*.json'))
assert set(base) == set(range(90))
models = [(f'expo_{t}', t, EXPO/f'task_{t}') for t in (2,28,29,60,79)]
models += [(f'sft_{t}', t, SFT/folder) for t,folder in ((2,'task_02_step_20000'),(28,'task_28_step_40000'),(29,'task_29_step_20000'))]
report = dict(base_successes=sum(r['successes'] for r in base.values()), models={})
for name, training_task, root in models:
    records = load(root.glob('shard_*/task_*.json'))
    row = dict(coverage=len(records), training_task=training_task,
               successes=sum(r['successes'] for r in records.values()))
    if set(records) != set(base):
        row['status'] = 'partial; excluded from full-suite aggregates'
        report['models'][name] = row
        continue
    losses = gains = both = neither = 0
    changes = []
    for task, result in sorted(records.items()):
        assert result['instruction'] == base[task]['instruction']
        b = {t['initial_state_index']: bool(t['success']) for t in base[task]['trials']}
        r = {t['initial_state_index']: bool(t['success']) for t in result['trials']}
        for i in b:
            losses += b[i] and not r[i]
            gains += r[i] and not b[i]
            both += b[i] and r[i]
            neither += not b[i] and not r[i]
        changes.append(dict(task=task, instruction=result['instruction'],
                            base_successes=base[task]['successes'],
                            successes=result['successes'],
                            delta_successes=result['successes']-base[task]['successes']))
    row.update(status='complete', suite_rate=row['successes']/1800,
               own_task_rate=records[training_task]['successes']/20,
               off_task_rate=(row['successes']-records[training_task]['successes'])/1780,
               paired_lost=losses, paired_gained=gains, paired_both_success=both,
               paired_both_failure=neither,
               zero_success_tasks=sum(r['successes']==0 for r in records.values()),
               tasks_improved=sum(x['delta_successes']>0 for x in changes),
               tasks_worsened=sum(x['delta_successes']<0 for x in changes),
               successful_tasks=[x for x in changes if x['successes']>0],
               per_task=changes)
    report['models'][name] = row
    print(name, {k:v for k,v in row.items() if k not in ('successful_tasks','per_task')}, flush=True)
(OUT/'suite_retention_audit.json').write_text(json.dumps(report,indent=2)+'\n')

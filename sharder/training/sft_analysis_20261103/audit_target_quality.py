"""Measure executed-prefix support and task-local official demo coverage."""
import json
from pathlib import Path
import h5py
import numpy as np
from libero.libero import benchmark

OUT = Path(__file__).parent
ROOT = Path('/work/hdd/bgqe/rpolonsky/dcrlcheckpoints/sft_agnostic_small_expo')
DEMOS = Path('/projects/bgqe/rpolonsky/dcrl/datasets')
suite = benchmark.get_benchmark_dict()['libero_90']()
report = {}
for task_id in (2,28,29,60):
    task = suite.get_task(task_id)
    source = next(ROOT.glob(f'task_{task_id:02d}_*')) / 'pi_base_data/success/teacher_chunks.hdf5'
    arrays, overlaps, executed, lengths = [], [], [], []
    with h5py.File(source,'r') as f:
        for g in f.values():
            a = np.asarray(g['actions'],np.float32)
            n = np.asarray(g['executed_prefix'],np.int32)
            arrays.append(a)
            executed.extend(n.tolist())
            lengths.append(int(g.attrs['steps']))
            if len(a)>1:
                overlaps.append(np.abs(a[:-1,5:]-a[1:,:5]))
    a = np.concatenate(arrays)
    row = dict(teacher_chunks=len(a), teacher_episode_steps_median=float(np.median(lengths)),
               fraction_target_actions_executed=sum(executed)/(10*len(a)),
               teacher_outside_controller_motion_bounds_prefix=np.mean(np.abs(a[:,:5,:6])>1,axis=(0,1)).tolist(),
               teacher_outside_controller_motion_bounds_tail=np.mean(np.abs(a[:,5:,:6])>1,axis=(0,1)).tolist(),
               proposal_tail_vs_next_prefix_mae_per_dim=np.mean(np.concatenate(overlaps),axis=(0,1)).tolist())
    path = DEMOS / task.problem_folder / f'{task.name}_demo.hdf5'
    row['official_demo_path'] = str(path)
    with h5py.File(path,'r') as f:
        lengths = [len(g['actions']) for g in f['data'].values()]
        demo_actions = np.concatenate([np.asarray(g['actions'][1:],np.float32) for g in f['data'].values()])
        row.update(official_episodes=len(lengths), official_aligned_timesteps=len(demo_actions),
                   official_10_5_chunks=sum(len(range(0,n-1,5)) for n in lengths),
                   official_episode_steps_median=float(np.median(lengths)),
                   official_action_quantiles=np.quantile(demo_actions,[0,.01,.5,.99,1],axis=0).tolist())
    report[str(task_id)] = row
    (OUT/'target_quality_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(task_id, json.dumps(row), flush=True)

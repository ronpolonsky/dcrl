"""Read-only audit of SFT collections, exposure, metrics and saved evals."""
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

ROOT = Path('/work/hdd/bgqe/rpolonsky/dcrlcheckpoints/sft_agnostic_small_expo')
OUT = Path(__file__).parent
report = {}
for run in sorted(ROOT.glob('task_*')):
    if not (run / 'sft_config.json').exists():
        continue
    cfg = json.loads((run / 'sft_config.json').read_text())
    source = run / 'pi_base_data/success/teacher_chunks.hdf5'
    entry = dict(config=cfg, source=str(source))
    actions, states, lengths, prefixes, initial_states = [], [], [], [], []
    with h5py.File(source, 'r') as f:
        entry['instruction'] = str(f.attrs['instruction'])
        for group in f.values():
            a = np.asarray(group['actions'])
            actions.append(a)
            states.append(np.asarray(group['state']))
            lengths.append(len(a))
            prefixes.extend(np.asarray(group['executed_prefix']).tolist())
            initial_states.append(int(group.attrs['initial_state_index']))
        first = next(iter(f.values()))
        entry['image_shapes'] = {key: list(first[key].shape) for key in ('base_0_rgb', 'left_wrist_0_rgb')}
        entry['image_dtypes'] = {key: str(first[key].dtype) for key in ('base_0_rgb', 'left_wrist_0_rgb')}
    a, s = np.concatenate(actions), np.concatenate(states)
    norm_path = run / 'student_assets/physical-intelligence/libero/norm_stats.json'
    norm = json.loads(norm_path.read_text())['norm_stats']['actions']
    lo, hi = np.array(norm['q01']), np.array(norm['q99'])
    normalized = 2 * (a - lo) / (hi - lo + 1e-6) - 1
    entry.update(
        episodes=len(lengths), chunks=len(a), unique_initial_states=len(set(initial_states)),
        rows_per_episode=dict(min=int(min(lengths)), median=float(np.median(lengths)), max=int(max(lengths))),
        action_shape=list(a.shape), all_finite=bool(np.isfinite(a).all() and np.isfinite(s).all()),
        action_quantiles=np.quantile(a, [0, .01, .5, .99, 1], axis=(0, 1)).tolist(),
        normalized_outside_unit_per_dim=np.mean(np.abs(normalized) > 1, axis=(0, 1)).tolist(),
        prefix_outside_unit_per_dim=np.mean(np.abs(normalized[:, :5]) > 1, axis=(0, 1)).tolist(),
        tail_outside_unit_per_dim=np.mean(np.abs(normalized[:, 5:]) > 1, axis=(0, 1)).tolist(),
        executed_prefix_counts={str(k): prefixes.count(k) for k in sorted(set(prefixes))},
        state_quantiles=np.quantile(s, [0, .5, 1], axis=0).tolist(),
        norm_sha256=hashlib.sha256(norm_path.read_bytes()).hexdigest(),
    )
    bins = {}
    last = None
    for line in (run / 'metrics.jsonl').open():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        last = row
        if 'actor_loss' in row:
            bins.setdefault(int(row['step']) // 1000, []).append(row['actor_loss'])
    entry['last_metrics'] = last
    entry['loss_median_by_1k'] = {str(k * 1000): float(np.median(v)) for k, v in bins.items()}
    entry['sample_exposure_at_last_step'] = last['step'] * cfg['batch_size'] / len(a)
    entry['sample_exposure_at_5k'] = 5000 * cfg['batch_size'] / len(a)
    entry['evals'] = [json.loads(p.read_text()) for p in sorted((run / 'evals').glob('*.json'))]
    report[run.name] = entry
    (OUT / 'data_metrics_audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: entry[key] for key in ('instruction','episodes','chunks','unique_initial_states','sample_exposure_at_last_step','sample_exposure_at_5k','normalized_outside_unit_per_dim')}, indent=2), flush=True)

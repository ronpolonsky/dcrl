"""Validate standalone SFT screens and report the mixed policy on the same tasks."""

import argparse
import hashlib
import json
from pathlib import Path

from dcrl.ocbench_env import TASKS, environment_id


def read_json(path):
    return json.loads(path.read_text())


def screen_summary(root, suite):
    rows = []
    for task_id in range(27):
        path = root / 'screens' / suite / f'task_{task_id}'
        result_path = path / 'evals' / '00002000.json'
        if not result_path.is_file():
            continue
        run = read_json(path / 'sft_config.json')
        result = read_json(result_path)
        archive = root / 'data' / suite / f'{environment_id(suite, task_id)}.hdf5'
        record = read_json(archive.with_suffix('.json'))
        with archive.open('rb') as file:
            if hashlib.file_digest(file, 'sha256').hexdigest() != record['archive_sha256']:
                raise ValueError(f'Selected dataset differs from its manifest: {archive}')
        if len(record['episodes']) != 200 or any(episode['return_'] <= 0 for episode in record['episodes']):
            raise ValueError(f'Expected exactly 200 successful official trajectories: {archive}')
        expected = {'source_suite': suite, 'source_task_id': task_id, 'seed': 42, 'eval_seed': 100000,
                    'eval_episodes': 20, 'eval_every': 2000, 'batch_size': 4, 'learning_rate': .00005,
                    'warmup_steps': 200, 'action_horizon': 10, 'replan_steps': 5, 'checkpoint_policy': 'none',
                    'zero_lora_b': True}
        if any(run[key] != value for key, value in expected.items()):
            raise ValueError(f'Unexpected screening configuration in {path}')
        if not (run['student_checkpoint'].endswith('/checkpoints/pi05_base')
                and run['collection']['task_id'] == task_id and result['step'] == 2000
                and len(result['trials']) == 20):
            raise ValueError(f'Checkpoint, data source, or evaluation differs from the protocol: {path}')
        score = result['eval/success_rate']
        rows.append({'task_id': task_id, 'name': TASKS[task_id][0], 'success_rate': score,
                     'include': score >= 0.10, 'archive_sha256': record['archive_sha256']})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--suite', choices=('ocbench', 'ocbench_lite'), default='ocbench_lite')
    parser.add_argument('--final-evals', type=Path)
    parser.add_argument('--write-selection', action='store_true')
    args = parser.parse_args()
    rows = screen_summary(args.root, args.suite)
    survivors = [result['task_id'] for result in rows if result['include']]
    if args.write_selection or args.final_evals is not None:
        if len(rows) != 27:
            raise SystemExit(f'{len(rows)}/27 standalone screens complete')
        if args.write_selection:
            path = args.root / f'{args.suite}-selection.json'
            contents = json.dumps({'survivors': survivors, 'results': rows}, indent=2) + '\n'
            if path.exists() and path.read_text() != contents:
                raise ValueError(f'The existing selection disagrees with the standalone results: {path}')
            path.write_text(contents)
    print(f'{len(rows)}/27 standalone screens complete')
    if args.final_evals is None:
        print('| Task | Standalone success | Include in mixture |\n|---|---:|---|')
        for row in rows:
            print(f"| {row['task_id']} {row['name']} | {row['success_rate']:.0%} | {'yes' if row['include'] else 'no'} |")
        return
    print('| Task | Standalone success | Mixed policy | In mixture |\n|---|---:|---:|---|')
    for task_id in range(len(TASKS)):
        final = read_json(args.final_evals / f'task_{task_id:02d}.json')
        if not (final['suite'] == args.suite and final['eval_seed'] == 100000 and final['episodes'] == 20):
            raise ValueError(f'Mixed policy evaluation differs from the protocol for task {task_id}')
        if task_id < 27:
            standalone = f"{rows[task_id]['success_rate']:.0%}"
            mixed = f"{final['eval/success_rate']:.0%}"
        else:
            standalone = '—'
            mixed = f"{final['eval/normalized_return']:.1f} return"
        print(f"| {task_id} {TASKS[task_id][0]} | {standalone} | {mixed} | {'yes' if task_id in survivors else 'no'} |")


if __name__ == '__main__':
    main()

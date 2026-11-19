"""Select reproducible SFT episodes from the official OCBench state release."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import h5py
import numpy as np
import ocbench
from ocbench.dataset_utils import DATASET_REPO, DATASET_REVISION

from dcrl.ocbench_env import environment_id


def download(relative, directory, *, size=None):
    path = directory / relative
    if path.is_file() and (size is None or path.stat().st_size == size):
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.part')
    url = f'https://huggingface.co/datasets/{DATASET_REPO}/resolve/{DATASET_REVISION}/{relative}'
    subprocess.run(['curl', '--fail', '--silent', '--show-error', '--location', '--retry', '3',
                    '--connect-timeout', '30', url, '-o', str(temporary)], check=True)
    if size is not None and temporary.stat().st_size != size:
        raise ValueError(f'Downloaded size differs from the release manifest: {relative}')
    temporary.replace(path)
    return path


def sha256_file(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def prepare(args):
    env_id = environment_id(args.suite, args.task_id)
    category = 'lite' if args.suite == 'ocbench_lite' and args.task_id != 27 else 'full'
    source_env = env_id.removeprefix('visual-').replace('-cpu-', '-', 1)
    directory = f'{category}/{source_env}'
    manifest = download('manifest.jsonl', args.cache)
    shards = sorted((row for row in map(json.loads, manifest.read_text().splitlines())
                     if row['path'].rsplit('/', 1)[0] == directory
                     and row['kind'] == 'data' and row['split'] == 'train'),
                    key=lambda row: row['path'])
    if not shards:
        raise ValueError(f'No official training data for {source_env}')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f'{env_id}.hdf5'
    index_path = output.with_suffix('.json')
    all_eligible = args.episodes == 0
    selection = 'episodes' if args.task_id == 27 else 'successful_episodes'
    protocol = dict(dataset=DATASET_REPO, revision=DATASET_REVISION, suite=args.suite,
                    task_id=args.task_id, selected_episodes=args.episodes, replan_steps=args.replan_steps,
                    selection=f'{"all" if all_eligible else "first"}_{selection}')
    if output.is_file() and index_path.is_file():
        record = json.loads(index_path.read_text())
        if record['protocol'] != protocol or record['archive_sha256'] != sha256_file(output):
            raise ValueError(f'Existing official archive does not match the requested split: {output}')
        return output
    if output.exists():
        raise FileExistsError(f'Archive is missing its matching index: {output}')

    env = ocbench.make(env_id)
    episodes = []
    temporary = output.with_suffix('.partial.hdf5')
    try:
        env.reset(seed=0)
        raw = env.unwrapped
        with h5py.File(temporary, 'w') as archive:
            archive.attrs.update(env_id=env_id, observation_source='released_state',
                                 observation_interval=args.replan_steps, protocol=json.dumps(protocol))
            for shard in shards:
                path = download(shard['path'], args.cache, size=shard['bytes'])
                print(f'Reading {shard["path"]}', flush=True)
                with np.load(path, allow_pickle=False) as data:
                    ends = np.flatnonzero(data['terminals']) + 1
                    starts = np.concatenate(([0], ends[:-1]))
                    rewards = data['rewards']
                    selected = np.arange(len(ends)) if args.task_id == 27 else np.flatnonzero(rewards[ends - 1] > 0)
                    if not all_eligible:
                        selected = selected[:args.episodes - len(episodes)]
                    if not len(selected):
                        continue
                    arrays = {key: data[key] for key in ('actions', 'qpos', 'qvel', 'dynamics_info', 'button_states')
                              if key in data}
                    if any(array.shape[0] != ends[-1] for array in arrays.values()):
                        raise ValueError(f'Expected one recorded simulation state per action in {path}')
                    for episode in selected:
                        start, stop = int(starts[episode]), int(ends[episode])
                        frames = slice(start, stop, args.replan_steps)
                        group = archive.create_group(f'episode_{len(episodes):06d}')
                        group.attrs['success'] = bool(args.task_id != 27)
                        group.create_dataset('actions', data=arrays['actions'][start:stop].astype(np.float32))
                        qpos = arrays['qpos'][frames]
                        joint_pos = qpos[:, raw._arm_joint_ids]
                        gripper = np.clip(qpos[:, raw._gripper_opening_joint_id] / 0.8, 0, 1)
                        group.create_dataset('proprioception', data=np.column_stack((joint_pos, gripper)).astype(np.float32))
                        for key in ('qpos', 'qvel', 'dynamics_info', 'button_states'):
                            if key in arrays:
                                group.create_dataset(key, data=arrays[key][frames], compression='lzf')
                        episodes.append(dict(shard=shard['path'], episode_index=int(episode),
                                             start=start, stop=stop, return_=float(rewards[start:stop].sum())))
                if not all_eligible and len(episodes) == args.episodes:
                    break
            if not episodes or (not all_eligible and len(episodes) != args.episodes):
                raise ValueError(f'Release supplied {len(episodes)} eligible episodes, expected {args.episodes}')
            archive.attrs['complete'] = True
    finally:
        env.close()
    record = dict(protocol=protocol, episodes=episodes, archive_sha256=sha256_file(temporary),
                  release_manifest_sha256=sha256_file(manifest))
    temp_index = index_path.with_suffix('.partial.json')
    temp_index.write_text(json.dumps(record, indent=2) + '\n')
    temporary.replace(output)
    temp_index.replace(index_path)
    print(f'Saved {len(episodes)} official episodes and their source indices at {output}', flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=('ocbench', 'ocbench_lite'), required=True)
    parser.add_argument('--task-id', type=int, required=True)
    parser.add_argument('--episodes', type=int, default=200,
                        help='Successful episode limit. Zero uses every eligible training episode.')
    parser.add_argument('--replan-steps', type=int, default=5)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.episodes < 0 or args.replan_steps <= 0:
        parser.error('Episodes must be nonnegative and replan steps must be positive.')
    prepare(args)


if __name__ == '__main__':
    main()

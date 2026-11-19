"""Collect dense visual OCBench demonstrations with upstream scripted policies."""

import argparse
from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np

from dcrl.benchmarks import max_episode_steps
from dcrl.ocbench_env import environment_id, proprioception
from dcrl.ocbench_env import demonstration_groups


def collect(suite, task_id, episodes, seed, output_dir):
    import ocbench
    from ocbench.oracles.controllers.block import BlockController
    from ocbench.oracles.controllers.bowling import BowlingController
    from ocbench.oracles.controllers.chamber import ChamberController
    from ocbench.oracles.controllers.hanoi import HanoiController
    from ocbench.oracles.controllers.switch import SwitchController

    env_id = environment_id(suite, task_id)
    controllers = dict(block=BlockController, bowling=BowlingController,
                       chamber=ChamberController, hanoi=HanoiController, switch=SwitchController)
    controller_type = controllers[env_id.split('-')[1]]
    horizon = max_episode_steps(suite, task_id)
    output = output_dir / f'{env_id}.hdf5'
    if output.exists():
        raise FileExistsError(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    env = ocbench.make(env_id)
    np.random.seed(seed)  # Upstream scripted policies use NumPy's global RNG.
    completed = successes = 0
    temporary = output.with_suffix('.partial.hdf5')
    try:
        with h5py.File(temporary, 'x') as dataset:
            dataset.attrs.update(env_id=env_id, seed=seed, observation_interval=1)
            for attempt in range(episodes * 4):
                ob, info = env.reset(seed=seed + attempt)
                controller = controller_type(env=env)
                controller.reset(ob, info, transition_idx=0)
                trajectory = {key: [] for key in (
                    'observations', 'proprioception', 'actions', 'rewards', 'terminals', 'masks')}
                for step in range(horizon):
                    trajectory['observations'].append(ob)
                    trajectory['proprioception'].append(proprioception(info))
                    action = np.asarray(controller.select_action(ob, info, transition_idx=step), np.float32)
                    action = np.clip(action, env.action_space.low, env.action_space.high)
                    ob, reward, terminated, truncated, info = env.step(action)
                    trajectory['actions'].append(action)
                    trajectory['rewards'].append(reward)
                    trajectory['terminals'].append(terminated or truncated)
                    trajectory['masks'].append(not terminated)
                    if terminated or truncated:
                        break
                if not info['healthy']:
                    continue
                if not trajectory['terminals'][-1]:
                    raise RuntimeError(f'{env_id} reached its registered horizon without truncation')
                trajectory['observations'].append(ob)
                trajectory['proprioception'].append(proprioception(info))
                group = dataset.create_group(f'episode_{completed:06d}')
                success = bool(info['success'] and not info['failure'])
                group.attrs.update(success=success, reset_seed=seed + attempt)
                for key, values in trajectory.items():
                    options = {'chunks': (1, *ob.shape), 'compression': 'lzf'} if key == 'observations' else {}
                    group.create_dataset(key, data=np.asarray(values), **options)
                completed += 1
                successes += int(success)
                print(f'{env_id}: {completed}/{episodes} episodes, {successes} successes', flush=True)
                if completed == episodes:
                    break
            if completed != episodes:
                raise RuntimeError(f'Collected only {completed} healthy episodes of {episodes} requested.')
            dataset.attrs['complete'] = True
    finally:
        env.close()
    temporary.replace(output)
    return output


def write_norm_stats(paths, output_dir, *, native_gripper_range=False):
    from openpi.shared import normalize
    running = {key: normalize.RunningStats() for key in ('state', 'actions')}
    for path in paths:
        if Path(path).suffix == '.hdf5':
            with h5py.File(path, 'r') as dataset:
                released_states = dataset.attrs.get('observation_source') == 'released_state'
                if released_states and not dataset.attrs.get('complete'):
                    raise ValueError(f'Incomplete released state archive at {path}')
                groups = dataset.values() if released_states else demonstration_groups(dataset, path)
                for group in groups:
                    states = group['proprioception'][:] if released_states else group['proprioception'][:-1]
                    running['state'].update(states)
                    running['actions'].update(group['actions'][:])
            continue
        with np.load(path) as dataset:
            actions = dataset['actions']
            # Each episode contributes one extra outcome observation.
            episode_ends = np.flatnonzero(dataset['terminals']) + 1
            states = np.delete(dataset['proprioception'], episode_ends + np.arange(len(episode_ends)), axis=0)
            if len(states) != len(actions):
                raise ValueError(f'Invalid state/action alignment in {path}')
            running['state'].update(states)
            running['actions'].update(actions)
    stats = {key: values.get_statistics() for key, values in running.items()}
    if native_gripper_range:
        actions = stats['actions']
        if len(actions.mean) != 7:
            raise ValueError('Expected six joint deltas and one gripper command')
        q01, q99 = actions.q01.copy(), actions.q99.copy()
        # OCBench already expresses gripper control in [-1, 1].
        q01[6], q99[6] = -1.0, 1.0
        stats['actions'] = replace(actions, q01=q01, q99=q99)
    normalize.save(output_dir, stats)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=('ocbench', 'ocbench_lite'), default='ocbench')
    parser.add_argument('--task-id', type=int)
    parser.add_argument('--episodes', type=int)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--norm-stats-dir', type=Path)
    parser.add_argument('--native-gripper-range', action='store_true',
                        help='Keep the gripper command in its native [-1, 1] range under quantile normalization.')
    parser.add_argument('--stats-from', type=Path, nargs='+', help='Fit shared stats from existing demo files.')
    args = parser.parse_args()
    if args.stats_from is not None:
        if args.norm_stats_dir is None or args.task_id is not None or args.episodes is not None:
            parser.error('--stats-from requires --norm-stats-dir and cannot be combined with collection options')
        write_norm_stats(args.stats_from, args.norm_stats_dir,
                         native_gripper_range=args.native_gripper_range)
        return
    if args.task_id is None or args.episodes is None or args.episodes < 1 or args.output_dir is None:
        parser.error('Set --task-id, positive --episodes, and --output-dir')
    path = collect(args.suite, args.task_id, args.episodes, args.seed, args.output_dir)
    if args.norm_stats_dir is not None:
        write_norm_stats([path], args.norm_stats_dir,
                         native_gripper_range=args.native_gripper_range)


if __name__ == '__main__':
    main()

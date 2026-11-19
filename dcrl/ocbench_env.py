"""OCBench visual CPU environments and dense demonstrations for DCRL."""

from pathlib import Path

import numpy as np
from openpi_client import image_tools

from dcrl.libero_env import CAMERA_KEYS, tokenize_instruction


# Stable zero-based IDs shared by the standard and lite suites. OCBench's
# individual environment names use one-based task suffixes.
TASKS = (
    ('block-single-task1', 'Move the cube to the marked target.'),
    ('block-double-task1', 'Place each of the two cubes at its marked target.'),
    ('block-double-task2', 'Stack the two cubes.'),
    ('block-triple-task1', 'Place each of the three cubes at its marked target.'),
    ('block-triple-task2', 'Stack the three cubes.'),
    ('block-quadruple-task1', 'Place each of the four cubes at its marked target.'),
    ('block-quadruple-task2', 'Stack the four cubes.'),
    ('block-quadruple-task3', 'Arrange the four cubes in the marked grid.'),
    ('chamber-easy-task1', 'Open the drawer and the window.'),
    ('chamber-easy-task2', 'Put the cube in the drawer, close it, and open the window.'),
    ('chamber-easy-task3', 'Take the cube out of the drawer and close the drawer and window.'),
    ('chamber-medium-task1', 'Put both cubes in the drawer, close it, and open the window.'),
    ('chamber-medium-task2', 'Stack the two cubes and close the drawer and window.'),
    ('chamber-hard-task1', 'Put all three cubes in the drawer, close it, and open the window.'),
    ('chamber-hard-task2', 'Stack the three cubes and close the drawer and window.'),
    ('switch-3x3-task1', 'Switch all buttons on the 3 by 3 board to blue.'),
    ('switch-3x3-task2', 'Switch all present buttons on the 3 by 3 board to blue.'),
    ('switch-4x4-task1', 'Switch all buttons on the 4 by 4 board to blue.'),
    ('switch-4x4-task2', 'Switch all present buttons on the 4 by 4 board to blue.'),
    ('switch-5x5-task1', 'Switch all buttons on the 5 by 5 board to blue.'),
    ('switch-5x5-task2', 'Switch all present buttons on the 5 by 5 board to blue.'),
    ('hanoi-single-task1', 'Move the disk from the center peg to either outer peg.'),
    ('hanoi-single-task2', 'Move the disk from the center peg to either outer peg.'),
    ('hanoi-double-task1', 'Move the two-disk tower to either outer peg without placing a larger disk on a smaller one.'),
    ('hanoi-double-task2', 'Move the two-disk tower to either outer peg without placing a larger disk on a smaller one.'),
    ('hanoi-triple-task1', 'Move the three-disk tower to either outer peg without placing a larger disk on a smaller one.'),
    ('hanoi-triple-task2', 'Move the three-disk tower to either outer peg without placing a larger disk on a smaller one.'),
    ('bowling-task1', 'Roll the ball to knock down the pins.'),
)


def environment_id(suite, task_id):
    if suite not in ('ocbench', 'ocbench_lite') or not 0 <= task_id < len(TASKS):
        raise ValueError(f'Invalid OCBench suite/task: {suite}/{task_id}')
    family, variant = TASKS[task_id][0].split('-', 1)
    # Bowling has one registration and the same dynamics in both suites.
    lite = 'lite-' if suite == 'ocbench_lite' and family != 'bowling' else ''
    return f'visual-{family}-cpu-{lite}{variant}-v0'


def demo_path(env_id, demos_root):
    if demos_root is None:
        raise ValueError('Set the OCBench demonstration root to a directory with dense visual demonstrations.')
    path = Path(demos_root) / f'{env_id}.hdf5'
    legacy_path = path.with_suffix('.npz')
    return legacy_path if not path.is_file() and legacy_path.is_file() else path


def proprioception(info):
    # Six arm joint angles and the opening of the gripper they control.
    return np.concatenate((info['proprio/joint_pos'], info['proprio/gripper_opening'])).astype(np.float32)


def observation(images, state, instruction, image_size):
    if images.ndim != 4 or images.shape[0] != 3 or images.shape[-1] != 3:
        raise ValueError(f'Expected OCBench front, side and wrist RGB views, got {images.shape}')
    tokens = tokenize_instruction(instruction)
    return {
        'image': {key: image_tools.resize_with_pad(images[index], image_size, image_size)
                  for key, index in zip(CAMERA_KEYS, (0, 2))},
        'state': state,
        'tokenized_prompt': tokens,
        'tokenized_prompt_mask': tokens != 0,
    }


class OCBenchTaskEnv:
    init_states = None

    def __init__(self, suite, task_id, *, seed, image_size, init_state_offset=0,
                 benchmark_eval=False):
        import ocbench

        self.env_id = environment_id(suite, task_id)
        self.instruction = TASKS[task_id][1]
        self.seed = seed
        self.image_size = image_size
        self.init_state_offset = init_state_offset
        self.episode_index = 0
        self.episode_done = False
        self.terminated = False
        self.env = ocbench.make(self.env_id)

    def reset_sequence(self):
        self.episode_index = 0

    def reset(self):
        pixels, info = self.env.reset(seed=self.seed + self.init_state_offset + self.episode_index)
        self.episode_index += 1
        self.episode_done = False
        self.terminated = False
        return observation(pixels, proprioception(info), self.instruction, self.image_size)

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape != (7,) or not np.isfinite(action).all():
            raise ValueError(f'Expected seven finite OCBench controls, got {action}')
        executed = np.clip(action, self.env.action_space.low, self.env.action_space.high)
        pixels, reward, terminated, truncated, info = self.env.step(executed)
        self.episode_done = bool(terminated or truncated)
        self.terminated = bool(terminated)
        success = bool(info['success'])
        return observation(pixels, proprioception(info), self.instruction, self.image_size), float(reward), success, executed

    def demo_path(self, demos_root=None):
        return demo_path(self.env_id, demos_root)

    def single_cube_positions(self):
        """Read the cube and goal positions for the single-cube placement task."""
        raw = self.env.unwrapped
        cube = raw._data.joint('object_joint_0').qpos[:3].copy()
        goal = raw._data.mocap_pos[raw._cube_target_mocap_ids[0]].copy()
        return cube, goal

    def close(self):
        self.env.close()


def evaluation_metrics(trials, horizon, *, score_return=False):
    """Episode success and throughput as defined by the upstream evaluator."""
    lengths = np.asarray([trial['steps'] for trial in trials], dtype=np.float64)
    mean_length = float(lengths.mean())
    if score_return:
        return {'eval/normalized_return': 100 * float(np.mean([trial['return'] for trial in trials])),
                'eval/mean_steps': mean_length}
    successes = np.asarray([trial['success'] for trial in trials], dtype=bool)
    success_rate = float(successes.mean())
    mean_success_length = float(lengths[successes].mean()) if successes.any() else None
    return {
        'eval/success_rate': success_rate,
        'eval/mean_steps': mean_length,
        'eval/mean_success_steps': mean_success_length,
        'eval/throughput': success_rate * horizon / mean_length,
        'eval/success_weighted_speed': (success_rate * horizon / mean_success_length
                                         if mean_success_length is not None else 0.0),
    }


def load_demonstrations(path, *, instruction, image_size, max_episodes=None):
    """Read dense visual OCBench files with the recorded robot proprioception."""
    return list(iter_demonstrations(path, instruction=instruction, image_size=image_size,
                                    max_episodes=max_episodes))


def iter_demonstrations(path, *, instruction, image_size, max_episodes=None):
    """Read at most one decompressed HDF5 episode at a time before replay insertion."""
    import ocbench
    from dcrl.libero_replay import Transition

    path = Path(path)
    if path.suffix == '.hdf5':
        import h5py

        with h5py.File(path, 'r') as dataset:
            groups = demonstration_groups(dataset, path)
            for ep, group in enumerate(groups[:max_episodes]):
                yield [Transition(observation(pixels, state, instruction, image_size), action,
                            float(reward), bool(done), ep, bool(group.attrs['success']),
                            truncated=bool(done and mask))
                 for pixels, state, action, reward, done, mask in zip(
                     group['observations'][:], group['proprioception'][:], group['actions'][:],
                     group['rewards'][:], group['terminals'][:], group['masks'][:])]
        return

    with np.load(path) as dataset:
        if int(dataset['observation_interval']) != 1 or 'proprioception' not in dataset:
            raise ValueError(f'{path} needs dense RGB and aligned proprioception. Generate it with dcrl.collect_ocbench_demos.')
        actions = dataset['actions'].astype(np.float32)
        pixels = dataset['observations']
        states = dataset['proprioception'].astype(np.float32)
        rewards = dataset['rewards']
        terminals = dataset['terminals']
        masks = dataset['masks']
        action_offsets, observation_offsets = ocbench.get_episode_offsets(dict(
            actions=actions, observations=pixels, terminals=terminals,
            observation_interval=dataset['observation_interval'],
        ))
    if actions.shape != (len(rewards), 7) or states.shape != (len(pixels), 7):
        raise ValueError(f'Unaligned OCBench action or proprioception arrays in {path}')
    for ep, (start, end, obs_start) in enumerate(zip(action_offsets, action_offsets[1:], observation_offsets)):
        if max_episodes is not None and ep >= max_episodes:
            break
        success = bool(rewards[end - 1] > 0)
        yield [
            Transition(observation(pixels[obs_start + i - start], states[obs_start + i - start], instruction, image_size),
                       actions[i], float(rewards[i]), bool(terminals[i]), ep, success,
                       truncated=bool(terminals[i] and masks[i]))
            for i in range(start, end)
        ]


def demonstration_groups(dataset, path):
    if not dataset.attrs.get('complete', False) or dataset.attrs.get('observation_interval') != 1:
        raise ValueError(f'Expected a completed dense visual OCBench dataset at {path}')
    groups = [dataset[key] for key in sorted(dataset)]
    if not groups:
        raise ValueError(f'No OCBench demonstration episodes at {path}')
    for episode in groups:
        length = len(episode['actions'])
        if (not length or episode['actions'].shape != (length, 7)
                or episode['proprioception'].shape != (length + 1, 7)
                or len(episode['observations']) != length + 1
                or any(episode[key].shape != (length,) for key in ('rewards', 'terminals', 'masks'))):
            raise ValueError(f'Unaligned OCBench observations or actions in {path}:{episode.name}')
    return groups

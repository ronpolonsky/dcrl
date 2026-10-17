"""LIBERO replay with the action chunks and r-step targets of EXPO-FT.

Rows are observations *before* actions. Sampling deliberately follows
expo_ft.data.replay_buffer.PiReplayBuffer.sample_jax: next state is row t+r,
rewards are gated by preceding continuation masks, and valids is the product
of the first r-1 masks. Action chunks stop at episode boundaries and repeat
their final executed action in the unfilled tail.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from dcrl.libero_env import CAMERA_KEYS, observation


@dataclass
class Transition:
    obs: dict
    action: np.ndarray  # command actually sent to LIBERO, before normalization
    reward: float
    done: bool
    episode_id: int
    success: bool = False


class ExpoLiberoReplay:
    def __init__(self, *, capacity: int, action_horizon: int, replan_steps: int,
                 padded_action_dim: int, discount: float, norm_stats: dict, seed: int):
        if capacity <= replan_steps or not 1 <= replan_steps <= action_horizon:
            raise ValueError('capacity must exceed replan_steps, and replan_steps must fit the chunk')
        from openpi import transforms
        self.rows: list[Transition] = []
        self.capacity = capacity
        self.horizon = action_horizon
        self.r = replan_steps
        self.padded_dim = padded_action_dim
        self.discount = discount
        self.normalize = transforms.Normalize(norm_stats, use_quantiles=True)
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.rows)

    def insert(self, row: Transition):
        self.rows.append(row)
        if len(self.rows) > self.capacity:
            del self.rows[0]

    def insert_episode(self, episode: list[Transition]):
        for row in episode:
            self.insert(row)

    def mark_episode_success(self, episode_id: int):
        for row in reversed(self.rows):
            if row.episode_id != episode_id:
                break
            row.success = True

    def can_sample(self, *, success_only: bool = False) -> bool:
        return len(self._eligible(success_only)) > 0

    def _eligible(self, success_only: bool) -> np.ndarray:
        # Same exclusive upper bound as EXPO-FT's randint(maxval=len-r).
        high = len(self.rows) - self.r
        if high <= 0:
            return np.empty(0, dtype=np.int64)
        indices = np.arange(high, dtype=np.int64)
        if success_only:
            indices = indices[[self.rows[i].success for i in indices]]
        return indices

    def sample(self, batch_size: int, *, success_only: bool = False) -> dict:
        eligible = self._eligible(success_only)
        if not len(eligible):
            raise ValueError('Replay has no eligible r-step windows')
        starts = self.rng.choice(eligible, size=batch_size, replace=True)
        return _stack([self._sample_row(int(t)) for t in starts])

    def _sample_row(self, t: int) -> dict:
        current = self.rows[t]
        future = self.rows[t + self.r]
        chunk = [current.action]
        for offset in range(1, self.horizon):
            previous = self.rows[t + offset - 1]
            if previous.done or t + offset >= len(self.rows):
                break
            candidate = self.rows[t + offset]
            if candidate.episode_id != current.episode_id:
                break
            chunk.append(candidate.action)
        actions = np.stack(chunk, axis=0).astype(np.float32)
        if len(actions) < self.horizon:
            actions = np.concatenate(
                [actions, np.repeat(actions[-1:], self.horizon - len(actions), axis=0)])
        actions = self.normalize({'actions': actions})['actions']
        actions = np.pad(actions, ((0, 0), (0, self.padded_dim - actions.shape[-1])))

        reward = np.float32(current.reward)
        mask = np.float32(not current.done)
        valid = np.float32(1)
        for offset in range(1, self.r):
            row = self.rows[t + offset]
            reward += np.float32(self.discount ** offset * row.reward * mask)
            valid = mask
            mask = min(mask, np.float32(not row.done))

        state = self.normalize({'state': np.asarray(current.obs['state'], np.float32)})['state']
        next_state = self.normalize({'state': np.asarray(future.obs['state'], np.float32)})['state']
        image = {key: _scale_image(current.obs['image'][key]) for key in CAMERA_KEYS}
        next_image = {key: _scale_image(future.obs['image'][key]) for key in CAMERA_KEYS}
        return {
            'image': image, 'next_image': next_image,
            'image_mask': {key: np.bool_(True) for key in CAMERA_KEYS},
            'next_image_mask': {key: np.bool_(True) for key in CAMERA_KEYS},
            'state': np.pad(state, (0, self.padded_dim - len(state))),
            'next_state': np.pad(next_state, (0, self.padded_dim - len(next_state))),
            'actions': actions,
            'tokenized_prompt': current.obs['tokenized_prompt'],
            'tokenized_prompt_mask': current.obs['tokenized_prompt_mask'],
            'rewards': reward, 'masks': np.float32(mask),
            'valids': np.float32(valid),
        }


def _scale_image(value):
    return np.asarray(value, np.float32) / 127.5 - 1.0


def _stack(rows):
    first = rows[0]
    return {key: _stack([row[key] for row in rows]) if isinstance(first[key], dict)
            else np.stack([row[key] for row in rows]) for key in first}


def concatenate_batches(online: dict | None, offline: dict | None, rng: np.random.Generator) -> dict:
    if online is None:
        return offline
    if offline is None:
        return online
    size = online['actions'].shape[0] + offline['actions'].shape[0]
    order = rng.permutation(size)
    def combine(a, b):
        if isinstance(a, dict):
            return {key: combine(a[key], b[key]) for key in a}
        return np.concatenate([a, b], axis=0)[order]
    return combine(online, offline)


def load_demonstrations(path: Path, *, instruction: str, image_size: int,
                        max_episodes: int | None = None) -> list[list[Transition]]:
    """Align post-action HDF5 observations with the following action and outcome."""
    if not path.exists():
        raise FileNotFoundError(f'LIBERO demonstrations not found: {path}')
    episodes = []
    with h5py.File(path, 'r') as file:
        names = sorted(file['data'], key=lambda name: int(name.split('_')[-1]))
        for episode_id, name in enumerate(names[:max_episodes] if max_episodes else names):
            group = file['data'][name]
            actions = np.asarray(group['actions'], np.float32)
            rewards = np.asarray(group['rewards'], np.float32)
            dones = np.asarray(group['dones'], np.bool_)
            if rewards.shape != (len(actions),) or dones.shape != (len(actions),):
                raise ValueError(f'Invalid reward/done length in {path}:{name}')
            if not dones[-1] or np.any(dones[:-1]):
                raise ValueError(f'Expected one terminal at the end of {path}:{name}')
            fields = {key: np.asarray(group['obs'][key]) for key in
                      ('agentview_rgb', 'eye_in_hand_rgb', 'ee_states', 'gripper_states')}
            if any(len(value) != len(actions) for value in fields.values()):
                raise ValueError(f'Observation/action length mismatch in {path}:{name}')
            observations = [
                observation({key: value[i] for key, value in fields.items()}, instruction, image_size)
                for i in range(len(actions))
            ]
            episode = [
                Transition(obs=observations[i], action=actions[i + 1],
                           reward=float(rewards[i + 1]), done=bool(dones[i + 1]),
                           episode_id=episode_id, success=bool(rewards[-1] > 0))
                for i in range(len(actions) - 1)
            ]
            if episode:
                episodes.append(episode)
    return episodes


def compute_norm_stats(path: Path) -> dict:
    """Use OpenPI RunningStats on every aligned demonstration, as EXPO-FT does."""
    if not path.exists():
        raise FileNotFoundError(f'LIBERO demonstrations not found: {path}')
    from openpi.shared import normalize
    running = {'state': normalize.RunningStats(), 'actions': normalize.RunningStats()}
    count = 0
    with h5py.File(path, 'r') as file:
        for group in file['data'].values():
            if len(group['actions']) < 3:
                continue
            states = np.concatenate((
                np.asarray(group['obs']['ee_states'][:-1], np.float32),
                np.asarray(group['obs']['gripper_states'][:-1], np.float32),
            ), axis=-1)
            actions = np.asarray(group['actions'][1:], np.float32)
            running['state'].update(states)
            running['actions'].update(actions)
            count += len(actions)
    if count < 2:
        raise ValueError(f'Not enough demonstrations to compute norm_stats: {path}')
    return {name: value.get_statistics() for name, value in running.items()}

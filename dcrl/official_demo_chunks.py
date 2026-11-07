"""Lazy sampler for same-task official LIBERO demonstrations as SFT chunks."""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np


class OfficialDemoChunkDataset:
    """Build Pi0.5 10/5 training rows from a task's official LIBERO HDF5 demos.

    LIBERO's saved observations are post-action. As in ``libero_replay``, row i
    is paired with action i+1. Windows begin every ``replan_steps`` and are
    padded by repeating their last action, matching the existing 10/5 SFT row
    contract. Images remain on disk and are decoded only when sampled.
    """

    OBSERVATION_FIELDS = (
        "agentview_rgb", "eye_in_hand_rgb", "ee_states", "gripper_states",
    )

    def __init__(self, path: str | Path, *, instruction: str, image_size: int,
                 action_horizon: int, replan_steps: int):
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"Official LIBERO demonstrations not found: {self.path}")
        self.instruction = instruction
        self.image_size = image_size
        self.action_horizon = action_horizon
        self.replan_steps = replan_steps
        self.file = h5py.File(self.path, "r")
        try:
            if "data" not in self.file:
                raise ValueError(f"Expected a 'data' group in {self.path}")
            names = sorted(self.file["data"], key=lambda name: int(name.split("_")[-1]))
            self.episodes: list[tuple[str, int]] = []
            self.indices: list[tuple[str, int]] = []
            for name in names:
                group = self.file["data"][name]
                if "actions" not in group or "obs" not in group:
                    raise ValueError(f"Missing actions/obs in {self.path}:{name}")
                actions = group["actions"]
                length = len(actions)
                if actions.ndim != 2 or actions.shape[1] != 7:
                    raise ValueError(f"Expected 7D actions in {self.path}:{name}, got {actions.shape}")
                if "rewards" in group and group["rewards"].shape != (length,):
                    raise ValueError(f"Invalid rewards length in {self.path}:{name}")
                if "dones" in group:
                    dones = np.asarray(group["dones"], dtype=np.bool_)
                    if dones.shape != (length,) or not dones[-1] or np.any(dones[:-1]):
                        raise ValueError(f"Expected one terminal at the end of {self.path}:{name}")
                for field in self.OBSERVATION_FIELDS:
                    if field not in group["obs"] or len(group["obs"][field]) != length:
                        raise ValueError(f"Missing or misaligned {field} in {self.path}:{name}")
                self.episodes.append((name, length))
                # Observation i is the state after action i; train it to predict
                # the following action, as in the repository's offline adapter.
                self.indices.extend((name, start) for start in range(0, length - 1, replan_steps))
            if not self.indices:
                raise ValueError(f"No eligible official demo chunks in {self.path}")
        except BaseException:
            self.file.close()
            raise

    def __len__(self) -> int:
        return len(self.indices)

    @property
    def episode_count(self) -> int:
        return len(self.episodes)

    def sample(self, rng: np.random.Generator, count: int) -> list[dict]:
        from dcrl.libero_env import observation

        rows = []
        for index in rng.integers(len(self.indices), size=count):
            name, start = self.indices[int(index)]
            group = self.file["data"][name]
            raw_obs = {key: group["obs"][key][start] for key in self.OBSERVATION_FIELDS}
            obs = observation(raw_obs, self.instruction, self.image_size)
            action_start = start + 1
            action_stop = min(action_start + self.action_horizon, len(group["actions"]))
            actions = np.asarray(group["actions"][action_start:action_stop], dtype=np.float32)
            if not len(actions):
                raise ValueError(f"Empty action chunk in {self.path}:{name} at {start}")
            if len(actions) < self.action_horizon:
                actions = np.concatenate(
                    (actions, np.repeat(actions[-1:], self.action_horizon - len(actions), axis=0)),
                    axis=0,
                )
            rows.append({
                "observation/image": obs["image"]["base_0_rgb"],
                "observation/wrist_image": obs["image"]["left_wrist_0_rgb"],
                "observation/state": obs["state"],
                "actions": actions,
                "prompt": self.instruction,
            })
        return rows

    def close(self):
        self.file.close()


def sample_mixed_sft_rows(target_dataset, demo_dataset, *, batch_size: int,
                          official_demo_ratio: float, rng: np.random.Generator):
    """Sample one batch; official_demo_ratio is its expected demo fraction."""
    if official_demo_ratio == 0:
        return target_dataset.sample(rng, batch_size), batch_size, 0
    exact_demo_count = batch_size * official_demo_ratio
    demo_count = int(exact_demo_count)
    if rng.random() < exact_demo_count - demo_count:
        demo_count += 1
    target_count = batch_size - demo_count
    rows = target_dataset.sample(rng, target_count) if target_count else []
    if demo_count:
        rows.extend(demo_dataset.sample(rng, demo_count))
    order = rng.permutation(batch_size)
    return [rows[int(index)] for index in order], target_count, demo_count

"""Disk-backed Pi0.5 chunk data for initializing SmallExpoLearner.

The Pi0.5 collector stores one row per replanning observation and the complete
action chunk predicted at that observation. This module keeps that contract
intact while applying the same state/action quantile statistics and image
scaling used by the LIBERO replay buffer.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from dcrl.libero_env import CAMERA_KEYS, tokenize_instruction


class Pi05DistillReplay:
    """Sample full Pi0.5 action chunks from completed collections.

    Both successful and failed trajectories are included by default. The
    purpose of this phase is to initialize the compact actor to the frozen
    Pi0.5 base policy, so filtering out failures would change that objective.
    success_only is retained as an explicit diagnostic option.
    """

    def __init__(
        self,
        paths: str | Path | list[str | Path],
        *,
        norm_stats: dict,
        action_horizon: int,
        padded_action_dim: int,
        seed: int,
        success_only: bool = False,
    ) -> None:
        if isinstance(paths, (str, Path)):
            paths = [paths]
        self.paths = [Path(path) for path in paths]
        if not self.paths:
            raise ValueError("at least one Pi0.5 collection is required")
        self.files: list[h5py.File] = []
        self.indices: list[tuple[int, str, int]] = []
        self.rng = np.random.default_rng(seed)
        self.action_horizon = int(action_horizon)
        self.padded_action_dim = int(padded_action_dim)
        if self.action_horizon < 1 or self.padded_action_dim < 7:
            raise ValueError("invalid action shape for Pi0.5 distillation")

        from openpi import transforms

        self.normalize = transforms.Normalize(norm_stats, use_quantiles=True)
        try:
            instruction = None
            metadata = None
            for file_index, path in enumerate(self.paths):
                if not path.exists():
                    raise FileNotFoundError(path)
                file = h5py.File(path, "r")
                self.files.append(file)
                if not bool(file.attrs.get("complete", False)):
                    raise ValueError(f"Pi0.5 collection is incomplete: {path}")
                current_instruction = str(file.attrs.get("instruction", ""))
                current_metadata = json.loads(file.attrs["metadata"])
                if instruction is None:
                    instruction = current_instruction
                    metadata = current_metadata
                elif current_instruction != instruction or current_metadata != metadata:
                    raise ValueError("Pi0.5 collections have incompatible metadata")
                for name, group in file.items():
                    if not bool(group.attrs.get("complete", False)):
                        continue
                    if success_only and not bool(group.attrs.get("success", False)):
                        continue
                    if set(CAMERA_KEYS) - set(group.keys()):
                        raise ValueError(f"missing camera data in {path}:{name}")
                    actions = group["actions"]
                    if actions.ndim != 3 or actions.shape[-1] != 7:
                        raise ValueError(f"invalid action shape in {path}:{name}: {actions.shape}")
                    if group["state"].shape[0] != actions.shape[0]:
                        raise ValueError(f"state/chunk count mismatch in {path}:{name}")
                    self.indices.extend(
                        (file_index, name, row) for row in range(actions.shape[0])
                    )
            if not self.indices:
                raise ValueError("Pi0.5 collections contain no eligible chunk rows")
            self.instruction = instruction or ""
            self.metadata = metadata or {}
            self.tokenized_prompt = tokenize_instruction(self.instruction)
        except BaseException:
            self.close()
            raise

    def __len__(self) -> int:
        return len(self.indices)

    def _pad_or_trim_chunk(self, actions: np.ndarray) -> np.ndarray:
        """Match learner horizon while preserving every teacher action."""
        actions = np.asarray(actions, np.float32)
        source_horizon = actions.shape[0]
        if source_horizon >= self.action_horizon:
            actions = actions[:self.action_horizon]
        else:
            # Pi0.5 LIBERO emits ten actions. A learner configured with a
            # longer horizon receives the final teacher action in the tail,
            # matching the replay buffer's terminal chunk padding convention.
            actions = np.concatenate(
                [actions, np.repeat(actions[-1:], self.action_horizon - source_horizon, axis=0)],
                axis=0,
            )
        return actions

    def _row(self, index: int) -> dict:
        file_index, name, row = self.indices[index]
        group = self.files[file_index][name]
        image = {
            key: np.asarray(group[key][row], np.float32) / 127.5 - 1.0
            for key in CAMERA_KEYS
        }
        state = np.asarray(group["state"][row], np.float32)
        state = np.asarray(self.normalize({"state": state})["state"], np.float32)
        if state.ndim != 1:
            state = state.reshape(-1)
        if state.shape[0] > self.padded_action_dim:
            raise ValueError("state dimension exceeds learner padded action dimension")
        state = np.pad(state, (0, self.padded_action_dim - state.shape[0]))

        # The actor loss consumes unpadded environment actions (7-D), while
        # only the critic replay representation pads actions to 32 dimensions.
        # Keep this distillation target aligned with SmallActor.prepare_batch_for_actor.
        actions = self._pad_or_trim_chunk(np.asarray(group["actions"][row], np.float32))
        actions = np.asarray(self.normalize({"actions": actions})["actions"], np.float32)
        return {
            "image": image,
            "state": state,
            "tokenized_prompt": self.tokenized_prompt,
            "tokenized_prompt_mask": self.tokenized_prompt != 0,
            "full_actions": actions.reshape(-1),
        }

    def sample(self, batch_size: int) -> dict:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        rows = [self._row(int(i)) for i in self.rng.integers(len(self), size=batch_size)]

        def stack(values):
            return np.stack(values, axis=0)

        return {
            "image": {key: stack([row["image"][key] for row in rows]) for key in CAMERA_KEYS},
            "state": stack([row["state"] for row in rows]),
            "tokenized_prompt": stack([row["tokenized_prompt"] for row in rows]),
            "tokenized_prompt_mask": stack([row["tokenized_prompt_mask"] for row in rows]),
            "full_actions": stack([row["full_actions"] for row in rows]),
        }

    def close(self) -> None:
        for file in getattr(self, "files", []):
            file.close()
        self.files = []

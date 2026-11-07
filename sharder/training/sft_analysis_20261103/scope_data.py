"""Isolated SFT data variants used by the retention investigation.

The production ``ChunkDataset`` preserves the teacher's complete 10-action
proposal.  ``executed_backfill`` reconstructs the 10-action target from the
actions actually executed at the current and subsequent 5-step replanning
windows.  It is intentionally confined to the audit trainer so the existing
SFT path is unchanged.
"""

from __future__ import annotations

import numpy as np

from dcrl.bc_distill_data import ChunkDataset as _CanonicalChunkDataset


class ChunkDataset(_CanonicalChunkDataset):
    """Canonical chunk dataset plus an explicit target-semantics ablation."""

    def __init__(self, paths, *, success_only=True, target_mode="full_proposal"):
        if target_mode not in ("full_proposal", "executed_backfill"):
            raise ValueError(f"unknown target_mode: {target_mode}")
        self.target_mode = target_mode
        super().__init__(paths, success_only=success_only)
        if self.target_mode == "executed_backfill":
            for file in self.files:
                for name, group in file.items():
                    if not bool(group.attrs.get("complete", False)):
                        continue
                    if "executed_prefix" not in group:
                        raise ValueError(
                            "executed_backfill requires collection metadata with "
                            "executed_prefix per row"
                        )

    def _executed_chunk(self, file_index: int, episode_name: str, row: int) -> np.ndarray:
        """Return a 10x7 target aligned to the row's actual rollout trajectory."""
        group = self.files[file_index][episode_name]
        actions = group["actions"]
        executed = np.asarray(group["executed_prefix"], dtype=np.int32)
        pieces = []
        for current in range(row, len(actions)):
            count = int(np.clip(executed[current], 0, actions.shape[1]))
            if count:
                pieces.append(np.asarray(actions[current][:count], dtype=np.float32))
            if sum(piece.shape[0] for piece in pieces) >= self.action_horizon:
                break
        if not pieces:
            raise ValueError(
                f"row {episode_name}:{row} has no executed actions for backfill"
            )
        result = np.concatenate(pieces, axis=0)[: self.action_horizon]
        if len(result) < self.action_horizon:
            result = np.concatenate(
                (result, np.repeat(result[-1:], self.action_horizon - len(result), axis=0)),
                axis=0,
            )
        return result

    def sample(self, rng: np.random.Generator, count: int) -> list[dict]:
        if self.target_mode == "full_proposal":
            return super().sample(rng, count)

        result = []
        for index in rng.integers(len(self.indices), size=count):
            file_index, name, row = self.indices[int(index)]
            group = self.files[file_index][name]
            result.append({
                "observation/image": group["base_0_rgb"][row],
                "observation/wrist_image": group["left_wrist_0_rgb"][row],
                "observation/state": group["state"][row],
                "actions": self._executed_chunk(file_index, name, row),
                "prompt": self.instruction,
            })
        return result

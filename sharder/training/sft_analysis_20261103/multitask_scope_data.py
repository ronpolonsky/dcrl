"""Isolated balanced multi-task Small EXPO dataset for the SFT audit.

The production SFT dataset intentionally rejects files with different
instructions.  That is correct for a task-specific run, but it prevents the
rehearsal needed by the generalization experiment.  This module keeps each
file's prompt and samples the four task collections uniformly by task.  It is
loaded only by the isolated wrapper; production data code is unchanged.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np


def configured_paths() -> list[Path]:
    value = os.environ.get("SFT_MULTITASK_SUCCESS_PATHS", "")
    if not value:
        raise RuntimeError(
            "SFT_MULTITASK_SUCCESS_PATHS must contain one success HDF5 path per task"
        )
    paths = [Path(item).expanduser().resolve() for item in value.split(os.pathsep) if item]
    if len(paths) < 2:
        raise ValueError("multi-task SFT requires at least two collection files")
    return paths


def audit_paths(paths: list[Path]) -> dict:
    records = []
    for path in paths:
        with h5py.File(path, "r") as file:
            if not bool(file.attrs.get("complete", False)):
                raise ValueError(f"collection is incomplete: {path}")
            metadata = json.loads(file.attrs["metadata"])
            instruction = str(file.attrs["instruction"])
            task_id = int(metadata.get("task_id", metadata["teacher_config"]["task_id"]))
            chunks = sum(len(group["actions"]) for group in file.values())
            episodes = sum(
                bool(group.attrs.get("complete", False)) for group in file.values()
            )
        records.append({
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "task_id": task_id,
            "instruction": instruction,
            "chunks": chunks,
            "episodes": episodes,
            "metadata": metadata,
        })
    return {"sampling": "uniform_task_then_uniform_chunk", "tasks": records}


class ChunkDataset:
    """Balanced multi-task HDF5 sampler with prompt-preserving rows."""

    def __init__(self, _paths, *, success_only=True, target_mode="full_proposal"):
        if not success_only:
            raise ValueError("the audit arm intentionally uses successful rows only")
        if target_mode not in ("full_proposal", "executed_backfill"):
            raise ValueError(f"unknown target mode: {target_mode}")
        self.target_mode = target_mode
        self.paths = configured_paths()
        self.files = [h5py.File(path, "r") for path in self.paths]
        try:
            self.records = []
            self.task_indices = []
            self.indices = []
            first_metadata = None
            first_instruction = None
            horizon = None
            for file_index, (path, file) in enumerate(zip(self.paths, self.files)):
                if not bool(file.attrs.get("complete", False)):
                    raise ValueError(f"collection is incomplete: {path}")
                metadata = json.loads(file.attrs["metadata"])
                instruction = str(file.attrs["instruction"])
                task_id = int(metadata.get("task_id", metadata["teacher_config"]["task_id"]))
                task_rows = []
                for name, group in file.items():
                    if not bool(group.attrs.get("complete", False)):
                        continue
                    if not bool(group.attrs.get("success", False)):
                        continue
                    if "actions" not in group:
                        raise ValueError(f"missing actions in {path}:{name}")
                    if target_mode == "executed_backfill" and "executed_prefix" not in group:
                        raise ValueError(
                            f"executed_backfill requires executed_prefix in {path}:{name}"
                        )
                    for row in range(len(group["actions"])):
                        index = (file_index, name, row)
                        task_rows.append(index)
                        self.indices.append(index)
                if not task_rows:
                    raise ValueError(f"no successful chunks in {path}")
                this_horizon = int(file[task_rows[0][1]]["actions"].shape[1])
                if horizon is None:
                    horizon = this_horizon
                    first_metadata = metadata
                    first_instruction = instruction
                elif this_horizon != horizon:
                    raise ValueError("all multi-task collections must have the same action horizon")
                self.task_indices.append(task_rows)
                self.records.append({
                    "path": str(path), "task_id": task_id,
                    "instruction": instruction, "chunks": len(task_rows),
                    "metadata": metadata,
                })
            self.action_horizon = int(horizon)
            self.instruction = first_instruction
            self.metadata = dict(first_metadata)
            self.metadata.update({
                "multi_task": True,
                "multi_task_ids": [record["task_id"] for record in self.records],
                "multi_task_paths": [record["path"] for record in self.records],
                "sampling": "uniform_task_then_uniform_chunk",
            })
            self.manifest = {
                "target_mode": target_mode,
                "sampling": "uniform_task_then_uniform_chunk",
                "tasks": self.records,
            }
        except BaseException:
            self.close()
            raise

    def __len__(self):
        return len(self.indices)

    def _executed_chunk(self, file_index, name, row):
        group = self.files[file_index][name]
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
            raise ValueError(f"no executed actions for {name}:{row}")
        result = np.concatenate(pieces, axis=0)[: self.action_horizon]
        if len(result) < self.action_horizon:
            result = np.concatenate(
                (result, np.repeat(result[-1:], self.action_horizon - len(result), axis=0)),
                axis=0,
            )
        return result

    def sample(self, rng: np.random.Generator, count: int) -> list[dict]:
        rows = []
        for _ in range(count):
            task_index = int(rng.integers(len(self.task_indices)))
            file_index, name, row = self.task_indices[task_index][
                int(rng.integers(len(self.task_indices[task_index])))
            ]
            group = self.files[file_index][name]
            actions = (
                self._executed_chunk(file_index, name, row)
                if self.target_mode == "executed_backfill"
                else np.asarray(group["actions"][row], dtype=np.float32)
            )
            rows.append({
                "observation/image": group["base_0_rgb"][row],
                "observation/wrist_image": group["left_wrist_0_rgb"][row],
                "observation/state": group["state"][row],
                "actions": actions,
                "prompt": self.records[task_index]["instruction"],
                "task_id": self.records[task_index]["task_id"],
            })
        return rows

    def close(self):
        for file in getattr(self, "files", []):
            file.close()

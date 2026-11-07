"""Create tiny trusted LIBERO HDF5 fixtures for a 3-phase trainer smoke run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--norm-stats", type=Path, required=True)
    args = parser.parse_args()

    task_name = "KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet"
    demo_path = args.root / "datasets" / "libero_90" / f"{task_name}_demo.hdf5"
    distill_path = args.root / "distill" / "pi05_task_00_smoke.hdf5"
    demo_path.parent.mkdir(parents=True, exist_ok=True)
    distill_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    steps = 8
    image = np.zeros((steps, 256, 256, 3), dtype=np.uint8)
    ee = np.zeros((steps, 6), dtype=np.float32)
    gripper = np.zeros((steps, 2), dtype=np.float32)
    actions = np.zeros((steps, 7), dtype=np.float32)
    actions[:, -1] = -1.0
    rewards = np.zeros(steps, dtype=np.float32)
    rewards[-1] = 1.0
    dones = np.zeros(steps, dtype=np.bool_)
    dones[-1] = True

    with h5py.File(demo_path, "w") as file:
        data = file.create_group("data")
        for episode in range(2):
            group = data.create_group(f"demo_{episode}")
            group.create_dataset("actions", data=actions)
            group.create_dataset("rewards", data=rewards)
            group.create_dataset("dones", data=dones)
            obs = group.create_group("obs")
            obs.create_dataset("agentview_rgb", data=image)
            obs.create_dataset("eye_in_hand_rgb", data=image)
            obs.create_dataset("ee_states", data=ee)
            obs.create_dataset("gripper_states", data=gripper)

    norm_stats = json.loads(args.norm_stats.read_text())
    metadata = json.dumps({"suite": "libero_90", "task_id": 0,
                           "action_horizon": 10, "replan_steps": 5,
                           "image_size": 224}, sort_keys=True)
    with h5py.File(distill_path, "w") as file:
        file.attrs["complete"] = True
        file.attrs["instruction"] = "close the top drawer of the cabinet"
        file.attrs["metadata"] = metadata
        file.attrs["normalization_stats"] = json.dumps(norm_stats)
        group = file.create_group("episode_000")
        group.attrs["complete"] = True
        group.attrs["success"] = True
        rows = 2
        group.create_dataset("base_0_rgb", data=rng.integers(0, 256, (rows, 224, 224, 3), dtype=np.uint8))
        group.create_dataset("left_wrist_0_rgb", data=rng.integers(0, 256, (rows, 224, 224, 3), dtype=np.uint8))
        group.create_dataset("state", data=np.zeros((rows, 8), dtype=np.float32))
        group.create_dataset("actions", data=np.zeros((rows, 10, 7), dtype=np.float32))

    print(demo_path)
    print(distill_path)


if __name__ == "__main__":
    main()

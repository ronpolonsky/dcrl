"""Data collection and storage for SmallExpo-to-Pi0.5 BC distillation.

Each row is one replanning observation and the teacher's complete target chunk.
The executed prefix is recorded as rollout metadata; it is not used to turn the
chunk into one-step behavior-cloning targets.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import multiprocessing as mp
import os

import h5py
import numpy as np
from tqdm import tqdm


CAMERA_DATASETS = ("base_0_rgb", "left_wrist_0_rgb")


def partition_collection(source: Path, success_path: Path, failure_path: Path) -> None:
    """Atomically split a complete mixed collection into success/failure files."""
    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(source)
    with h5py.File(source, "r") as src:
        if not bool(src.attrs.get("complete", False)):
            raise ValueError(f"collection is incomplete: {source}")
        metadata = src.attrs["metadata"]
        for destination, wanted_success in ((success_path, True), (failure_path, False)):
            destination = Path(destination)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                with h5py.File(destination, "r") as existing:
                    if (not bool(existing.attrs.get("complete", False))
                            or existing.attrs.get("metadata") != metadata):
                        raise ValueError(f"partition does not match collection: {destination}")
                continue
            partial = destination.with_suffix(".partial.hdf5")
            with h5py.File(partial, "w") as dst:
                for key, value in src.attrs.items():
                    dst.attrs[key] = value
                dst.attrs["partition"] = "success" if wanted_success else "failure"
                for name, group in src.items():
                    if bool(group.attrs.get("success", False)) == wanted_success:
                        src.copy(name, dst, name=name)
                dst.attrs["complete"] = True
            partial.replace(destination)


def _episode_worker(connection, suite, task_id, seed, image_size, init_state_index, max_steps):
    """Run one MuJoCo episode in a process with its own EGL context."""
    env = None
    try:
        from dcrl.libero_env import LiberoTaskEnv

        env = LiberoTaskEnv(suite, task_id, seed=seed, image_size=image_size,
                            init_state_offset=init_state_index)
        observation = env.reset()
        elapsed = 0
        connection.send(("observation", observation))
        while elapsed < max_steps:
            actions = connection.recv()
            success = False
            executed = 0
            for action in actions:
                observation, _, success, _ = env.step(action)
                elapsed += 1
                executed += 1
                if success or elapsed >= max_steps:
                    break
            connection.send(("result", observation, success, executed, elapsed))
            if success or elapsed >= max_steps:
                break
    except BaseException as exc:
        try:
            connection.send(("error", repr(exc)))
        except (BrokenPipeError, EOFError, OSError):
            pass
        raise
    finally:
        if env is not None:
            env.close()
        connection.close()


def collect_teacher_chunks(
    teacher_dir: Path,
    checkpoint_step: int,
    output: Path,
    episodes: int,
    seed: int,
    *,
    target_action_horizon: int = 10,
    target_replan_steps: int = 5,
    target_successes: int | None = None,
    target_failures: int | None = None,
) -> None:
    """Collect 10/5 chunks from a frozen SmallExpo checkpoint.

    The teacher is constructed with the action/replan shapes stored in its
    checkpoint so Orbax restores exactly. For legacy 16/8 checkpoints, only
    the first 10 predicted actions are written and the first 5 are executed.
    When success/failure targets are supplied, only the requested number of
    completed trajectories from each partition is stored. Extra rollout
    attempts are discarded, matching the direct Pi0.5 collector semantics.
    """
    import jax
    import orbax.checkpoint as ocp
    from openpi.shared import normalize
    from openpi.training import sharding
    from dcrl.agents.pi_expo_learner import restore_checkpoint
    from dcrl.agents.small_expo_learner import SmallExpoLearner
    from dcrl.libero_env import LIBERO_SUITE_MAX_STEPS, LiberoTaskEnv

    teacher_dir = Path(teacher_dir)
    output = Path(output)
    if (target_successes is None) != (target_failures is None):
        raise ValueError("provide both target_successes and target_failures, or neither")
    if target_successes is not None and (target_successes <= 0 or target_failures <= 0):
        raise ValueError("target_successes and target_failures must be positive")
    # ``episodes`` remains the user-visible minimum attempt budget. Quota mode
    # gets the same four-times quota allowance as the Pi0.5 collector so a
    # difficult task has room to obtain both outcomes.
    max_attempts = int(episodes)
    if target_successes is not None:
        max_attempts = max(max_attempts, 4 * (int(target_successes) + int(target_failures)))
    config = json.loads((teacher_dir / "run_config.json").read_text())
    teacher_horizon = int(config["action_horizon"])
    teacher_replan = int(config["replan_steps"])
    if not (1 <= target_replan_steps <= target_action_horizon <= teacher_horizon):
        raise ValueError("require 1 <= target_replan_steps <= target_action_horizon <= teacher horizon")
    norm_bytes = (teacher_dir / "norm_stats.json").read_bytes()
    metadata = dict(
        teacher_dir=str(teacher_dir.resolve()),
        checkpoint_step=int(checkpoint_step),
        seed=int(seed),
        episodes=int(episodes),
        target_successes=(None if target_successes is None else int(target_successes)),
        target_failures=(None if target_failures is None else int(target_failures)),
        max_attempts=int(max_attempts),
        teacher_config=config,
        target_action_horizon=int(target_action_horizon),
        target_replan_steps=int(target_replan_steps),
        norm_sha256=hashlib.sha256(norm_bytes).hexdigest(),
        format_version=2,
    )

    if output.exists():
        with h5py.File(output, "r") as existing:
            if (json.loads(existing.attrs["metadata"]) != metadata
                    or not bool(existing.attrs.get("complete", False))):
                raise ValueError("existing collection does not match this request")
        partition_collection(output, output.parent / "success/teacher_chunks.hdf5",
                            output.parent / "failures/teacher_chunks.hdf5")
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(".partial.hdf5")
    if partial.exists():
        # A preempted process can leave a short, invalid HDF5 header. Remove
        # that artifact so a later run can restart collection safely; valid
        # partial collections remain resumable below.
        try:
            with h5py.File(partial, "r"):
                pass
        except OSError:
            partial.unlink()
    bootstrap = LiberoTaskEnv(config["suite"], config["task_id"], seed=seed,
                              image_size=config["image_size"])
    try:
        example_observation = bootstrap.reset()
        instruction = bootstrap.instruction
        initial_state_count = len(bootstrap.init_states)
        bootstrap.close()
        bootstrap = None

        mesh = sharding.make_mesh(1)
        data_sharding = jax.sharding.NamedSharding(
            mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
        replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        teacher = SmallExpoLearner.from_examples(
            seed=int(config["seed"]) + int(config["task_id"]),
            example_observation=example_observation,
            example_action=np.zeros(7, np.float32),
            example_state=example_observation["state"],
            mesh=mesh,
            data_sharding=data_sharding,
            replicated_sharding=replicated_sharding,
            image_size=int(config["image_size"]),
            action_horizon=teacher_horizon,
            replan_steps=teacher_replan,
            actor_success_only=True,
            norm_stats=normalize.load(teacher_dir),
        )
        with ocp.CheckpointManager(
            teacher_dir / "checkpoints",
            item_handlers={"agent": ocp.PyTreeCheckpointHandler(),
                           "params": ocp.PyTreeCheckpointHandler()},
            options=ocp.CheckpointManagerOptions(create=False, read_only=True),
        ) as manager:
            if checkpoint_step not in manager.all_steps():
                raise ValueError(f"teacher checkpoint {checkpoint_step} does not exist")
            teacher = restore_checkpoint(manager, teacher, checkpoint_step).cache_infer_params()
        print(f"Restored frozen teacher checkpoint {checkpoint_step}", flush=True)
        teacher = teacher.replace(rng=jax.random.PRNGKey(seed))

        # Some H100 EGL stacks abort when MuJoCo rendering happens in a
        # spawned child after JAX has initialized the GPU. Keep the worker
        # path available, but use one same-process LIBERO environment by
        # default for production collection.
        use_worker = os.environ.get("DCRL_EPISODE_WORKER", "0") == "1"
        direct_env = None
        if not use_worker:
            direct_env = LiberoTaskEnv(config["suite"], config["task_id"], seed=seed,
                                       image_size=int(config["image_size"]))
        with h5py.File(partial, "a") as file:
            if "metadata" in file.attrs and json.loads(file.attrs["metadata"]) != metadata:
                raise ValueError("partial collection does not match request")
            file.attrs["metadata"] = json.dumps(metadata, sort_keys=True)
            file.attrs["instruction"] = instruction
            file.attrs["teacher_norm_stats"] = norm_bytes.decode()
            complete = [name for name in file if bool(file[name].attrs.get("complete", False))]
            if complete:
                last = max(complete, key=lambda name: int(name.rsplit("_", 1)[1]))
                saved_rng = file[last].attrs.get("teacher_rng_after")
                if saved_rng is not None:
                    teacher = teacher.replace(rng=jax.numpy.asarray(saved_rng, dtype=jax.numpy.uint32))

            for episode in tqdm(range(max_attempts), desc="Collect 10/5 teacher episodes"):
                if target_successes is not None:
                    success_count = sum(
                        bool(file[name].attrs.get("success", False))
                        for name in file
                        if bool(file[name].attrs.get("complete", False))
                    )
                    failure_count = sum(
                        not bool(file[name].attrs.get("success", False))
                        for name in file
                        if bool(file[name].attrs.get("complete", False))
                    )
                    if success_count >= target_successes and failure_count >= target_failures:
                        break
                name = f"episode_{episode:06d}"
                if name in file and bool(file[name].attrs.get("complete", False)):
                    continue
                if name in file:
                    del file[name]
                rows = []
                success = False
                elapsed = 0
                limit = LIBERO_SUITE_MAX_STEPS[config["suite"]]
                if use_worker:
                    context = mp.get_context("spawn")
                    parent, child = context.Pipe()
                    worker = context.Process(
                        target=_episode_worker,
                        args=(child, config["suite"], int(config["task_id"]), seed + episode,
                              int(config["image_size"]), episode % initial_state_count, limit),
                    )
                    worker.start()
                    child.close()
                    try:
                        reply = parent.recv()
                        if reply[0] != "observation":
                            raise RuntimeError(f"LIBERO worker failed to initialize: {reply[1]}")
                        observation = reply[1]
                        while elapsed < limit and not success:
                            start = observation
                            chunk, teacher, _ = teacher.sample_actions(observation)
                            chunk = np.asarray(jax.device_get(chunk), np.float32)
                            if chunk.shape != (teacher_horizon, 7) or not np.isfinite(chunk).all():
                                raise ValueError("teacher returned an invalid action chunk")
                            target = chunk[:target_action_horizon]
                            parent.send(target[:min(target_replan_steps, limit - elapsed)])
                            reply = parent.recv()
                            if reply[0] == "error":
                                raise RuntimeError(f"LIBERO worker failed: {reply[1]}")
                            _, observation, success, executed, elapsed = reply
                            rows.append((start, target, executed))
                    finally:
                        parent.close()
                        worker.join()
                    if worker.exitcode != 0:
                        raise RuntimeError(f"LIBERO episode worker exited with {worker.exitcode}")
                else:
                    # Reuse one EGL context in this process. This avoids the
                    # H100 child-process render abort and is faster as well.
                    direct_env.seed = seed + episode
                    direct_env.init_state_offset = episode % initial_state_count
                    direct_env.episode_index = 0
                    observation = direct_env.reset()
                    while elapsed < limit and not success:
                        start = observation
                        chunk, teacher, _ = teacher.sample_actions(observation)
                        chunk = np.asarray(jax.device_get(chunk), np.float32)
                        if chunk.shape != (teacher_horizon, 7) or not np.isfinite(chunk).all():
                            raise ValueError("teacher returned an invalid action chunk")
                        target = chunk[:target_action_horizon]
                        executed = 0
                        for action in target[:min(target_replan_steps, limit - elapsed)]:
                            observation, _, success, _ = direct_env.step(action)
                            elapsed += 1
                            executed += 1
                            if success or elapsed >= limit:
                                break
                        rows.append((start, target, executed))
                if target_successes is not None:
                    success_count = sum(
                        bool(file[n].attrs.get("success", False))
                        for n in file
                        if bool(file[n].attrs.get("complete", False))
                    )
                    failure_count = sum(
                        not bool(file[n].attrs.get("success", False))
                        for n in file
                        if bool(file[n].attrs.get("complete", False))
                    )
                    # Do not retain samples after their requested partition is
                    # full; the rollout was still executed to preserve the
                    # environment/teacher state progression.
                    if ((success and success_count >= target_successes)
                            or (not success and failure_count >= target_failures)):
                        continue
                group = file.create_group(name)
                group.attrs.update(success=bool(success), steps=int(elapsed),
                                   initial_state_index=episode % initial_state_count,
                                   teacher_rng_after=np.asarray(jax.device_get(teacher.rng), np.uint32),
                                   complete=True)
                for key in CAMERA_DATASETS:
                    group.create_dataset(key, data=np.stack([row[0]["image"][key] for row in rows]),
                                         compression="gzip")
                group.create_dataset("state", data=np.stack([row[0]["state"] for row in rows]))
                group.create_dataset("actions", data=np.stack([row[1] for row in rows]))
                group.create_dataset("executed_prefix", data=[row[2] for row in rows])
                file.flush()
            if target_successes is not None:
                success_count = sum(
                    bool(file[name].attrs.get("success", False))
                    for name in file
                    if bool(file[name].attrs.get("complete", False))
                )
                failure_count = sum(
                    not bool(file[name].attrs.get("success", False))
                    for name in file
                    if bool(file[name].attrs.get("complete", False))
                )
                if success_count < target_successes or failure_count < target_failures:
                    raise RuntimeError(
                        f"reached collection attempt limit with {success_count} successes and "
                        f"{failure_count} failures; requested "
                        f"{target_successes}/{target_failures}"
                    )
            file.attrs["complete"] = True
        if direct_env is not None:
            direct_env.close()
            direct_env = None
        partial.replace(output)
        partition_collection(output, output.parent / "success/teacher_chunks.hdf5",
                            output.parent / "failures/teacher_chunks.hdf5")
    finally:
        if 'direct_env' in locals() and direct_env is not None:
            direct_env.close()
        if bootstrap is not None:
            bootstrap.close()


class ChunkDataset:
    """Disk-backed replanning rows with full action chunks as targets."""

    def __init__(self, paths, *, success_only=True):
        paths = [paths] if isinstance(paths, (str, Path)) else list(paths)
        self.files = [h5py.File(path, "r") for path in paths]
        try:
            if any(not bool(file.attrs.get("complete", False)) for file in self.files):
                raise ValueError("collection partition is incomplete")
            self.metadata = json.loads(self.files[0].attrs["metadata"])
            self.instruction = self.files[0].attrs["instruction"]
            for file in self.files[1:]:
                if (json.loads(file.attrs["metadata"]) != self.metadata
                        or file.attrs["instruction"] != self.instruction):
                    raise ValueError("collection partitions do not match")
            self.indices = [
                (file_index, name, row)
                for file_index, file in enumerate(self.files)
                for name, group in file.items()
                if not success_only or bool(group.attrs.get("success", False))
                for row in range(len(group["actions"]))
            ]
            if not self.indices:
                raise ValueError("no eligible chunks")
            file_index, name, _ = self.indices[0]
            self.action_horizon = int(self.files[file_index][name]["actions"].shape[1])
        except Exception:
            self.close()
            raise

    def sample(self, rng: np.random.Generator, count: int) -> list[dict]:
        result = []
        for index in rng.integers(len(self.indices), size=count):
            file_index, name, row = self.indices[int(index)]
            group = self.files[file_index][name]
            result.append({
                "observation/image": group["base_0_rgb"][row],
                "observation/wrist_image": group["left_wrist_0_rgb"][row],
                "observation/state": group["state"][row],
                "actions": group["actions"][row],
                "prompt": self.instruction,
            })
        return result

    def close(self):
        for file in getattr(self, "files", []):
            file.close()

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
        from dcrl.benchmarks import make_env

        env = make_env(suite, task_id, seed=seed, image_size=image_size,
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
                if env.episode_done or success or elapsed >= max_steps:
                    break
            done = env.episode_done or success
            connection.send(("result", observation, success, done, executed, elapsed))
            if done or elapsed >= max_steps:
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
    max_attempts_override: int | None = None,
    progress_callback=None,
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
    from dcrl.benchmarks import is_ocbench, make_env, max_episode_steps

    teacher_dir, checkpoint_root, checkpoint_step = _resolve_small_expo_checkpoint(
        teacher_dir, checkpoint_step
    )
    output = Path(output)
    if (target_successes is None) != (target_failures is None):
        raise ValueError("provide both target_successes and target_failures, or neither")
    if target_successes is not None and (target_successes <= 0 or target_failures < 0):
        raise ValueError("target_successes must be positive and target_failures non-negative")
    if target_successes == 0 and target_failures == 0:
        raise ValueError("at least one collection target must be positive")
    # ``episodes`` remains the user-visible minimum attempt budget. Quota mode
    # gets the same four-times quota allowance as the Pi0.5 collector so a
    # difficult task has room to obtain both outcomes.
    max_attempts = int(episodes)
    if target_successes is not None:
        max_attempts = max(max_attempts, 4 * (int(target_successes) + int(target_failures)))
    if max_attempts_override is not None:
        if int(max_attempts_override) < max_attempts:
            raise ValueError("max_attempts_override cannot be smaller than the quota attempt budget")
        max_attempts = int(max_attempts_override)
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
    ocbench = is_ocbench(config['suite'])
    if ocbench:
        metadata.update(suite=config['suite'], task_id=config['task_id'], image_size=config['image_size'])

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
    bootstrap = make_env(config["suite"], config["task_id"], seed=seed,
                              image_size=config["image_size"])
    try:
        example_observation = bootstrap.reset()
        instruction = bootstrap.instruction
        initial_state_count = None if ocbench else len(bootstrap.init_states)
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
            checkpoint_root,
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
            direct_env = make_env(config["suite"], config["task_id"], seed=seed,
                                       image_size=int(config["image_size"]))
        with h5py.File(partial, "a") as file:
            if "metadata" in file.attrs:
                existing_metadata = json.loads(file.attrs["metadata"])
                # A failed quota collection can be resumed with a larger
                # attempt budget. It can also be finalized with a lower
                # failure target when the existing partition already meets it.
                comparable_existing = dict(existing_metadata)
                comparable_requested = dict(metadata)
                for key in ("episodes", "max_attempts", "target_failures"):
                    comparable_existing.pop(key, None)
                    comparable_requested.pop(key, None)
                if comparable_existing != comparable_requested:
                    raise ValueError("partial collection does not match request")
                old_failures = existing_metadata.get("target_failures")
                new_failures = metadata.get("target_failures")
                if (old_failures is not None and new_failures is not None
                        and int(new_failures) > int(old_failures)):
                    raise ValueError("cannot increase the failure target while resuming a partial collection")
            file.attrs["metadata"] = json.dumps(metadata, sort_keys=True)
            file.attrs["instruction"] = instruction
            file.attrs["teacher_norm_stats"] = norm_bytes.decode()
            complete = [name for name in file if bool(file[name].attrs.get("complete", False))]
            if complete:
                last = max(complete, key=lambda name: int(name.rsplit("_", 1)[1]))
                saved_rng = file[last].attrs.get("teacher_rng_after")
                if saved_rng is not None:
                    teacher = teacher.replace(rng=jax.numpy.asarray(saved_rng, dtype=jax.numpy.uint32))
                if progress_callback is not None:
                    success_count = sum(bool(file[n].attrs.get("success", False)) for n in complete)
                    progress_callback({
                        "episode": max(int(n.rsplit("_", 1)[1]) for n in complete) + 1,
                        "successes": success_count,
                        "failures": len(complete) - success_count,
                    })

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
                limit = max_episode_steps(config['suite'], int(config['task_id']))
                episode_seed = seed if ocbench else seed + episode
                state_index = episode if ocbench else episode % initial_state_count
                if use_worker:
                    context = mp.get_context("spawn")
                    parent, child = context.Pipe()
                    worker = context.Process(
                        target=_episode_worker,
                        args=(child, config["suite"], int(config["task_id"]), episode_seed,
                              int(config["image_size"]), state_index, limit),
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
                            _, observation, success, done, executed, elapsed = reply
                            rows.append((start, target, executed))
                            if done:
                                break
                    finally:
                        parent.close()
                        worker.join()
                    if worker.exitcode != 0:
                        raise RuntimeError(f"LIBERO episode worker exited with {worker.exitcode}")
                else:
                    # Reuse one EGL context in this process. This avoids the
                    # H100 child-process render abort and is faster as well.
                    direct_env.seed = episode_seed
                    direct_env.init_state_offset = state_index
                    direct_env.episode_index = 0
                    observation = direct_env.reset()
                    while elapsed < limit and not (success or direct_env.episode_done):
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
                            if direct_env.episode_done or success or elapsed >= limit:
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
                                   initial_state_index=state_index,
                                   teacher_rng_after=np.asarray(jax.device_get(teacher.rng), np.uint32),
                                   complete=True)
                for key in CAMERA_DATASETS:
                    group.create_dataset(key, data=np.stack([row[0]["image"][key] for row in rows]),
                                         compression="gzip")
                group.create_dataset("state", data=np.stack([row[0]["state"] for row in rows]))
                group.create_dataset("actions", data=np.stack([row[1] for row in rows]))
                group.create_dataset("executed_prefix", data=[row[2] for row in rows])
                file.flush()
                if progress_callback is not None:
                    complete_groups = [n for n in file if bool(file[n].attrs.get("complete", False))]
                    success_count = sum(bool(file[n].attrs.get("success", False)) for n in complete_groups)
                    progress_callback({
                        "episode": episode + 1,
                        "successes": success_count,
                        "failures": len(complete_groups) - success_count,
                    })
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


def _resolve_small_expo_checkpoint(path: Path, step: int | None = None):
    """Resolve a Small EXPO task/checkpoint path to (task, root, step).

    The project has both ``checkpoints_best`` and ``checkpoints_last`` trees.
    A numeric path is preferred for reproducibility; a tree path is accepted
    for convenience and resolves to its latest numeric step.
    """
    path = Path(path).expanduser().resolve()
    if step is not None:
        for slot in ("checkpoints", "checkpoints_best", "checkpoints_last"):
            root = path / slot
            if (root / str(step)).is_dir():
                return path, root, step
        raise FileNotFoundError(f"Small EXPO checkpoint step {step} is not present below {path}")
    if path.name.isdigit() and path.parent.name.startswith("checkpoints"):
        root = path.parent
        step = int(path.name)
    else:
        if path.name.startswith("checkpoints"):
            root = path
        elif (path / "checkpoints_best").is_dir():
            root = path / "checkpoints_best"
        elif (path / "checkpoints_last").is_dir():
            root = path / "checkpoints_last"
        elif (path / "checkpoints").is_dir():
            root = path / "checkpoints"
        else:
            root = path
        numeric_steps = [int(child.name) for child in root.iterdir()
                         if child.is_dir() and child.name.isdigit()]
        if not numeric_steps:
            raise FileNotFoundError(f"No numeric checkpoints found below {root}")
        step = max(numeric_steps)
    if not root.exists():
        raise FileNotFoundError(f"Small EXPO checkpoint directory does not exist: {root}")
    if not (root / str(step)).is_dir():
        raise FileNotFoundError(f"Small EXPO checkpoint step {step} is not present in {root}")
    return root.parent, root, step


def collect_hybrid_teacher_chunks(
    small_expo_checkpoint: Path,
    pi05_checkpoint: str,
    output: Path,
    episodes: int,
    seed: int,
    *,
    suite: str = "libero_90",
    task_id: int,
    image_size: int = 224,
    target_action_horizon: int = 10,
    target_replan_steps: int = 5,
    target_successes: int | None = None,
    target_failures: int | None = None,
    max_attempts_override: int | None = None,
    progress_callback=None,
) -> None:
    """Collect Pi0.5 targets selected by a frozen Small EXPO editor/critic.

    The large official Pi0.5 actor proposes ten-action chunks.  The compact
    Small EXPO checkpoint supplies only the editor, critic, and their sampling
    settings.  The resulting hybrid is used for rollout and target collection;
    the SFT trainer later trains/evaluates the large Pi0.5 actor alone.
    """
    import dataclasses
    import types

    import jax
    import orbax.checkpoint as ocp
    from openpi.shared import normalize
    from openpi.training import sharding

    from dcrl.agents.pi_expo_learner import restore_checkpoint
    from dcrl.eval_pi_expo_with_editor import (
        _build_compact_agent,
        _build_large_agent,
        _checkpoint_manager,
        _editor_rollout_config,
        _hybrid_agent,
        _load_stats,
    )
    from dcrl.benchmarks import asset_id, initial_state_index, is_ocbench, make_env, max_episode_steps

    if (target_successes is None) != (target_failures is None):
        raise ValueError("provide both target_successes and target_failures, or neither")
    if target_successes is not None and (target_successes <= 0 or target_failures < 0):
        raise ValueError("target_successes must be positive and target_failures non-negative")
    if target_successes == 0 and target_failures == 0:
        raise ValueError("at least one collection target must be positive")
    if not 1 <= target_replan_steps <= target_action_horizon:
        raise ValueError("require 1 <= target_replan_steps <= target_action_horizon")

    max_attempts = int(episodes)
    if target_successes is not None:
        max_attempts = max(max_attempts, 4 * (int(target_successes) + int(target_failures)))
    if max_attempts_override is not None:
        if int(max_attempts_override) < max_attempts:
            raise ValueError("max_attempts_override cannot be smaller than the quota attempt budget")
        max_attempts = int(max_attempts_override)

    task_dir, checkpoint_root, checkpoint_step = _resolve_small_expo_checkpoint(
        small_expo_checkpoint
    )
    run_config_path = task_dir / "run_config.json"
    norm_stats_path = task_dir / "norm_stats.json"
    if not run_config_path.exists() or not norm_stats_path.exists():
        raise FileNotFoundError(
            f"Small EXPO task directory must contain run_config.json and norm_stats.json: {task_dir}"
        )
    small_config = json.loads(run_config_path.read_text())
    if small_config.get("suite", suite) != suite or int(small_config.get("task_id", task_id)) != int(task_id):
        raise ValueError(
            "Small EXPO checkpoint task does not match the requested collection task: "
            f"checkpoint=({small_config.get('suite')},{small_config.get('task_id')}), "
            f"requested=({suite},{task_id})"
        )
    small_horizon = int(small_config["action_horizon"])
    small_replan = int(small_config["replan_steps"])
    if target_action_horizon > small_horizon:
        raise ValueError(
            f"target horizon {target_action_horizon} exceeds compact checkpoint horizon {small_horizon}"
        )

    output = Path(output)
    metadata = {
        "source": "pi05_small_expo_editor",
        "small_expo_checkpoint": str((checkpoint_root / str(checkpoint_step)).resolve()),
        "small_expo_task_dir": str(task_dir.resolve()),
        "small_expo_checkpoint_step": int(checkpoint_step),
        "pi05_checkpoint": str(pi05_checkpoint),
        "suite": suite,
        "task_id": int(task_id),
        "seed": int(seed),
        "episodes": int(episodes),
        "target_successes": None if target_successes is None else int(target_successes),
        "target_failures": None if target_failures is None else int(target_failures),
        "max_attempts": int(max_attempts),
        "image_size": int(image_size),
        "render_resolution": 224 if is_ocbench(suite) else 256,
        "action_horizon": int(target_action_horizon),
        "replan_steps": int(target_replan_steps),
        "small_expo_action_horizon": small_horizon,
        "small_expo_replan_steps": small_replan,
        "official_pi05_libero": not is_ocbench(suite),
        "norm_sha256": hashlib.sha256(norm_stats_path.read_bytes()).hexdigest(),
        "format_version": 1,
    }
    success_path = output.parent / "success" / output.name
    failure_path = output.parent / "failures" / output.name
    if output.exists():
        with h5py.File(output, "r") as existing:
            if json.loads(existing.attrs["metadata"]) != metadata or not bool(existing.attrs.get("complete", False)):
                raise ValueError("existing hybrid collection does not match this request")
        partition_collection(output, success_path, failure_path)
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(".partial.hdf5")
    if partial.exists():
        try:
            with h5py.File(partial, "r"):
                pass
        except OSError:
            partial.unlink()

    env = None
    try:
        env = make_env(suite, task_id, seed=seed, image_size=image_size)
        observation = env.reset()
        mesh = sharding.make_mesh(1)
        data_sharding = jax.sharding.NamedSharding(
            mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS)
        )
        replicated_sharding = jax.sharding.NamedSharding(
            mesh, jax.sharding.PartitionSpec()
        )

        # Use the official LIBERO EXPO wiring, but point its Pi0.5 loader at
        # the requested official base checkpoint. No large EXPO checkpoint is
        # restored here: only the compact editor/critic is restored.
        pi05_params = str(pi05_checkpoint).rstrip("/")
        if not pi05_params.endswith("/params"):
            pi05_params += "/params"
        large_args = types.SimpleNamespace(
            seed=int(seed),
            replan_steps=int(target_replan_steps),
            pi05_weight_loader_path=pi05_params,
            pi05_config_module="configs.model.expo_ft_pi_libero_config",
        )
        if is_ocbench(suite):
            large_args.pi05_asset_id = asset_id(suite)
            large_args.pi05_assets_dir = output.parent / 'pi05_assets'
            actor_stats = large_args.pi05_assets_dir / large_args.pi05_asset_id / 'norm_stats.json'
            actor_stats.parent.mkdir(parents=True, exist_ok=True)
            actor_stats.write_bytes(norm_stats_path.read_bytes())
        large = _build_large_agent(
            large_args, observation, env.instruction, mesh,
            data_sharding, replicated_sharding,
        )

        stats, _ = _load_stats(checkpoint_root, None)
        compact_args = types.SimpleNamespace(
            seed=int(small_config.get("seed", seed)),
            image_size=int(small_config.get("image_size", image_size)),
        )
        compact = _build_compact_agent(
            compact_args, observation, stats, mesh,
            data_sharding, replicated_sharding,
            action_horizon=small_horizon, replan_steps=small_replan,
        )
        manager, _, root = _checkpoint_manager(checkpoint_root)
        try:
            compact = restore_checkpoint(manager, compact, checkpoint_step)
        finally:
            manager.close()
        print(
            f"Restored Small EXPO editor/critic checkpoint {checkpoint_step} from {root}",
            flush=True,
        )
        hybrid = dataclasses.replace(
            _hybrid_agent(large, compact),
            eval_candidate_steps=target_replan_steps,
        ).cache_infer_params()
        print(
            f"Initialized official Pi0.5 base + Small EXPO editor/critic; "
            f"targets={target_action_horizon}/{target_replan_steps}",
            flush=True,
        )

        instruction = env.instruction
        limit = max_episode_steps(suite, task_id)
        norm_bytes = norm_stats_path.read_bytes()
        with h5py.File(partial, "a") as file:
            if "metadata" in file.attrs:
                old = json.loads(file.attrs["metadata"])
                if old != metadata:
                    raise ValueError("partial hybrid collection does not match this request")
            file.attrs["metadata"] = json.dumps(metadata, sort_keys=True)
            file.attrs["instruction"] = instruction
            file.attrs["normalization_stats"] = norm_bytes.decode()
            for episode in tqdm(range(max_attempts), desc="Collect hybrid 10/5 episodes"):
                complete = [name for name in file if bool(file[name].attrs.get("complete", False))]
                success_count = sum(bool(file[name].attrs.get("success", False)) for name in complete)
                failure_count = len(complete) - success_count
                if target_successes is not None and success_count >= target_successes and failure_count >= target_failures:
                    break
                name = f"episode_{episode:06d}"
                if name in file and bool(file[name].attrs.get("complete", False)):
                    continue
                if name in file:
                    del file[name]
                env.seed = seed if is_ocbench(suite) else seed + episode
                env.init_state_offset = initial_state_index(env, episode)
                env.episode_index = 0
                observation = env.reset()
                rows, success, elapsed = [], False, 0
                while elapsed < limit and not (success or env.episode_done):
                    start = observation
                    chunk, hybrid, sample_info = hybrid.sample_actions(observation)
                    chunk = np.asarray(jax.device_get(chunk), np.float32)
                    if chunk.shape != (target_action_horizon, 7) or not np.isfinite(chunk).all():
                        raise ValueError(f"hybrid teacher returned invalid action chunk shape={chunk.shape}")
                    executed = 0
                    for action in chunk[:min(target_replan_steps, limit - elapsed)]:
                        observation, _, success, _ = env.step(action)
                        elapsed += 1
                        executed += 1
                        if env.episode_done or success or elapsed >= limit:
                            break
                    rows.append((start, chunk, executed, sample_info))
                if target_successes is not None:
                    complete = [name for name in file if bool(file[name].attrs.get("complete", False))]
                    success_count = sum(bool(file[name].attrs.get("success", False)) for name in complete)
                    failure_count = len(complete) - success_count
                    if ((success and success_count >= target_successes)
                            or (not success and failure_count >= target_failures)):
                        continue
                group = file.create_group(name)
                group.attrs.update(
                    success=bool(success), steps=int(elapsed),
                    initial_state_index=initial_state_index(env, episode),
                    selected_action_types=json.dumps([
                        str(row[3].get("selected_action_type", "unknown")) for row in rows
                    ]),
                    selected_candidate_indices=json.dumps([
                        int(row[3].get("selected_candidate_index", -1)) for row in rows
                    ]),
                    complete=True,
                )
                for key in CAMERA_DATASETS:
                    group.create_dataset(
                        key, data=np.stack([row[0]["image"][key] for row in rows]),
                        compression="gzip",
                    )
                group.create_dataset("state", data=np.stack([row[0]["state"] for row in rows]))
                group.create_dataset("actions", data=np.stack([row[1] for row in rows]))
                group.create_dataset("executed_prefix", data=[row[2] for row in rows])
                file.flush()
                if progress_callback is not None:
                    complete = [name for name in file if bool(file[name].attrs.get("complete", False))]
                    success_count = sum(bool(file[name].attrs.get("success", False)) for name in complete)
                    progress_callback({
                        "episode": episode + 1,
                        "successes": success_count,
                        "failures": len(complete) - success_count,
                    })
            complete = [name for name in file if bool(file[name].attrs.get("complete", False))]
            success_count = sum(bool(file[name].attrs.get("success", False)) for name in complete)
            failure_count = len(complete) - success_count
            if target_successes is not None and (success_count < target_successes or failure_count < target_failures):
                raise RuntimeError(
                    f"reached collection attempt limit with {success_count} successes and "
                    f"{failure_count} failures; requested {target_successes}/{target_failures}"
                )
            file.attrs["complete"] = True
        partial.replace(output)
        partition_collection(output, success_path, failure_path)
    finally:
        if env is not None:
            env.close()


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

#!/usr/bin/env python3
"""Evaluate an SFT actor, compact local EXPO agent, or full Pi EXPO agent over LIBERO-90.

The evaluator deliberately uses the same rollout boundary as the training
evaluators: the policy predicts a ten-action chunk and LIBERO executes the
first five actions before replanning. A process can evaluate the complete
suite, an explicit task list, or one modulo shard. The latter is intended for
Slurm arrays, for example ``--shard-index 2 --num-shards 7``.

``--mode actor`` expects an SFT checkpoint. It loads the actor (including its
LoRA leaves) through the Pi0.5 weight loader and evaluates it without creating
or using an EXPO critic/editor.

``--mode full`` expects a checkpoint written by the full Pi EXPO trainer. It
constructs the EXPO learner, restores the actor, editor, critic, temperature,
target networks, and optimizer state, and evaluates the normal candidate
sampling/critic-selection path.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
from collections import deque
from pathlib import Path
from typing import Any, Iterable

import numpy as np

LOGGER = logging.getLogger("full_90_libero_evals")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _flat_observation(observation: dict, instruction: str) -> dict:
    """Convert the local LIBERO adapter output to the Pi0.5 raw input schema."""
    return {
        "image": np.asarray(observation["image"]["base_0_rgb"], dtype=np.uint8),
        "wrist_image": np.asarray(
            observation["image"]["left_wrist_0_rgb"], dtype=np.uint8
        ),
        "state": np.asarray(observation["state"], dtype=np.float32),
        "prompt": instruction,
    }


def _make_mesh(fsdp_devices: int):
    import jax
    import openpi.training.sharding as openpi_sharding

    mesh = openpi_sharding.make_mesh(fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec(openpi_sharding.DATA_AXIS)
    )
    replicated_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec()
    )
    return mesh, data_sharding, replicated_sharding


def _numeric_steps(root: Path) -> list[int]:
    return sorted(
        int(child.name)
        for child in root.iterdir()
        if child.is_dir() and child.name.isdigit()
    )


def _resolve_sft_state(path: Path) -> Path:
    """Resolve an SFT output/checkpoint path to an Orbax ``state`` item."""
    path = path.expanduser().resolve()
    if path.name in {"state", "params"}:
        return path

    if path.is_dir() and path.name.isdigit():
        for item in ("state", "params"):
            if (path / item).exists():
                return path / item

    checkpoints = path / "checkpoints" if (path / "checkpoints").is_dir() else path
    steps = _numeric_steps(checkpoints) if checkpoints.is_dir() else []
    if not steps:
        raise FileNotFoundError(
            f"Could not find numeric SFT checkpoints below {path}"
        )
    step_dir = checkpoints / str(max(steps))
    for item in ("state", "params"):
        if (step_dir / item).exists():
            return step_dir / item
    raise FileNotFoundError(f"SFT checkpoint {step_dir} has no state or params item")


def _find_sft_assets(path: Path) -> Path | None:
    """Find the copied OpenPI assets/norm-stats directory beside an SFT run."""
    path = path.expanduser().resolve()
    for candidate in (path, *path.parents):
        assets = candidate / "student_assets"
        if assets.is_dir():
            return assets
    return None


def _resolve_full_root(path: Path) -> tuple[Path, int | None]:
    """Resolve a full EXPO task/checkpoint path to its manager root and step."""
    path = path.expanduser().resolve()
    step_hint = int(path.name) if path.name.isdigit() else None
    if path.name.isdigit() and path.parent.name == "checkpoints":
        return path.parent, step_hint
    if path.name == "checkpoints":
        return path, None
    if (path / "checkpoints").is_dir():
        return path / "checkpoints", None
    # Current train_pi_robo_libero launchers put run_name between the task
    # directory and checkpoints (taskXX/<run_name>/checkpoints). Accept that
    # task-level path as well, but only select a child that actually has steps.
    if path.is_dir():
        candidates = sorted(
            child / "checkpoints"
            for child in path.iterdir()
            if child.is_dir() and (child / "checkpoints").is_dir()
        )
        candidates = [candidate for candidate in candidates if _numeric_steps(candidate)]
        if len(candidates) == 1:
            return candidates[0], None
        if len(candidates) > 1:
            raise ValueError(
                f"Multiple full-agent checkpoint runs below {path}; pass the run directory explicitly: {candidates}"
            )
    return path, None


def _resolve_compact_root(path: Path) -> tuple[Path, int | None]:
    """Resolve a local SmallExpo checkpoint path to its manager root and step."""
    path = path.expanduser().resolve()
    step_hint = int(path.name) if path.name.isdigit() else None
    if path.name.isdigit() and path.parent.name == "checkpoints":
        return path.parent, step_hint
    if path.name == "checkpoints":
        return path, None
    if (path / "checkpoints").is_dir():
        return path / "checkpoints", None
    return path, step_hint


def _find_compact_stats(path: Path) -> Path:
    """Find norm_stats.json beside a compact task/checkpoint directory."""
    path = path.expanduser().resolve()
    candidates = [path]
    if path.name.isdigit():
        candidates.extend([path.parent, path.parent.parent])
    elif path.name == "checkpoints":
        candidates.append(path.parent)
    else:
        candidates.extend([path.parent, path.parent.parent])
    for candidate in candidates:
        if (candidate / "norm_stats.json").exists():
            return candidate
    searched = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Could not find compact norm_stats.json; searched: {searched}")


def _build_config(mode: str, checkpoint: Path, assets_dir: Path | None):
    from configs.model import expo_ft_pi_libero_config

    config = copy.deepcopy(expo_ft_pi_libero_config.get_config())
    if assets_dir is not None:
        config.pi05_assets_dir = str(assets_dir)
        config.pi05_asset_id = "physical-intelligence/libero"
    if mode == "actor":
        config.pi05_weight_loader_path = str(_resolve_sft_state(checkpoint))
    return config


def _build_agent(
    *,
    mode: str,
    checkpoint: Path,
    assets_dir: Path | None,
    task_description: str,
    example_observation: dict,
    mesh: Any,
    data_sharding: Any,
    replicated_sharding: Any,
    seed: int,
    replan_steps: int,
):
    """Build actor-only, compact local EXPO, or full Pi EXPO agent."""
    if mode == "compact":
        from openpi.shared import normalize
        from dcrl.agents.pi_expo_learner import restore_checkpoint
        from dcrl.agents.small_expo_learner import SmallExpoLearner
        from dcrl.expo.agents.alg.agent import initialize_checkpoint_dir

        stats = normalize.load(_find_compact_stats(checkpoint))
        agent = SmallExpoLearner.from_examples(
            seed=seed,
            example_observation=example_observation,
            example_action=np.zeros((7,), dtype=np.float32),
            example_state=np.asarray(example_observation["state"], dtype=np.float32),
            mesh=mesh,
            data_sharding=data_sharding,
            replicated_sharding=replicated_sharding,
            image_size=int(example_observation["image"]["base_0_rgb"].shape[0]),
            action_horizon=10,
            replan_steps=replan_steps,
            actor_success_only=True,
            norm_stats=stats,
        )
        root, step_hint = _resolve_compact_root(checkpoint)
        manager, _ = initialize_checkpoint_dir(
            root, keep_period=None, overwrite=False, resume=True
        )
        steps = tuple(int(step) for step in manager.all_steps())
        if not steps:
            manager.close()
            raise FileNotFoundError(f"No compact checkpoints found below {root}")
        step = step_hint if step_hint is not None else max(steps)
        if step not in steps:
            manager.close()
            raise ValueError(f"Checkpoint step {step} is not present below {root}; steps={steps}")
        agent = restore_checkpoint(manager, agent, step)
        if hasattr(agent, "cache_infer_params"):
            agent = agent.cache_infer_params()
        LOGGER.info("Restored compact EXPO checkpoint step %d from %s", step, root)
        return agent, manager

    from expo_ft.agents.vla.pi05 import build_pi05

    config = _build_config(mode, checkpoint, assets_dir)
    actor, actor_train_state, target_actor_params, agent_kwargs, metadata = build_pi05(
        config,
        seed,
        mesh,
        data_sharding,
        replicated_sharding,
        False,
        task_description,
    )

    action_dim = 7
    raw_state = np.asarray(example_observation["state"], dtype=np.float32).reshape(-1)
    # The Pi replay buffer initializes EXPO with the model's padded state width
    # (32 for Pi0.5), while the live LIBERO observation contains only 8 values.
    # Match that convention for full-agent checkpoint restoration. BCLearner
    # actor-only checkpoints retain the raw state width used by their loader.
    if mode == "full":
        padded_state_dim = int(getattr(actor.model_config, "action_dim", raw_state.shape[-1]))
        state = np.pad(raw_state, (0, max(0, padded_state_dim - raw_state.shape[-1])))[:padded_state_dim]
    else:
        state = raw_state
    actor.action_dim = action_dim
    actor.state_dim = int(state.shape[-1])
    example_action = np.zeros((action_dim,), dtype=np.float32)
    critic_observation = np.concatenate(
        [
            np.asarray(example_observation["image"], dtype=np.uint8),
            np.asarray(example_observation["wrist_image"], dtype=np.uint8),
        ],
        axis=-1,
    )

    if mode == "actor":
        from expo_ft.agents.alg.bc import load_agent

        agent = load_agent(
            seed=seed,
            example_observation=critic_observation,
            example_action=example_action,
            example_state=state,
            actor=actor,
            actor_train_state=actor_train_state,
            target_actor_params=target_actor_params,
            agent_kwargs=agent_kwargs,
            metadata=metadata,
            mesh=mesh,
            data_sharding=data_sharding,
            replicated_sharding=replicated_sharding,
            resume=False,
            replan_steps=replan_steps,
            default_prompt=task_description,
        )
        return agent, None

    from expo_ft.agents.alg.agent import initialize_checkpoint_dir
    from expo_ft.agents.alg.expo_ft import load_agent, restore_checkpoint

    agent_kwargs["critic_camera_keys"] = ("base_0_rgb", "left_wrist_0_rgb")
    agent = load_agent(
        seed=seed,
        example_observation=critic_observation,
        example_action=example_action,
        example_state=state,
        actor=actor,
        actor_train_state=actor_train_state,
        target_actor_params=target_actor_params,
        agent_kwargs=agent_kwargs,
        metadata=metadata,
        mesh=mesh,
        data_sharding=data_sharding,
        replicated_sharding=replicated_sharding,
        resume=True,
        replan_steps=replan_steps,
        default_prompt=task_description,
        edit_action_xyzg=False,
    )
    root, step_hint = _resolve_full_root(checkpoint)
    manager, _ = initialize_checkpoint_dir(
        root, keep_period=None, overwrite=False, resume=True
    )
    steps = tuple(int(step) for step in manager.all_steps())
    if not steps:
        manager.close()
        raise FileNotFoundError(f"No full-agent checkpoints found below {root}")
    step = step_hint if step_hint is not None else max(steps)
    if step not in steps:
        manager.close()
        raise ValueError(f"Checkpoint step {step} is not present below {root}; steps={steps}")
    agent = restore_checkpoint(manager, agent, step)
    if hasattr(agent, "cache_infer_params"):
        agent = agent.cache_infer_params()
    LOGGER.info("Restored full EXPO checkpoint step %d from %s", step, root)
    return agent, manager


def _evaluate_task(
    agent,
    *,
    mode: str,
    suite: str,
    task_id: int,
    episodes: int,
    max_steps: int,
    replan_steps: int,
    image_size: int,
    eval_seed: int,
):
    import jax
    from dcrl.libero_env import LiberoTaskEnv

    env = LiberoTaskEnv(
        suite,
        task_id,
        seed=eval_seed,
        image_size=image_size,
        benchmark_eval=True,
    )
    if episodes > len(env.init_states):
        env.close()
        raise ValueError(
            f"task {task_id}: episodes={episodes} exceeds the "
            f"{len(env.init_states)} official initial states"
        )

    env.episode_index = 0
    env.env.seed(eval_seed)
    eval_agent = agent.replace(rng=jax.random.PRNGKey(eval_seed))
    successes = 0
    lengths: list[int] = []
    trials: list[dict[str, Any]] = []
    instruction = env.instruction
    try:
        for episode in range(episodes):
            local_observation = env.reset()
            observation = (local_observation if mode == "compact"
                           else _flat_observation(local_observation, instruction))
            action_plan: deque[np.ndarray] = deque()
            for step in range(max_steps):
                if not action_plan:
                    if mode == "actor":
                        chunk, eval_agent, _ = eval_agent.sample_actions(
                            observation, only_base_actions=True
                        )
                    else:
                        chunk, eval_agent, _ = eval_agent.sample_actions(
                            observation, only_base_actions=False
                        )
                    action_plan.extend(
                        np.asarray(jax.device_get(chunk))[:replan_steps]
                    )
                raw_observation, _, success, _ = env.step(action_plan.popleft())
                observation = (raw_observation if mode == "compact"
                               else _flat_observation(raw_observation, instruction))
                if success:
                    successes += 1
                    lengths.append(step + 1)
                    trials.append(
                        {
                            "episode": episode,
                            "initial_state_index": episode,
                            "success": True,
                            "steps": step + 1,
                        }
                    )
                    break
            else:
                lengths.append(max_steps)
                trials.append(
                    {
                        "episode": episode,
                        "initial_state_index": episode,
                        "success": False,
                        "steps": max_steps,
                    }
                )
    finally:
        env.close()

    return {
        "suite": suite,
        "task_id": int(task_id),
        "instruction": instruction,
        "episodes": int(episodes),
        "max_steps": int(max_steps),
        "eval_seed": int(eval_seed),
        "action_horizon": 10,
        "replan_steps": int(replan_steps),
        "successes": int(successes),
        "eval/success_rate": float(successes / episodes),
        "eval/mean_steps": float(np.mean(lengths)),
        "trials": trials,
    }


def _task_ids(args, count: int) -> list[int]:
    if args.task_id is not None:
        values = [args.task_id]
    elif args.task_ids:
        if args.task_ids.strip().lower() == "all":
            values = list(range(count))
        else:
            values = [int(value) for value in args.task_ids.split(",") if value.strip()]
    elif args.shard_index is not None:
        if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
            raise ValueError("shard-index must be in [0, num-shards)")
        values = list(range(args.shard_index, count, args.num_shards))
    else:
        values = list(range(count))
    if not values:
        raise ValueError("No task IDs selected")
    invalid = [value for value in values if not 0 <= value < count]
    if invalid:
        raise ValueError(f"Task IDs outside [0, {count}): {invalid}")
    return values


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("actor", "compact", "full"), required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="SFT checkpoint for actor mode, compact SmallExpo checkpoint for compact mode, or full EXPO checkpoint for full mode",
    )
    parser.add_argument("--assets-dir", type=Path, default=None,
                        help="Optional OpenPI assets directory; actor mode auto-detects student_assets")
    parser.add_argument("--suite", default="libero_90")
    parser.add_argument("--task-id", type=int, default=None)
    parser.add_argument("--task-ids", default=None,
                        help="Comma-separated IDs or 'all'; ignored when --task-id is set")
    parser.add_argument("--shard-index", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=0,
                        help="0 uses the official suite horizon")
    parser.add_argument("--eval-seed", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--action-horizon", type=int, default=10,
                        help="Must remain 10 for the official Pi0.5 LIBERO setup")
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--fsdp-devices", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    if args.action_horizon != 10:
        raise ValueError("LIBERO evaluation uses the official Pi0.5 action horizon of 10")
    if not 1 <= args.replan_steps <= args.action_horizon:
        raise ValueError("replan-steps must be in [1, 10]")
    if args.episodes < 1:
        raise ValueError("episodes must be positive")

    from libero.libero import benchmark
    from dcrl.libero_env import LIBERO_SUITE_MAX_STEPS

    if args.suite not in LIBERO_SUITE_MAX_STEPS:
        raise ValueError(f"Unknown LIBERO suite {args.suite!r}")
    suite = benchmark.get_benchmark_dict()[args.suite]()
    count = suite.get_num_tasks()
    task_ids = _task_ids(args, count)
    max_steps = args.max_steps or LIBERO_SUITE_MAX_STEPS[args.suite]

    checkpoint = args.checkpoint.expanduser().resolve()
    assets_dir = args.assets_dir.expanduser().resolve() if args.assets_dir else None
    if args.mode == "actor" and assets_dir is None:
        assets_dir = _find_sft_assets(checkpoint)
        if assets_dir:
            LOGGER.info("Using SFT checkpoint assets and norm stats from %s", assets_dir)

    # Build once per shard. Prompts are supplied in each observation, so the
    # same actor/EXPO parameters can be evaluated on all tasks in that shard.
    from dcrl.libero_env import LiberoTaskEnv

    first_env = LiberoTaskEnv(
        args.suite,
        task_ids[0],
        seed=args.eval_seed,
        image_size=args.image_size,
        benchmark_eval=True,
    )
    first_local_observation = first_env.reset()
    first_observation = (first_local_observation if args.mode == "compact"
                         else _flat_observation(first_local_observation, first_env.instruction))
    first_instruction = first_env.instruction
    first_env.close()

    mesh, data_sharding, replicated_sharding = _make_mesh(args.fsdp_devices)
    agent, manager = _build_agent(
        mode=args.mode,
        checkpoint=checkpoint,
        assets_dir=assets_dir,
        task_description=first_instruction,
        example_observation=first_observation,
        mesh=mesh,
        data_sharding=data_sharding,
        replicated_sharding=replicated_sharding,
        seed=args.seed,
        replan_steps=args.replan_steps,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    try:
        for task_id in task_ids:
            task_output = args.output_dir / f"task_{task_id:02d}.json"
            try:
                result = _evaluate_task(
                    agent,
                    mode=args.mode,
                    suite=args.suite,
                    task_id=task_id,
                    episodes=args.episodes,
                    max_steps=max_steps,
                    replan_steps=args.replan_steps,
                    image_size=args.image_size,
                    eval_seed=args.eval_seed,
                )
                result.update({
                    "mode": args.mode,
                    "checkpoint": str(checkpoint),
                    "assets_dir": None if assets_dir is None else str(assets_dir),
                })
                _atomic_json(task_output, result)
                results.append(result)
                LOGGER.info(
                    "task %02d: %.1f%% success, mean steps %.2f",
                    task_id,
                    100.0 * result["eval/success_rate"],
                    result["eval/mean_steps"],
                )
            except Exception as error:
                LOGGER.exception("task %02d failed", task_id)
                failure = {"task_id": task_id, "error": repr(error)}
                failed.append(failure)
                _atomic_json(task_output, failure)

        shard_name = (
            f"shard_{args.shard_index:02d}.json"
            if args.shard_index is not None
            else "summary.json"
        )
        aggregate = {
            "mode": args.mode,
            "suite": args.suite,
            "checkpoint": str(checkpoint),
            "assets_dir": None if assets_dir is None else str(assets_dir),
            "task_ids": task_ids,
            "episodes_per_task": args.episodes,
            "max_steps": max_steps,
            "eval_seed": args.eval_seed,
            "action_horizon": args.action_horizon,
            "replan_steps": args.replan_steps,
            "completed_tasks": len(results),
            "failed_tasks": failed,
            "mean_task_success_rate": (
                float(np.mean([r["eval/success_rate"] for r in results]))
                if results else None
            ),
            "results": [
                {
                    "task_id": r["task_id"],
                    "instruction": r["instruction"],
                    "successes": r["successes"],
                    "episodes": r["episodes"],
                    "eval/success_rate": r["eval/success_rate"],
                    "eval/mean_steps": r["eval/mean_steps"],
                }
                for r in results
            ],
        }
        _atomic_json(args.output_dir / shard_name, aggregate)
        if failed:
            raise RuntimeError(f"{len(failed)} task evaluations failed; see {args.output_dir}")
    finally:
        if manager is not None:
            manager.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

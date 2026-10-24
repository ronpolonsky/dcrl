"""One-shot LIBERO evaluation of PiExpo with a supplied compact editor/critic.

The large Pi0.5 actor remains the action-chunk generator.  The supplied compact
checkpoint contributes the shared critic encoder, edit policy, critic ensemble,
and their candidate-sampling settings.  This is useful for measuring the
hybrid before distilling the compact policy back into the large actor.

Examples::

    python -m dcrl.eval_pi_expo_with_editor \
      --editor-checkpoint-dir /path/to/small/task_60/checkpoints \
      --pi-checkpoint-dir /path/to/large/task_60/checkpoints \
      --task-id 60 --output /path/to/hybrid_eval.json

If ``--pi-checkpoint-dir`` is omitted, the large actor is initialized from the
Pi05 weight loader configured in ``configs/model/expo_ft_pi_config.py``.  The
compact checkpoint directory may be either its ``checkpoints`` directory or
the task directory containing ``checkpoints`` and ``norm_stats.json``.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from dcrl.libero_env import DEFAULT_TASK_IDS, LIBERO_SUITE_MAX_STEPS, LiberoTaskEnv

LOGGER = logging.getLogger(__name__)


def _checkpoint_root(path: Path) -> Path:
    """Accept a task directory, checkpoints directory, or numeric step path."""
    path = path.expanduser().resolve()
    if path.name.isdigit() and path.parent.name == "checkpoints":
        return path.parent
    if (path / "checkpoints").is_dir():
        return path / "checkpoints"
    return path


def _checkpoint_manager(path: Path):
    from dcrl.expo.agents.alg.agent import initialize_checkpoint_dir

    root = _checkpoint_root(path)
    if not root.exists():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {root}")
    manager, _ = initialize_checkpoint_dir(
        root, keep_period=None, overwrite=False, resume=True
    )
    steps = tuple(manager.all_steps())
    if not steps:
        manager.close()
        raise FileNotFoundError(f"No checkpoints found below {root}")
    return manager, max(int(step) for step in steps), root


def _load_stats(editor_checkpoint_dir: Path, explicit: Path | None):
    from openpi.shared import normalize

    candidates = []
    if explicit is not None:
        candidates.append(explicit.expanduser().resolve())
    root = editor_checkpoint_dir.expanduser().resolve()
    if root.name == "checkpoints":
        candidates.append(root.parent)
    if (root / "checkpoints").is_dir():
        candidates.append(root)
    candidates.append(root.parent)
    for candidate in candidates:
        if (candidate / "norm_stats.json").exists():
            LOGGER.info("Using compact actor normalization statistics from %s", candidate)
            return normalize.load(candidate), candidate
    searched = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        "Could not find compact norm_stats.json. Pass --editor-norm-stats-dir; "
        f"searched: {searched}"
    )


def _make_mesh():
    import jax
    from openpi.training import sharding as openpi_sharding

    mesh = openpi_sharding.make_mesh(1)
    data_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec(openpi_sharding.DATA_AXIS)
    )
    replicated_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec()
    )
    return mesh, data_sharding, replicated_sharding


def _large_config(weight_loader_path: str | None):
    from configs.model import expo_ft_pi_config

    config = copy.deepcopy(expo_ft_pi_config.get_config())
    if weight_loader_path is not None:
        config.pi05_weight_loader_path = weight_loader_path
    return config


def _build_large_agent(args, obs, task_description, mesh, data_sharding, replicated_sharding):
    import numpy as np
    from dcrl.agents.pi_expo_learner import PiExpoLearner

    camera_keys = ("base_0_rgb", "left_wrist_0_rgb")
    critic_image = np.concatenate(
        [np.asarray(obs["image"][key]) for key in camera_keys], axis=-1
    )
    config = _large_config(args.pi05_weight_loader_path)
    agent = PiExpoLearner.from_pi05(
        config=config,
        seed=args.seed,
        mesh=mesh,
        data_sharding=data_sharding,
        replicated_sharding=replicated_sharding,
        task_description=task_description,
        replan_steps=args.replan_steps,
        example_observation=critic_image,
        example_action=np.zeros(7, dtype=np.float32),
        example_state=np.asarray(obs["state"], dtype=np.float32),
        critic_camera_keys=camera_keys,
        resume=False,
    )
    return agent


def _editor_rollout_config(checkpoint_dir: Path, args):
    """Read the compact checkpoint shape; execution remains the evaluator's 10/5."""
    root = checkpoint_dir.expanduser().resolve()
    candidates = []
    if root.name == "checkpoints":
        candidates.append(root.parent / "run_config.json")
    if (root / "checkpoints").is_dir():
        candidates.append(root / "run_config.json")
    candidates.append(root.parent / "run_config.json")
    for config_path in candidates:
        if not config_path.exists():
            continue
        try:
            config = json.loads(config_path.read_text())
        except json.JSONDecodeError:
            continue
        horizon = int(config.get("action_horizon", args.action_horizon))
        replan = int(config.get("replan_steps", args.replan_steps))
        return horizon, replan
    return args.action_horizon, args.replan_steps


def _build_compact_agent(args, obs, stats, mesh, data_sharding, replicated_sharding,
                         *, action_horizon: int, replan_steps: int):
    from dcrl.agents.small_expo_learner import SmallExpoLearner

    return SmallExpoLearner.from_examples(
        seed=args.seed + 100003,
        example_observation=obs,
        example_action=np.zeros(7, dtype=np.float32),
        example_state=np.asarray(obs["state"], dtype=np.float32),
        mesh=mesh,
        data_sharding=data_sharding,
        replicated_sharding=replicated_sharding,
        image_size=args.image_size,
        action_horizon=action_horizon,
        replan_steps=replan_steps,
        actor_success_only=True,
        norm_stats=stats,
    )


def _restore(manager, agent, step):
    from dcrl.agents.pi_expo_learner import restore_checkpoint

    return restore_checkpoint(manager, agent, step)


def _hybrid_agent(large, compact):
    """Keep large Pi05 actor; replace all editor/critic-side compact fields."""
    compact_fields = (
        "batch_encoder", "edit_actor", "critic", "target_critic", "temp",
        "N", "n_edit_samples", "edit_scale", "edit_action_xyzg", "batch_split",
        "encode_batch_split", "actor_tau", "tau", "discount", "target_entropy",
        "entropy_scale", "num_qs", "num_min_qs", "full_action_dim",
        "replan_steps", "resize_size", "default_prompt",
        "batch_encoder_sharding", "freeze_critic_encoder", "actor_success_only",
        "critic_camera_keys",
    )
    values = {name: getattr(compact, name) for name in compact_fields}
    # The actor and its optimizer state deliberately remain from the large
    # learner.  _infer_cache is invalid after swapping the compact modules.
    values["_infer_cache"] = None
    hybrid = dataclasses.replace(large, **values)
    if hybrid.action_dim != compact.action_dim or hybrid.state_dim != compact.state_dim:
        raise ValueError(
            "Large and compact checkpoints have different action/state dimensions: "
            f"large=({hybrid.action_dim},{hybrid.state_dim}), "
            f"compact=({compact.action_dim},{compact.state_dim})"
        )
    large_horizon = int(getattr(large.actor.model_config, "action_horizon", -1))
    if compact.replan_steps > large_horizon:
        raise ValueError(
            "The compact editor scores a prefix of "
            f"{compact.replan_steps} actions, but the large Pi05 actor emits "
            f"only {large_horizon}."
        )
    # The large actor owns the returned chunk length. A legacy compact 16/8
    # checkpoint can therefore score its trained 8-action prefix while the
    # evaluator executes only the official 5 actions from each 10-action chunk.
    values["action_horizon"] = large_horizon
    if compact.replan_steps != large.replan_steps:
        LOGGER.warning(
            "Hybrid uses compact editor prefix %d for scoring and evaluator "
            "execution prefix %d", compact.replan_steps, large.replan_steps
        )
    return dataclasses.replace(hybrid, action_horizon=large_horizon)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--editor-checkpoint-dir", type=Path, required=True,
                        help="Compact SmallExpo task directory or checkpoints directory")
    parser.add_argument("--editor-step", type=int, default=None,
                        help="Compact checkpoint step; default is the latest")
    parser.add_argument("--editor-norm-stats-dir", type=Path, default=None,
                        help="Directory containing the compact norm_stats.json")
    parser.add_argument("--pi-checkpoint-dir", type=Path, default=None,
                        help="Large PiExpo task/checkpoints directory; default initializes Pi05")
    parser.add_argument("--pi-step", type=int, default=None,
                        help="Large checkpoint step; default is the latest")
    parser.add_argument("--pi05-weight-loader-path", default=None,
                        help="Optional Pi05 base checkpoint when no large checkpoint is restored")
    parser.add_argument("--suite", default="libero_90")
    parser.add_argument("--task-id", type=int, choices=range(90), default=DEFAULT_TASK_IDS[0])
    parser.add_argument("--episodes", type=int, default=20,
                        help="Fixed official LIBERO initial states to evaluate")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Per-episode horizon; default is the official suite horizon")
    parser.add_argument("--eval-seed", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--action-horizon", type=int, default=10)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--output", type=Path, default=None,
                        help="Optional JSON output path")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if args.replan_steps < 1 or args.replan_steps > args.action_horizon:
        raise ValueError("--replan-steps must be between 1 and --action-horizon")
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    import jax
    from dcrl.train_libero import _evaluate

    max_steps = args.max_steps or LIBERO_SUITE_MAX_STEPS[args.suite]
    env = LiberoTaskEnv(
        args.suite, args.task_id, seed=args.seed + args.task_id,
        image_size=args.image_size,
    )
    eval_env = LiberoTaskEnv(
        args.suite, args.task_id, seed=args.eval_seed,
        image_size=args.image_size, benchmark_eval=True,
    )
    large_manager = compact_manager = None
    try:
        if args.episodes > len(eval_env.init_states):
            raise ValueError(
                f"--episodes={args.episodes} exceeds the {len(eval_env.init_states)} "
                "official initial states for this task"
            )
        obs = env.reset()
        mesh, data_sharding, replicated_sharding = _make_mesh()
        large = _build_large_agent(
            args, obs, env.instruction, mesh, data_sharding, replicated_sharding
        )
        large_step = None
        if args.pi_checkpoint_dir is not None:
            large_manager, latest, root = _checkpoint_manager(args.pi_checkpoint_dir)
            large_step = latest if args.pi_step is None else args.pi_step
            if large_step not in tuple(large_manager.all_steps()):
                raise ValueError(f"Large checkpoint step {large_step} is not present in {root}")
            large = _restore(large_manager, large, large_step)
            LOGGER.info("Restored large PiExpo checkpoint step %d from %s", large_step, root)

        stats, stats_dir = _load_stats(args.editor_checkpoint_dir, args.editor_norm_stats_dir)
        editor_horizon, editor_replan = _editor_rollout_config(args.editor_checkpoint_dir, args)
        compact = _build_compact_agent(
            args, obs, stats, mesh, data_sharding, replicated_sharding,
            action_horizon=editor_horizon, replan_steps=editor_replan,
        )
        compact_manager, latest, root = _checkpoint_manager(args.editor_checkpoint_dir)
        compact_step = latest if args.editor_step is None else args.editor_step
        if compact_step not in tuple(compact_manager.all_steps()):
            raise ValueError(f"Compact checkpoint step {compact_step} is not present in {root}")
        compact = _restore(compact_manager, compact, compact_step)
        LOGGER.info("Restored compact editor/critic checkpoint step %d from %s", compact_step, root)

        hybrid = dataclasses.replace(
            _hybrid_agent(large, compact),
            eval_candidate_steps=args.replan_steps,
        ).cache_infer_params()
        result = _evaluate(
            hybrid, eval_env, episodes=args.episodes, max_steps=max_steps,
            replan_steps=args.replan_steps, seed=args.eval_seed,
        )
        output = {
            "suite": args.suite,
            "task_id": args.task_id,
            "instruction": env.instruction,
            "episodes": args.episodes,
            "max_steps": max_steps,
            "image_size": args.image_size,
            "action_horizon": args.action_horizon,
            "replan_steps": args.replan_steps,
            "editor_action_horizon": int(compact.action_horizon),
            "editor_replan_steps": int(compact.replan_steps),
            "eval_seed": args.eval_seed,
            "pi_checkpoint_dir": None if args.pi_checkpoint_dir is None else str(_checkpoint_root(args.pi_checkpoint_dir)),
            "pi_step": large_step,
            "editor_checkpoint_dir": str(_checkpoint_root(args.editor_checkpoint_dir)),
            "editor_step": compact_step,
            "editor_norm_stats_dir": str(stats_dir),
            **{key: float(value) for key, value in result.items()},
        }
        rendered = json.dumps(output, indent=2) + "\n"
        if args.output is None:
            print(rendered, end="")
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered)
            print(rendered, end="")
        return output
    finally:
        eval_env.close()
        env.close()
        if large_manager is not None:
            large_manager.close()
        if compact_manager is not None:
            compact_manager.close()


if __name__ == "__main__":
    main()

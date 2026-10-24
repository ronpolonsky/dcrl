"""Upgrade pre-dual-optimizer SmallExpoLearner checkpoints.

The first 50k initialization-distillation checkpoints were written before the
learner stored separate optimizer histories for distillation and RL.  This
utility restores that legacy tree, keeps its actor/critic/editor parameters and
distillation moments, initializes a fresh RL optimizer, and writes a checkpoint
that the three-phase trainer can resume directly.

Run this on a GPU allocation.  It intentionally writes to a separate output
root and never modifies the source checkpoint.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np


SOURCE_STEP = 50_000


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--suite", default="libero_90")
    parser.add_argument("--task-ids", type=int, nargs="+", required=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--action-horizon", type=int, default=10)
    parser.add_argument("--replan-steps", type=int, default=5)
    return parser.parse_args()


def _load_legacy_module() -> object:
    """Load the pre-dual-optimizer PiExpoLearner from the checked-in commit."""
    import subprocess

    source = subprocess.check_output(
        ["git", "show", "HEAD:dcrl/agents/pi_expo_learner.py"], text=True
    )
    with tempfile.NamedTemporaryFile("w", suffix="_legacy_pi_expo.py", delete=False) as file:
        file.write(source)
        path = file.name
    spec = importlib.util.spec_from_file_location("dcrl._legacy_pi_expo_learner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the legacy PiExpoLearner source")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _make_agents(task_id: int, suite: str, stats, image_size: int,
                 action_horizon: int, replan_steps: int):
    import jax
    from openpi.training import sharding as openpi_sharding
    from dcrl.agents.small_expo_learner import SmallExpoLearner
    from dcrl.libero_env import LiberoTaskEnv

    env = LiberoTaskEnv(suite, task_id, seed=42 + task_id, image_size=image_size)
    observation = env.reset()
    mesh = openpi_sharding.make_mesh(1)
    data_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec(openpi_sharding.DATA_AXIS)
    )
    replicated_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec()
    )
    common = dict(
        seed=42 + task_id,
        example_observation=observation,
        example_action=np.zeros(7, np.float32),
        example_state=observation["state"],
        mesh=mesh,
        data_sharding=data_sharding,
        replicated_sharding=replicated_sharding,
        image_size=image_size,
        action_horizon=action_horizon,
        replan_steps=replan_steps,
        actor_success_only=True,
        norm_stats=stats,
    )
    current_agent = SmallExpoLearner.from_examples(**common)

    # Bind the current compact constructor to the legacy parent class.  The
    # resulting pytree has exactly the fields present in the old checkpoint.
    legacy_module = _load_legacy_module()
    class LegacySmallExpoLearner(legacy_module.PiExpoLearner):
        pass
    LegacySmallExpoLearner.from_examples = classmethod(
        SmallExpoLearner.from_examples.__func__
    )
    legacy_agent = LegacySmallExpoLearner.from_examples(**common)
    return env, current_agent, legacy_agent


def _restore_legacy(source_checkpoint: Path, legacy_agent):
    import orbax.checkpoint as ocp
    from dcrl.agents.pi_expo_learner import _merge_params

    manager = ocp.CheckpointManager(
        str(source_checkpoint.parent),
        item_handlers={
            "agent": ocp.PyTreeCheckpointHandler(),
            "params": ocp.PyTreeCheckpointHandler(),
        },
        options=ocp.CheckpointManagerOptions(create=False, read_only=True),
    )
    try:
        # Use the legacy module's splitter/restore implementation.  It knows
        # the on-disk tree has no dual optimizer fields.
        legacy_module = sys.modules["dcrl._legacy_pi_expo_learner"]
        return legacy_module.restore_checkpoint(manager, legacy_agent, SOURCE_STEP)
    finally:
        manager.close()


def _upgrade_one(args: argparse.Namespace, task_id: int) -> None:
    import jax
    from openpi.shared import normalize
    from dcrl.agents.pi_expo_learner import restore_checkpoint, save_checkpoint
    from dcrl.expo.agents.alg.agent import initialize_checkpoint_dir

    source_task = args.source_root / args.suite / f"task_{task_id:02d}"
    source_checkpoint = source_task / "checkpoints" / str(SOURCE_STEP)
    if not source_checkpoint.exists():
        raise FileNotFoundError(f"missing source checkpoint: {source_checkpoint}")
    target_task = args.target_root / args.suite / f"task_{task_id:02d}"
    target_checkpoint = target_task / "checkpoints" / str(SOURCE_STEP)
    if target_checkpoint.exists():
        print(f"task {task_id}: target already exists, skipping", flush=True)
        return

    target_task.mkdir(parents=True, exist_ok=True)
    source_stats = source_task / "norm_stats.json"
    if not source_stats.exists():
        raise FileNotFoundError(f"missing source normalization stats: {source_stats}")
    shutil.copy2(source_stats, target_task / "norm_stats.json")
    for name in ("run_config.json", "metrics.jsonl"):
        source_file = source_task / name
        if source_file.exists():
            shutil.copy2(source_file, target_task / name)
    replay = source_task / f"replay_{SOURCE_STEP}.pkl.zst"
    if not replay.exists():
        replay = source_task / f"replay_{SOURCE_STEP}.pkl"
    if not replay.exists():
        raise FileNotFoundError(f"missing replay sidecar: {source_task}")
    shutil.copy2(replay, target_task / replay.name)

    stats = normalize.load(source_task)
    env, current_agent, legacy_agent = _make_agents(
        task_id, args.suite, stats, args.image_size,
        args.action_horizon, args.replan_steps,
    )
    try:
        restored = _restore_legacy(source_checkpoint, legacy_agent)
        # Keep current code's static actor adapter and fresh RL optimizer, but
        # copy all learned dynamic components and the legacy actor state.
        current_actor_state = dataclasses.replace(
            current_agent.actor_train_state,
            params=restored.actor_train_state.params,
            ema_params=restored.actor_train_state.ema_params,
            opt_state=restored.actor_train_state.opt_state,
            step=restored.actor_train_state.step,
        )
        upgraded = current_agent.replace(
            rng=restored.rng,
            critic=restored.critic,
            batch_encoder=restored.batch_encoder,
            target_critic=restored.target_critic,
            actor_train_state=current_actor_state,
            target_actor_params=restored.target_actor_params,
            edit_actor=restored.edit_actor,
            temp=restored.temp,
            distill_actor_opt_state=restored.actor_train_state.opt_state,
            distill_actor_step=int(np.asarray(restored.actor_train_state.step)),
            # The current constructor created this state at step zero.  It is
            # deliberately not copied from the legacy distillation optimizer.
            rl_actor_step=0,
        )
        if int(np.asarray(upgraded.distill_actor_step)) != SOURCE_STEP:
            raise AssertionError("legacy distillation optimizer step was not preserved")
        if int(np.asarray(upgraded.rl_actor_step)) != 0:
            raise AssertionError("RL optimizer was not initialized at step zero")
        source_actor_params = legacy_agent.actor.get_params(restored.actor_train_state)
        upgraded_actor_params = current_agent.actor.get_params(upgraded.actor_train_state)
        leaves = jax.tree_util.tree_leaves(
            jax.tree_util.tree_map(lambda a, b: np.array_equal(np.asarray(a), np.asarray(b)),
                                   source_actor_params, upgraded_actor_params)
        )
        if not all(bool(leaf) for leaf in leaves):
            raise AssertionError("upgraded actor weights differ from the 50k source weights")

        manager, _ = initialize_checkpoint_dir(
            target_task / "checkpoints", keep_period=None,
            overwrite=False, resume=False,
        )
        try:
            save_checkpoint(manager, upgraded, SOURCE_STEP)
            manager.wait_until_finished()
            roundtrip = restore_checkpoint(manager, current_agent, SOURCE_STEP)
            roundtrip_params = current_agent.actor.get_params(roundtrip.actor_train_state)
            roundtrip_equal = jax.tree_util.tree_leaves(
                jax.tree_util.tree_map(
                    lambda a, b: np.array_equal(np.asarray(a), np.asarray(b)),
                    source_actor_params, roundtrip_params,
                )
            )
            if not all(bool(leaf) for leaf in roundtrip_equal):
                raise AssertionError("serialized upgrade changed the 50k actor weights")
            if (int(np.asarray(roundtrip.distill_actor_step)) != SOURCE_STEP
                    or int(np.asarray(roundtrip.rl_actor_step)) != 0):
                raise AssertionError("serialized optimizer phase counters are incorrect")
        finally:
            manager.close()
    finally:
        env.close()
    metadata = {
        "source": str(source_checkpoint),
        "step": SOURCE_STEP,
        "distill_steps_done": SOURCE_STEP,
        "offline_steps_done": 0,
        "online_steps_done": 0,
        "rl_actor_step": 0,
        "schema": "dual_actor_optimizers_v1",
        "actor_weight_checksum": float(sum(
            np.asarray(leaf, dtype=np.float64).sum()
            for leaf in jax.tree_util.tree_leaves(source_actor_params)
        )),
    }
    (target_task / "upgrade_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(f"task {task_id}: upgraded -> {target_checkpoint}", flush=True)


def main() -> None:
    args = _parse_args()
    for task_id in args.task_ids:
        _upgrade_one(args, task_id)


if __name__ == "__main__":
    main()

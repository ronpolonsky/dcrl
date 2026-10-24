"""Learner-agnostic Pi0.5 full-chunk SFT trainer.

This standalone counterpart to train_bc_distill.py keeps the same data
collection, normalization, tokenization, 10/5 LIBERO rollout, evaluation,
metrics, checkpoint, and resume behavior while delegating learner construction
and updates to a configured learner API.
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from collections import deque

import numpy as np
from tqdm import tqdm

PI05_HORIZON = 10
PI05_REPLAN = 5
PI05_BATCH_SIZE = 16
PI05_LR = 5e-5
PI05_WARMUP = 10_000
PI05_DECAY = 1_000_000
PI05_STEPS = 500_000
PI05_EVAL_EPISODES = 20


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("all", "collect", "train"), default="all")
    parser.add_argument("--collection-source", choices=("small_expo", "pi_base"), default="small_expo")
    parser.add_argument("--teacher-dir", type=Path, default=None)
    parser.add_argument("--teacher-step", type=int, default=None)
    parser.add_argument("--source-checkpoint", default="gs://openpi-assets/checkpoints/pi05_base")
    parser.add_argument("--norm-stats", default="gs://openpi-assets/checkpoints/pi05_libero/assets/physical-intelligence/libero/norm_stats.json")
    parser.add_argument("--source-suite", default="libero_90")
    parser.add_argument("--source-task-id", type=int, default=29)
    parser.add_argument("--collection-successes", type=int, default=100)
    parser.add_argument("--collection-failures", type=int, default=25)
    parser.add_argument("--collection-episodes", type=int, default=200)
    parser.add_argument("--collection-max-attempts", type=int, default=None)
    parser.add_argument("--include-failures", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--student-checkpoint", default="gs://openpi-assets/checkpoints/pi05_base")
    parser.add_argument("--learner-config", default="configs.sft.my_expo_libero")
    parser.add_argument("--action-horizon", type=int, default=PI05_HORIZON)
    parser.add_argument("--replan-steps", type=int, default=PI05_REPLAN)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-sft-steps", type=int, default=PI05_STEPS)
    parser.add_argument("--batch-size", type=int, default=PI05_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=PI05_LR)
    parser.add_argument("--warmup-steps", type=int, default=PI05_WARMUP)
    parser.add_argument("--lr-decay-steps", type=int, default=PI05_DECAY)
    parser.add_argument("--finetune-mode", choices=("lora", "full"), default="lora")
    parser.add_argument("--eval-every", type=int, default=25_000)
    parser.add_argument("--eval-episodes", type=int, default=PI05_EVAL_EPISODES)
    parser.add_argument("--checkpoint-every", type=int, default=25_000)
    parser.add_argument("--max-to-keep", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-seed", type=int, default=7)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    for name in ("collection_episodes", "num_sft_steps", "batch_size",
                 "warmup_steps", "lr_decay_steps", "eval_every",
                 "eval_episodes", "checkpoint_every"):
        if getattr(args, name) <= 0:
            parser.error(f"{name} must be positive")
    if args.action_horizon != PI05_HORIZON or args.replan_steps != PI05_REPLAN:
        parser.error("this trainer requires the official Pi0.5 LIBERO 10/5 interface")
    return args


def _raw_input(obs, instruction):
    """Flatten a LIBERO environment observation like the existing SFT path."""
    return {
        "observation/image": obs["image"]["base_0_rgb"],
        "observation/wrist_image": obs["image"]["left_wrist_0_rgb"],
        "observation/state": obs["state"],
        "prompt": instruction,
    }


def _evaluate_actor(learner, env, *, episodes, max_steps, replan_steps, seed):
    import jax

    eval_learner = learner.replace(rng=jax.random.PRNGKey(seed))
    env.episode_index = 0
    env.env.seed(env.seed)
    successes, lengths, trials = 0, [], []
    for episode in tqdm(range(episodes), desc="Evaluate learner actor", leave=False):
        obs, plan = env.reset(), deque()
        for step in range(max_steps):
            if not plan:
                chunk, eval_learner, _ = eval_learner.eval_actor_only(
                    _raw_input(obs, env.instruction)
                )
                plan.extend(np.asarray(jax.device_get(chunk))[:replan_steps])
            obs, _, success, _ = env.step(plan.popleft())
            if success:
                successes += 1
                lengths.append(step + 1)
                trials.append({"episode": episode, "initial_state_index": episode,
                               "success": True, "steps": step + 1})
                break
        else:
            lengths.append(max_steps)
            trials.append({"episode": episode, "initial_state_index": episode,
                           "success": False, "steps": max_steps})
    return {"eval/success_rate": successes / episodes,
            "eval/mean_steps": float(np.mean(lengths)), "trials": trials}


def _paths(args):
    name = "teacher_chunks.hdf5" if args.collection_source == "small_expo" else "pi05_chunks.hdf5"
    mixed = args.output_dir / "pi_base_data" / name
    return mixed, mixed.parent / "success" / name, mixed.parent / "failures" / name


def _collect(args):
    mixed, _, _ = _paths(args)
    if args.collection_source == "small_expo":
        if args.teacher_dir is None or args.teacher_step is None:
            raise ValueError("small_expo collection requires --teacher-dir and --teacher-step")
        from dcrl.bc_distill_data import collect_teacher_chunks
        collect_teacher_chunks(
            args.teacher_dir, args.teacher_step, mixed, args.collection_episodes,
            args.eval_seed, target_action_horizon=args.action_horizon,
            target_replan_steps=args.replan_steps,
            target_successes=args.collection_successes,
            target_failures=args.collection_failures,
            max_attempts_override=args.collection_max_attempts)
    else:
        from dcrl.collect_pi05_libero import collect
        collect(checkpoint=args.source_checkpoint, suite=args.source_suite,
                norm_stats=args.norm_stats, task_id=args.source_task_id,
                episodes=None, seed=args.eval_seed, output=mixed,
                successes=args.collection_successes, failures=args.collection_failures)


def train(args):
    import jax
    import orbax.checkpoint as ocp
    from openpi.shared import download, normalize
    from openpi.training import sharding
    from dcrl.bc_distill_data import ChunkDataset, partition_collection
    from dcrl.libero_env import LIBERO_SUITE_MAX_STEPS, LiberoTaskEnv
    from dcrl.agents.my_expo_learner import restore_checkpoint, save_checkpoint

    mixed, success_file, failure_file = _paths(args)
    if not success_file.exists():
        partition_collection(mixed, success_file, failure_file)
    dataset = ChunkDataset(
        [success_file] + ([failure_file] if args.include_failures else []),
        success_only=not args.include_failures)
    collection = dataset.metadata
    if args.collection_source == "small_expo":
        if args.teacher_dir is None or args.teacher_step is None:
            raise ValueError("small_expo training requires --teacher-dir and --teacher-step")
        if collection.get("teacher_dir") != str(args.teacher_dir.resolve()):
            raise ValueError("dataset teacher directory does not match the requested one")
        if int(collection["checkpoint_step"]) != args.teacher_step:
            raise ValueError("dataset teacher checkpoint does not match the requested one")
    elif collection.get("source") != "pi05_base_actor":
        raise ValueError("dataset was not collected from the Pi0.5 base actor")
    elif collection.get("checkpoint") != args.source_checkpoint:
        raise ValueError("dataset Pi0.5 checkpoint does not match the requested source checkpoint")
    horizon = collection.get("target_action_horizon", collection.get("action_horizon"))
    replan = collection.get("target_replan_steps", collection.get("replan_steps"))
    if horizon != args.action_horizon or replan != args.replan_steps:
        raise ValueError("dataset was not collected with the Pi0.5 10/5 interface")
    if dataset.action_horizon != args.action_horizon:
        raise ValueError("dataset action horizon does not match the learner")
    if int(collection.get("image_size", args.image_size)) != args.image_size:
        raise ValueError("dataset image size differs from --image-size")

    manifest = {k: (str(v) if isinstance(v, Path) else v)
                for k, v in vars(args).items() if k not in ("phase", "resume", "num_sft_steps")}
    manifest.update(collection=collection, loss="full_chunk_flow_matching",
                    student_action_horizon=args.action_horizon,
                    student_replan_steps=args.replan_steps,
                    official_pi05_libero=True)
    manifest_file = args.output_dir / "sft_config.json"
    if args.resume:
        if not manifest_file.exists() or json.loads(manifest_file.read_text()) != manifest:
            raise ValueError("resume configuration differs from saved SFT configuration")
    elif manifest_file.exists():
        raise FileExistsError("output exists; use --resume")

    stats_dir = args.output_dir / "student_assets/physical-intelligence/libero"
    if not args.resume:
        source = download.maybe_download(args.norm_stats)
        stats_dir.mkdir(parents=True, exist_ok=True)
        (stats_dir / "norm_stats.json").write_bytes(Path(source).read_bytes())
    stats = normalize.load(stats_dir)
    if len(stats["actions"].mean) != 7 or len(stats["state"].mean) != 8:
        raise ValueError("Pi0.5 LIBERO normalization must have 7 actions and 8 state values")

    mesh = sharding.make_mesh(1)
    data_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec())
    if args.batch_size % jax.device_count():
        raise ValueError("batch size must divide the device count")

    config = importlib.import_module(args.learner_config)
    learner, transform, _ = config.build_learner(
        args=args, output_dir=args.output_dir, instruction=dataset.instruction,
        stats_dir=stats_dir, mesh=mesh, data_sharding=data_sharding,
        replicated_sharding=replicated_sharding)
    suite_name = collection.get("suite", args.source_suite)
    env = LiberoTaskEnv(suite_name,
                        int(collection.get("task_id", args.source_task_id)),
                        seed=args.eval_seed, image_size=args.image_size,
                        benchmark_eval=True)
    if args.eval_episodes > len(env.init_states):
        raise ValueError("evaluation episodes exceed official initial states")

    manager = ocp.CheckpointManager(
        args.output_dir / "checkpoints",
        item_handlers={"agent": ocp.PyTreeCheckpointHandler(),
                       "params": ocp.PyTreeCheckpointHandler()},
        options=ocp.CheckpointManagerOptions(max_to_keep=args.max_to_keep, create=True))
    try:
        latest = manager.latest_step() if args.resume else None
        if latest is not None:
            # Keep the restored actor on its training sharding until the next
            # BC update. Eagerly copying the complete inference cache here
            # doubles large parameter trees and can exhaust 48-GiB GPUs before
            # the first resumed update. eval_actor_only() works directly from
            # the train state, and bc_update() invalidates stale caches.
            learner = restore_checkpoint(manager, learner, latest)
        elif args.resume:
            raise ValueError("resume requested but no checkpoint exists")
        else:
            manifest_file.write_text(json.dumps(manifest, indent=2) + "\n")
        start = int(latest or 0)
        if start >= args.num_sft_steps:
            raise ValueError("requested SFT steps must exceed restored step")

        def evaluate_and_save(step):
            result = _evaluate_actor(
                learner, env, episodes=args.eval_episodes,
                max_steps=LIBERO_SUITE_MAX_STEPS[suite_name],
                replan_steps=args.replan_steps, seed=args.eval_seed)
            eval_dir = args.output_dir / "evals"
            eval_dir.mkdir(exist_ok=True)
            (eval_dir / f"{step:08d}.json").write_text(
                json.dumps({"step": step, **result}, indent=2))
            return {key: value for key, value in result.items() if key != "trials"}

        if latest is None:
            evaluate_and_save(0)
        with (args.output_dir / "metrics.jsonl").open("a") as log:
            for step in tqdm(range(start + 1, args.num_sft_steps + 1),
                             total=args.num_sft_steps, initial=start,
                             desc="Learner BC"):
                rng = np.random.default_rng(np.random.SeedSequence([args.seed, step]))
                rows = [transform(row) for row in dataset.sample(rng, args.batch_size)]
                batch = jax.tree.map(
                    lambda *values: jax.device_put(np.stack(values), data_sharding), *rows)
                target = batch.pop("actions")
                # LiberoInputs pads the 7D environment action to Pi0.5's
                # model action dimension (32).  bc_update() accepts the
                # learner-facing 7D chunk and applies that padding itself.
                if target.shape[-1] < 7:
                    raise ValueError(f"action target has only {target.shape[-1]} dimensions")
                target = target[..., :7]
                batch["full_actions"] = target.reshape(target.shape[0], -1)
                if batch["full_actions"].shape[-1] != args.action_horizon * 7:
                    raise ValueError(
                        f"learner BC target has shape {batch['full_actions'].shape}; "
                        f"expected final width {args.action_horizon * 7}"
                    )
                update_key = jax.device_put(
                    jax.random.fold_in(jax.random.PRNGKey(args.seed), step),
                    replicated_sharding,
                )
                learner, info = learner.bc_update(batch, rng=update_key)
                metrics = {k: float(np.asarray(v)) for k, v in info.items()
                           if np.asarray(v).size == 1}
                if not all(np.isfinite(v) for v in metrics.values()):
                    raise FloatingPointError(f"nonfinite BC metrics at step {step}: {metrics}")
                if step % args.eval_every == 0 or step == args.num_sft_steps:
                    metrics.update(evaluate_and_save(step))
                log.write(json.dumps({"step": step, **metrics}) + "\n")
                log.flush()
                if step % args.checkpoint_every == 0 or step == args.num_sft_steps:
                    save_checkpoint(manager, learner, step)
                    manager.wait_until_finished()
    finally:
        manager.close()
        env.close()
        dataset.close()


def main():
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    if args.teacher_dir is not None:
        args.teacher_dir = args.teacher_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.phase in ("all", "collect"):
        _collect(args)
    if args.phase in ("all", "train"):
        train(args)


if __name__ == "__main__":
    main()

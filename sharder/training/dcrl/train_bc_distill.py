"""Distill a frozen SmallExpo RL policy into the Pi0.5 base actor.

The default LIBERO interface is Pi0.5's 10-action prediction with 5-action
execution/replanning. Collection stores one full chunk per replanning state;
training updates only the base actor with flow-matching BC. Every update can
write metrics, run the official fixed-state LIBERO evaluation, and save a full
Orbax state for exact resume.
"""
from __future__ import annotations

import argparse
from collections import deque
import dataclasses
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from tqdm import tqdm


PI05_HORIZON = 10
PI05_REPLAN = 5
PI05_BATCH_SIZE = 256
PI05_LR = 5e-5
PI05_WARMUP = 10_000
PI05_DECAY = 1_000_000
PI05_STEPS = 1_000_000
PI05_EVAL_EPISODES = 20


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("all", "collect", "train"), default="all")
    parser.add_argument("--collection-source", choices=("small_expo", "pi_base"), default="small_expo")
    parser.add_argument("--teacher-dir", type=Path, default=None,
                        help="SmallExpo source directory when collection-source=small_expo.")
    parser.add_argument("--teacher-step", type=int, default=None)
    parser.add_argument("--source-checkpoint", default="gs://openpi-assets/checkpoints/pi05_base")
    parser.add_argument("--norm-stats", default="gs://openpi-assets/checkpoints/pi05_libero/assets/physical-intelligence/libero/norm_stats.json")
    parser.add_argument("--source-suite", default="libero_90")
    parser.add_argument("--source-task-id", type=int, default=29)
    parser.add_argument("--collection-successes", type=int, default=100)
    parser.add_argument("--collection-failures", type=int, default=25)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--collection-episodes", type=int, default=200)
    parser.add_argument("--collection-max-attempts", type=int, default=None,
                        help="Override the quota attempt budget when resuming a partial collection.")
    parser.add_argument("--include-failures", action="store_true")
    parser.add_argument("--student-checkpoint", default="gs://openpi-assets/checkpoints/pi05_base")
    parser.add_argument("--num-sft-steps", type=int, default=PI05_STEPS)
    parser.add_argument("--batch-size", type=int, default=PI05_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=PI05_LR)
    parser.add_argument("--warmup-steps", type=int, default=PI05_WARMUP)
    parser.add_argument("--lr-decay-steps", type=int, default=PI05_DECAY)
    parser.add_argument("--finetune-mode", choices=("lora", "full"), default="lora")
    parser.add_argument("--eval-every", type=int, default=25_000)
    parser.add_argument("--eval-episodes", type=int, default=PI05_EVAL_EPISODES)
    parser.add_argument("--checkpoint-every", type=int, default=25_000)
    parser.add_argument("--max-to-keep", type=int, default=None,
                        help="Maximum number of full checkpoints to retain (default: all).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-seed", type=int, default=7)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    for name in ("collection_episodes", "num_sft_steps", "batch_size", "warmup_steps",
                 "lr_decay_steps", "eval_every", "eval_episodes", "checkpoint_every"):
        if getattr(args, name) <= 0:
            parser.error(f"{name} must be positive")
    return args


def _raw_input(obs, instruction):
    return {
        "observation/image": obs["image"]["base_0_rgb"],
        "observation/wrist_image": obs["image"]["left_wrist_0_rgb"],
        "observation/state": obs["state"],
        "prompt": instruction,
    }


def _evaluate(actor, state, transform, output_transform, env, env_horizon, episodes, seed):
    import jax
    from openpi.models import model

    env.episode_index = 0
    env.env.seed(env.seed)
    rng = jax.random.PRNGKey(seed)
    trials = []
    for episode in tqdm(range(episodes), desc="Evaluate Pi0.5 actor", leave=False):
        obs = env.reset()
        plan = deque()
        success = False
        for step in range(env_horizon):
            if not plan:
                inputs = transform(_raw_input(obs, env.instruction))
                inputs = jax.tree.map(lambda value: np.asarray(value)[None], inputs)
                packed = model.Observation.from_dict(inputs).to_dict()
                rng, key = jax.random.split(rng)
                actions, _ = actor.sample_actions(packed, state, key, num_samples=1)
                actions = np.asarray(jax.device_get(actions))[0]
                actions = output_transform({
                    "actions": actions,
                    "state": np.asarray(inputs["state"])[0],
                })["actions"]
                plan.extend(actions[:PI05_REPLAN])
            obs, _, success, _ = env.step(plan.popleft())
            if success:
                break
        trials.append({"episode": episode, "initial_state_index": episode,
                       "success": bool(success), "steps": step + 1})
    return {
        "eval/success_rate": float(np.mean([trial["success"] for trial in trials])),
        "eval/mean_steps": float(np.mean([trial["steps"] for trial in trials])),
        "trials": trials,
    }


def _manifest(args, collection_metadata):
    values = {key: (str(value) if isinstance(value, Path) else value)
              for key, value in vars(args).items()
              if key not in ("phase", "resume", "num_sft_steps")}
    values.update({
        "collection": collection_metadata,
        "loss": "full_chunk_flow_matching",
        "student_action_horizon": PI05_HORIZON,
        "student_replan_steps": PI05_REPLAN,
        "official_pi05_libero": True,
    })
    return values


def train(args):
    import jax
    import orbax.checkpoint as ocp
    from openpi import transforms
    from openpi.models import model
    from openpi.shared import download, normalize
    from openpi.training import config as configs, optimizer, sharding, weight_loaders
    from dcrl.bc_distill_data import ChunkDataset, partition_collection
    from dcrl.expo.agents.vla.pi05 import Pi05Agent
    from dcrl.libero_env import LIBERO_SUITE_MAX_STEPS, LiberoTaskEnv

    if args.collection_source == "small_expo":
        mixed = args.output_dir / "pi_base_data/teacher_chunks.hdf5"
        success_file = args.output_dir / "pi_base_data/success/teacher_chunks.hdf5"
        failure_file = args.output_dir / "pi_base_data/failures/teacher_chunks.hdf5"
    else:
        mixed = args.output_dir / "pi_base_data/pi05_chunks.hdf5"
        success_file = args.output_dir / "pi_base_data/success/pi05_chunks.hdf5"
        failure_file = args.output_dir / "pi_base_data/failures/pi05_chunks.hdf5"
    if not success_file.exists():
        partition_collection(mixed, success_file, failure_file)
    paths = [success_file] + ([failure_file] if args.include_failures else [])
    dataset = ChunkDataset(paths, success_only=not args.include_failures)
    collection = dataset.metadata
    if args.collection_source == "small_expo":
        if (collection.get("teacher_dir") != str(args.teacher_dir.resolve())
                or int(collection["checkpoint_step"]) != args.teacher_step):
            raise ValueError("dataset teacher does not match requested SmallExpo checkpoint")
        collection_suite = collection["teacher_config"]["suite"]
        collection_task_id = collection["teacher_config"]["task_id"]
        collection_image_size = collection["teacher_config"]["image_size"]
    else:
        if collection.get("source") != "pi05_base_actor":
            raise ValueError("dataset was not collected from the Pi0.5 base actor")
        if collection.get("checkpoint") != args.source_checkpoint:
            raise ValueError("dataset source checkpoint does not match requested Pi0.5 checkpoint")
        collection_suite = collection["suite"]
        collection_task_id = collection["task_id"]
        collection_image_size = collection["image_size"]
    collection_horizon = collection.get("target_action_horizon", collection.get("action_horizon"))
    collection_replan = collection.get("target_replan_steps", collection.get("replan_steps"))
    if (collection_horizon != PI05_HORIZON or collection_replan != PI05_REPLAN):
        raise ValueError("dataset was not collected with the official Pi0.5 10/5 interface")
    if dataset.action_horizon != PI05_HORIZON:
        raise ValueError(f"expected {PI05_HORIZON}-action targets, got {dataset.action_horizon}")

    manifest = _manifest(args, collection)
    manifest_file = args.output_dir / "sft_config.json"
    if args.resume:
        if not manifest_file.exists() or json.loads(manifest_file.read_text()) != manifest:
            raise ValueError("resume configuration differs from saved SFT configuration")
    elif manifest_file.exists():
        raise FileExistsError("SFT output already exists; use --resume")

    stats_dir = args.output_dir / "student_assets/physical-intelligence/libero"
    if not args.resume:
        source = download.maybe_download(args.norm_stats)
        stats_dir.mkdir(parents=True, exist_ok=True)
        (stats_dir / "norm_stats.json").write_bytes(Path(source).read_bytes())
    stats = normalize.load(stats_dir)
    if len(stats["actions"].mean) != 7 or len(stats["state"].mean) != 8:
        raise ValueError("Pi0.5 LIBERO normalization must have 7 actions and 8 state values")

    print(f"BC dataset: {len(dataset.indices)} chunks; student={args.student_checkpoint}", flush=True)
    base = configs.get_config("pi05_libero")
    student_model = dataclasses.replace(base.model, action_horizon=PI05_HORIZON)
    if args.finetune_mode == "lora":
        student_model = dataclasses.replace(
            student_model, paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora")
    config = dataclasses.replace(
        base,
        model=student_model,
        freeze_filter=student_model.get_freeze_filter(),
        ema_decay=None if args.finetune_mode == "lora" else base.ema_decay,
        data=dataclasses.replace(base.data, assets=configs.AssetsConfig(
            assets_dir=str(args.output_dir / "student_assets"), asset_id="physical-intelligence/libero")),
        weight_loader=weight_loaders.CheckpointWeightLoader(args.student_checkpoint.rstrip("/") + "/params"),
        lr_schedule=optimizer.CosineDecaySchedule(
            warmup_steps=args.warmup_steps, peak_lr=args.learning_rate,
            decay_steps=args.lr_decay_steps, decay_lr=args.learning_rate),
        batch_size=args.batch_size,
        num_train_steps=args.num_sft_steps,
    )
    mesh = sharding.make_mesh(1)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    if args.batch_size % jax.device_count():
        raise ValueError("batch size must divide the device count")
    actor, state, _ = Pi05Agent.initialize(
        config, mesh, jax.random.PRNGKey(args.seed),
        resume=False, data_sharding=data_sharding, replicated_sharding=replicated_sharding,
        default_prompt=dataset.instruction, freeze_pi05_encoder=False,
    )
    actor.action_dim, actor.state_dim = 7, 8
    dc = actor.data_config
    transform = transforms.compose([
        *dc.data_transforms.inputs,
        transforms.Normalize(stats, use_quantiles=dc.use_quantile_norm),
        *dc.model_transforms.inputs,
    ])
    output_transform = transforms.compose([
        *dc.model_transforms.outputs,
        transforms.Unnormalize(stats, use_quantiles=dc.use_quantile_norm),
        *dc.data_transforms.outputs,
    ])
    env = LiberoTaskEnv(collection_suite, collection_task_id,
                        seed=args.eval_seed, image_size=collection_image_size,
                        benchmark_eval=True)
    checkpoint_dir = args.output_dir / "checkpoints"
    manager = ocp.CheckpointManager(
        checkpoint_dir,
        item_handlers={"state": ocp.PyTreeCheckpointHandler()},
        options=ocp.CheckpointManagerOptions(max_to_keep=args.max_to_keep, create=True),
    )
    try:
        if args.eval_episodes > len(env.init_states):
            raise ValueError("evaluation episodes exceed available official initial states")
        restore_existing = args.resume and manager.latest_step() is not None
        if restore_existing:
            latest = manager.latest_step()
            state = manager.restore(
                latest, items={"state": state},
                restore_kwargs={"state": {"restore_args": ocp.checkpoint_utils.construct_restore_args(state)}},
            )["state"]
        else:
            manifest_file.write_text(json.dumps(manifest, indent=2) + "\n")
        start = int(np.asarray(state.step))
        if start >= args.num_sft_steps:
            raise ValueError("requested SFT steps must exceed restored step")

        def evaluate_and_save(step):
            result = _evaluate(actor, state, transform, output_transform, env,
                               LIBERO_SUITE_MAX_STEPS[collection_suite],
                               args.eval_episodes, args.eval_seed)
            eval_dir = args.output_dir / "evals"
            eval_dir.mkdir(exist_ok=True)
            (eval_dir / f"{step:08d}.json").write_text(
                json.dumps({"step": step, **result}, indent=2))
            return {key: value for key, value in result.items() if key != "trials"}

        if not restore_existing:
            evaluate_and_save(0)
        with (args.output_dir / "metrics.jsonl").open("a") as log:
            for step in tqdm(range(start + 1, args.num_sft_steps + 1),
                             total=args.num_sft_steps, initial=start, desc="Pi0.5 BC"):
                rng = np.random.default_rng(np.random.SeedSequence([args.seed, step]))
                rows = [transform(row) for row in dataset.sample(rng, args.batch_size)]
                batch = jax.tree.map(lambda *values: jax.device_put(np.stack(values), data_sharding), *rows)
                targets = batch.pop("actions")[:, :PI05_HORIZON]
                observation = model.Observation.from_dict(batch)
                key = jax.device_put(jax.random.fold_in(jax.random.PRNGKey(args.seed), step), replicated_sharding)
                with sharding.set_mesh(mesh):
                    state, info = actor.train_step(key, state, (observation, targets))
                metrics = {key: float(np.asarray(value)) for key, value in info.items()}
                if not all(np.isfinite(value) for value in metrics.values()):
                    raise FloatingPointError(f"nonfinite BC metrics at step {step}: {metrics}")
                if step % args.eval_every == 0 or step == args.num_sft_steps:
                    metrics.update(evaluate_and_save(step))
                log.write(json.dumps({"step": step, **metrics}) + "\n")
                log.flush()
                if step % args.checkpoint_every == 0 or step == args.num_sft_steps:
                    manager.save(step, {"state": state})
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
    if args.collection_source == "small_expo" and (args.teacher_dir is None or args.teacher_step is None):
        raise ValueError("small_expo collection requires --teacher-dir and --teacher-step")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.phase == "all":
        subprocess.run([sys.executable, "-m", "dcrl.train_bc_distill", *sys.argv[1:], "--phase", "collect"], check=True)
        subprocess.run([sys.executable, "-m", "dcrl.train_bc_distill", *sys.argv[1:], "--phase", "train"], check=True)
    elif args.phase == "collect":
        if args.collection_source == "small_expo":
            from dcrl.bc_distill_data import collect_teacher_chunks
            collect_teacher_chunks(
                args.teacher_dir, args.teacher_step, args.output_dir / "pi_base_data/teacher_chunks.hdf5",
                args.collection_episodes, args.eval_seed,
                target_action_horizon=PI05_HORIZON, target_replan_steps=PI05_REPLAN,
                target_successes=args.collection_successes,
                target_failures=args.collection_failures,
                max_attempts_override=args.collection_max_attempts,
            )
        else:
            from dcrl.collect_pi05_libero import collect
            collect(
                checkpoint=args.source_checkpoint, suite=args.source_suite,
                norm_stats=args.norm_stats,
                task_id=args.source_task_id, episodes=None, seed=args.eval_seed,
                output=args.output_dir / "pi_base_data" / "pi05_chunks.hdf5",
                successes=args.collection_successes, failures=args.collection_failures,
            )
    else:
        train(args)


if __name__ == "__main__":
    main()

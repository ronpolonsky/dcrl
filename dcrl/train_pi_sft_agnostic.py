"""Learner-agnostic Pi0.5 full-chunk SFT trainer.

This standalone counterpart to train_bc_distill.py keeps the same data
collection, normalization, tokenization, 10/5 LIBERO rollout, evaluation,
metrics, checkpoint, and resume behavior while delegating learner construction
and updates to a configured learner API.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
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

_OPENPI_LOCAL_CACHE_ROOTS = (
    Path("/projects/bgqe/rpolonsky/dcrlcheckpoints/openpi_cache/openpi-assets"),
    Path("/work/hdd/bgqe/rpolonsky/dcrlcheckpoints/openpi_cache/openpi-assets"),
)


def _manifest_openpi_asset_identity(value):
    """Keep cached local paths resume-compatible with their canonical GCS URIs."""
    value = str(value)
    if value.startswith("gs://"):
        return value
    path = Path(value).expanduser()
    try:
        resolved = path.resolve()
    except OSError:
        return value
    for root in _OPENPI_LOCAL_CACHE_ROOTS:
        try:
            relative = resolved.relative_to(root.resolve())
        except ValueError:
            continue
        return f"gs://openpi-assets/{relative.as_posix()}"
    return value


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("all", "collect", "train"), default="all")
    parser.add_argument(
        "--collection-source",
        choices=("small_expo", "pi_base", "pi05_small_expo_editor", "benchmark_demos"),
        default="small_expo",
    )
    parser.add_argument("--teacher-dir", type=Path, default=None)
    parser.add_argument("--teacher-step", type=int, default=None)
    parser.add_argument(
        "--small-expo-checkpoint", type=Path, default=None,
        help="Numeric Small EXPO checkpoint for the hybrid Pi0.5+editor/critic collector.",
    )
    parser.add_argument(
        "--pi05-checkpoint", default="gs://openpi-assets/checkpoints/pi05_libero",
        help="Official Pi0.5 LIBERO checkpoint used as the hybrid teacher actor.",
    )
    parser.add_argument("--source-checkpoint", default="gs://openpi-assets/checkpoints/pi05_base")
    parser.add_argument("--norm-stats", default="gs://openpi-assets/checkpoints/pi05_libero/assets/physical-intelligence/libero/norm_stats.json")
    parser.add_argument("--source-suite", default="libero_90")
    parser.add_argument('--benchmark-demo-task-ids',
                        help='Additional OCBench demo tasks for initial Pi0.5 SFT, as comma-separated IDs or all.')
    parser.add_argument('--eval-benchmark-demo-tasks', action='store_true',
                        help='Evaluate the same actor on all tasks in an OCBench demo mixture.')
    parser.add_argument(
        "--source-task-id", type=int, required=True,
        help="LIBERO task ID used for collection and actor-only evaluation; must match the selected teacher checkpoint.",
    )
    parser.add_argument("--collection-successes", type=int, default=100)
    parser.add_argument("--collection-failures", type=int, default=25)
    parser.add_argument("--collection-episodes", type=int, default=200)
    parser.add_argument("--collection-max-attempts", type=int, default=None)
    parser.add_argument("--include-failures", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--student-checkpoint", default="gs://openpi-assets/checkpoints/pi05_libero")
    parser.add_argument("--learner-config", default="configs.sft.my_expo_libero")
    parser.add_argument("--action-horizon", type=int, default=PI05_HORIZON)
    parser.add_argument("--replan-steps", type=int, default=PI05_REPLAN)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-sft-steps", type=int, default=PI05_STEPS)
    parser.add_argument("--batch-size", type=int, default=PI05_BATCH_SIZE)
    parser.add_argument(
        "--official-demo-ratio", type=float, default=0.0,
        help="Expected fraction of each SFT batch from official demos for the same task; the rest comes from SmallExpo targets.",
    )
    parser.add_argument(
        "--official-demo-root", type=Path, default=None,
        help="Root containing official LIBERO suite folders (defaults to LIBERO's configured datasets root).",
    )
    parser.add_argument("--learning-rate", type=float, default=PI05_LR)
    parser.add_argument('--ocbench-task-prompt',
                        help='Instruction used for both training and rollout in a standalone OCBench demo run.')
    parser.add_argument('--ocbench-reference-action-normalization', action='store_true',
                        help='Clip normalized arm actions to [-5, 5] and keep the gripper in [-1, 1].')
    parser.add_argument("--warmup-steps", type=int, default=PI05_WARMUP)
    parser.add_argument("--lr-decay-steps", type=int, default=PI05_DECAY)
    parser.add_argument("--finetune-mode", choices=("lora", "full"), default="lora")
    parser.add_argument(
        "--trainable-scope", choices=("existing", "lora_only"),
        default="lora_only",
        help="Gradient scope; lora_only matches the tuned historical SFT runs.",
    )
    parser.add_argument(
        "--zero-lora-b", action="store_true",
        help=("Zero newly initialized LoRA-B leaves before the first update, "
              "matching the tuned SFT trainer's initialization."),
    )
    parser.add_argument("--eval-every", type=int, default=25_000)
    parser.add_argument('--skip-initial-eval', action='store_true')
    parser.add_argument('--checkpoint-policy', choices=('best', 'final', 'none'), default='best')
    parser.add_argument('--eval-max-steps', type=int, default=0,
                        help='Use the benchmark task horizon when zero.')
    parser.add_argument("--eval-episodes", type=int, default=PI05_EVAL_EPISODES)
    parser.add_argument(
        "--checkpoint-every", type=int, default=25_000,
        help="Deprecated compatibility option; best checkpoints are saved at evaluation steps.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-seed", type=int, default=7)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--wandb-log-every", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.resume and args.checkpoint_policy == 'none':
        parser.error('--resume requires a checkpoint policy that saves a checkpoint.')
    for name in ("collection_episodes", "num_sft_steps", "batch_size",
                 "warmup_steps", "lr_decay_steps", "eval_every",
                 "eval_episodes", "checkpoint_every", "wandb_log_every"):
        if getattr(args, name) <= 0:
            parser.error(f"{name} must be positive")
    ocbench_demos = (args.source_suite in ('ocbench', 'ocbench_lite')
                     and args.collection_source == 'benchmark_demos')
    if args.ocbench_reference_action_normalization and not ocbench_demos:
        parser.error('--ocbench-reference-action-normalization requires OCBench benchmark_demos')
    if args.ocbench_task_prompt is not None:
        args.ocbench_task_prompt = args.ocbench_task_prompt.strip()
        if not ocbench_demos or args.benchmark_demo_task_ids or not args.ocbench_task_prompt:
            parser.error('--ocbench-task-prompt requires a nonempty instruction and standalone OCBench benchmark_demos')
    if args.eval_benchmark_demo_tasks and (not ocbench_demos or not args.benchmark_demo_task_ids):
        parser.error('--eval-benchmark-demo-tasks requires an OCBench demo mixture')
    if ocbench_demos:
        if not 1 <= args.replan_steps <= args.action_horizon:
            parser.error('--replan-steps must be in [1, action-horizon]')
    elif args.action_horizon != PI05_HORIZON or args.replan_steps != PI05_REPLAN:
        parser.error("this collection path requires the official Pi0.5 LIBERO 10/5 interface")
    if not 0.0 <= args.official_demo_ratio <= 1.0:
        parser.error("official_demo_ratio must be in [0, 1]")
    if args.trainable_scope == "lora_only" and args.finetune_mode != "lora":
        parser.error("--trainable-scope lora_only requires --finetune-mode lora")
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

    env_id = getattr(env, 'env_id', '')
    score_return = env_id.startswith('visual-bowling-')
    track_cube = env_id in ('visual-block-cpu-lite-single-task1-v0', 'visual-block-cpu-single-task1-v0')
    eval_learner = learner.replace(rng=jax.random.PRNGKey(seed))
    env.reset_sequence()
    successes, lengths, trials = 0, [], []
    for episode in tqdm(range(episodes), desc="Evaluate learner actor", leave=False):
        obs, plan = env.reset(), deque()
        episode_return = 0.0
        if track_cube:
            start, goal = env.single_cube_positions()
            initial_distance = min_distance = float(np.linalg.norm(start - goal))
            max_lift = max_displacement = 0.0
        for step in range(max_steps):
            if not plan:
                chunk, eval_learner, _ = eval_learner.eval_actor_only(
                    _raw_input(obs, env.instruction)
                )
                plan.extend(np.asarray(jax.device_get(chunk))[:replan_steps])
            obs, reward, success, _ = env.step(plan.popleft())
            if score_return:
                episode_return += reward
            if track_cube:
                cube, _ = env.single_cube_positions()
                min_distance = min(min_distance, float(np.linalg.norm(cube - goal)))
                max_lift = max(max_lift, float(cube[2] - start[2]))
                max_displacement = max(max_displacement, float(np.linalg.norm(cube - start)))
            if env.episode_done or success:
                successes += int(success)
                lengths.append(step + 1)
                trials.append({"episode": episode, "initial_state_index": episode,
                               "success": bool(success), "steps": step + 1})
                break
        else:
            lengths.append(max_steps)
            trials.append({"episode": episode, "initial_state_index": episode,
                           "success": False, "steps": max_steps})
        if score_return:
            trials[-1]['return'] = episode_return
        if track_cube:
            trials[-1].update(cube_initial_goal_distance_m=initial_distance,
                              cube_min_goal_distance_m=min_distance, cube_max_lift_m=max_lift,
                              cube_max_displacement_m=max_displacement)
    if score_return:
        from dcrl.ocbench_env import evaluation_metrics

        return {**evaluation_metrics(trials, max_steps, score_return=True), 'trials': trials}
    metrics = {"eval/success_rate": successes / episodes,
               "eval/mean_steps": float(np.mean(lengths)), "trials": trials}
    if track_cube:
        for key in ('cube_min_goal_distance_m', 'cube_max_lift_m', 'cube_max_displacement_m'):
            metrics[f'eval/mean_{key}'] = float(np.mean([trial[key] for trial in trials]))
    return metrics


def _paths(args):
    if args.collection_source == "small_expo":
        name = "teacher_chunks.hdf5"
    elif args.collection_source == "pi_base":
        name = "pi05_chunks.hdf5"
    else:
        name = "hybrid_teacher_chunks.hdf5"
    mixed = args.output_dir / "pi_base_data" / name
    return mixed, mixed.parent / "success" / name, mixed.parent / "failures" / name


def _start_wandb(args):
    if not args.wandb_project:
        return None
    import wandb

    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    init_kwargs = dict(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name,
        mode=args.wandb_mode,
        dir=str(args.output_dir / "wandb"),
        config=config,
    )
    run_id_file = args.output_dir / "wandb_run_id"
    if args.resume and run_id_file.exists():
        init_kwargs.update(id=run_id_file.read_text().strip(), resume="must")
    run = wandb.init(**init_kwargs)
    run_id_file.write_text(f"{run.id}\n")
    run.define_metric("collection/episode")
    run.define_metric("collection/*", step_metric="collection/episode")
    run.define_metric("sft/step")
    run.define_metric("sft/*", step_metric="sft/step")
    return run


def _collect(args, wandb_run=None):
    mixed, _, _ = _paths(args)
    if args.collection_source == "small_expo":
        if args.teacher_dir is None or args.teacher_step is None:
            raise ValueError("small_expo collection requires --teacher-dir and --teacher-step")
        from dcrl.bc_distill_data import collect_teacher_chunks

        def log_collection(progress):
            if wandb_run is not None:
                wandb_run.log({
                    "collection/episode": progress["episode"],
                    "collection/successes": progress["successes"],
                    "collection/failures": progress["failures"],
                })

        collect_teacher_chunks(
            args.teacher_dir, args.teacher_step, mixed, args.collection_episodes,
            args.eval_seed, target_action_horizon=args.action_horizon,
            target_replan_steps=args.replan_steps,
            target_successes=args.collection_successes,
            target_failures=args.collection_failures,
            max_attempts_override=args.collection_max_attempts,
            progress_callback=log_collection if wandb_run is not None else None)
    elif args.collection_source == "pi_base":
        from dcrl.collect_pi05_libero import collect
        collect(checkpoint=args.source_checkpoint, suite=args.source_suite,
                norm_stats=args.norm_stats, task_id=args.source_task_id,
                episodes=None, seed=args.eval_seed, output=mixed,
                successes=args.collection_successes, failures=args.collection_failures)
    else:
        if args.small_expo_checkpoint is None:
            raise ValueError(
                "pi05_small_expo_editor collection requires --small-expo-checkpoint"
            )
        from dcrl.bc_distill_data import collect_hybrid_teacher_chunks

        def log_collection(progress):
            if wandb_run is not None:
                wandb_run.log({
                    "collection/episode": progress["episode"],
                    "collection/successes": progress["successes"],
                    "collection/failures": progress["failures"],
                })

        collect_hybrid_teacher_chunks(
            args.small_expo_checkpoint, args.pi05_checkpoint, mixed,
            args.collection_episodes, args.eval_seed,
            suite=args.source_suite, task_id=args.source_task_id,
            image_size=args.image_size,
            target_action_horizon=args.action_horizon,
            target_replan_steps=args.replan_steps,
            target_successes=args.collection_successes,
            target_failures=args.collection_failures,
            max_attempts_override=args.collection_max_attempts,
            progress_callback=log_collection if wandb_run is not None else None,
        )
    if wandb_run is not None:
        import h5py

        with h5py.File(mixed, "r") as collection:
            complete = [group for group in collection.values()
                        if bool(group.attrs.get("complete", False))]
            successes = sum(bool(group.attrs.get("success", False)) for group in complete)
            failures = len(complete) - successes
        wandb_run.log({
            "collection/episode": len(complete),
            "collection/successes": successes,
            "collection/failures": failures,
        })
        wandb_run.summary["collection/successes"] = successes
        wandb_run.summary["collection/failures"] = failures


def train(args, wandb_run=None):
    import jax
    import orbax.checkpoint as ocp
    from openpi.shared import download, normalize
    from openpi.training import sharding
    from dcrl.bc_distill_data import ChunkDataset, partition_collection
    from dcrl.best_checkpoint import (
        create_best_checkpoint_manager, evaluation_checkpoint_metrics,
    )
    from dcrl.benchmarks import asset_id, is_ocbench, make_env, max_episode_steps, validate_eval_episodes
    from dcrl.agents.my_expo_learner import restore_checkpoint, save_checkpoint

    ocbench = is_ocbench(args.source_suite)
    task_prompt = getattr(args, 'ocbench_task_prompt', None)
    if ocbench:
        from dcrl.benchmark_demo_chunks import OCBenchDemoChunkDataset as DemoDataset
    else:
        from dcrl.official_demo_chunks import OfficialDemoChunkDataset as DemoDataset

    demo_tasks = []
    if args.collection_source == 'benchmark_demos':
        demo_tasks = [args.source_task_id]
        kwargs = dict(image_size=args.image_size, action_horizon=args.action_horizon,
                      replan_steps=args.replan_steps)
        if ocbench:
            from dcrl.ocbench_env import TASKS, demo_path as ocbench_demo_path, environment_id

            if args.benchmark_demo_task_ids:
                selection = (range(len(TASKS)) if args.benchmark_demo_task_ids == 'all'
                             else map(int, args.benchmark_demo_task_ids.split(',')))
                demo_tasks.extend(sorted(set(selection) - {args.source_task_id}))
            if args.eval_benchmark_demo_tasks and 27 in demo_tasks:
                raise ValueError('Bowling uses normalized return and cannot enter a success-rate average')
            paths = [ocbench_demo_path(environment_id(args.source_suite, task), args.official_demo_root)
                     for task in demo_tasks]
            demo_path = paths[0]
            instruction = task_prompt or TASKS[args.source_task_id][1]
            dataset = DemoDataset(demo_path, instruction=instruction,
                                  additional_tasks=[(path, TASKS[task][1])
                                                    for path, task in zip(paths[1:], demo_tasks[1:])],
                                  **kwargs)
        else:
            with contextlib.closing(make_env(args.source_suite, args.source_task_id,
                                            seed=args.eval_seed, image_size=args.image_size)) as demo_env:
                demo_path = demo_env.demo_path(args.official_demo_root)
                dataset = DemoDataset(demo_path, instruction=demo_env.instruction, **kwargs)
        collection = dict(source='benchmark_demos', path=str(demo_path), suite=args.source_suite,
                          task_id=args.source_task_id, action_horizon=args.action_horizon,
                          replan_steps=args.replan_steps)
        if len(demo_tasks) > 1:
            collection['demo_task_ids'] = demo_tasks
    else:
        mixed, success_file, failure_file = _paths(args)
        if not success_file.exists():
            partition_collection(mixed, success_file, failure_file)
        dataset = ChunkDataset(
            [success_file] + ([failure_file] if args.include_failures else []),
            success_only=not args.include_failures)
        collection = dataset.metadata
    if (ocbench or is_ocbench(collection.get('suite'))) and (
            collection.get('suite') != args.source_suite or int(collection.get('task_id', -1)) != args.source_task_id):
        raise ValueError('OCBench training data must match --source-suite and --source-task-id.')
    if args.collection_source == "small_expo":
        if args.teacher_dir is None or args.teacher_step is None:
            raise ValueError("small_expo training requires --teacher-dir and --teacher-step")
        if collection.get("teacher_dir") != str(args.teacher_dir.resolve()):
            raise ValueError("dataset teacher directory does not match the requested one")
        if int(collection["checkpoint_step"]) != args.teacher_step:
            raise ValueError("dataset teacher checkpoint does not match the requested one")
    elif args.collection_source == "pi_base" and collection.get("source") != "pi05_base_actor":
        raise ValueError("dataset was not collected from the Pi0.5 base actor")
    elif args.collection_source == "pi_base" and collection.get("checkpoint") != args.source_checkpoint:
        raise ValueError("dataset Pi0.5 checkpoint does not match the requested source checkpoint")
    elif args.collection_source == "pi05_small_expo_editor":
        if args.small_expo_checkpoint is None:
            raise ValueError("hybrid training requires --small-expo-checkpoint")
        if collection.get("source") != "pi05_small_expo_editor":
            raise ValueError("dataset was not collected from the Pi0.5+Small EXPO hybrid")
        if collection.get("small_expo_checkpoint") != str(args.small_expo_checkpoint.resolve()):
            raise ValueError("dataset Small EXPO checkpoint does not match the requested checkpoint")
        if collection.get("pi05_checkpoint") != args.pi05_checkpoint:
            raise ValueError("dataset Pi0.5 checkpoint does not match the requested hybrid source")
    horizon = collection.get("target_action_horizon", collection.get("action_horizon"))
    replan = collection.get("target_replan_steps", collection.get("replan_steps"))
    if horizon != args.action_horizon or replan != args.replan_steps:
        raise ValueError("dataset action horizon or replanning interval differs from the learner")
    if getattr(dataset, 'action_horizon', horizon) != args.action_horizon:
        raise ValueError("dataset action horizon does not match the learner")
    if int(collection.get("image_size", args.image_size)) != args.image_size:
        raise ValueError("dataset image size differs from --image-size")

    manifest = {k: (str(v) if isinstance(v, Path) else v)
                for k, v in vars(args).items() if k not in ("phase", "resume", "num_sft_steps")}
    for optional in ('eval_max_steps', 'benchmark_demo_task_ids', 'eval_benchmark_demo_tasks',
                     'skip_initial_eval', 'ocbench_reference_action_normalization', 'ocbench_task_prompt'):
        if not manifest.get(optional):
            manifest.pop(optional, None)
    if args.checkpoint_policy == 'best':
        manifest.pop('checkpoint_policy')
    # SFT runs may load these assets directly from the project filesystem or
    # through OpenPI's local cache. Record the same canonical asset identity so
    # changing between those equivalent paths does not invalidate a resume.
    for key in ("source_checkpoint", "student_checkpoint", "norm_stats", "pi05_checkpoint"):
        if key in manifest:
            manifest[key] = _manifest_openpi_asset_identity(manifest[key])
    manifest.update(collection=collection, loss="full_chunk_flow_matching",
                    student_action_horizon=args.action_horizon,
                    student_replan_steps=args.replan_steps,
                    official_pi05_libero=not ocbench)
    manifest_file = args.output_dir / "sft_config.json"
    if args.resume:
        if not manifest_file.exists() or json.loads(manifest_file.read_text()) != manifest:
            raise ValueError("resume configuration differs from saved SFT configuration")
    elif manifest_file.exists():
        raise FileExistsError("output exists; use --resume")

    stats_dir = args.output_dir / 'student_assets' / asset_id(args.source_suite)
    if not args.resume:
        source = download.maybe_download(args.norm_stats)
        stats_dir.mkdir(parents=True, exist_ok=True)
        (stats_dir / "norm_stats.json").write_bytes(Path(source).read_bytes())
    if ocbench and 'norm_sha256' in collection:
        stats_hash = hashlib.sha256((stats_dir / 'norm_stats.json').read_bytes()).hexdigest()
        if collection['norm_sha256'] != stats_hash:
            raise ValueError('The OCBench SFT normalization must match the teacher collection.')
    stats = normalize.load(stats_dir)
    state_dim = 7 if ocbench else 8
    if len(stats['actions'].mean) != 7 or len(stats['state'].mean) != state_dim:
        raise ValueError(f'Expected normalization with 7 actions and {state_dim} state values')

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
    task_id = int(collection.get('task_id', args.source_task_id))
    score_key = 'eval/normalized_return' if ocbench and task_id == 27 else 'eval/success_rate'
    env = make_env(suite_name,
                        task_id,
                        seed=args.eval_seed, image_size=args.image_size,
                        benchmark_eval=True)
    if task_prompt:
        env.instruction = dataset.instruction
    validate_eval_episodes(env, args.eval_episodes)

    demo_dataset = None

    manager = None
    if args.checkpoint_policy == 'best':
        manager = create_best_checkpoint_manager(args.output_dir / 'checkpoints', score_key=score_key)
    elif args.checkpoint_policy == 'final':
        manager = ocp.CheckpointManager(
            args.output_dir / 'checkpoints',
            item_handlers={'agent': ocp.PyTreeCheckpointHandler(), 'params': ocp.PyTreeCheckpointHandler()},
            options=ocp.CheckpointManagerOptions(max_to_keep=1, create=True),
        )
    try:
        if args.official_demo_ratio > 0:
            demo_path = env.demo_path(
                str(args.official_demo_root) if args.official_demo_root is not None else None
            )
            demo_dataset = DemoDataset(
                demo_path, instruction=env.instruction, image_size=args.image_size,
                action_horizon=args.action_horizon, replan_steps=args.replan_steps,
            )
            if env.instruction != dataset.instruction:
                raise ValueError("official-demo task instruction differs from the SmallExpo target instruction")
        print(
            "SFT data loaded: "
            f"small_expo_chunks={len(dataset.indices)}; "
            f"training_episodes={getattr(dataset, 'episode_count', 'unavailable')}; "
            f"official_demo_chunks={len(demo_dataset) if demo_dataset is not None else 0}; "
            f"official_demo_episodes={demo_dataset.episode_count if demo_dataset is not None else 0}; "
            f"official_demo_ratio={args.official_demo_ratio}; "
            f"official_demo_path={demo_dataset.path if demo_dataset is not None else 'disabled'}",
            flush=True,
        )
        if wandb_run is not None:
            wandb_run.summary["sft/small_expo_chunks"] = len(dataset.indices)
            wandb_run.summary["sft/official_demo_chunks"] = len(demo_dataset) if demo_dataset is not None else 0
            wandb_run.summary["sft/official_demo_episodes"] = demo_dataset.episode_count if demo_dataset is not None else 0
            wandb_run.summary["sft/official_demo_ratio"] = args.official_demo_ratio
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

        best_eval_score, best_eval_step = float("-inf"), -1
        if latest is not None:
            latest_eval = args.output_dir / "evals" / f"{latest:08d}.json"
            if latest_eval.exists():
                try:
                    record = json.loads(latest_eval.read_text())
                    best_eval_score = float(record[score_key])
                    best_eval_step = int(record.get("step", latest))
                except (KeyError, ValueError, json.JSONDecodeError):
                    pass

        def evaluate_and_save(step):
            def evaluate_task(task_env, current_task):
                return _evaluate_actor(
                    learner, task_env, episodes=args.eval_episodes,
                    max_steps=args.eval_max_steps or max_episode_steps(suite_name, current_task),
                    replan_steps=args.replan_steps, seed=args.eval_seed)

            result = evaluate_task(env, task_id)
            if args.eval_benchmark_demo_tasks:
                per_task = {task_id: result}
                for extra_task in demo_tasks[1:]:
                    with contextlib.closing(make_env(
                        suite_name, extra_task, seed=args.eval_seed,
                        image_size=args.image_size, benchmark_eval=True,
                    )) as extra_env:
                        per_task[extra_task] = evaluate_task(extra_env, extra_task)
                result = {
                    'eval/success_rate': float(np.mean([
                        row['eval/success_rate'] for row in per_task.values()
                    ])),
                    'eval/mean_steps': float(np.mean([
                        row['eval/mean_steps'] for row in per_task.values()
                    ])),
                    'task_results': per_task,
                }
            eval_dir = args.output_dir / "evals"
            eval_dir.mkdir(exist_ok=True)
            (eval_dir / f"{step:08d}.json").write_text(
                json.dumps({"step": step, **result}, indent=2))
            return {key: value for key, value in result.items() if key.startswith('eval/')}

        def save_best_evaluation(step, result):
            nonlocal best_eval_score, best_eval_step
            if manager is None or (args.checkpoint_policy == 'final' and step != args.num_sft_steps):
                return
            if score_key not in result:
                return
            score = float(result[score_key])
            # Keep the first checkpoint attaining the best score.  Do not
            # replace it with a later evaluation that merely ties it.
            if args.checkpoint_policy == 'best' and score <= best_eval_score:
                return
            save_checkpoint(
                manager, learner, step,
                metrics=evaluation_checkpoint_metrics(score, step, score_key=score_key),
            )
            manager.wait_until_finished()
            if step in manager.all_steps():
                best_eval_score, best_eval_step = score, step
                if wandb_run is not None:
                    wandb_run.summary[score_key.replace('eval/', 'best_eval/', 1)] = best_eval_score
                    wandb_run.summary["best_eval/step"] = best_eval_step

        if latest is None and not args.skip_initial_eval:
            initial_eval = evaluate_and_save(0)
            if args.collection_source != 'benchmark_demos':
                save_best_evaluation(0, initial_eval)
            if wandb_run is not None:
                wandb_run.log({"sft/step": 0, **{f"sft/{key}": value
                                                for key, value in initial_eval.items()}})
        with (args.output_dir / "metrics.jsonl").open("a") as log:
            for step in tqdm(range(start + 1, args.num_sft_steps + 1),
                             total=args.num_sft_steps, initial=start,
                             desc="Learner BC"):
                rng = np.random.default_rng(np.random.SeedSequence([args.seed, step]))
                from dcrl.official_demo_chunks import sample_mixed_sft_rows
                sampled_rows, target_count, demo_count = sample_mixed_sft_rows(
                    dataset, demo_dataset, batch_size=args.batch_size,
                    official_demo_ratio=args.official_demo_ratio, rng=rng,
                )
                rows = [transform(row) for row in sampled_rows]
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
                metrics["data/small_expo_chunks"] = float(target_count)
                metrics["data/official_demo_chunks"] = float(demo_count)
                if not all(np.isfinite(v) for v in metrics.values()):
                    raise FloatingPointError(f"nonfinite BC metrics at step {step}: {metrics}")
                evaluation = None
                if step % args.eval_every == 0 or step == args.num_sft_steps:
                    evaluation = evaluate_and_save(step)
                    metrics.update(evaluation)
                    save_best_evaluation(step, evaluation)
                log.write(json.dumps({"step": step, **metrics}) + "\n")
                log.flush()
                if (wandb_run is not None
                        and (step % args.wandb_log_every == 0 or evaluation is not None)):
                    wandb_metrics = {"sft/step": step}
                    for key, value in metrics.items():
                        if key.startswith("eval/"):
                            wandb_metrics[f"sft/{key}"] = value
                        else:
                            wandb_metrics[f"sft/train/{key}"] = value
                    wandb_run.log(wandb_metrics)
    finally:
        if manager is not None:
            manager.close()
        env.close()
        dataset.close()
        if demo_dataset is not None:
            demo_dataset.close()


def main():
    args = parse_args()
    from dcrl.benchmarks import is_ocbench

    if args.eval_max_steps < 0:
        raise ValueError('--eval-max-steps must be nonnegative.')
    if args.collection_source == 'benchmark_demos' and (
            args.phase != 'train' or args.official_demo_ratio or args.official_demo_root is None):
        raise ValueError('Use --phase train and --official-demo-root for benchmark_demos.')
    if args.benchmark_demo_task_ids and (not is_ocbench(args.source_suite)
                                        or args.collection_source != 'benchmark_demos'):
        raise ValueError('--benchmark-demo-task-ids requires OCBench benchmark_demos training.')
    if is_ocbench(args.source_suite):
        paths = [args.norm_stats, args.student_checkpoint]
        if args.collection_source == 'pi05_small_expo_editor':
            paths.append(args.pi05_checkpoint)
        if any('pi05_libero' in path or 'physical-intelligence/libero' in path for path in paths):
            raise ValueError('Provide OCBench normalization and non-LIBERO Pi0.5 checkpoint paths.')
    args.output_dir = args.output_dir.resolve()
    if args.teacher_dir is not None:
        args.teacher_dir = args.teacher_dir.resolve()
    if args.small_expo_checkpoint is not None:
        from dcrl.bc_distill_data import _resolve_small_expo_checkpoint

        _, checkpoint_root, checkpoint_step = _resolve_small_expo_checkpoint(
            args.small_expo_checkpoint
        )
        args.small_expo_checkpoint = (checkpoint_root / str(checkpoint_step)).resolve()
    if args.official_demo_root is not None:
        args.official_demo_root = args.official_demo_root.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    wandb_run = _start_wandb(args)
    exit_code = 0
    try:
        if args.phase in ("all", "collect"):
            _collect(args, wandb_run)
        if args.phase in ("all", "train"):
            train(args, wandb_run)
    except BaseException:
        exit_code = 1
        raise
    finally:
        if wandb_run is not None:
            wandb_run.finish(exit_code=exit_code)


if __name__ == "__main__":
    main()

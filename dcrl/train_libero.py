"""Synchronous EXPO-FT training of SmallExpoLearner on LIBERO.

The control structure follows upstream train_pi_robo.py: a single environment
step loop with chunk inference, executed-action insertion, configurable update
trigger, evaluation, and checkpoint checks. The learner itself retains the
upstream EXPO critic, flow-actor, edit-policy, and temperature update rules.
"""

from __future__ import annotations

import argparse
from collections import deque
import json
import logging
from pathlib import Path
import pickle
import shutil
import subprocess

import numpy as np
from tqdm import tqdm

from dcrl.libero_env import DEFAULT_TASK_IDS, LIBERO_SUITE_MAX_STEPS, LiberoTaskEnv
from dcrl.libero_replay import (
    ExpoLiberoReplay, Transition, compute_norm_stats, concatenate_batches,
    load_demonstrations,
)


# Replay snapshots contain the complete online buffer (including raw camera
# arrays), so an uncompressed snapshot is roughly 3 GB at the current replay
# capacity. Keep every snapshot for exact resumption, but stream it through
# zstd when available so the 1M-step runs do not exhaust shared storage.
_ZSTD = shutil.which("zstd")
if _ZSTD is None:
    _fallback_zstd = Path("/iris/u/ronpo/miniconda3/bin/zstd")
    if _fallback_zstd.exists():
        _ZSTD = str(_fallback_zstd)


class _ReplayUnpickler(pickle.Unpickler):
    """Load replay snapshots written with either NumPy core module layout."""

    def find_class(self, module, name):
        # NumPy 2.x serializes some helpers under numpy._core, while the
        # current training image exposes them under numpy.core.  Keep replay
        # snapshots portable without changing their serialized contents.
        if module == "numpy._core" or module.startswith("numpy._core."):
            module = "numpy.core" + module[len("numpy._core"): ]
        return super().find_class(module, name)


def _load_pickle(file):
    return _ReplayUnpickler(file).load()


def _save_replay_state(task_dir: Path, step: int, state: dict, *, slot: str | None = None) -> None:
    stem = f"replay_{slot}" if slot is not None else f"replay_{step}"
    if _ZSTD is None:
        path = task_dir / f"{stem}.pkl"
        temp = task_dir / f"{stem}.pkl.tmp"
        with temp.open("wb") as file:
            pickle.dump(state, file, protocol=pickle.HIGHEST_PROTOCOL)
        temp.replace(path)
        return

    path = task_dir / f"{stem}.pkl.zst"
    temp = task_dir / f"{stem}.pkl.zst.tmp"
    with temp.open("wb") as output:
        process = subprocess.Popen(
            [_ZSTD, "-1", "-T0", "-q", "-c"],
            stdin=subprocess.PIPE,
            stdout=output,
        )
        try:
            pickle.dump(state, process.stdin, protocol=pickle.HIGHEST_PROTOCOL)
            process.stdin.close()
            return_code = process.wait()
        except BaseException:
            process.kill()
            process.wait()
            raise
    if return_code != 0:
        temp.unlink(missing_ok=True)
        raise RuntimeError(f"zstd replay compression failed with exit code {return_code}")
    temp.replace(path)


def _load_replay_state(task_dir: Path, step: int, *, slot: str | None = None) -> dict:
    stem = f"replay_{slot}" if slot is not None else f"replay_{step}"
    compressed = task_dir / f"{stem}.pkl.zst"
    if compressed.exists():
        if _ZSTD is None:
            raise RuntimeError(f"Cannot read compressed replay without zstd: {compressed}")
        process = subprocess.Popen(
            [_ZSTD, "-d", "-q", "-c", str(compressed)],
            stdout=subprocess.PIPE,
        )
        try:
            state = _load_pickle(process.stdout)
        finally:
            process.stdout.close()
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"zstd replay decompression failed with exit code {return_code}")
        return state

    path = task_dir / f"{stem}.pkl"
    with path.open("rb") as file:
        return _load_pickle(file)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', default='libero_90')
    parser.add_argument('--task-id', type=int, default=DEFAULT_TASK_IDS[0],
                        help='One zero-based LIBERO task ID per run (default: 60)')
    parser.add_argument('--online-rl-steps', dest='online_rl_steps', type=int, default=10000, help='Online RL environment actions')
    parser.add_argument('--offline-rl-steps', type=int, default=0,
                        help='EXPO learner updates from demos before collecting online actions')
    parser.add_argument('--init-distill-steps', type=int, default=0,
                        help='Pi0.5 base-chunk distillation updates before EXPO offline/RL')
    parser.add_argument('--distill-data-root', type=str,
                        default='/nlp/scr/ronpo/dcrlcheckpoints/pi_base_data/fixed125',
                        help='Directory containing pi05_task_<id>_*.hdf5 collections')
    parser.add_argument('--distill-data-path', type=str, default=None,
                        help='Explicit completed Pi0.5 mixed HDF5 collection')
    parser.add_argument('--distill-success-only', action='store_true',
                        help='Diagnostic option: use only successful Pi0.5 chunks')
    parser.add_argument(
        '--norm-stats-dir',
        default='/scratch/bgqe/rpolonsky/dcrlcheckpoints/openpi_cache/openpi-assets/checkpoints/pi05_libero/assets/physical-intelligence/libero',
        help='Local directory containing the official Pi0.5 LIBERO norm_stats.json; never recomputed from demos')
    parser.add_argument('--pi-distill-eval-interval', type=int, default=None,
                        help='Pi0.5-to-compact distillation eval cadence; defaults to offline-rl-eval-interval')
    parser.add_argument('--pi-distill-chckpt-interval', type=int, default=None,
                        help='Pi0.5-to-compact distillation checkpoint cadence; defaults to offline-rl-checkpoint-interval')
    parser.add_argument('--max-episode-steps', type=int, default=None,
                        help='Training action limit; default matches the official suite limit')
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--action-horizon', type=int, default=10)
    parser.add_argument('--replan-steps', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--utd-ratio', type=int, default=20)
    parser.add_argument('--update-type', choices=('episode', 'step', 'batch'), default='episode')
    parser.add_argument('--num-updates', type=int, default=1)
    parser.add_argument('--num-batch', type=int, default=1, help='Episodes per batch update')
    parser.add_argument('--warmup-episodes', type=int, default=10)
    parser.add_argument('--offline-ratio', type=float, default=0.0)
    parser.add_argument('--num-demos', type=int, default=0, help='0 loads all task demonstrations')
    parser.add_argument('--demos-root')
    parser.add_argument('--replay-capacity', type=int, default=10000)
    parser.add_argument('--actor-lr', type=float, default=3e-4)
    parser.add_argument('--critic-lr', type=float, default=3e-4)
    parser.add_argument('--temp-lr', type=float, default=3e-4)
    parser.add_argument('--edit-scale', type=float, default=0.2)
    parser.add_argument('--adjust-target-entropy', action='store_true')
    parser.add_argument('--offline-rl-eval-interval', type=int, default=500,
                        help='Evaluate every N offline updates and at phase end; 0 disables')
    parser.add_argument('--offline-rl-checkpoint-interval', type=int, default=5000,
                        help='Checkpoint every N offline updates; 0 disables')
    parser.add_argument('--online-rl-eval-interval', type=int, default=1000)
    # Use the same 20 fixed benchmark starts as the SFT/official Pi-facing
    # evaluation defaults.  The evaluator still preserves their official
    # ordering and reset semantics.
    parser.add_argument('--eval-episodes', type=int, default=20)
    parser.add_argument('--eval-seed', type=int, default=7)
    parser.add_argument('--online-rl-checkpot-interval', type=int, default=5000)
    parser.add_argument('--wandb-entity', default='distill_rl')
    parser.add_argument('--wandb-project', default='small_learner')
    parser.add_argument('--wandb-name', default=None)
    parser.add_argument('--wandb-mode', choices=('online', 'offline', 'disabled'), default='online')
    parser.add_argument('--output-dir', type=Path, default=Path('runs/small_expo_libero'))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--resume', action='store_true')
    return parser.parse_args(argv)



def _norm_stats(task_dir: Path, demo_path: Path, *, state_dim: int, overwrite: bool):
    from openpi.shared import normalize
    path = task_dir / 'norm_stats.json'
    if path.exists() and not overwrite:
        stats = normalize.load(task_dir)
    else:
        stats = compute_norm_stats(demo_path)
        normalize.save(task_dir, stats)
    for name, width in (('state', state_dim), ('actions', 7)):
        if name not in stats or stats[name].q01 is None or stats[name].q99 is None:
            raise ValueError(f'Missing OpenPI quantile stats for {name}: {path}')
        if np.asarray(stats[name].q01).shape != (width,) or np.asarray(stats[name].q99).shape != (width,):
            raise ValueError(f'Wrong {name} stats dimension in {path}')
    return stats


def _pi05_norm_stats(task_dir: Path, paths: list[Path], *, state_dim: int,
                     overwrite: bool):
    """Load the shared official Pi0.5 stats embedded in collector files.

    Pi0.5 collections are produced with one shared LIBERO statistics file.  A
    task-local demo estimate can have a zero quantile range (for example the
    task-28 gripper dimension), which makes quantile normalization amplify
    small Pi0.5 actions by 1e6.  Reusing the collector's stats keeps the
    distillation targets and the compact learner in the same coordinate
    system for all three phases.
    """
    from openpi.shared import normalize
    import h5py

    path = task_dir / 'norm_stats.json'
    if path.exists() and not overwrite:
        stats = normalize.load(task_dir)
    else:
        with h5py.File(paths[0], 'r') as file:
            encoded = file.attrs.get('normalization_stats')
        if encoded is None:
            raise ValueError(f'Pi0.5 collection has no embedded normalization stats: {paths[0]}')
        if isinstance(encoded, bytes):
            encoded = encoded.decode()
        stats = normalize.deserialize_json(str(encoded))
        normalize.save(task_dir, stats)
    for name, width in (('state', state_dim), ('actions', 7)):
        if name not in stats or stats[name].q01 is None or stats[name].q99 is None:
            raise ValueError(f'Missing Pi0.5 quantile stats for {name}: {path}')
        if np.asarray(stats[name].q01).shape != (width,) or np.asarray(stats[name].q99).shape != (width,):
            raise ValueError(f'Wrong {name} stats dimension in {path}')
    return stats


def _official_norm_stats(task_dir: Path, source_dir: Path, *, state_dim: int):
    """Load and record the same fixed stats used by sft_agnostic.

    The SFT trainer uses the official ``pi05_libero`` stats, downloaded from
    the checkpoint assets.  RL must use that exact file too: estimating stats
    from task demonstrations changes the action/state coordinate system.
    """
    from openpi.shared import normalize

    source_dir = Path(source_dir)
    source_file = source_dir / 'norm_stats.json'
    if not source_file.is_file():
        raise FileNotFoundError(
            f'Official Pi0.5 normalization file is missing: {source_file}. '
            'Populate the local checkpoint cache before starting training.')
    stats = normalize.load(source_dir)
    # Store an exact local copy alongside the run for resumption/audit.  This
    # is a copy of the official file, not a recomputation from demonstrations.
    destination = task_dir / 'norm_stats.json'
    destination.write_bytes(source_file.read_bytes())
    for name, width in (('state', state_dim), ('actions', 7)):
        if name not in stats or stats[name].q01 is None or stats[name].q99 is None:
            raise ValueError(f'Missing official quantile stats for {name}: {source_file}')
        if np.asarray(stats[name].q01).shape != (width,) or np.asarray(stats[name].q99).shape != (width,):
            raise ValueError(f'Wrong official {name} stats dimension in {source_file}')
    logging.info(
        'Using official Pi0.5 LIBERO normalization from %s; '
        'state.q01=%s state.q99=%s actions.q01=%s actions.q99=%s',
        source_file,
        np.asarray(stats['state'].q01).tolist(), np.asarray(stats['state'].q99).tolist(),
        np.asarray(stats['actions'].q01).tolist(), np.asarray(stats['actions'].q99).tolist())
    return stats


def _device_batch(batch, sharding):
    import jax
    return jax.tree_util.tree_map(lambda value: jax.device_put(np.asarray(value), sharding), batch)


def _metrics(info):
    return {name: float(np.asarray(value)) for name, value in info.items()
            if np.asarray(value).size == 1}


def _resolve_distill_paths(args, task_id: int) -> list[Path]:
    if args.distill_data_path:
        return [Path(args.distill_data_path)]
    root = Path(args.distill_data_root)
    paths = sorted(
        path for path in root.glob(f'pi05_task_{task_id}_*.hdf5')
        if not path.name.endswith('.partial.hdf5')
    )
    if not paths:
        raise FileNotFoundError(
            f'No completed Pi0.5 collection for task {task_id} under {root}'
        )
    if len(paths) > 1:
        raise ValueError(
            f'Multiple Pi0.5 collections match task {task_id}: {paths}; '
            'pass --distill-data-path explicitly'
        )
    return paths


def _sample_batches(online, offline, *, batch_size, utd_ratio, offline_ratio,
                    actor_success_only, rng):
    total = batch_size * utd_ratio
    offline_count = int(total * offline_ratio)
    online_count = total - offline_count
    critic_online = online.sample(online_count) if online_count else None
    critic_offline = offline.sample(offline_count) if offline_count else None
    critic = concatenate_batches(critic_online, critic_offline, rng)

    if not actor_success_only:
        return critic, None
    offline_actor_count = int(batch_size * offline_ratio)
    online_actor_count = batch_size - offline_actor_count
    actor_online = (online.sample(online_actor_count, success_only=True)
                    if online_actor_count and online.can_sample(success_only=True) else None)
    actor_offline = (offline.sample(offline_actor_count, success_only=True)
                     if offline_actor_count and offline.can_sample(success_only=True) else None)
    if actor_online is None and offline.can_sample(success_only=True):
        actor_offline = offline.sample(batch_size, success_only=True)
    elif actor_offline is None and online.can_sample(success_only=True):
        actor_online = online.sample(batch_size, success_only=True)
    return critic, concatenate_batches(actor_online, actor_offline, rng)


def _evaluate(agent, env, *, episodes: int, max_steps: int, replan_steps: int, seed: int):
    import jax
    eval_agent = agent.replace(rng=jax.random.PRNGKey(seed))
    # Reuse the same benchmark initial states at every checkpoint.
    env.episode_index = 0
    env.env.seed(env.seed)
    successes = 0
    lengths = []
    for _ in range(episodes):
        obs = env.reset()
        plan = deque()
        for step in range(max_steps):
            if not plan:
                chunk, eval_agent, _ = eval_agent.sample_actions(obs)
                plan.extend(np.asarray(jax.device_get(chunk))[:replan_steps])
            obs, _, success, _ = env.step(plan.popleft())
            if success:
                successes += 1
                lengths.append(step + 1)
                break
        else:
            lengths.append(max_steps)
    return {'eval/success_rate': successes / episodes,
                   'eval/mean_steps': float(np.mean(lengths))}


def _evaluate_actor(agent, env, *, episodes: int, max_steps: int, replan_steps: int, seed: int):
    """Evaluate only the base actor using the official LIBERO rollout loop.

    Initialization distillation trains the compact base actor with BC.  Its
    eval must therefore not use the compact learner's critic/edit candidates;
    otherwise the reported score would include components that were not
    trained by the distillation update.  The reset, fixed initial-state order,
    horizon, and action-prefix execution remain identical to ``_evaluate``.
    """
    import jax
    eval_agent = agent.replace(rng=jax.random.PRNGKey(seed))
    env.episode_index = 0
    env.env.seed(env.seed)
    successes = 0
    lengths = []
    for _ in range(episodes):
        obs = env.reset()
        plan = deque()
        for step in range(max_steps):
            if not plan:
                chunk, eval_agent, _ = eval_agent.eval_actor_only(obs)
                plan.extend(np.asarray(jax.device_get(chunk))[:replan_steps])
            obs, _, success, _ = env.step(plan.popleft())
            if success:
                successes += 1
                lengths.append(step + 1)
                break
        else:
            lengths.append(max_steps)
    return {'eval/actor_success_rate': successes / episodes,
            'eval/actor_mean_steps': float(np.mean(lengths))}


def _save(checkpoint_manager, agent, online, offline, *, task_dir: Path, step: int,
          offline_steps_done: int, online_steps_done: int,
          completed_episodes: int, episode_id: int, episodes_since_update: int, rng,
          distill_steps_done: int = 0, distill_rng=None, slot: str | None = None):
    from dcrl.agents.pi_expo_learner import save_checkpoint
    state = dict(rows=online.rows, replay_rng=online.rng.bit_generator.state,
                 offline_rng=offline.rng.bit_generator.state,
                 combine_rng=rng.bit_generator.state,
                 distill_rng=(None if distill_rng is None else distill_rng.bit_generator.state),
                 completed_episodes=completed_episodes, episode_id=episode_id,
                 episodes_since_update=episodes_since_update,
                 distill_steps_done=distill_steps_done,
                 offline_steps_done=offline_steps_done, online_steps_done=online_steps_done)
    _save_replay_state(task_dir, step, state, slot=slot)
    # Publish the model checkpoint last: a visible checkpoint always has replay.
    save_checkpoint(checkpoint_manager, agent, step)
    checkpoint_manager.wait_until_finished()


def _load_best_record(task_dir: Path) -> dict:
    path = task_dir / 'best_record.json'
    if not path.exists():
        return {'score': None, 'step': None}
    try:
        record = json.loads(path.read_text())
        return {'score': record.get('score'), 'step': record.get('step')}
    except (OSError, json.JSONDecodeError):
        logging.warning('Could not read best record %s; treating best as unset', path)
        return {'score': None, 'step': None}


def _save_best_record(task_dir: Path, *, score: float, step: int) -> None:
    path = task_dir / 'best_record.json'
    temp = task_dir / 'best_record.json.tmp'
    temp.write_text(json.dumps({'score': float(score), 'step': int(step)}, indent=2) + '\n')
    temp.replace(path)


def train_task(args, suite: str, task_id: int, wandb_run=None):
    import jax
    from openpi.training import sharding as openpi_sharding
    from dcrl.agents.pi_expo_learner import restore_checkpoint
    from dcrl.agents.small_expo_learner import SmallExpoLearner
    from dcrl.expo.agents.alg.agent import initialize_checkpoint_dir

    official_max_steps = LIBERO_SUITE_MAX_STEPS[suite]
    train_max_steps = args.max_episode_steps or official_max_steps
    task_dir = args.output_dir / suite / f'task_{task_id:02d}'
    task_dir.mkdir(parents=True, exist_ok=True)
    last_checkpoint_dir = task_dir / 'checkpoints_last'
    best_checkpoint_dir = task_dir / 'checkpoints_best'
    if (last_checkpoint_dir.exists() or best_checkpoint_dir.exists()) and not (args.resume or args.overwrite):
        raise FileExistsError(f'Checkpoint slots already exist under {task_dir}')
    config_path = task_dir / 'run_config.json'
    fixed = {key: getattr(args, key) for key in (
        'seed', 'image_size', 'action_horizon', 'replan_steps', 'batch_size',
        'utd_ratio', 'offline_ratio', 'num_demos', 'replay_capacity',
        'max_episode_steps', 'demos_root', 'update_type', 'num_updates', 'num_batch',
        'warmup_episodes', 'init_distill_steps', 'distill_data_root',
        'distill_data_path', 'distill_success_only', 'norm_stats_dir', 'pi_distill_eval_interval',
        'pi_distill_chckpt_interval', 'offline_rl_steps', 'offline_rl_eval_interval',
        'online_rl_eval_interval', 'eval_episodes', 'eval_seed', 'online_rl_checkpot_interval', 'offline_rl_checkpoint_interval', 'online_rl_steps',
        'actor_lr', 'critic_lr', 'temp_lr', 'edit_scale', 'adjust_target_entropy')}
    fixed.update(suite=suite, task_id=task_id)
    if args.resume:
        if not config_path.exists():
            raise ValueError(f'Resume requested but configuration is missing: {config_path}')
        existing_config = json.loads(config_path.read_text())
        # A checkpoint can be used as a warm start for any later phase.
        # The phase budgets may be changed (including set to zero), while all
        # model, data, and optimization settings must remain compatible.
        phase_keys = {
            'init_distill_steps', 'offline_rl_steps', 'online_rl_steps',
            'offline_rl_checkpoint_interval', 'online_rl_checkpot_interval',
        }
        phase_transition = (
            {k: v for k, v in existing_config.items() if k not in phase_keys}
                == {k: v for k, v in fixed.items() if k not in phase_keys}
        )
        if existing_config != fixed and not phase_transition:
            raise ValueError(f'Resume configuration differs from {config_path}')
    else:
        config_path.write_text(json.dumps(fixed, indent=2) + '\n')
    env = LiberoTaskEnv(suite, task_id, seed=args.seed + task_id,
                        image_size=args.image_size)
    pi_distill_eval_interval = (
        args.pi_distill_eval_interval
        if args.pi_distill_eval_interval is not None
        else args.offline_rl_eval_interval
    )
    pi_distill_chckpt_interval = (
        args.pi_distill_chckpt_interval
        if args.pi_distill_chckpt_interval is not None
        else args.offline_rl_checkpoint_interval
    )
    eval_enabled = (
        args.online_rl_eval_interval
        or (args.offline_rl_steps and args.offline_rl_eval_interval)
        or (args.init_distill_steps and pi_distill_eval_interval)
    )
    eval_env = (LiberoTaskEnv(suite, task_id, seed=args.eval_seed,
                              image_size=args.image_size, benchmark_eval=True)
                if eval_enabled and args.eval_episodes else None)
    manager = None
    best_manager = None
    try:
        if eval_env is not None and args.eval_episodes > len(eval_env.init_states):
            raise ValueError(f'Official evaluation has only {len(eval_env.init_states)} initial states')
        obs = env.reset()
        demo_path = env.demo_path(args.demos_root)
        distill_paths = (
            _resolve_distill_paths(args, task_id) if args.init_distill_steps else None
        )
        stats = _official_norm_stats(
            task_dir, Path(args.norm_stats_dir), state_dim=len(obs['state']))
        demos = load_demonstrations(
            demo_path, instruction=env.instruction, image_size=args.image_size,
            max_episodes=args.num_demos or None)
        if not demos:
            raise ValueError(f'No demonstration transitions: {demo_path}')

        mesh = openpi_sharding.make_mesh(1)
        data_sharding = jax.sharding.NamedSharding(
            mesh, jax.sharding.PartitionSpec(openpi_sharding.DATA_AXIS))
        replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        agent = SmallExpoLearner.from_examples(
            seed=args.seed + task_id, example_observation=obs,
            example_action=np.zeros(7, np.float32), example_state=obs['state'],
            mesh=mesh, data_sharding=data_sharding,
            replicated_sharding=replicated_sharding,
            image_size=args.image_size, action_horizon=args.action_horizon,
            replan_steps=args.replan_steps, actor_success_only=True,
            actor_lr=args.actor_lr, critic_lr=args.critic_lr, temp_lr=args.temp_lr,
            edit_scale=args.edit_scale,
            adjust_target_entropy=args.adjust_target_entropy,
            norm_stats=stats)
        replay_kwargs = dict(
            capacity=args.replay_capacity, action_horizon=args.action_horizon,
            replan_steps=args.replan_steps,
            padded_action_dim=agent.actor.model_config.action_dim,
            discount=float(agent.discount), norm_stats=stats)
        online = ExpoLiberoReplay(**replay_kwargs, seed=args.seed + task_id)
        offline = ExpoLiberoReplay(**replay_kwargs, seed=args.seed + 1000 + task_id)
        for demo in demos:
            offline.insert_episode(demo)
        del demos

        distill_replay = None
        if args.init_distill_steps:
            from dcrl.pi05_distill_data import Pi05DistillReplay
            distill_replay = Pi05DistillReplay(
                distill_paths,
                norm_stats=stats,
                action_horizon=args.action_horizon,
                padded_action_dim=agent.actor.model_config.action_dim,
                seed=args.seed + 3000 + task_id,
                success_only=args.distill_success_only,
            )

        manager, resuming = initialize_checkpoint_dir(
            last_checkpoint_dir, keep_period=None,
            overwrite=args.overwrite, resume=args.resume, max_to_keep=1)
        best_manager, _ = initialize_checkpoint_dir(
            best_checkpoint_dir, keep_period=None,
            overwrite=args.overwrite, resume=args.resume, max_to_keep=1)
        best_record = _load_best_record(task_dir) if args.resume else {'score': None, 'step': None}
        checkpoint_step = 0
        distill_steps_done = 0
        offline_steps_done = 0
        online_steps_done = 0
        completed_episodes = 0
        episode_id = 0
        episodes_since_update = 0
        combine_rng = np.random.default_rng(args.seed + 2000 + task_id)
        if resuming:
            steps = tuple(manager.all_steps())
            if not steps:
                raise ValueError('Resume requested but no model checkpoint exists')
            checkpoint_step = max(steps)
            agent = restore_checkpoint(manager, agent, checkpoint_step).cache_infer_params()
            try:
                state = _load_replay_state(task_dir, checkpoint_step, slot='last')
            except FileNotFoundError:
                state = _load_replay_state(task_dir, checkpoint_step)
            online.rows = state['rows']
            online.rng.bit_generator.state = state['replay_rng']
            offline.rng.bit_generator.state = state['offline_rng']
            if distill_replay is not None and state.get('distill_rng') is not None:
                distill_replay.rng.bit_generator.state = state['distill_rng']
            episodes_since_update = state['episodes_since_update']
            distill_steps_done = state.get('distill_steps_done', 0)
            offline_steps_done = state['offline_steps_done']
            online_steps_done = state['online_steps_done']
            if checkpoint_step != distill_steps_done + offline_steps_done + online_steps_done:
                raise ValueError('Checkpoint phase counters do not match its step')
            combine_rng.bit_generator.state = state['combine_rng']
            completed_episodes = state['completed_episodes']
            episode_id = state['episode_id']
            if online_steps_done:
                env.episode_index = completed_episodes
                obs = env.reset()
            logging.info('Task %d resumed: distill=%d offline=%d online=%d',
                         task_id, distill_steps_done, offline_steps_done, online_steps_done)
        # A zero phase budget means "do not run this phase". If a resumed
        # checkpoint already contains work in that phase, retain its counter
        # as the global-step offset and skip additional updates. Positive
        # budgets remain absolute per-phase targets for exact resumption.
        init_target = (distill_steps_done if args.resume and args.init_distill_steps == 0
                       else args.init_distill_steps)
        offline_target = (offline_steps_done if args.resume and args.offline_rl_steps == 0
                          else args.offline_rl_steps)
        online_target = (online_steps_done if args.resume and args.online_rl_steps == 0
                         else args.online_rl_steps)
        if resuming and offline_steps_done and args.init_distill_steps not in (0, distill_steps_done):
            raise ValueError(
                'cannot change the distillation target after offline RL has started; '
                'pass --init-distill-steps 0 to skip it'
            )
        if resuming and online_steps_done and args.offline_rl_steps not in (0, offline_steps_done):
            raise ValueError(
                'cannot change the offline RL target after online RL has started; '
                'pass --offline-rl-steps 0 to skip it'
            )
        if distill_steps_done > init_target:
            raise ValueError('checkpoint has more distillation steps than requested')
        if offline_steps_done > offline_target:
            raise ValueError('checkpoint has more offline steps than requested')
        if online_steps_done > online_target:
            raise ValueError('checkpoint has more online steps than requested')
        logging.info('Phase targets: distill=%d offline=%d online=%d',
                     init_target, offline_target, online_target)

        # Select the optimizer state that belongs to the phase being resumed.
        # A checkpoint always contains both actor optimizer states; the phase
        # counter determines which one is active, including the case where
        # init_distill_steps=0 intentionally skips remaining distillation.
        agent = agent.activate_actor_optimizer(
            'distill' if distill_steps_done < init_target else 'rl'
        )

        def maybe_save_best(metrics, step):
            """Save the full best slot whenever eval score is not lower."""
            nonlocal best_record
            score = metrics.get('eval/success_rate')
            if score is None:
                return
            score = float(score)
            previous = best_record.get('score')
            # Ties intentionally move best to the most recent step.
            if previous is not None and score < float(previous):
                return
            _save(
                best_manager, agent, online, offline, task_dir=task_dir,
                step=step,
                distill_steps_done=distill_steps_done,
                offline_steps_done=offline_steps_done,
                online_steps_done=online_steps_done,
                completed_episodes=completed_episodes,
                episode_id=episode_id,
                episodes_since_update=episodes_since_update,
                rng=combine_rng,
                distill_rng=(None if distill_replay is None else distill_replay.rng),
                slot='best')
            _save_best_record(task_dir, score=score, step=step)
            best_record = {'score': score, 'step': int(step)}
            logging.info('Updated best eval score %.4f at global step %d', score, step)

        plan = deque()
        episode_steps = 0
        last_saved_step = checkpoint_step
        log_mode = 'a' if resuming else 'w'
        with (task_dir / 'metrics.jsonl').open(log_mode) as log_file:
            # Initialization distillation uses the same tqdm/eval/checkpoint
            # structure as the EXPO offline phase, but calls bc_update so only
            # the compact base actor learns from frozen Pi0.5 chunks.
            if args.init_distill_steps and distill_replay is None:
                raise RuntimeError('distillation replay was not initialized')
            distill_progress = tqdm(
                range(distill_steps_done, init_target),
                desc=f'{suite} task {task_id} init-distill')
            for distill_index in distill_progress:
                distill_batch = distill_replay.sample(args.batch_size)
                agent, info = agent.bc_update(
                    _device_batch(distill_batch, data_sharding)
                )
                distill_steps_done = distill_index + 1
                global_step = distill_steps_done
                metrics = dict(
                    phase='init_distill',
                    step=distill_steps_done,
                    global_step=global_step,
                    **_metrics(info),
                )
                if (eval_env is not None and pi_distill_eval_interval
                        and (distill_steps_done % pi_distill_eval_interval == 0
                             or distill_steps_done == init_target)):
                    metrics.update(_evaluate_actor(
                        agent, eval_env, episodes=args.eval_episodes,
                        max_steps=official_max_steps, replan_steps=args.replan_steps,
                        seed=args.eval_seed))
                log_file.write(json.dumps(metrics) + '\n')
                log_file.flush()
                if wandb_run is not None:
                    wandb_run.log(metrics, step=global_step)
                if (pi_distill_chckpt_interval
                        and distill_steps_done % pi_distill_chckpt_interval == 0):
                    _save(
                        manager, agent, online, offline, task_dir=task_dir,
                        step=global_step,
                        distill_steps_done=distill_steps_done,
                        offline_steps_done=offline_steps_done,
                        online_steps_done=online_steps_done,
                        completed_episodes=completed_episodes,
                        episode_id=episode_id,
                        episodes_since_update=episodes_since_update,
                        rng=combine_rng,
                        distill_rng=distill_replay.rng,
                        slot='last',
                    )
                    last_saved_step = global_step

            if init_target and last_saved_step < distill_steps_done:
                _save(
                    manager, agent, online, offline, task_dir=task_dir,
                    step=distill_steps_done,
                    distill_steps_done=distill_steps_done,
                    offline_steps_done=offline_steps_done,
                    online_steps_done=online_steps_done,
                    completed_episodes=completed_episodes,
                    episode_id=episode_id,
                    episodes_since_update=episodes_since_update,
                    rng=combine_rng,
                    distill_rng=distill_replay.rng,
                    slot='last',
                )
                last_saved_step = distill_steps_done

            # Distillation and RL use different actor optimizer histories.
            # Switch once, after the distillation loop, before either offline
            # or online RL updates. Offline and online RL then share this RL
            # optimizer state.
            agent = agent.activate_actor_optimizer('rl')

            # Offline initialization uses the same EXPO update API, but all
            # critic and success-only actor examples come from demonstrations.
            demo_replay = offline
            if offline_target and not demo_replay.can_sample(success_only=True):
                raise ValueError('Offline initialization needs at least r+1 demo transitions')
            offline_progress = tqdm(
                range(offline_steps_done, offline_target),
                desc=f'{suite} task {task_id} offline')
            for offline_index in offline_progress:
                critic = demo_replay.sample(args.batch_size * args.utd_ratio)
                actor = demo_replay.sample(args.batch_size, success_only=True)
                agent, info = agent.update(
                    agent, _device_batch(critic, data_sharding), args.utd_ratio,
                    _device_batch(actor, data_sharding))
                offline_steps_done = offline_index + 1
                global_step = distill_steps_done + offline_steps_done
                metrics = dict(phase='offline', step=offline_steps_done,
                               global_step=global_step, **_metrics(info))
                if (eval_env is not None and args.offline_rl_eval_interval
                        and (offline_steps_done % args.offline_rl_eval_interval == 0
                             or offline_steps_done == offline_target)):
                    metrics.update(_evaluate(
                        agent, eval_env, episodes=args.eval_episodes,
                        max_steps=official_max_steps, replan_steps=args.replan_steps,
                        seed=args.eval_seed))
                maybe_save_best(metrics, global_step)
                log_file.write(json.dumps(metrics) + '\n')
                log_file.flush()
                if wandb_run is not None:
                    wandb_run.log(metrics, step=global_step)
                if args.offline_rl_checkpoint_interval and offline_steps_done % args.offline_rl_checkpoint_interval == 0:
                    _save(manager, agent, online, offline, task_dir=task_dir,
                          step=global_step,
                          distill_steps_done=distill_steps_done,
                          offline_steps_done=offline_steps_done,
                          online_steps_done=online_steps_done,
                          completed_episodes=completed_episodes, episode_id=episode_id,
                          episodes_since_update=episodes_since_update,
                          rng=combine_rng,
                          distill_rng=(None if distill_replay is None else distill_replay.rng),
                          slot='last')
                    last_saved_step = global_step

            if offline_target and last_saved_step < distill_steps_done + offline_target:
                _save(manager, agent, online, offline, task_dir=task_dir,
                      step=distill_steps_done + offline_target,
                      distill_steps_done=distill_steps_done,
                      offline_steps_done=offline_target,
                      online_steps_done=online_steps_done,
                      completed_episodes=completed_episodes, episode_id=episode_id,
                      episodes_since_update=episodes_since_update,
                      rng=combine_rng,
                      distill_rng=(None if distill_replay is None else distill_replay.rng),
                      slot='last')
                last_saved_step = distill_steps_done + offline_target

            # Keep both schedules on the online-step counter. With episode
            # updates, run each due eval/checkpoint at the next completed
            # episode, after its learner update.
            next_eval_online = (
                (online_steps_done // args.online_rl_eval_interval + 1) * args.online_rl_eval_interval
                if args.online_rl_eval_interval else None)
            next_checkpoint_online = (
                (online_steps_done // args.online_rl_checkpot_interval + 1) * args.online_rl_checkpot_interval
                if args.online_rl_checkpot_interval else None)
            progress = tqdm(
                range(online_steps_done, online_target),
                desc=f'{suite} task {task_id} online')
            for index in progress:
                step = index + 1
                global_step = distill_steps_done + offline_target + step
                if not plan:
                    chunk, agent, _ = agent.sample_actions(obs)
                    plan.extend(np.asarray(jax.device_get(chunk))[:args.replan_steps])
                proposed = plan.popleft()
                next_obs, reward, success, executed = env.step(proposed)
                episode_steps += 1
                done = success or episode_steps >= train_max_steps or step == online_target
                online.insert(Transition(obs=obs, action=executed, reward=reward,
                                         done=done, episode_id=episode_id))
                if done and success:
                    online.mark_episode_success(episode_id)
                obs = next_obs

                total_critic = args.batch_size * args.utd_ratio
                offline_count = int(total_critic * args.offline_ratio)
                online_count = total_critic - offline_count
                critic_ready = ((online_count == 0 or online.can_sample())
                                and (offline_count == 0 or offline.can_sample()))
                success_ready = (online.can_sample(success_only=True)
                                 or offline.can_sample(success_only=True))
                can_update = (completed_episodes >= args.warmup_episodes
                              and step >= args.batch_size
                              and critic_ready and success_ready)
                metrics = dict(phase='online', step=step, global_step=global_step,
                               episode=episode_id, reward=reward,
                               success=bool(success), replay_steps=len(online))
                trigger = ((args.update_type == 'step')
                           or (done and args.update_type == 'episode')
                           or (done and args.update_type == 'batch'
                               and episodes_since_update + 1 >= args.num_batch))
                if can_update and trigger:
                    for _ in range(args.num_updates):
                        critic, actor = _sample_batches(
                            online, offline, batch_size=args.batch_size,
                            utd_ratio=args.utd_ratio, offline_ratio=args.offline_ratio,
                            actor_success_only=agent.actor_success_only, rng=combine_rng)
                        agent, info = agent.update(
                            agent, _device_batch(critic, data_sharding), args.utd_ratio,
                            _device_batch(actor, data_sharding) if actor is not None else None)
                    metrics.update(_metrics(info))

                online_steps_done = step
                if done:
                    metrics['episode_steps'] = episode_steps
                    completed_episodes += 1
                    episode_id += 1
                    episode_steps = 0
                    plan.clear()
                    if args.update_type == 'batch' and can_update:
                        episodes_since_update = 0 if trigger else episodes_since_update + 1
                    if step < online_target:
                        obs = env.reset()
                    progress.set_postfix(episodes=completed_episodes, success=int(success))

                if (done and eval_env is not None and next_eval_online is not None
                        and step >= next_eval_online):
                    metrics.update(_evaluate(
                        agent, eval_env, episodes=args.eval_episodes,
                        max_steps=official_max_steps, replan_steps=args.replan_steps,
                        seed=args.eval_seed))
                    maybe_save_best(metrics, global_step)
                    next_eval_online = (step // args.online_rl_eval_interval + 1) * args.online_rl_eval_interval

                if done and next_checkpoint_online is not None and step >= next_checkpoint_online:
                    _save(manager, agent, online, offline, task_dir=task_dir,
                          step=global_step,
                          distill_steps_done=distill_steps_done,
                          offline_steps_done=offline_steps_done,
                          online_steps_done=online_steps_done,
                          completed_episodes=completed_episodes, episode_id=episode_id,
                          episodes_since_update=episodes_since_update,
                          rng=combine_rng,
                          distill_rng=(None if distill_replay is None else distill_replay.rng),
                          slot='last')
                    last_saved_step = global_step
                    next_checkpoint_online = (
                        (step // args.online_rl_checkpot_interval + 1) * args.online_rl_checkpot_interval)

                log_file.write(json.dumps(metrics) + '\n')
                log_file.flush()
                if wandb_run is not None:
                    wandb_run.log(metrics, step=global_step)

            final_step = distill_steps_done + offline_target + online_target
            if last_saved_step != final_step and final_step > 0:
                _save(manager, agent, online, offline, task_dir=task_dir,
                      step=final_step,
                      distill_steps_done=distill_steps_done,
                      offline_steps_done=offline_steps_done,
                      online_steps_done=online_steps_done,
                      completed_episodes=completed_episodes, episode_id=episode_id,
                      episodes_since_update=episodes_since_update,
                      rng=combine_rng,
                      distill_rng=(None if distill_replay is None else distill_replay.rng),
                      slot='last')
    finally:
        if manager is not None:
            manager.wait_until_finished()
            manager.close()
        if best_manager is not None:
            best_manager.wait_until_finished()
            best_manager.close()
        env.close()
        if eval_env is not None:
            eval_env.close()
        if distill_replay is not None:
            distill_replay.close()


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    if args.online_rl_steps < 0 or args.batch_size < 1 or args.utd_ratio < 1 or args.num_updates < 1:
        raise ValueError('online-rl-steps must be nonnegative; batch-size, utd-ratio, and num-updates must be positive')
    if (args.max_episode_steps is not None and args.max_episode_steps < 1) or args.num_batch < 1 or args.replay_capacity < 2:
        raise ValueError('max-episode-steps, num-batch, and replay-capacity must be positive')
    if not 1 <= args.replan_steps <= args.action_horizon:
        raise ValueError('replan-steps must be in [1, action-horizon]')
    if not 0 <= args.offline_ratio <= 1:
        raise ValueError('offline-ratio must be in [0, 1]')
    if args.actor_lr <= 0 or args.critic_lr <= 0 or args.temp_lr <= 0 or args.edit_scale <= 0:
        raise ValueError('actor-lr, critic-lr, temp-lr, and edit-scale must be positive')
    if (args.init_distill_steps < 0 or args.offline_rl_steps < 0 or args.offline_rl_eval_interval < 0
            or args.pi_distill_eval_interval is not None and args.pi_distill_eval_interval < 0
            or args.pi_distill_chckpt_interval is not None and args.pi_distill_chckpt_interval < 0
            or args.offline_rl_checkpoint_interval < 0
            or args.warmup_episodes < 0 or args.num_demos < 0 or args.online_rl_eval_interval < 0
            or args.eval_episodes < 0 or args.online_rl_checkpot_interval < 0):
        raise ValueError('phase steps, warmup, demo count, eval settings, and checkpoint intervals must be nonnegative')
    if args.resume and args.overwrite:
        raise ValueError('resume and overwrite cannot be combined')
    if args.online_rl_steps + args.offline_rl_steps + args.init_distill_steps == 0:
        raise ValueError('at least one distillation, offline, or online step is required')
    from libero.libero import benchmark
    if args.suite not in LIBERO_SUITE_MAX_STEPS:
        raise ValueError(f'Unknown official LIBERO suite: {args.suite}')
    suite = benchmark.get_benchmark_dict()[args.suite]()
    if not 0 <= args.task_id < suite.get_num_tasks():
        raise ValueError(f'Task ID must be in [0, {suite.get_num_tasks()})')
    import wandb
    run_name = args.wandb_name or f'small_expo_{args.suite}_task_{args.task_id:02d}'
    wandb_run = wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        name=run_name,
        mode=args.wandb_mode,
        config={key: (str(value) if isinstance(value, Path) else value)
                for key, value in vars(args).items()
                if key not in {'overwrite', 'resume'}},
        resume='allow' if args.resume else None,
    )
    try:
        train_task(args, args.suite, args.task_id, wandb_run=wandb_run)
    finally:
        wandb_run.finish()


if __name__ == '__main__':
    main()

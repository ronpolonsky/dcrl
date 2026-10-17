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


def _save_replay_state(task_dir: Path, step: int, state: dict) -> None:
    if _ZSTD is None:
        path = task_dir / f"replay_{step}.pkl"
        temp = task_dir / f"replay_{step}.pkl.tmp"
        with temp.open("wb") as file:
            pickle.dump(state, file, protocol=pickle.HIGHEST_PROTOCOL)
        temp.replace(path)
        return

    path = task_dir / f"replay_{step}.pkl.zst"
    temp = task_dir / f"replay_{step}.pkl.zst.tmp"
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


def _load_replay_state(task_dir: Path, step: int) -> dict:
    compressed = task_dir / f"replay_{step}.pkl.zst"
    if compressed.exists():
        if _ZSTD is None:
            raise RuntimeError(f"Cannot read compressed replay without zstd: {compressed}")
        process = subprocess.Popen(
            [_ZSTD, "-d", "-q", "-c", str(compressed)],
            stdout=subprocess.PIPE,
        )
        try:
            state = pickle.load(process.stdout)
        finally:
            process.stdout.close()
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"zstd replay decompression failed with exit code {return_code}")
        return state

    path = task_dir / f"replay_{step}.pkl"
    with path.open("rb") as file:
        return pickle.load(file)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', default='libero_90')
    parser.add_argument('--task-id', type=int, default=DEFAULT_TASK_IDS[0],
                        help='One zero-based LIBERO task ID per run (default: 60)')
    parser.add_argument('--num-steps', type=int, default=10000, help='Online environment actions')
    parser.add_argument('--num-offline-steps', type=int, default=0,
                        help='EXPO learner updates from demos before collecting online actions')
    parser.add_argument('--max-episode-steps', type=int, default=None,
                        help='Training action limit; default matches the official suite limit')
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--action-horizon', type=int, default=16)
    parser.add_argument('--replan-steps', type=int, default=8)
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
    parser.add_argument('--offline-eval-interval', type=int, default=500,
                        help='Evaluate every N offline updates and at phase end; 0 disables')
    parser.add_argument('--eval-every', type=int, default=1000)
    parser.add_argument('--eval-episodes', type=int, default=50)
    parser.add_argument('--eval-seed', type=int, default=7)
    parser.add_argument('--checkpoint-every', type=int, default=5000)
    parser.add_argument('--wandb-entity', default='distill_rl')
    parser.add_argument('--wandb-project', default='dcrl')
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


def _device_batch(batch, sharding):
    import jax
    return jax.tree_util.tree_map(lambda value: jax.device_put(np.asarray(value), sharding), batch)


def _metrics(info):
    return {name: float(np.asarray(value)) for name, value in info.items()
            if np.asarray(value).size == 1}


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


def _save(checkpoint_manager, agent, online, offline, *, task_dir: Path, step: int,
          offline_steps_done: int, online_steps_done: int,
          completed_episodes: int, episode_id: int, episodes_since_update: int, rng):
    from dcrl.agents.pi_expo_learner import save_checkpoint
    state = dict(rows=online.rows, replay_rng=online.rng.bit_generator.state,
                 offline_rng=offline.rng.bit_generator.state,
                 combine_rng=rng.bit_generator.state,
                 completed_episodes=completed_episodes, episode_id=episode_id,
                 episodes_since_update=episodes_since_update,
                 offline_steps_done=offline_steps_done, online_steps_done=online_steps_done)
    _save_replay_state(task_dir, step, state)
    # Publish the model checkpoint last: a visible checkpoint always has replay.
    save_checkpoint(checkpoint_manager, agent, step)
    checkpoint_manager.wait_until_finished()


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
    if (task_dir / 'checkpoints').exists() and not (args.resume or args.overwrite):
        raise FileExistsError(f'Checkpoint directory exists: {task_dir / "checkpoints"}')
    config_path = task_dir / 'run_config.json'
    fixed = {key: getattr(args, key) for key in (
        'seed', 'image_size', 'action_horizon', 'replan_steps', 'batch_size',
        'utd_ratio', 'offline_ratio', 'num_demos', 'replay_capacity',
        'max_episode_steps', 'demos_root', 'update_type', 'num_updates', 'num_batch',
        'warmup_episodes', 'num_offline_steps', 'offline_eval_interval',
        'eval_every', 'eval_episodes', 'eval_seed', 'checkpoint_every', 'num_steps')}
    fixed.update(suite=suite, task_id=task_id)
    if args.resume:
        if not config_path.exists() or json.loads(config_path.read_text()) != fixed:
            raise ValueError(f'Resume configuration differs from {config_path}')
    else:
        config_path.write_text(json.dumps(fixed, indent=2) + '\n')
    env = LiberoTaskEnv(suite, task_id, seed=args.seed + task_id,
                        image_size=args.image_size)
    eval_enabled = args.eval_every or (args.num_offline_steps and args.offline_eval_interval)
    eval_env = (LiberoTaskEnv(suite, task_id, seed=args.eval_seed,
                              image_size=args.image_size, benchmark_eval=True)
                if eval_enabled and args.eval_episodes else None)
    manager = None
    try:
        if eval_env is not None and args.eval_episodes > len(eval_env.init_states):
            raise ValueError(f'Official evaluation has only {len(eval_env.init_states)} initial states')
        obs = env.reset()
        demo_path = env.demo_path(args.demos_root)
        stats = _norm_stats(task_dir, demo_path, state_dim=len(obs['state']),
                            overwrite=args.overwrite)
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

        manager, resuming = initialize_checkpoint_dir(
            task_dir / 'checkpoints', keep_period=None,
            overwrite=args.overwrite, resume=args.resume)
        checkpoint_step = 0
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
            state = _load_replay_state(task_dir, checkpoint_step)
            online.rows = state['rows']
            online.rng.bit_generator.state = state['replay_rng']
            offline.rng.bit_generator.state = state['offline_rng']
            episodes_since_update = state['episodes_since_update']
            offline_steps_done = state['offline_steps_done']
            online_steps_done = state['online_steps_done']
            if checkpoint_step != offline_steps_done + online_steps_done:
                raise ValueError('Checkpoint phase counters do not match its step')
            combine_rng.bit_generator.state = state['combine_rng']
            completed_episodes = state['completed_episodes']
            episode_id = state['episode_id']
            if online_steps_done:
                env.episode_index = completed_episodes
                obs = env.reset()
            logging.info('Task %d resumed: offline=%d online=%d',
                         task_id, offline_steps_done, online_steps_done)
        if online_steps_done >= args.num_steps:
            raise ValueError('num-steps must exceed the resumed online step')

        plan = deque()
        episode_steps = 0
        last_saved_step = checkpoint_step
        log_mode = 'a' if resuming else 'w'
        with (task_dir / 'metrics.jsonl').open(log_mode) as log_file:
            # Offline initialization uses the same EXPO update API, but all
            # critic and success-only actor examples come from demonstrations.
            demo_replay = offline
            if args.num_offline_steps and not demo_replay.can_sample(success_only=True):
                raise ValueError('Offline initialization needs at least r+1 demo transitions')
            offline_progress = tqdm(
                range(offline_steps_done, args.num_offline_steps),
                desc=f'{suite} task {task_id} offline')
            for offline_index in offline_progress:
                critic = demo_replay.sample(args.batch_size * args.utd_ratio)
                actor = demo_replay.sample(args.batch_size, success_only=True)
                agent, info = agent.update(
                    agent, _device_batch(critic, data_sharding), args.utd_ratio,
                    _device_batch(actor, data_sharding))
                offline_steps_done = offline_index + 1
                metrics = dict(phase='offline', step=offline_steps_done,
                               global_step=offline_steps_done, **_metrics(info))
                if (eval_env is not None and args.offline_eval_interval
                        and (offline_steps_done % args.offline_eval_interval == 0
                             or offline_steps_done == args.num_offline_steps)):
                    metrics.update(_evaluate(
                        agent, eval_env, episodes=args.eval_episodes,
                        max_steps=official_max_steps, replan_steps=args.replan_steps,
                        seed=args.eval_seed))
                log_file.write(json.dumps(metrics) + '\n')
                log_file.flush()
                if wandb_run is not None:
                    wandb_run.log(metrics, step=offline_steps_done)
                if args.checkpoint_every and offline_steps_done % args.checkpoint_every == 0:
                    _save(manager, agent, online, offline, task_dir=task_dir,
                          step=offline_steps_done,
                          offline_steps_done=offline_steps_done,
                          online_steps_done=online_steps_done,
                          completed_episodes=completed_episodes, episode_id=episode_id,
                          episodes_since_update=episodes_since_update,
                          rng=combine_rng)
                    last_saved_step = offline_steps_done

            if args.num_offline_steps and last_saved_step < args.num_offline_steps:
                _save(manager, agent, online, offline, task_dir=task_dir,
                      step=args.num_offline_steps,
                      offline_steps_done=args.num_offline_steps,
                      online_steps_done=online_steps_done,
                      completed_episodes=completed_episodes, episode_id=episode_id,
                      episodes_since_update=episodes_since_update,
                          rng=combine_rng)
                last_saved_step = args.num_offline_steps

            # Keep both schedules on the online-step counter. With episode
            # updates, run each due eval/checkpoint at the next completed
            # episode, after its learner update.
            next_eval_online = (
                (online_steps_done // args.eval_every + 1) * args.eval_every
                if args.eval_every else None)
            next_checkpoint_online = (
                (online_steps_done // args.checkpoint_every + 1) * args.checkpoint_every
                if args.checkpoint_every else None)
            progress = tqdm(
                range(online_steps_done, args.num_steps),
                desc=f'{suite} task {task_id} online')
            for index in progress:
                step = index + 1
                global_step = args.num_offline_steps + step
                if not plan:
                    chunk, agent, _ = agent.sample_actions(obs)
                    plan.extend(np.asarray(jax.device_get(chunk))[:args.replan_steps])
                proposed = plan.popleft()
                next_obs, reward, success, executed = env.step(proposed)
                episode_steps += 1
                done = success or episode_steps >= train_max_steps or step == args.num_steps
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
                    if step < args.num_steps:
                        obs = env.reset()
                    progress.set_postfix(episodes=completed_episodes, success=int(success))

                if (done and eval_env is not None and next_eval_online is not None
                        and step >= next_eval_online):
                    metrics.update(_evaluate(
                        agent, eval_env, episodes=args.eval_episodes,
                        max_steps=official_max_steps, replan_steps=args.replan_steps,
                        seed=args.eval_seed))
                    next_eval_online = (step // args.eval_every + 1) * args.eval_every

                if done and next_checkpoint_online is not None and step >= next_checkpoint_online:
                    _save(manager, agent, online, offline, task_dir=task_dir,
                          step=global_step,
                          offline_steps_done=offline_steps_done,
                          online_steps_done=online_steps_done,
                          completed_episodes=completed_episodes, episode_id=episode_id,
                          episodes_since_update=episodes_since_update,
                          rng=combine_rng)
                    last_saved_step = global_step
                    next_checkpoint_online = (
                        (step // args.checkpoint_every + 1) * args.checkpoint_every)

                log_file.write(json.dumps(metrics) + '\n')
                log_file.flush()
                if wandb_run is not None:
                    wandb_run.log(metrics, step=global_step)

            final_step = args.num_offline_steps + args.num_steps
            if last_saved_step != final_step:
                _save(manager, agent, online, offline, task_dir=task_dir,
                      step=final_step,
                      offline_steps_done=offline_steps_done,
                      online_steps_done=online_steps_done,
                      completed_episodes=completed_episodes, episode_id=episode_id,
                      episodes_since_update=episodes_since_update,
                          rng=combine_rng)
    finally:
        if manager is not None:
            manager.wait_until_finished()
            manager.close()
        env.close()
        if eval_env is not None:
            eval_env.close()


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    if args.num_steps < 1 or args.batch_size < 1 or args.utd_ratio < 1 or args.num_updates < 1:
        raise ValueError('num-steps, batch-size, utd-ratio, and num-updates must be positive')
    if (args.max_episode_steps is not None and args.max_episode_steps < 1) or args.num_batch < 1 or args.replay_capacity < 2:
        raise ValueError('max-episode-steps, num-batch, and replay-capacity must be positive')
    if not 1 <= args.replan_steps <= args.action_horizon:
        raise ValueError('replan-steps must be in [1, action-horizon]')
    if not 0 <= args.offline_ratio <= 1:
        raise ValueError('offline-ratio must be in [0, 1]')
    if args.num_offline_steps < 0 or args.offline_eval_interval < 0 or args.warmup_episodes < 0 or args.num_demos < 0 or args.eval_every < 0 or args.eval_episodes < 0 or args.checkpoint_every < 0:
        raise ValueError('offline steps, warmup, demo count, eval settings, and checkpoint interval must be nonnegative')
    if args.resume and args.overwrite:
        raise ValueError('resume and overwrite cannot be combined')
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

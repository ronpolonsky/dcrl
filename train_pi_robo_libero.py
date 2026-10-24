#! /usr/bin/env python
"""Online RL fine-tuning of a Pi0.5 policy on a LIBERO task.

This file starts from Expo-FT's ``train_pi_robo.py``.  The learner, replay
buffer, batch processor, update schedule, checkpointing, and logging loop are
unchanged; only the DROID dataset/client boundary is replaced by the local
LIBERO environment and demonstrations.
"""
import logging
import os
import threading
import json
from pathlib import Path
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import tqdm
from absl import app, flags

from ml_collections import config_flags

import jax
import etils.epath as epath

import wandb
from expo_ft.agents import initialize_checkpoint_dir
from expo_ft.agents.alg.batch_utils import CRITIC_CAMERA_KEYS
from expo_ft.data.replay_buffer import create_replay_buffer, _critic_key_to_storage
from expo_ft.data.batch_processor import BatchProcessor
from expo_ft.utils.log_utils import EpisodeState, TrainingStats
from expo_ft.utils.loop_utils import AsyncChunkSampler, EpisodeSink
from expo_ft.utils.timer import Timer
from expo_ft.utils.train_utils import (
    get_batch_info, init_logging, init_wandb, set_compilation_cache_dir,
)
from expo_ft.agents.vla.pi05 import build_pi05

import openpi.training.sharding as openpi_sharding

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

FLAGS = flags.FLAGS

flags.DEFINE_string("project_name", "expo-ft", "wandb project name.")
flags.DEFINE_string("run_name", None, "Optional wandb run name.")
flags.DEFINE_float("offline_ratio", 0.0, "Offline batch fraction; 0 inserts dataset into online replay buffer.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_enum("update_type", "episode", ["episode", "step", "batch"], "When to run gradient updates: per episode, per step, or per batch of episodes.")
flags.DEFINE_integer("num_batch", 1, "Number of episodes per update batch (only used when update_type=batch).")
flags.DEFINE_integer("num_updates", 0, "Upstream-style fixed number of gradient updates per trigger (episode/step/batch). 0 = derive the count from --step_interval instead.")
flags.DEFINE_integer("step_interval", 1, "One gradient update per this many collected transitions. update_type=step runs it every step_interval env steps; episode/batch runs the accumulated equivalent at the episode boundary (remainder carries over).")
flags.DEFINE_integer("batch_size", 64, "Mini batch size.")
flags.DEFINE_integer("max_steps", 500_000, "Number of training steps.")
flags.DEFINE_integer("num_data", 0, "Max number of offline demo episodes to load (0 = all).")
flags.DEFINE_boolean("tqdm", True, "Use tqdm progress bar.")
flags.DEFINE_boolean("checkpoint_model", False, "Save agent checkpoint during training.")
flags.DEFINE_integer("checkpoint_interval", 0, "Save agent checkpoint every N steps. When 0 and checkpoint_model=True, no interval saving (save at end only).")
flags.DEFINE_boolean("checkpoint_buffer", False, "Save agent replay buffer on evaluation.")
flags.DEFINE_integer("utd_ratio", 20, "Update to data ratio.")
flags.DEFINE_integer("keep_period", None, "Keep checkpoints every N steps.")
flags.DEFINE_integer("checkpoint_max_to_keep", 100, "Maximum number of checkpoints retained by Orbax.")
flags.DEFINE_boolean("checkpoint_keep_without_metrics", True, "Keep checkpoints that have no saved metrics.")
flags.DEFINE_boolean("overwrite", False, "Overwrite existing checkpoint directory.")
flags.DEFINE_boolean("resume", False, "Resume training from checkpoint.")
flags.DEFINE_string("output_dir", "./logs", "Directory for logs and checkpoints.")
flags.DEFINE_integer("fsdp_devices", 1, "Number of FSDP devices for sharding.")
flags.DEFINE_integer("eval_interval", 0, "Evaluate on fixed LIBERO initial states every N environment steps; 0 disables.")
flags.DEFINE_integer("eval_episodes", 20, "Number of fixed LIBERO initial states per evaluation.")
flags.DEFINE_integer("eval_seed", 7, "Seed for deterministic fixed-state LIBERO evaluation.")
flags.DEFINE_integer("eval_max_steps", 0, "Evaluation horizon; 0 uses the official suite horizon.")

flags.DEFINE_string("client_host", "0.0.0.0", "Bind host to listen on; the rollout client dials in.")
flags.DEFINE_integer("client_port", 8102, "Bind port to listen on.")

flags.DEFINE_integer("replan_steps", 5, "LIBERO replanning prefix length.")
flags.DEFINE_integer(
    "delay", 0,
    "Simulated main-actor inference latency (env-steps) for RealTimeEXPOFTLearner. "
    "0 <= delay <= replan_steps. >0 enables the delayed + prefix-inpainted "
    "rollout and the matching delayed critic backup. Ignored by other learners.",
)
flags.DEFINE_float("sim_latency", 0.0, "Simulated extra inference latency in ms added to each sample_actions call; 0 disables.")

flags.DEFINE_string("dataset_path", "", "Unused upstream DROID dataset path.")
flags.DEFINE_string("suite", "libero_90", "LIBERO benchmark suite.")
flags.DEFINE_integer("task_id", 29, "Zero-based LIBERO task index.")
flags.DEFINE_string("demos_root", "datasets", "Root containing LIBERO demonstration HDF5 files.")
flags.DEFINE_string("demo_path", "", "Optional explicit LIBERO demonstration HDF5 path.")
flags.DEFINE_integer("image_size", 224, "Pi0.5 input image size.")
flags.DEFINE_integer("max_episode_steps", 400, "LIBERO training episode horizon.")
flags.DEFINE_integer("control_hz", 0, "LIBERO control frequency; zero disables real-time sleeping.")
config_flags.DEFINE_config_file(
    "config",
    "configs/model/expo_ft_pi_libero_config.py",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)




def _flat_libero_observation(obs: dict, instruction: str) -> dict:
    """Convert the local LIBERO adapter output to OpenPI's raw input keys."""
    images = obs["image"]
    # Pi0.5's LIBERO RepackTransform maps these raw keys into the canonical
    # observation/image, observation/wrist_image, and observation/state paths.
    return {
        "image": np.asarray(images["base_0_rgb"], dtype=np.uint8),
        "wrist_image": np.asarray(images["left_wrist_0_rgb"], dtype=np.uint8),
        "state": np.asarray(obs["state"], dtype=np.float32),
        "prompt": instruction,
    }


def _load_libero_dataset(path: Path, instruction: str, image_size: int,
                         max_episodes: int) -> list[dict]:
    """Load aligned LIBERO demonstrations in PiReplayBuffer's input schema."""
    from dcrl.libero_replay import load_demonstrations

    episodes = load_demonstrations(
        path, instruction=instruction, image_size=image_size,
        max_episodes=max_episodes if max_episodes > 0 else None,
    )
    dataset = []
    for episode in episodes:
        for transition in episode:
            dataset.append({
                "observations": _flat_libero_observation(transition.obs, instruction),
                "actions": np.asarray(transition.action, dtype=np.float32),
                "rewards": np.float32(transition.reward),
                "masks": np.float32(not transition.done),
                "dones": np.bool_(transition.done),
            })
    return dataset


class _LiberoEnvClient:
    """Small local adapter exposing the EnvClientWrapper methods used upstream."""

    def __init__(self, suite: str, task_id: int, *, seed: int, image_size: int,
                 max_episode_steps: int, benchmark_eval: bool = False):
        from dcrl.libero_env import LiberoTaskEnv

        self._env = LiberoTaskEnv(
            suite, task_id, seed=seed, image_size=image_size,
            benchmark_eval=benchmark_eval,
        )
        self.env_id = f"{suite}:task_{task_id}"
        self.task_description = self._env.instruction
        self.max_episode_steps = int(max_episode_steps)
        self._observation = None
        self._episode_steps = 0
        self._last_info = (False, False, 0.0, 1.0)
        self.network_ms = 0.0

    def reset(self):
        self._observation = _flat_libero_observation(
            self._env.reset(), self.task_description
        )
        self._episode_steps = 0
        self._last_info = (False, False, 0.0, 1.0)
        return self._observation

    def get_observation(self):
        if self._observation is None:
            return self.reset()
        return self._observation

    def get_info_for_step(self):
        info = self._last_info
        self._last_info = (False, False, 0.0, 1.0)
        return info

    def step(self, action):
        raw_observation, reward, success, executed = self._env.step(np.asarray(action))
        self._observation = _flat_libero_observation(
            raw_observation, self.task_description
        )
        self._episode_steps += 1
        done = bool(success or self._episode_steps >= self.max_episode_steps)
        self._last_info = (done, bool(success), float(reward), float(not done))
        return np.asarray(executed, dtype=np.float32), "policy"

    def pop_network_ms(self):
        value, self.network_ms = self.network_ms, 0.0
        return value

    def close(self):
        self._env.close()


def _evaluate(agent, env, *, episodes: int, max_steps: int, replan_steps: int,
              seed: int):
    """Evaluate the complete EXPO agent on fixed official LIBERO starts.

    Evaluation uses the same ten-action Pi0.5 chunks as training and executes
    five actions before replanning.  The benchmark environment enumerates the
    fixed initial states in order, and resetting its episode index makes every
    checkpoint comparable.
    """
    eval_agent = agent.replace(rng=jax.random.PRNGKey(seed))
    env._env.episode_index = 0
    env._env.env.seed(env._env.seed)
    successes = 0
    lengths = []
    trials = []
    for episode in range(episodes):
        observation = env.reset()
        action_plan = deque()
        for step in range(max_steps):
            if not action_plan:
                chunk, eval_agent, _ = eval_agent.sample_actions(observation)
                action_plan.extend(np.asarray(jax.device_get(chunk))[:replan_steps])
            env.step(action_plan.popleft())
            done, success, _, _ = env.get_info_for_step()
            observation = env.get_observation()
            if success:
                successes += 1
                lengths.append(step + 1)
                trials.append({"episode": episode, "initial_state_index": episode,
                               "success": True, "steps": step + 1})
                break
            if done:
                lengths.append(step + 1)
                trials.append({"episode": episode, "initial_state_index": episode,
                               "success": False, "steps": step + 1})
                break
        else:
            lengths.append(max_steps)
            trials.append({"episode": episode, "initial_state_index": episode,
                           "success": False, "steps": max_steps})
    return {
        "eval/success_rate": float(successes / episodes),
        "eval/mean_steps": float(np.mean(lengths)),
        "trials": trials,
    }


def main(_):
    init_logging()
    assert FLAGS.offline_ratio >= 0.0 and FLAGS.offline_ratio <= 1.0

    if FLAGS.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {FLAGS.batch_size} must be divisible by "
            f"the number of devices {jax.device_count()}"
        )
    set_compilation_cache_dir(f"sync-{FLAGS.config.model_cls}")

    mesh = openpi_sharding.make_mesh(FLAGS.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec(openpi_sharding.DATA_AXIS)
    )
    replicated_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec()
    )

    log_dir = os.path.join(FLAGS.output_dir, FLAGS.run_name)
    os.makedirs(log_dir, exist_ok=True)
    train_video_dir = os.path.join(log_dir, "train_videos")
    os.makedirs(train_video_dir, exist_ok=True)
    checkpoint_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_dir_path = epath.Path(checkpoint_dir)
    checkpoint_manager, resuming = initialize_checkpoint_dir(
        checkpoint_dir_path,
        keep_period=FLAGS.keep_period,
        overwrite=FLAGS.overwrite,
        resume=FLAGS.resume,
        max_to_keep=FLAGS.checkpoint_max_to_keep,
        keep_checkpoints_without_metrics=FLAGS.checkpoint_keep_without_metrics,
    )

    init_wandb(checkpoint_dir_path, resuming, FLAGS.project_name, FLAGS.run_name)
    wandb.config.update(FLAGS.flag_values_dict(), allow_val_change=resuming)

    # LIBERO replaces the upstream DROID dataset/client boundary.  The replay
    # buffer and BatchProcessor below remain the Expo-FT implementations.
    from dcrl.libero_env import LIBERO_SUITE_MAX_STEPS, LiberoTaskEnv

    probe_env = LiberoTaskEnv(
        FLAGS.suite, FLAGS.task_id, seed=FLAGS.seed, image_size=FLAGS.image_size,
        benchmark_eval=True,
    )
    task_description = probe_env.instruction
    demo_path = Path(FLAGS.demo_path) if FLAGS.demo_path else probe_env.demo_path(FLAGS.demos_root)
    probe_env.close()
    dataset = _load_libero_dataset(
        demo_path, task_description, FLAGS.image_size, FLAGS.num_data,
    )
    if not dataset:
        raise ValueError(f"No LIBERO demonstrations loaded from {demo_path}.")
    example_action = dataset[0]['actions'][np.newaxis]

    model_cls = FLAGS.config.model_cls
    # BCLearner uses human-intervention chunks for the actor batch only (no critic).
    use_dagger_hil_sampling = model_cls == "BCLearner"
    if model_cls == "BCLearner":
        from expo_ft.agents.alg.bc import load_agent, restore_checkpoint, save_checkpoint
    elif model_cls == "EXPOLearner":
        from expo_ft.agents.alg.expo_ft import load_agent, restore_checkpoint, save_checkpoint
    elif model_cls == "RealTimeEXPOFTLearner":
        from expo_ft.agents.alg.realtime_expo_ft import load_agent, restore_checkpoint, save_checkpoint
    else:
        raise ValueError(f"Unsupported model class: {model_cls}")

    actor, actor_train_state, target_actor_params, agent_kwargs, vla_metadata = build_pi05(
        FLAGS.config, FLAGS.seed, mesh, data_sharding, replicated_sharding,
        resuming, task_description,
    )

    # LIBERO has the two official cameras used by Pi0.5; the Pi replay buffer
    # pads the absent right-wrist view according to the official model transform.
    critic_camera_keys = ("base_0_rgb", "left_wrist_0_rgb")
    rb_args = dict(
        config=FLAGS.config,
        example_action=example_action,
        capacity=FLAGS.max_steps,
        task_description=task_description,
        replan_steps=FLAGS.replan_steps,
        seed=FLAGS.seed,
        delay=FLAGS.delay,
        critic_camera_keys=critic_camera_keys,
    )
    replay_buffer = create_replay_buffer(**rb_args)
    offline_replay_buffer = create_replay_buffer(**rb_args)

    actor_success_only = getattr(FLAGS.config, "actor_success_only", False)
    batch_processor = BatchProcessor(
        replay_buffer=replay_buffer,
        offline_replay_buffer=offline_replay_buffer,
        data_sharding=data_sharding,
        batch_size=FLAGS.batch_size,
        utd_ratio=FLAGS.utd_ratio,
        offline_ratio=FLAGS.offline_ratio,
        actor_success_only=actor_success_only,
        use_dagger_hil_sampling=use_dagger_hil_sampling,
        dataset=dataset,
    )

    critic_example = {
        _critic_key_to_storage(k): offline_replay_buffer.dataset_dict[_critic_key_to_storage(k)][0][np.newaxis]
        for k in critic_camera_keys
    }
    critic_example["state"] = offline_replay_buffer.dataset_dict['state'][0][np.newaxis]
    critic_example["actions"] = offline_replay_buffer.dataset_dict['actions'][0][np.newaxis]
    agent_example_observation, agent_example_state, agent_example_action = offline_replay_buffer.convert_to_critic_format(
        critic_example)
    actor.action_dim = agent_example_action.squeeze().shape[-1]
    actor.state_dim = agent_example_state.squeeze().shape[-1]

    if model_cls == "RealTimeEXPOFTLearner":
        agent_kwargs["delay"] = FLAGS.delay
    agent_kwargs["critic_camera_keys"] = critic_camera_keys

    agent = load_agent(
        seed=FLAGS.seed,
        example_observation=agent_example_observation.squeeze(),
        example_action=agent_example_action.squeeze(),
        example_state=agent_example_state.squeeze(),
        actor=actor,
        actor_train_state=actor_train_state,
        target_actor_params=target_actor_params,
        agent_kwargs=agent_kwargs,
        metadata=vla_metadata,
        mesh=mesh,
        data_sharding=data_sharding,
        replicated_sharding=replicated_sharding,
        resume=resuming,
        replan_steps=FLAGS.replan_steps,
        default_prompt=task_description,
        # LIBERO actions use all xyz/rotation/gripper dimensions.
        edit_action_xyzg=False,
    )

    start_step = 0
    if resuming:
        agent = restore_checkpoint(checkpoint_manager, agent)
        if hasattr(agent, "cache_infer_params"):
            agent = agent.cache_infer_params()
        steps = tuple(checkpoint_manager.all_steps())
        latest_step = max(steps) if steps else None
        if latest_step is not None:
            # The checkpoint is written after completing ``latest_step``.  The
            # next resumed loop iteration must therefore advance to the next
            # environment step instead of replaying that transition.
            start_step = latest_step + 1
            logging.info("Resuming from step %d", start_step)
        batch_processor.restore(checkpoint_dir_path, up_to_step=latest_step)

    # The environment is created only now, after the slow Pi0.5 build.
    logging.info("Creating LIBERO environment...")
    env = _LiberoEnvClient(
        FLAGS.suite, FLAGS.task_id, seed=FLAGS.seed, image_size=FLAGS.image_size,
        max_episode_steps=FLAGS.max_episode_steps,
    )
    env.reset()
    logging.info("Created training environment %s (%s)", env.env_id, task_description)
    eval_env = None
    eval_horizon = FLAGS.eval_max_steps or LIBERO_SUITE_MAX_STEPS[FLAGS.suite]
    eval_dir = Path(log_dir) / "evals"
    eval_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = Path(log_dir) / "metrics.jsonl"

    def record_local_metrics(step, metrics):
        scalar = {}
        for key, value in metrics.items():
            array = np.asarray(value)
            # Evaluation results also contain the per-trial records.  Persist
            # only numeric scalar metrics in metrics.jsonl; trial details are
            # written to the per-evaluation JSON file above.
            if array.size == 1 and np.issubdtype(array.dtype, np.number):
                scalar[key] = float(array.reshape(()))
        with metrics_path.open("a") as file:
            file.write(json.dumps({"step": int(step), **scalar}) + "\n")

    if FLAGS.eval_interval > 0 and FLAGS.eval_episodes > 0:
        eval_env = _LiberoEnvClient(
            FLAGS.suite, FLAGS.task_id, seed=FLAGS.eval_seed,
            image_size=FLAGS.image_size,
            max_episode_steps=eval_horizon,
            benchmark_eval=True,
        )
        if FLAGS.eval_episodes > len(eval_env._env.init_states):
            raise ValueError(
                f"eval_episodes={FLAGS.eval_episodes} exceeds the "
                f"{len(eval_env._env.init_states)} official initial states"
            )
    # On resume, start evaluations at the next interval boundary rather than
    # immediately after the first resumed episode.
    next_eval_step = (
        ((start_step // FLAGS.eval_interval) + 1) * FLAGS.eval_interval
        if FLAGS.eval_interval > 0 else None
    )

    episode_log = EpisodeState()
    training_log = TrainingStats(
        ep_count=replay_buffer.count_episodes_chronological() if resuming else 0,
    )
    if resuming:
        logging.info("Resuming: ep_count set to %d (episodes in replay buffer).", training_log.ep_count)

    # Guards the replay buffer: the prefetch thread samples while the loop inserts.
    buffer_lock = threading.Lock()

    batch_processor.on_episode_start()
    timer = Timer()
    # Runs main-actor inference in the background while the robot executes the last `delay` actions.
    sampler = AsyncChunkSampler(
        agent, delay=FLAGS.delay, replan_steps=FLAGS.replan_steps,
        sim_latency_ms=FLAGS.sim_latency, timer=timer,
    )
    # Buffer inserts, NFS buffer saves, wandb logs and checkpoints queue here and flush at episode end.
    # ``start_step`` is the first step after the saved checkpoint.  Anchor the
    # checkpoint cadence to the step that was actually saved so resumption does
    # not shift interval checkpoints by one environment step.
    checkpoint_anchor = max(0, start_step - 1) if resuming else start_step
    sink = EpisodeSink(
        batch_processor, buffer_lock, checkpoint_manager, checkpoint_dir_path, save_checkpoint,
        save_buffer=FLAGS.checkpoint_buffer, checkpoint_model=FLAGS.checkpoint_model,
        checkpoint_interval=FLAGS.checkpoint_interval, start_step=checkpoint_anchor,
    )
    if eval_env is not None and not resuming:
        initial_eval = _evaluate(
            agent, eval_env, episodes=FLAGS.eval_episodes,
            max_steps=eval_horizon, replan_steps=FLAGS.replan_steps,
            seed=FLAGS.eval_seed,
        )
        (eval_dir / "00000000.json").write_text(
            json.dumps({"step": 0, **initial_eval}, indent=2) + "\n"
        )
        sink.record_log(0, {
            key: value for key, value in initial_eval.items()
            if key != "trials"
        })
        record_local_metrics(0, initial_eval)
    # One-ahead batch prefetch: the CPU-bound replay image gather overlaps the GPU update.
    prefetch_pool = ThreadPoolExecutor(max_workers=1)
    pending_batch = None

    def _fetch(rng):
        with buffer_lock:
            return batch_processor.next_batch(rng)

    def run_agent_updates(num_updates: int, metrics: dict):
        nonlocal agent, combine_rng, pending_batch
        sink.flush_transitions()
        for _ in tqdm.tqdm(range(num_updates), disable=not FLAGS.tqdm):
            update_start = time.time()
            if pending_batch is None:
                pending_batch = prefetch_pool.submit(_fetch, combine_rng)
            batch, actor_batch, combine_rng = pending_batch.result()
            # Kick off the next gather now so it overlaps this update.
            pending_batch = prefetch_pool.submit(_fetch, combine_rng)
            metrics["batch_info"] = get_batch_info(batch)
            agent = agent.replace(rng=jax.device_put(agent.rng, replicated_sharding))
            agent, update_info = agent.update(agent, batch, FLAGS.utd_ratio, actor_batch)
            training_log.record_update_time(time.time() - update_start, metrics)
            for k, v in update_info.items():
                metrics[f"training/{k}"] = v

    def updates_due():
        """Upstream --num_updates: a fixed count per trigger; else one per step_interval transitions."""
        nonlocal transitions_since_update
        if FLAGS.num_updates > 0:
            return FLAGS.num_updates
        n_updates, transitions_since_update = divmod(transitions_since_update, FLAGS.step_interval)
        return n_updates

    # LIBERO is a local simulator; unlike the robot client it does not need
    # real-time pacing. Set --control_hz>0 only when deliberately pacing it.
    dt = 1.0 / FLAGS.control_hz if FLAGS.control_hz > 0 else 0.0
    done = False
    last_control_start = time.time()
    # LiberoTaskEnv.reset() already performs the official settle steps; the
    # upstream robot warm-up action is intentionally omitted.
    action_plan = deque()
    action_type = "policy"
    episodes_since_update = 0
    transitions_since_update = 0
    combine_rng = jax.random.PRNGKey(FLAGS.seed + 100)
    first_step_of_episode = True

    def finish_episode(step, episode_success, metrics):
        """Finalize a terminal LIBERO episode before issuing another action."""
        nonlocal agent, episodes_since_update, transitions_since_update, next_eval_step
        sink.flush_transitions()
        with buffer_lock:
            batch_processor.on_episode_done(episode_success)
        sampler.on_episode_end()
        training_log.on_episode_done(episode_log, episode_success, metrics)

        can_update = training_log.ep_count >= 10 and step >= FLAGS.batch_size
        if not can_update:
            transitions_since_update = 0
        if FLAGS.update_type == "episode" and can_update:
            run_agent_updates(updates_due(), metrics)
        elif FLAGS.update_type == "batch" and can_update:
            episodes_since_update += 1
            if episodes_since_update >= FLAGS.num_batch:
                run_agent_updates(updates_due(), metrics)
                episodes_since_update = 0

        if (eval_env is not None and next_eval_step is not None
                and step >= next_eval_step):
            eval_result = _evaluate(
                agent, eval_env, episodes=FLAGS.eval_episodes,
                max_steps=eval_horizon, replan_steps=FLAGS.replan_steps,
                seed=FLAGS.eval_seed,
            )
            metrics.update({
                key: value for key, value in eval_result.items()
                if key != "trials"
            })
            (eval_dir / f"{step:08d}.json").write_text(
                json.dumps({"step": step, **eval_result}, indent=2) + "\n"
            )
            next_eval_step = (step // FLAGS.eval_interval + 1) * FLAGS.eval_interval

        episode_log.reset()
        with buffer_lock:
            batch_processor.on_episode_start()
        sink.flush_episode(step, agent)

    for i in tqdm.tqdm(
        range(start_step, FLAGS.max_steps + 1), smoothing=0.1, disable=not FLAGS.tqdm
    ):
        step_metrics = {}
        timer.reset()
        timer.tick("loop_time")
        timer.tick("total")

        with timer.context("obs"):
            observation = env.get_observation()
        with timer.context("info"):
            done, success, reward, mask = env.get_info_for_step()
        sink.label_last(reward, mask, done)  # outcome of the previous action

        if done:
            finish_episode(i, success, step_metrics)
            observation = env.reset()
            env.pop_network_ms()  # discard reset latency from the new step
            action_plan.clear()
            action_type = "policy"
            last_control_start = None
            first_step_of_episode = True
            done = False
            reward, mask = 0.0, 1.0

        # Observation history for the replan_steps < delay fallback (no-op otherwise).
        sampler.observe(observation)

        # Start the next chunk's inference once exactly `delay` actions remain in the plan.
        sampler.launch(agent, observation, len(action_plan), action_type)

        with timer.context("plan"):
            # Skip model inference while human is controlling.
            if not action_plan and action_type != "human":
                sample_start = time.time()
                action_chunk, agent, sample_info = sampler.sample(agent, observation, step_metrics)
                episode_log.sample_info_history.append(sample_info)
                action_chunk = np.asarray(jax.device_get(action_chunk[:FLAGS.replan_steps]))
                training_log.record_sample_time(time.time() - sample_start, step_metrics)
                action_plan.extend(action_chunk)
                # Re-check with the refilled plan (covers delay == replan_steps).
                sampler.launch(agent, observation, len(action_plan), action_type)
            else:
                episode_log.sample_info_history.append(
                    episode_log.sample_info_history[-1] if episode_log.sample_info_history else None
                )

        # Pacing is anchored to the previous step's dispatch time, not its return, so latency does not accumulate.
        sleep_left = 0.0 if last_control_start is None else dt - (time.time() - last_control_start)
        step_metrics["timing/sleep_left_ms"] = sleep_left * 1000.0
        with timer.context("wait"):
            if sleep_left > 0:
                time.sleep(sleep_left)

        has_action = bool(action_plan)
        action = action_plan.popleft() if has_action else np.zeros_like(example_action.squeeze())

        with timer.context("act"):
            last_control_start = time.time()
            real_action, action_type = env.step(action.tolist())

        # Wire time of this step's env RPCs, as measured by the client wrapper.
        step_metrics["timing/network_ms"] = env.pop_network_ms()
        timer.tock("total")
        timer.tick("post_step")
        episode_log.record_step(observation, len(action_plan), action_type, real_action, reward)

        # A human override discards the plan and any in-flight inference.
        if action_type == "human":
            action_plan.clear()
            sampler.on_human_takeover()

        if (has_action or action_type == "human") and not done:
            transition = dict(
                observations=observation,
                actions=real_action,
                is_hil=(action_type == "human"),
            )
            # Queued only; the buffer insert happens before the next update or at episode end.
            sink.record_transition(i, transition)
            transitions_since_update += 1

        can_update = training_log.ep_count >= 10 and i >= FLAGS.batch_size
        if FLAGS.update_type == "step" and can_update:
            if FLAGS.num_updates > 0:
                run_agent_updates(FLAGS.num_updates, step_metrics)
            elif i % FLAGS.step_interval == 0:
                run_agent_updates(1, step_metrics)

        log_timing = not first_step_of_episode
        first_step_of_episode = False

        timer.tock("post_step")
        timer.tock("loop_time")
        step_metrics.update({f"timing/{k}_ms": v for k, v in timer.get_times_ms().items()})

        if not log_timing:
            step_metrics = {k: v for k, v in step_metrics.items()
                            if not k.startswith("timing/")}
        training_log.maybe_add_success_rate(i, step_metrics)
        record_local_metrics(i, step_metrics)
        sink.record_log(i, step_metrics)

    # The final command may have terminated the episode on the last loop
    # iteration, leaving its result queued until the next observation.  Flush
    # that terminal outcome so success marks and replay persistence are exact.
    final_done, final_success, final_reward, final_mask = env.get_info_for_step()
    if final_done:
        final_metrics = {}
        sink.label_last(final_reward, final_mask, final_done)
        finish_episode(FLAGS.max_steps, final_success, final_metrics)
        record_local_metrics(FLAGS.max_steps, final_metrics)
        sink.record_log(FLAGS.max_steps, final_metrics)

    sink.flush_all()

    if FLAGS.checkpoint_model:
        try:
            save_checkpoint(checkpoint_manager, agent, FLAGS.max_steps)
            logging.info(f"Saved final agent checkpoint at step {FLAGS.max_steps}")
        except Exception as e:
            logging.error(f"Could not save final checkpoint: {e}")
        logging.info("Waiting for checkpoint manager to finish")
        checkpoint_manager.wait_until_finished()

    sampler.close()
    prefetch_pool.shutdown(wait=False)
    env.close()
    if eval_env is not None:
        eval_env.close()


if __name__ == "__main__":
    app.run(main)

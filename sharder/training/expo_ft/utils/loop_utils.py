"""Helpers that keep train_pi_robo.py's control loop flat.

`AsyncChunkSampler` hides main-actor inference latency behind chunk execution;
`EpisodeSink` holds the work deferred off the control loop until episode end.
"""
import logging
import select
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import jax
import numpy as np
import wandb

from expo_ft.agents import save_replay_buffer_transition


class AsyncChunkSampler:
    """Samples action chunks, hiding main-actor inference latency behind execution.

    With ``delay == 0`` this is plain ``agent.sample_actions(obs)``. With ``delay > 0``
    (RealTimeEXPOFTLearner) ``sample_pre_cache`` -- the delayed, prefix-inpainted main
    actor -- runs on a background thread while the robot executes the last ``delay``
    actions of the current plan, and ``sample_actions`` finishes the chunk at the replan
    boundary. If ``replan_steps < delay`` more than one chunk would be in flight, which
    the single-slot async path cannot represent, so it falls back to boundary-synchronous
    delayed inference from the executed history (sim-equivalent).

    Agents are immutable pytrees, so every method takes the current agent and returns
    the (rng-advanced) one, exactly like ``agent.sample_actions``.
    """

    def __init__(self, agent, *, delay, replan_steps, sim_latency_ms=0.0, timer=None):
        self.delay = int(delay)
        self.replan_steps = int(replan_steps)
        self.sim_latency_ms = float(sim_latency_ms)
        self.timer = timer
        self.use_pre_cache = hasattr(agent, "sample_pre_cache") and self.delay > 0
        self.sync_mode = self.use_pre_cache and self.replan_steps < self.delay
        self.async_enabled = self.use_pre_cache and not self.sync_mode
        self.executor = ThreadPoolExecutor(max_workers=1) if self.async_enabled else None
        self.prefix_cache = None  # executed window of the last chunk (normalized + padded)
        self.pending = None       # in-flight background inference
        self.obs_hist = deque(maxlen=self.delay + 1)  # sync mode: obs_hist[-1-d] = obs d steps ago
        self.exec_hist = deque(maxlen=self.delay)     # sync mode: executed padded rows, oldest first
        if self.use_pre_cache:
            logging.info("RealTimeEXPOFTLearner: %s main-actor inference, delay=%d env-steps",
                         "boundary-synchronous" if self.sync_mode else "async", self.delay)

    # ------------------------------------------------------------------ loop hooks
    def observe(self, observation):
        """Once per control step with the fresh observation (history for sync mode)."""
        if self.sync_mode:
            self.obs_hist.append(observation)

    def launch(self, agent, observation, plan_len, action_type):
        """Start background inference once exactly ``delay`` actions remain in the plan."""
        if not self.async_enabled or self.prefix_cache is None or self.pending is not None:
            return
        if plan_len != self.delay or action_type == "human":
            return
        self.pending = self.executor.submit(self._run_pre_cache, agent, observation, self.prefix_cache)

    def sample(self, agent, observation, metrics):
        """Next chunk at a replan boundary. Returns ``(action_chunk, agent, sample_info)``."""
        if self.use_pre_cache:
            return self._sample_pre_cached(agent, observation, metrics)
        action_chunk, agent, sample_info = agent.sample_actions(observation)
        self._simulate_latency()
        return action_chunk, agent, sample_info

    def on_human_takeover(self):
        """The plan was discarded: forget the executed prefix and in-flight inference."""
        self.prefix_cache = None
        self.exec_hist.clear()
        self._cancel_pending()

    def on_episode_end(self):
        self.on_human_takeover()
        self.obs_hist.clear()

    def close(self):
        if self.executor is not None:
            self.executor.shutdown(wait=True)

    # ------------------------------------------------------------------ internals
    def _simulate_latency(self):
        if self.sim_latency_ms > 0:
            time.sleep(self.sim_latency_ms / 1000.0)

    def _cancel_pending(self):
        if self.pending is not None:
            self.pending.cancel()
            self.pending = None

    def _wait_pending(self):
        if self.timer is not None:
            with self.timer.context("precache_wait"):
                result = self.pending.result()
        else:
            result = self.pending.result()
        self.pending = None
        return result

    def _run_pre_cache(self, agent, observation, prefix):
        t0 = time.time()
        precached, agent = agent.sample_pre_cache(observation, prefix_padded=prefix)
        precached = jax.block_until_ready(precached)
        return precached, agent, (time.time() - t0) * 1000.0

    def _sample_pre_cached(self, agent, observation, metrics):
        r, d = self.replan_steps, self.delay
        if self.sync_mode:
            # Warm-up ramp: effective delay = executed history length, so the first
            # chunks condition on the shorter prefix that actually exists.
            d_eff = min(d, len(self.exec_hist))
            if d_eff == 0:  # boot: vanilla actor
                precached, agent = agent.sample_pre_cache(observation, prefix_padded=None)
            else:
                prefix = np.stack(list(self.exec_hist)[-d_eff:])
                delayed_obs = self.obs_hist[-1 - d_eff]
                t0 = time.time()
                precached, agent = agent.sample_pre_cache(delayed_obs, prefix_padded=prefix)
                metrics["timing/precache_compute_ms"] = (time.time() - t0) * 1000.0
            action_chunk, agent, sample_info = agent.sample_actions(
                observation, precached=precached, delay=d_eff)
            self._simulate_latency()
            for row in np.asarray(jax.device_get(sample_info["executed_padded"]))[:r]:
                self.exec_hist.append(row)
            self.prefix_cache = None
            return action_chunk, agent, sample_info

        if self.prefix_cache is None:  # boot: vanilla actor
            precached, agent = agent.sample_pre_cache(observation, prefix_padded=None)
            action_chunk, agent, sample_info = agent.sample_actions(
                observation, precached=precached, delay=0)
        else:
            if self.pending is None:
                logging.warning("No pending async pre-cache at replan boundary; running it synchronously.")
                self.pending = self.executor.submit(self._run_pre_cache, agent, observation, self.prefix_cache)
            precached, precache_agent, compute_ms = self._wait_pending()
            metrics["timing/precache_compute_ms"] = compute_ms
            agent = agent.replace(rng=precache_agent.rng)
            action_chunk, agent, sample_info = agent.sample_actions(
                observation, precached=precached, delay=d)
        self._simulate_latency()
        self.prefix_cache = sample_info["executed_padded"][r - d:r]
        return action_chunk, agent, sample_info


class EpisodeSink:
    """Work deferred off the control loop and flushed at episode end.

    Per step the loop only appends: transitions (inserted into the replay buffer under
    the buffer lock before the next update), their on-disk copies when the buffer is
    being saved, and wandb metrics. Interval checkpoints are also taken here, at
    episode end, so the control loop never blocks on Orbax or NFS.
    """

    def __init__(self, batch_processor, buffer_lock, checkpoint_manager, checkpoint_dir_path,
                 save_checkpoint_fn, *, save_buffer, checkpoint_model, checkpoint_interval,
                 start_step):
        self.batch_processor = batch_processor
        self.buffer_lock = buffer_lock
        self.checkpoint_manager = checkpoint_manager
        self.checkpoint_dir_path = checkpoint_dir_path
        self.save_checkpoint_fn = save_checkpoint_fn
        self.save_buffer = save_buffer
        self.checkpoint_model = checkpoint_model
        self.checkpoint_interval = checkpoint_interval
        self.last_ckpt_step = start_step
        self._transitions = []  # (step, transition)
        self._disk_saves = []
        self._logs = []         # (step, metrics)

    def record_transition(self, step, transition):
        self._transitions.append((step, transition))

    def label_last(self, reward, mask, done):
        if self._transitions:
            self._transitions[-1][1].update(rewards=reward, masks=mask, dones=done)

    def record_log(self, step, metrics):
        self._logs.append((step, metrics))

    def flush_transitions(self):
        """Insert queued transitions into the replay buffer (call before any update)."""
        # Keep the newest transition queued until it is terminal: the next step may still relabel it.
        held = self._transitions[-1:] if self._transitions and not self._transitions[-1][1].get("dones") else []
        ready = self._transitions[:len(self._transitions) - len(held)]
        with self.buffer_lock:
            for _, transition in ready:
                self.batch_processor.insert_transition(transition)
        if self.save_buffer:
            self._disk_saves.extend(ready)
        self._transitions = held

    def flush_episode(self, step, agent):
        self.flush_transitions()
        self._flush_disk_saves()
        self._flush_logs()
        self.maybe_save_checkpoint(step, agent)

    def flush_all(self):
        self.flush_transitions()
        self._flush_disk_saves()
        self._flush_logs()

    def maybe_save_checkpoint(self, step, agent):
        # Press Enter in this terminal to force a checkpoint at the next episode end,
        # regardless of checkpoint_model / checkpoint_interval. select() on a non-tty
        # stdin (/dev/null under sbatch) always reports readable, hence the isatty guard.
        forced = sys.stdin.isatty() and bool(select.select([sys.stdin], [], [], 0)[0])
        if forced:
            sys.stdin.readline()
        else:
            if not self.checkpoint_model or self.checkpoint_interval <= 0:
                return
            if step - self.last_ckpt_step < self.checkpoint_interval:
                return
        try:
            self.save_checkpoint_fn(self.checkpoint_manager, agent, step)
            self.checkpoint_manager.wait_until_finished()
            self.last_ckpt_step = step
            logging.info("Saved agent checkpoint at step %d%s", step, " (forced via Enter)" if forced else "")
        except Exception as e:
            logging.error("Could not save model checkpoint: %s", e)

    def _flush_disk_saves(self):
        try:
            while self._disk_saves:
                step, transition = self._disk_saves[0]
                save_replay_buffer_transition(self.checkpoint_dir_path, transition, step=step)
                self._disk_saves.pop(0)
        except Exception:
            logging.exception("Could not save agent buffer; will retry at next flush.")

    def _flush_logs(self):
        # Detach first: an update thread may record while this flushes. wandb needs
        # non-decreasing steps, so log in step order.
        logs, self._logs = self._logs, []
        for step, metrics in sorted(logs, key=lambda item: item[0]):
            wandb.log(metrics, step=step)

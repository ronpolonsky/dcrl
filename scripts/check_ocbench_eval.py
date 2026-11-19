"""Compare DCRL and upstream OCBench scoring on identical action sequences."""

import argparse
from collections import defaultdict
import importlib.util
import json
from pathlib import Path

import gymnasium
import numpy as np
import ocbench
from ocbench.oracles.controllers.block import BlockController
from ocbench.oracles.controllers.bowling import BowlingController

from dcrl.ocbench_env import environment_id, evaluation_metrics
from full_90_libero_evals import _evaluate_task


def load_module(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FixedSeed(gymnasium.ObservationWrapper):
    """Stream only a scalar to the reference logger, since the actions are prerecorded."""

    def __init__(self, env, seed):
        super().__init__(env)
        self.seed = seed

    def reset(self, **kwargs):
        return super().reset(seed=self.seed, **kwargs)

    def observation(self, observation):
        return np.zeros((1,), dtype=np.float32)


class RecordedActions:
    def __init__(self, actions):
        self.actions = actions
        self.index = 0

    def _next(self):
        action = self.actions[self.index]
        self.index += 1
        return action

    def sample_actions(self, observations, *, seed=None):
        return self._next()[None]

    def replace(self, **kwargs):
        return RecordedActions(self.actions)

    def eval_actor_only(self, observation):
        return self._next()[None], self, {}


def record_actions(env_id, seed, controller_type):
    env = ocbench.make(env_id)
    np.random.seed(seed)
    try:
        ob, info = env.reset(seed=seed)
        controller = controller_type(env=env) if controller_type is not None else None
        if controller is not None:
            controller.reset(ob, info)
        actions = []
        done = False
        while not done:
            action = controller.select_action(ob, info) if controller is not None else np.zeros(7)
            action = np.clip(action, env.action_space.low, env.action_space.high).astype(np.float32)
            ob, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            actions.append(action)
        return actions
    finally:
        env.close()


def check(suite, task, seed, count, upstream):
    evaluator = load_module(upstream / 'impls/utils/evaluation.py')
    monitor = load_module(upstream / 'impls/envs/env_utils.py').EpisodeMonitor
    env_id = environment_id(suite, task)
    bowling = env_id.startswith('visual-bowling-')
    controller_type = BowlingController if bowling else BlockController
    _, _, _, horizon = ocbench.parse_env_spec(env_id)
    reference, trials = defaultdict(list), []
    for index in range(count):
        reset_seed = seed + index
        use_oracle = index % 2 == 0
        actions = record_actions(env_id, reset_seed, controller_type if use_oracle else None)
        env = monitor(FixedSeed(ocbench.make(env_id), reset_seed),
                      filter_regexes=['.*privileged.*', '.*proprio.*'])
        try:
            stats, _, _ = evaluator.evaluate(RecordedActions(actions), env, num_eval_episodes=1)
        finally:
            env.close()
        result = _evaluate_task(RecordedActions(actions), mode='sft_pi', suite=suite, task_id=task,
                                episodes=1, max_steps=horizon, replan_steps=1,
                                image_size=224, eval_seed=reset_seed)
        trial = result['trials'][0]
        assert trial['success'] == stats['success'], (trial, stats)
        assert trial['steps'] == stats['episode.length'] == len(actions), (trial, stats)
        if bowling:
            np.testing.assert_allclose(trial['return'], stats['episode.return'])
        trials.append(dict(trial, seed=reset_seed, policy='oracle' if use_oracle else 'zero_action'))
        reference['success'].append(stats['success'])
        reference['episode.length'].append(stats['episode.length'])
        reference['episode.return'].append(stats['episode.return'])
    metrics = evaluation_metrics(trials, horizon, score_return=bowling)
    if bowling:
        np.testing.assert_allclose(metrics['eval/normalized_return'],
                                   100 * np.mean(reference['episode.return']))
        return dict(env_id=env_id, horizon=horizon, trials=trials, metrics=metrics)
    evaluator.summarize_eval_metrics(reference, horizon)
    for local_key, upstream_key in [('success_rate', 'success'), ('mean_steps', 'mean_steps'),
                                    ('throughput', 'throughput'), ('success_weighted_speed', 'success_weighted_speed')]:
        np.testing.assert_allclose(metrics[f'eval/{local_key}'], np.mean(reference[upstream_key]), rtol=1e-6)
    return dict(env_id=env_id, horizon=horizon, trials=trials, metrics=metrics)


def check_table5(shard):
    """Verify the released training shard for block-lite-single-task1-v0."""
    with np.load(shard, allow_pickle=False) as dataset:
        rewards = dataset['rewards']
        ends = np.flatnonzero(dataset['terminals'])
    assert ends.size and ends[-1] == rewards.size - 1
    lengths = np.diff(ends, prepend=-1)
    trials = [dict(success=bool(rewards[end]), steps=int(length))
              for end, length in zip(ends, lengths)]
    _, _, _, horizon = ocbench.parse_env_spec(environment_id('ocbench_lite', 0))
    metrics = evaluation_metrics(trials, horizon)
    assert round(100 * metrics['eval/success_rate']) == 97
    assert round(metrics['eval/mean_steps'], 1) == 77.2
    assert round(metrics['eval/throughput'], 2) == 2.50
    assert len(trials) == 10_000 and round(rewards.size / 1_000_000, 1) == 0.8
    return dict(episodes=len(trials), transitions=rewards.size, metrics=metrics)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    checks = parser.add_mutually_exclusive_group(required=True)
    checks.add_argument('--upstream-checkout', type=Path)
    checks.add_argument('--table5-shard', type=Path,
                        help='Released training shard lite/block-lite-single-task1-v0/000.npz.')
    parser.add_argument('--suite', choices=('ocbench', 'ocbench_lite'), default='ocbench')
    parser.add_argument('--task-id', type=int, choices=(*range(8), 27), default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--episodes', type=int, default=4)
    args = parser.parse_args()
    if args.table5_shard is not None:
        print(json.dumps(check_table5(args.table5_shard), indent=2))
        return
    if args.episodes < 2:
        parser.error('Use at least two episodes to test both policies.')
    print(json.dumps(check(args.suite, args.task_id, args.seed, args.episodes,
                           args.upstream_checkout), indent=2))


if __name__ == '__main__':
    main()

"""Environment and demonstration dispatch shared by the training entry points."""

from dcrl.libero_env import LIBERO_SUITE_MAX_STEPS, LiberoTaskEnv


def is_ocbench(suite):
    return suite in ('ocbench', 'ocbench_lite')


def task_count(suite):
    if is_ocbench(suite):
        from dcrl.ocbench_env import TASKS
        return len(TASKS)
    from libero.libero import benchmark
    return benchmark.get_benchmark_dict()[suite]().get_num_tasks()


def max_episode_steps(suite, task_id):
    if is_ocbench(suite):
        import ocbench
        from dcrl.ocbench_env import environment_id
        return ocbench.parse_env_spec(environment_id(suite, task_id))[3]
    return LIBERO_SUITE_MAX_STEPS[suite]


def make_env(suite, task_id, **kwargs):
    if is_ocbench(suite):
        from dcrl.ocbench_env import OCBenchTaskEnv
        return OCBenchTaskEnv(suite, task_id, **kwargs)
    return LiberoTaskEnv(suite, task_id, **kwargs)


def iter_demonstrations(suite, path, **kwargs):
    if is_ocbench(suite):
        from dcrl.ocbench_env import iter_demonstrations

        yield from iter_demonstrations(path, **kwargs)
    else:
        from dcrl.libero_replay import load_demonstrations

        yield from load_demonstrations(path, **kwargs)


def validate_eval_episodes(env, episodes):
    if env.init_states is not None and episodes > len(env.init_states):
        raise ValueError(f'Benchmark evaluation has only {len(env.init_states)} initial states')


def initial_state_index(env, episode):
    return episode if env.init_states is None else episode % len(env.init_states)


def asset_id(suite):
    return f'dcrl/{suite}' if is_ocbench(suite) else 'physical-intelligence/libero'

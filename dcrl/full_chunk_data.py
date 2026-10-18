"""Frozen EXPO teacher collection: one target per actual replan observation."""
from pathlib import Path
import hashlib
import json
import multiprocessing as mp

import h5py
import numpy as np
from tqdm import tqdm


def partition_collection(source: Path, success_path: Path, failure_path: Path):
    """Split a complete mixed collection into success and failure HDF5 files."""
    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(source)
    source.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(source, 'r') as src:
        if not src.attrs.get('complete', False):
            raise ValueError(f'Collection is incomplete: {source}')
        metadata = src.attrs.get('metadata')
        for destination, wanted_success in ((success_path, True), (failure_path, False)):
            destination = Path(destination)
            destination.parent.mkdir(parents=True, exist_ok=True)
            temp = destination.with_suffix('.partial.hdf5')
            if destination.exists():
                with h5py.File(destination, 'r') as existing:
                    if (not existing.attrs.get('complete', False)
                            or existing.attrs.get('metadata') != metadata):
                        raise ValueError(f'Existing partition does not match collection: {destination}')
                continue
            with h5py.File(temp, 'w') as dst:
                for key, value in src.attrs.items():
                    dst.attrs[key] = value
                dst.attrs['partition'] = 'success' if wanted_success else 'failure'
                for name, group in src.items():
                    if bool(group.attrs.get('success', False)) == wanted_success:
                        src.copy(name, dst, name=name)
                dst.attrs['complete'] = True
            temp.replace(destination)


def _episode_worker(connection, suite, task_id, seed, image_size, initial_state_index, max_steps):
    """Own one MuJoCo/EGL context in an isolated process for a complete episode."""
    env = None
    try:
        from dcrl.libero_env import LiberoTaskEnv
        env = LiberoTaskEnv(suite, task_id, seed=seed, image_size=image_size,
                            init_state_offset=initial_state_index)
        obs = env.reset()
        elapsed = 0
        connection.send(("observation", obs))
        while elapsed < max_steps:
            actions = connection.recv()
            success = False
            executed = 0
            for action in actions:
                obs, _, success, _ = env.step(action)
                elapsed += 1
                executed += 1
                if success or elapsed >= max_steps:
                    break
            connection.send(("result", obs, success, executed, elapsed))
            if success or elapsed >= max_steps:
                break
    except BaseException as exc:
        try:
            connection.send(("error", repr(exc)))
        except (BrokenPipeError, EOFError, OSError):
            pass
        raise
    finally:
        if env is not None:
            env.close()
        connection.close()


def collect(teacher_dir: Path, checkpoint_step: int, output: Path, episodes: int, seed: int):
    import jax
    import orbax.checkpoint as ocp
    from openpi.shared import normalize
    from openpi.training import sharding
    from dcrl.agents.small_expo_learner import SmallExpoLearner
    from dcrl.agents.pi_expo_learner import restore_checkpoint
    from dcrl.libero_env import LiberoTaskEnv, LIBERO_SUITE_MAX_STEPS, CAMERA_KEYS

    config = json.loads((teacher_dir / 'run_config.json').read_text())
    norm_bytes = (teacher_dir / 'norm_stats.json').read_bytes()
    metadata = dict(teacher_dir=str(teacher_dir.resolve()), checkpoint_step=checkpoint_step,
                    seed=seed, episodes=episodes, config=config,
                    norm_sha256=hashlib.sha256(norm_bytes).hexdigest(), format_version=1)
    if output.exists():
        with h5py.File(output, 'r') as f:
            if json.loads(f.attrs['metadata']) != metadata or not f.attrs.get('complete', False):
                raise ValueError('Existing collection does not match this request')
        partition_collection(output, output.parent / 'data/success/teacher_chunks.hdf5',
                            output.parent / 'data/failures/teacher_chunks.hdf5')
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    # Publish only complete collections; interrupted temporary files are not training inputs.
    temp = output.with_suffix('.partial.hdf5')
    bootstrap_env = LiberoTaskEnv(config['suite'], config['task_id'], seed=seed,
                                  image_size=config['image_size'])
    try:
        obs = bootstrap_env.reset()
        instruction = bootstrap_env.instruction
        initial_state_count = len(bootstrap_env.init_states)
        bootstrap_env.close()
        bootstrap_env = None
        mesh = sharding.make_mesh(1)
        data = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
        rep = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        teacher = SmallExpoLearner.from_examples(
            seed=config['seed'] + config['task_id'], example_observation=obs,
            example_action=np.zeros(7, np.float32), example_state=obs['state'],
            mesh=mesh, data_sharding=data, replicated_sharding=rep,
            image_size=config['image_size'], action_horizon=config['action_horizon'],
            replan_steps=config['replan_steps'], actor_success_only=True,
            norm_stats=normalize.load(teacher_dir))
        with ocp.CheckpointManager(teacher_dir / 'checkpoints',
                item_handlers={'agent': ocp.PyTreeCheckpointHandler(), 'params': ocp.PyTreeCheckpointHandler()},
                options=ocp.CheckpointManagerOptions(create=False, read_only=True)) as manager:
            if checkpoint_step not in manager.all_steps():
                raise ValueError(f'Teacher checkpoint {checkpoint_step} does not exist')
            # from_examples() seeds an inference cache before checkpoint restore;
            # refresh it so rollout uses the restored, trained teacher params.
            teacher = restore_checkpoint(manager, teacher, checkpoint_step).cache_infer_params()
        print(f'Restored frozen teacher checkpoint {checkpoint_step}', flush=True)
        teacher = teacher.replace(rng=jax.random.PRNGKey(seed))
        with h5py.File(temp, 'a') as f:
            if 'metadata' in f.attrs and json.loads(f.attrs['metadata']) != metadata:
                raise ValueError('Partial collection does not match request')
            f.attrs['metadata'] = json.dumps(metadata, sort_keys=True)
            f.attrs['teacher_norm_stats'] = norm_bytes.decode()
            f.attrs['instruction'] = instruction
            # A partial collection can be resumed after a preemption. Restore
            # the teacher's post-episode RNG so the remaining episodes use the
            # same continuous sampling stream as an uninterrupted collection.
            complete = [name for name in f if f[name].attrs.get('complete', False)]
            if complete:
                last_name = max(complete, key=lambda name: int(name.rsplit('_', 1)[1]))
                saved_rng = f[last_name].attrs.get('teacher_rng_after')
                if saved_rng is not None:
                    teacher = teacher.replace(rng=jax.numpy.asarray(saved_rng, dtype=jax.numpy.uint32))
            for episode in tqdm(range(episodes), desc='Collect frozen teacher'):
                name = f'episode_{episode:06d}'
                if name in f and f[name].attrs.get('complete', False):
                    continue
                if name in f:
                    del f[name]
                # Keep the eval RNG stream continuous across episodes, exactly as
                # train_libero._evaluate does after seeding the teacher once.
                context = mp.get_context('spawn')
                parent, child = context.Pipe()
                worker = context.Process(
                    target=_episode_worker,
                    args=(child, config['suite'], config['task_id'], seed + episode,
                          config['image_size'], episode % initial_state_count,
                          LIBERO_SUITE_MAX_STEPS[config['suite']]),
                )
                worker.start()
                child.close()
                try:
                    reply = parent.recv()
                    if reply[0] != 'observation':
                        raise RuntimeError(f'LIBERO worker failed to initialize: {reply[1]}')
                    obs = reply[1]
                    rows = []
                    success = False
                    elapsed = 0
                    limit = LIBERO_SUITE_MAX_STEPS[config['suite']]
                    while elapsed < limit and not success:
                        start_obs = obs
                        chunk, teacher, _ = teacher.sample_actions(obs)
                        chunk = np.asarray(jax.device_get(chunk), np.float32)
                        if chunk.shape != (config['action_horizon'], 7) or not np.isfinite(chunk).all():
                            raise ValueError('Invalid teacher action chunk')
                        actions = chunk[:min(config['replan_steps'], limit - elapsed)]
                        parent.send(actions)
                        reply = parent.recv()
                        if reply[0] == 'error':
                            raise RuntimeError(f'LIBERO worker failed: {reply[1]}')
                        _, obs, success, executed, elapsed = reply
                        rows.append((start_obs, chunk, executed))
                finally:
                    parent.close()
                    worker.join()
                if worker.exitcode != 0:
                    raise RuntimeError(f'LIBERO episode worker exited with {worker.exitcode}')
                group = f.create_group(name)
                group.attrs.update(success=success, steps=elapsed, initial_state_index=episode % initial_state_count)
                for key in CAMERA_KEYS:
                    group.create_dataset(key, data=np.stack([r[0]['image'][key] for r in rows]), compression='gzip')
                group.create_dataset('state', data=np.stack([r[0]['state'] for r in rows]))
                group.create_dataset('actions', data=np.stack([r[1] for r in rows]))
                group.create_dataset('executed_prefix', data=[r[2] for r in rows])
                group.attrs['teacher_rng_after'] = np.asarray(jax.device_get(teacher.rng), dtype=np.uint32)
                group.attrs['complete'] = True
                f.flush()
            f.attrs['complete'] = True
        temp.replace(output)
        partition_collection(output, output.parent / 'data/success/teacher_chunks.hdf5',
                            output.parent / 'data/failures/teacher_chunks.hdf5')
    finally:
        if bootstrap_env is not None:
            bootstrap_env.close()


class ChunkDataset:
    """Disk-backed selected chunks, never reconstructed from executed transitions."""
    def __init__(self, path, success_only=True):
        paths = [path] if isinstance(path, (str, Path)) else list(path)
        self.files = [h5py.File(p, 'r') for p in paths]
        if any(not file.attrs.get('complete', False) for file in self.files):
            self.close()
            raise ValueError('Collection is incomplete')
        self.metadata = json.loads(self.files[0].attrs['metadata'])
        self.instruction = self.files[0].attrs['instruction']
        for file in self.files[1:]:
            if (json.loads(file.attrs['metadata']) != self.metadata
                    or file.attrs['instruction'] != self.instruction):
                self.close()
                raise ValueError('Collection partitions do not match')
        self.indices = [(file_index, name, i)
                        for file_index, file in enumerate(self.files)
                        for name, group in file.items()
                        if not success_only or group.attrs['success']
                        for i in range(len(group['actions']))]
        if not self.indices:
            self.close()
            raise ValueError('No eligible teacher chunks; collect more episodes or explicitly allow failures')

    def sample(self, rng, count):
        result = []
        for index in rng.integers(len(self.indices), size=count):
            file_index, name, row = self.indices[index]
            group = self.files[file_index][name]
            result.append({'observation/image': group['base_0_rgb'][row],
                           'observation/wrist_image': group['left_wrist_0_rgb'][row],
                           'observation/state': group['state'][row],
                           'actions': group['actions'][row], 'prompt': self.instruction})
        return result

    def close(self):
        for file in getattr(self, 'files', []):
            file.close()

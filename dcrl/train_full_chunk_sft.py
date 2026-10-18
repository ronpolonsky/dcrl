"""Distill frozen SmallExpoLearner decisions into the PiExpoLearner pi0.5 actor.

Run --phase collect, then --phase train (or --phase all, which isolates teacher
collection in a subprocess). Full selected chunks are targets, without prefix
masking. Student preprocessing retains its pretrained LIBERO normalization.
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


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=['all', 'collect', 'train'], default='all')
    p.add_argument('--teacher-dir', type=Path, required=True)
    p.add_argument('--teacher-step', type=int, required=True, help='Exact checkpoint global step')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--collection-episodes', type=int, default=200)
    p.add_argument('--include-failures', action='store_true')
    p.add_argument('--student-checkpoint', default='gs://openpi-assets/checkpoints/pi05_libero')
    p.add_argument('--num-sft-steps', type=int, default=10000)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--learning-rate', type=float, default=1e-5)
    p.add_argument('--finetune-mode', choices=['lora', 'full'], default='lora')
    p.add_argument('--eval-every', type=int, default=1000)
    p.add_argument('--eval-episodes', type=int, default=20)
    p.add_argument('--checkpoint-every', type=int, default=1000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--eval-seed', type=int, default=7)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    for name in ['collection_episodes', 'num_sft_steps', 'batch_size', 'eval_every',
                 'eval_episodes', 'checkpoint_every']:
        if getattr(args, name) <= 0:
            p.error(f'{name} must be positive')
    return args


def raw_input(obs, instruction):
    return {'observation/image': obs['image']['base_0_rgb'],
            'observation/wrist_image': obs['image']['left_wrist_0_rgb'],
            'observation/state': obs['state'], 'prompt': instruction}


def evaluate(actor, state, transform, output_transform, env, horizon, replan, episodes, seed):
    import jax
    from openpi.models import model
    env.episode_index = 0
    env.env.seed(env.seed)
    trials = []
    rng = jax.random.PRNGKey(seed)
    for episode in tqdm(range(episodes), desc='Evaluate student actor', leave=False):
        obs = env.reset()
        plan = deque()
        for step in range(horizon):
            if not plan:
                inputs = transform(raw_input(obs, env.instruction))
                inputs = jax.tree.map(lambda x: np.asarray(x)[None], inputs)
                packed = model.Observation.from_dict(inputs).to_dict()
                rng, key = jax.random.split(rng)
                actions, _ = actor.sample_actions(packed, state, key, num_samples=1)
                actions = np.asarray(jax.device_get(actions))[0]
                actions = output_transform({'actions': actions,
                                            'state': np.asarray(inputs['state'])[0]})['actions']
                plan.extend(actions[:replan])
            obs, _, success, _ = env.step(plan.popleft())
            if success:
                break
        trials.append(dict(episode=episode, initial_state_index=episode,
                           success=bool(success), steps=step + 1))
    return {'eval/success_rate': float(np.mean([x['success'] for x in trials])),
            'eval/mean_steps': float(np.mean([x['steps'] for x in trials])), 'trials': trials}


def train(args):
    import jax
    import orbax.checkpoint as ocp
    from openpi import transforms
    from openpi.models import model
    from openpi.shared import download, normalize
    from openpi.training import config as configs, optimizer, sharding, weight_loaders
    from dcrl.expo.agents.vla.pi05 import Pi05Agent
    from dcrl.full_chunk_data import ChunkDataset
    from dcrl.libero_env import LiberoTaskEnv, LIBERO_SUITE_MAX_STEPS

    dataset_path = args.output_dir / 'teacher_chunks.hdf5'
    dataset = ChunkDataset(dataset_path, success_only=not args.include_failures)
    teacher = dataset.metadata['config']
    if (dataset.metadata['teacher_dir'] != str(args.teacher_dir.resolve())
            or dataset.metadata['checkpoint_step'] != args.teacher_step):
        raise ValueError('Dataset teacher does not match requested checkpoint')
    metadata = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
                if k not in ('phase', 'resume', 'num_sft_steps')}
    metadata['collection'] = dataset.metadata
    metadata['loss'] = 'full_chunk_flow_matching'
    metadata['student_action_horizon'] = teacher['action_horizon']
    manifest = args.output_dir / 'sft_config.json'
    if args.resume:
        if not manifest.exists() or json.loads(manifest.read_text()) != metadata:
            raise ValueError('Resume configuration differs from saved SFT configuration')
    elif manifest.exists():
        raise FileExistsError('SFT output already exists; use --resume')

    # Load only normalization assets here; weight loader downloads parameters later.
    stats_dir = args.output_dir / 'student_assets' / 'physical-intelligence' / 'libero'
    if not args.resume:
        source = download.maybe_download(args.student_checkpoint.rstrip('/') +
                '/assets/physical-intelligence/libero/norm_stats.json')
        stats_dir.mkdir(parents=True, exist_ok=True)
        (stats_dir / 'norm_stats.json').write_bytes(Path(source).read_bytes())
    stats = normalize.load(stats_dir)
    if len(stats['actions'].mean) != 7 or len(stats['state'].mean) != 8:
        raise ValueError('Expected LIBERO state/action normalization dimensions 8/7')
    print(f'SFT dataset: {len(dataset.indices)} eligible chunks; loading student {args.student_checkpoint}', flush=True)
    base = configs.get_config('pi05_libero')
    student_model = dataclasses.replace(base.model, action_horizon=teacher['action_horizon'])
    if args.finetune_mode == 'lora':
        student_model = dataclasses.replace(student_model, paligemma_variant='gemma_2b_lora',
                                            action_expert_variant='gemma_300m_lora')
    # Same flow-matching train_step as PiExpoLearner.update_actor.
    cfg = dataclasses.replace(base,
        model=student_model,
        freeze_filter=student_model.get_freeze_filter(),
        ema_decay=None if args.finetune_mode == 'lora' else base.ema_decay,
        data=dataclasses.replace(base.data, assets=configs.AssetsConfig(
            assets_dir=str(args.output_dir / 'student_assets'), asset_id='physical-intelligence/libero')),
        weight_loader=weight_loaders.CheckpointWeightLoader(args.student_checkpoint.rstrip('/') + '/params'),
        lr_schedule=optimizer.CosineDecaySchedule(warmup_steps=10, peak_lr=args.learning_rate,
            decay_steps=max(11, args.num_sft_steps), decay_lr=args.learning_rate),
        batch_size=args.batch_size, num_train_steps=args.num_sft_steps)
    # Constant post-warmup LR makes an extended --num-sft-steps resume consistent.
    cfg = dataclasses.replace(cfg, lr_schedule=optimizer.CosineDecaySchedule(
        warmup_steps=10, peak_lr=args.learning_rate, decay_steps=1000000000, decay_lr=args.learning_rate))
    mesh = sharding.make_mesh(1)
    ds = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    rs = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    if args.batch_size % jax.device_count():
        raise ValueError('Batch size must divide device count')
    checkpoint_root = args.output_dir / 'checkpoints'
    restore_existing = args.resume and checkpoint_root.exists() and any(
        p.is_dir() and p.name.isdigit() and (p / '_CHECKPOINT_METADATA').exists()
        for p in checkpoint_root.iterdir())
    actor, state, _ = Pi05Agent.initialize(cfg, mesh, jax.random.PRNGKey(args.seed),
        resume=restore_existing, data_sharding=ds, replicated_sharding=rs,
        default_prompt=dataset.instruction, freeze_pi05_encoder=False)
    actor.action_dim, actor.state_dim = 7, 8
    dc = actor.data_config
    # Collection already uses inference keys and upright uint8 images: no dataset repack/rotation.
    transform = transforms.compose([*dc.data_transforms.inputs,
        transforms.Normalize(stats, use_quantiles=dc.use_quantile_norm), *dc.model_transforms.inputs])
    output_transform = transforms.compose([*dc.model_transforms.outputs,
        transforms.Unnormalize(stats, use_quantiles=dc.use_quantile_norm), *dc.data_transforms.outputs])
    env = LiberoTaskEnv(teacher['suite'], teacher['task_id'], seed=args.eval_seed,
                       image_size=teacher['image_size'], benchmark_eval=True)
    manager = ocp.CheckpointManager(args.output_dir / 'checkpoints',
        item_handlers={'state': ocp.PyTreeCheckpointHandler()},
        options=ocp.CheckpointManagerOptions(max_to_keep=None, create=True))
    try:
        if args.eval_episodes > len(env.init_states):
            raise ValueError('Too many evaluation initial states requested')
        if restore_existing:
            if manager.latest_step() is None:
                raise ValueError('No SFT checkpoint available to resume')
            state = manager.restore(manager.latest_step(), items={'state': state},
                restore_kwargs={'state': {'restore_args': ocp.checkpoint_utils.construct_restore_args(state)}})['state']
        else:
            manifest.write_text(json.dumps(metadata, indent=2) + '\n')
        start = int(np.asarray(state.step))
        if start >= args.num_sft_steps:
            raise ValueError('Requested training steps must exceed the restored step')
        def run_eval(step):
            result = evaluate(actor, state, transform, output_transform, env,
                LIBERO_SUITE_MAX_STEPS[teacher['suite']], teacher['replan_steps'],
                args.eval_episodes, args.eval_seed)
            path = args.output_dir / 'evals'
            path.mkdir(exist_ok=True)
            (path / f'{step:08d}.json').write_text(json.dumps({'step': step, **result}, indent=2))
            return {k: v for k, v in result.items() if k != 'trials'}
        if not restore_existing:
            run_eval(0)
        with (args.output_dir / 'metrics.jsonl').open('a') as log:
            for step in tqdm(range(start + 1, args.num_sft_steps + 1), desc='Full chunk SFT', initial=start, total=args.num_sft_steps):
                # Sampling and training randomness are reproducible from seed + saved update step.
                rng = np.random.default_rng(np.random.SeedSequence([args.seed, step]))
                rows = [transform(row) for row in dataset.sample(rng, args.batch_size)]
                batch = jax.tree.map(lambda *xs: jax.device_put(np.stack(xs), ds), *rows)
                targets = batch.pop('actions')
                observation = model.Observation.from_dict(batch)
                key = jax.device_put(jax.random.fold_in(jax.random.PRNGKey(args.seed), step), rs)
                with sharding.set_mesh(mesh):
                    state, info = actor.train_step(key, state, (observation, targets))
                metrics = {k: float(np.asarray(v)) for k, v in info.items()}
                if not all(np.isfinite(v) for v in metrics.values()):
                    raise FloatingPointError(f'Nonfinite SFT metrics at step {step}: {metrics}')
                if step % args.eval_every == 0 or step == args.num_sft_steps:
                    metrics.update(run_eval(step))
                log.write(json.dumps({'step': step, **metrics}) + '\n')
                log.flush()
                if step % args.checkpoint_every == 0 or step == args.num_sft_steps:
                    manager.save(step, {'state': state})
                    manager.wait_until_finished()
    finally:
        manager.close()
        env.close()
        dataset.close()


def main():
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    args.teacher_dir = args.teacher_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.phase == 'all':
        command = [sys.executable, '-m', 'dcrl.train_full_chunk_sft', *sys.argv[1:], '--phase', 'collect']
        subprocess.run(command, check=True)
        subprocess.run([sys.executable, '-m', 'dcrl.train_full_chunk_sft',
                        *sys.argv[1:], '--phase', 'train'], check=True)
        return
    elif args.phase == 'collect':
        from dcrl.full_chunk_data import collect
        collect(args.teacher_dir, args.teacher_step, args.output_dir / 'teacher_chunks.hdf5',
                args.collection_episodes, args.eval_seed)
        return
    train(args)


if __name__ == '__main__':
    main()

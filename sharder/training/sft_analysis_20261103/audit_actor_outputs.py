"""Compare initialization and trained SFT actors on identical stored inputs/noise."""
import dataclasses
import gc
import json
from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from openpi.shared import nnx_utils
from openpi.training import config as configs, optimizer, weight_loaders
from dcrl.expo.agents.vla.pi05 import Pi05Agent
from expo_ft.utils.train_utils import FlexibleCheckpointWeightLoader

OUT = Path(__file__).parent
BASE = Path('/projects/bgqe/rpolonsky/dcrlcheckpoints/openpi_cache/openpi-assets/checkpoints/pi05_libero')
ROOT = Path('/work/hdd/bgqe/rpolonsky/dcrlcheckpoints/sft_agnostic_small_expo')
samples = []
for task in (2, 28):
    run = next(ROOT.glob(f'task_{task:02d}_*'))
    with h5py.File(run / 'pi_base_data/success/teacher_chunks.hdf5', 'r') as f:
        for name in list(f)[:2]:
            g = f[name]
            for i in (0, len(g['actions']) - 1):
                samples.append(dict(task=task, episode=name, row=i, image=np.asarray(g['base_0_rgb'][i]), wrist_image=np.asarray(g['left_wrist_0_rgb'][i]), state=np.asarray(g['state'][i]), prompt=str(f.attrs['instruction']), target=np.asarray(g['actions'][i])))
# Use OpenPI's own axis names/layout.
from openpi.training import sharding
mesh = sharding.make_mesh(1)
data = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
results = {}
for mode in ('official', 'fresh_lora', 'sft28_40k'):
    print(f'BEGIN {mode}', flush=True)
    cfg = configs.get_config('pi05_libero')
    if mode != 'official':
        model = dataclasses.replace(cfg.model, paligemma_variant='gemma_2b_lora', action_expert_variant='gemma_300m_lora')
        cfg = dataclasses.replace(cfg, model=model, freeze_filter=model.get_freeze_filter())
    else:
        cfg = dataclasses.replace(cfg, freeze_filter=nnx.All(nnx.Param))
    loader = weight_loaders.CheckpointWeightLoader(str(BASE / 'params'))
    if mode == 'sft28_40k':
        run = next(ROOT.glob('task_28_*'))
        loader = FlexibleCheckpointWeightLoader(str(run / 'checkpoints/40000/params'))
    cfg = dataclasses.replace(cfg, optimizer=optimizer.SGD(momentum=0), ema_decay=None, weight_loader=loader,
        data=dataclasses.replace(cfg.data, assets=configs.AssetsConfig(assets_dir=str(BASE/'assets'), asset_id='physical-intelligence/libero')))
    actor, state, _ = Pi05Agent.initialize(cfg, mesh, jax.random.PRNGKey(42), data_sharding=data, replicated_sharding=replicated, freeze_pi05_encoder=False)
    actor.action_dim, actor.state_dim = 7, 8
    modes = [mode] if mode != 'fresh_lora' else ['fresh_lora', 'fresh_lora_zero_b', 'fresh_lora_zero_b_all_bf16']
    for variant in modes:
        if variant == 'fresh_lora_zero_b':
            params = nnx_utils.state_map(state.params, nnx_utils.PathRegex('.*lora_b.*'), lambda p: p.replace(jnp.zeros_like(p.value)))
            state = dataclasses.replace(state, params=params)
        if variant == 'fresh_lora_zero_b_all_bf16':
            params = nnx_utils.state_map(state.params, nnx.Param, lambda p: p.replace(p.value.astype(jnp.bfloat16)))
            state = dataclasses.replace(state, params=params)
        outputs = []
        for sample in samples:
            raw = {k: sample[k] for k in ('image','wrist_image','state','prompt')}
            observation = actor.process_raw_inputs(raw, 7, 224)
            action, _ = actor.sample_actions(observation, state, jax.random.PRNGKey(7), train=False, num_samples=1)
            action = np.asarray(actor.process_transformed_outputs(action))[0]
            outputs.append(dict(task=sample['task'], episode=sample['episode'], row=sample['row'], actions=action.tolist(), teacher_prefix_mae=float(np.abs(action[:5]-sample['target'][:5]).mean())))
        results[variant] = outputs
        (OUT/'actor_output_audit.json').write_text(json.dumps(results, indent=2)+'\n')
        print(f'DONE {variant}', flush=True)
    del actor, state
    gc.collect()
    jax.clear_caches()
for name, outputs in results.items():
    if name == 'official': continue
    diff = np.array([r['actions'] for r in outputs]) - np.array([r['actions'] for r in results['official']])
    print(f'{name} vs official: action MAE={np.abs(diff).mean():.6f} max={np.abs(diff).max():.6f}', flush=True)

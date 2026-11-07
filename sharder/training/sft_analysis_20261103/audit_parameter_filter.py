"""Count actual trainable parameters without allocating model weight arrays."""
import collections
import dataclasses
import json
import math
from pathlib import Path

import jax
from flax import nnx
from openpi.shared import nnx_utils
from configs.sft.pi05_base_libero import get_config

cfg = get_config()
model = dataclasses.replace(cfg.model, paligemma_variant='gemma_2b_lora', action_expert_variant='gemma_300m_lora')
cfg = dataclasses.replace(cfg, model=model, freeze_filter=model.get_freeze_filter(), ema_decay=None)
state = jax.eval_shape(lambda key: nnx.state(model.create(key)), jax.random.PRNGKey(42))
trainable = state.filter(cfg.trainable_filter)
strict = state.filter(nnx.All(nnx.Param, nnx_utils.PathRegex('.*lora.*')))
def summarize(tree):
    counts = collections.Counter()
    leaves = []
    for path, variable in tree.flat_state().items():
        value = variable.value
        n = math.prod(value.shape)
        name = '/'.join(map(str, path))
        counts['/'.join(map(str, path[:2]))] += n
        leaves.append(dict(name=name, shape=list(value.shape), parameters=n))
    return dict(total=sum(counts.values()), by_module=dict(counts), leaves=leaves)
result = dict(all=summarize(state), current_trainable=summarize(trainable), strict_lora=summarize(strict))
Path(__file__).with_name('parameter_filter_audit.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({k: {x:y for x,y in v.items() if x != 'leaves'} for k,v in result.items()}, indent=2))

"""Separate gradient scope from storage precision for a controlled SFT audit."""
import dataclasses
import hashlib
import json
import math
from pathlib import Path

import jax
import numpy as np
from flax import nnx
from openpi.shared import nnx_utils
from openpi.training.config import TrainConfig


@dataclasses.dataclass(frozen=True)
class AuditTrainConfig(TrainConfig):
    audit_scope: str = 'existing'

    @property
    def trainable_filter(self):
        if self.audit_scope == 'lora_only':
            return nnx.All(nnx.Param, nnx_utils.PathRegex('.*lora.*'))
        if self.audit_scope == 'existing':
            return super().trainable_filter
        raise ValueError(f'Unknown scope {self.audit_scope}')


def record_initialization(params, output_dir, scope):
    """Fingerprint samples from every leaf; no large model copy is written."""
    path = Path(output_dir) / 'initialization_audit.json'
    if path.exists():
        return
    leaves = []
    digest = hashlib.sha256()
    for key, variable in params.flat_state().items():
        array = variable.value
        values = np.asarray(jax.device_get(array.reshape(-1)[:32]))
        name = '/'.join(map(str, key))
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(values.tobytes())
        leaves.append(dict(name=name, shape=list(array.shape), dtype=str(array.dtype), parameters=math.prod(array.shape)))
    payload = dict(scope=scope, sampled_parameter_sha256=digest.hexdigest(), leaves=leaves,
                   note='A sampled fingerprint tests construction parity; it is not a hash of every model byte.')
    path.write_text(json.dumps(payload, indent=2)+'\n')
    print(f'INITIALIZATION_AUDIT scope={scope} sampled_parameter_sha256={digest.hexdigest()}', flush=True)

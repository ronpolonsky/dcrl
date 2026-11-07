"""Read checkpoint actor arrays on CPU and measure module-level drift from base."""
import gc
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
from flax import traverse_util
from expo_ft.utils.train_utils import _restore_pi05_params

OUT = Path(__file__).parent
base_path = '/projects/bgqe/rpolonsky/dcrlcheckpoints/openpi_cache/openpi-assets/checkpoints/pi05_libero/params'
print('Loading official base arrays', flush=True)
base = traverse_util.flatten_dict(_restore_pi05_params(base_path))
paths = {
 'sft28_40k': '/work/hdd/bgqe/rpolonsky/dcrlcheckpoints/sft_agnostic_small_expo/task_28_teacher_275001_success150_sft200k/checkpoints/40000/params',
 'expo28_40095': '/work/hdd/bgqe/rpolonsky/dcrlcheckpoints/pi_robo_libero_unsharded_h200_b16_200k/pi_robo_libero_unsharded_h200_b16_task28/checkpoints/40095/params',
}
results = {}
for name,path in paths.items():
 print(f'Loading {name}', flush=True)
 tree = traverse_util.flatten_dict(_restore_pi05_params(path))
 stats = defaultdict(lambda: dict(parameters=0, delta_sq=0., base_sq=0., max_abs_delta=0., changed=0))
 for key, value in tree.items():
  group = '/'.join(map(str, key[:2]))
  if key not in base:
   group = 'adapter/' + group
   original = np.zeros_like(value)
  else:
   original = base[key].astype(value.dtype)
  # Work one leaf at a time, avoiding a second full float32 model tree.
  a = np.asarray(value, np.float32)
  b = np.asarray(original, np.float32)
  diff = a-b
  s = stats[group]
  s['parameters'] += a.size
  s['delta_sq'] += float(np.sum(np.square(diff), dtype=np.float64))
  s['base_sq'] += float(np.sum(np.square(b), dtype=np.float64))
  s['max_abs_delta'] = max(s['max_abs_delta'], float(np.max(np.abs(diff))))
  s['changed'] += int(np.count_nonzero(diff))
 for s in stats.values():
  s['delta_rms'] = (s['delta_sq']/s['parameters'])**.5
  s['relative_l2'] = (s['delta_sq']/s['base_sq'])**.5 if s['base_sq'] else None
 results[name] = dict(checkpoint=path, modules=dict(stats))
 (OUT/'weight_drift_audit.json').write_text(json.dumps(results, indent=2)+'\n')
 print(json.dumps(results[name], indent=2), flush=True)
 del tree
 gc.collect()

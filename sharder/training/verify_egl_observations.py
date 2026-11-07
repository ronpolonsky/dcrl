"""Compare original and explicitly bound EGL rendering on identical actions."""

import ast
import resource
import random

resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

import numpy as np
from dcrl.libero_env import LiberoTaskEnv, LIBERO_RENDER_RESOLUTION
from robosuite.utils import binding_utils

with open('/work/hdd/bgqe/rpolonsky/unsharded_h200_b16_task29_debug.log') as file:
    actions = [np.asarray(ast.literal_eval(line.split('values=', 1)[1]), np.float32)
               for line in file if line.startswith('[ACTION CHECK]')]


def compare(left, right, label):
    if isinstance(left, dict):
        assert left.keys() == right.keys(), label
        for key in left:
            compare(left[key], right[key], f'{label}/{key}')
    elif isinstance(left, tuple):
        for index, (a, b) in enumerate(zip(left, right, strict=True)):
            compare(a, b, f'{label}/{index}')
    else:
        np.testing.assert_array_equal(left, right, err_msg=label)


def rollout():
    # LIBERO/robosuite constructor also consumes process-global RNGs before
    # env.seed() is called. Match fresh-process seed initialization here.
    random.seed(7)
    np.random.seed(7)
    env = LiberoTaskEnv('libero_90', 29, seed=7, image_size=224, benchmark_eval=True)
    try:
        results = [env.reset()]
        for index in range(100):
            results.append(env.step(actions[min(index, len(actions)-1)]))
        return results
    finally:
        env.close()


original = binding_utils.MjRenderContext.render


def bound(self, *args, **kwargs):
    self.gl_ctx.make_current()
    return original(self, *args, **kwargs)


# Compare captures of the SAME simulation state. Separate LIBERO constructors
# vary their initial XML/visuals even with the same process RNG seeds, so a
# comparison across constructors cannot establish rendering equivalence.
env = LiberoTaskEnv('libero_90', 29, seed=7, image_size=224, benchmark_eval=True)
try:
    env.reset()
    for index in range(100):
        binding_utils.MjRenderContext.render = original
        env.step(actions[min(index, len(actions)-1)])
        state_before = env.env.sim.get_state().flatten().copy()
        for camera in ('agentview', 'robot0_eye_in_hand'):
            binding_utils.MjRenderContext.render = original
            baseline = env.env.sim.render(width=256, height=256, camera_name=camera)
            binding_utils.MjRenderContext.render = bound
            candidate = env.env.sim.render(width=256, height=256, camera_name=camera)
            compare(baseline, candidate, f'step/{index}/camera/{camera}')
        compare(state_before, env.env.sim.get_state().flatten(), f'step/{index}/state')
    print(f'EXACT_RGB_PASS render={LIBERO_RENDER_RESOLUTION} '
          'states=100 cameras=2: raw RGB bit-identical; physics state unchanged', flush=True)
finally:
    env.close()

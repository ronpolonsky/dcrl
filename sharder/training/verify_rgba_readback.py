"""Compare RGB and RGBA transfers from the same framebuffer, without rerender."""

import json
import os
import numpy as np
from dcrl.libero_env import LiberoTaskEnv
from train_pi_robo_libero_rgba_readback import rgba_read_pixels
if os.getenv('VERIFY_PBO') == '1':
    from train_pi_robo_libero_pbo_readback import pbo_read_pixels as rgba_read_pixels

with open('/work/hdd/bgqe/rpolonsky/pi05_render_trace_task29.log') as file:
    actions = [json.loads(line.split('] ', 1)[1])['action']
               for line in file if line.startswith('[RENDER ACTION]')]
env = LiberoTaskEnv('libero_90', 29, seed=7, image_size=224, benchmark_eval=True)
try:
    env.reset()
    context = env.env.sim._render_context_offscreen
    for index, action in enumerate(actions):
        env.step(np.asarray(action, np.float32))
        for camera in ('agentview', 'robot0_eye_in_hand'):
            baseline = env.env.sim.render(width=256, height=256, camera_name=camera)
            candidate = rgba_read_pixels(context, 256, 256)
            np.testing.assert_array_equal(baseline, candidate,
                                          err_msg=f'step={index}, camera={camera}')
    print(f'READBACK_EXACT_PASS (PBO={os.getenv("VERIFY_PBO", "0")}): '
          f'{len(actions)} states x 2 cameras, 256x256 RGB bit-identical', flush=True)
finally:
    env.close()

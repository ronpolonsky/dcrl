"""Use NVIDIA's RGBA transfer path, returning unchanged 256x256 RGB pixels."""

import resource
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

import numpy as np
import mujoco
from robosuite.utils import binding_utils

_original_read_pixels = binding_utils.MjRenderContext.read_pixels
_original_render = binding_utils.MjRenderContext.render


def render_with_current_context(self, *args, **kwargs):
    self.gl_ctx.make_current()
    return _original_render(self, *args, **kwargs)


def rgba_read_pixels(self, width, height, depth=False, segmentation=False):
    # Preserve the original API for modes unused by LIBERO's two RGB cameras.
    if depth or segmentation:
        return _original_read_pixels(self, width, height, depth=depth, segmentation=segmentation)
    rgba = np.empty((height, width, 4), dtype=np.uint8)
    previous_format = self.con.readPixelFormat
    try:
        self.con.readPixelFormat = 0x1908  # GL_RGBA
        mujoco.mjr_readPixels(rgb=rgba, depth=None,
                             viewport=mujoco.MjrRect(0, 0, width, height), con=self.con)
    finally:
        self.con.readPixelFormat = previous_format
    return np.ascontiguousarray(rgba[:, :, :3])


def install():
    binding_utils.MjRenderContext.render = render_with_current_context
    binding_utils.MjRenderContext.read_pixels = rgba_read_pixels


if __name__ == '__main__':
    install()
    import train_pi_robo_libero_logged as logged
    print('[RGBA READBACK] same renderer, framebuffer, cameras and RGB; alpha discarded', flush=True)
    logged.trainer.app.run(logged.trainer.main)

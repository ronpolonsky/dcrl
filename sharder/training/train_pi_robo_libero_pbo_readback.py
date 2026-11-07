"""Preserve MuJoCo's RGB rendering, stage readback through a pixel buffer."""

import ctypes
import resource
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

import mujoco
import numpy as np
from OpenGL import GL
from OpenGL.raw.GL.VERSION.GL_1_0 import glReadPixels
from robosuite.utils import binding_utils

_original_read_pixels = binding_utils.MjRenderContext.read_pixels
_original_render = binding_utils.MjRenderContext.render


def pbo_read_pixels(self, width, height, depth=False, segmentation=False):
    if depth or segmentation or self.con.currentBuffer != mujoco.mjtFramebuffer.mjFB_OFFSCREEN:
        return _original_read_pixels(self, width, height, depth=depth, segmentation=segmentation)
    result = np.empty((height, width, 3), np.uint8)
    previous_pbo = int(GL.glGetIntegerv(GL.GL_PIXEL_PACK_BUFFER_BINDING))
    if not hasattr(self, '_diagnostic_readback_pbo'):
        self._diagnostic_readback_pbo = int(GL.glGenBuffers(1))
    try:
        # Match MuJoCo's resolve and framebuffer selection exactly.
        GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, self.con.offFBO)
        GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT0)
        if self.con.offSamples:
            GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self.con.offFBO_r)
            GL.glDrawBuffer(GL.GL_COLOR_ATTACHMENT0)
            GL.glBlitFramebuffer(0, 0, width, height, 0, 0, width, height,
                                 GL.GL_COLOR_BUFFER_BIT, GL.GL_NEAREST)
            GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, self.con.offFBO_r)
            GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT0)
        GL.glBindBuffer(GL.GL_PIXEL_PACK_BUFFER, self._diagnostic_readback_pbo)
        GL.glBufferData(GL.GL_PIXEL_PACK_BUFFER, result.nbytes, None, GL.GL_STREAM_READ)
        glReadPixels(0, 0, width, height, GL.GL_RGB, GL.GL_UNSIGNED_BYTE, ctypes.c_void_p(0))
        GL.glGetBufferSubData(GL.GL_PIXEL_PACK_BUFFER, 0, result.nbytes, result)
    finally:
        GL.glBindBuffer(GL.GL_PIXEL_PACK_BUFFER, previous_pbo)
        mujoco.mjr_restoreBuffer(self.con)
    return result


def render_with_current_context(self, *args, **kwargs):
    self.gl_ctx.make_current()
    return _original_render(self, *args, **kwargs)


def install():
    binding_utils.MjRenderContext.render = render_with_current_context
    binding_utils.MjRenderContext.read_pixels = pbo_read_pixels


if __name__ == '__main__':
    install()
    import train_pi_robo_libero_logged as logged
    print('[PBO READBACK] unchanged RGB framebuffer; pixel-buffer transfer', flush=True)
    logged.trainer.app.run(logged.trainer.main)

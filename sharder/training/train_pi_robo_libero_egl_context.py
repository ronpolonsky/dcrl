"""Isolated EGL-context diagnostic; leaves model and image processing intact."""

import resource
import os

resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

from robosuite.utils import binding_utils

_original_render = binding_utils.MjRenderContext.render


def _render_with_current_context(self, *args, **kwargs):
    # EGL contexts belong to a thread. Explicitly bind this environment's
    # context when alternating training/evaluation environments or GPU work.
    self.gl_ctx.make_current()
    return _original_render(self, *args, **kwargs)


binding_utils.MjRenderContext.render = _render_with_current_context

from OpenGL import GL
from dcrl import libero_env

if os.getenv('EGL_SYNC_READBACK') == '1':
    _original_read_pixels = binding_utils.MjRenderContext.read_pixels

    def _synchronized_read_pixels(self, *args, **kwargs):
        self.gl_ctx.make_current()
        GL.glFinish()
        result = _original_read_pixels(self, *args, **kwargs)
        GL.glFinish()
        return result

    binding_utils.MjRenderContext.read_pixels = _synchronized_read_pixels

_original_reset = libero_env.LiberoTaskEnv.reset
_reported = False


def _reset_with_renderer_report(self, *args, **kwargs):
    global _reported
    result = _original_reset(self, *args, **kwargs)
    if not _reported:
        print("[EGL CHECK] render_resolution=%d vendor=%r renderer=%r version=%r"
              % (libero_env.LIBERO_RENDER_RESOLUTION,
                 GL.glGetString(GL.GL_VENDOR), GL.glGetString(GL.GL_RENDERER),
                 GL.glGetString(GL.GL_VERSION)), flush=True)
        _reported = True
    return result


libero_env.LiberoTaskEnv.reset = _reset_with_renderer_report

import train_pi_robo_libero_logged as logged


if __name__ == "__main__":
    print("[EGL CHECK] explicitly binding each environment before rendering", flush=True)
    logged.trainer.app.run(logged.trainer.main)

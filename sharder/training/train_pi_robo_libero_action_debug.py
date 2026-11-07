"""Temporary isolated wrapper for diagnosing LIBERO policy actions."""

import numpy as np

import train_pi_robo_libero as trainer


_original_step = trainer._LiberoEnvClient.step
_step_count = 0


def _step_with_action_report(self, action):
    global _step_count
    values = np.asarray(action, dtype=np.float64)
    safe_action = np.clip(values, -1.0, 1.0).astype(np.float32)
    if _step_count < 40:
        print(
            "[ACTION CHECK] step=%d finite=%s raw_min=%s raw_max=%s clipped=%s values=%s"
            % (
                _step_count,
                bool(np.isfinite(values).all()),
                float(np.nanmin(values)),
                float(np.nanmax(values)),
                bool(np.any(values != safe_action)),
                values.tolist(),
            ),
            flush=True,
        )
    _step_count += 1
    return _original_step(self, safe_action)


trainer._LiberoEnvClient.step = _step_with_action_report


if __name__ == "__main__":
    trainer.app.run(trainer.main)

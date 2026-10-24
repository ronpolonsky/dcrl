"""Small smoke test for Orbax checkpoint retention.

This exercises the same checkpoint-manager construction used by the LIBERO
trainer, without loading a model, environment, replay buffer, or evaluator.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
from expo_ft.agents.alg.agent import initialize_checkpoint_dir


def main() -> None:
    root = Path(tempfile.mkdtemp(prefix="dcrl_checkpoint_retention_"))
    manager = None
    try:
        manager, resuming = initialize_checkpoint_dir(
            root,
            keep_period=None,
            overwrite=True,
            resume=False,
            max_to_keep=5,
            keep_checkpoints_without_metrics=False,
        )
        assert not resuming

        # Save more than the retention limit. These are the same two top-level
        # items used by expo_ft's save_checkpoint helper.
        for step in range(1, 8):
            manager.save(
                step,
                {
                    "agent": {"optimizer_step": np.asarray(step, dtype=np.int32)},
                    "params": {"weight": np.asarray([step], dtype=np.float32)},
                },
            )
            manager.wait_until_finished()

        steps = list(manager.all_steps())
        assert steps == [3, 4, 5, 6, 7], f"unexpected retained steps: {steps}"

        # Verify that every retained checkpoint is complete and corresponds to
        # its own step, rather than merely checking directory names.
        for step in steps:
            restored = manager.restore(
                step,
                items={
                    "agent": {"optimizer_step": np.asarray(0, dtype=np.int32)},
                    "params": {"weight": np.asarray([0], dtype=np.float32)},
                },
            )
            assert int(restored["agent"]["optimizer_step"]) == step
            assert int(restored["params"]["weight"][0]) == step

        print(f"PASS: retained exactly {steps}; every checkpoint restored correctly")
    finally:
        if manager is not None:
            manager.close()
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()

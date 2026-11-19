"""Shared Orbax setup for retaining a trainer's best evaluation checkpoint."""
from __future__ import annotations

from pathlib import Path

import orbax.checkpoint as ocp


def evaluation_checkpoint_metrics(score: float, step: int, *,
                                  score_key: str = "eval/success_rate") -> dict[str, float | int]:
    return {score_key: float(score), "global_step": int(step)}


def create_best_checkpoint_manager(directory: Path | str, *, item_handlers: dict | None = None,
                                   score_key: str = "eval/success_rate"):
    """Keep one highest-scoring checkpoint; later steps win exact score ties."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if item_handlers is None:
        item_handlers = {
            "agent": ocp.PyTreeCheckpointHandler(),
            "params": ocp.PyTreeCheckpointHandler(),
        }
    return ocp.CheckpointManager(
        directory,
        item_handlers=item_handlers,
        options=ocp.CheckpointManagerOptions(
            max_to_keep=1,
            best_fn=lambda metrics: (
                float(metrics[score_key])
                + int(metrics["global_step"]) * 1e-12
            ),
            best_mode="max",
            keep_checkpoints_without_metrics=False,
            create=True,
            async_options=ocp.AsyncOptions(timeout_secs=7200),
        ),
    )

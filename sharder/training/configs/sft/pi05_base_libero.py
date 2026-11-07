"""Explicit Pi0.5-base configuration for LIBERO training.

This is the official ``pi05_libero`` setup with an unambiguous name and an
explicit ``pi05_base`` weight loader.  The model and preprocessing settings
are kept identical; only the configuration name is local to DCRL.
"""

from __future__ import annotations

from openpi.models import pi0_config
from openpi.training import config as openpi_config
from openpi.training import optimizer as _optimizer
from openpi.training import weight_loaders


def get_config() -> openpi_config.TrainConfig:
    """Build the official Pi0.5 LIBERO config with base weights."""

    return openpi_config.TrainConfig(
        name="pi05_base_libero",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
        ),
        data=openpi_config.LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=openpi_config.DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        pytorch_weight_path="/path/to/your/pytorch_weight_path",
        num_train_steps=30_000,
    )

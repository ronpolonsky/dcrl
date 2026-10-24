"""Wiring for learner-agnostic Pi0.5 LIBERO SFT.

The learner itself lives in ``dcrl.agents.my_expo_learner``.  This module is
only a factory/configuration boundary so the SFT loop does not contain
learner-specific construction code.
"""

from __future__ import annotations

import dataclasses
import importlib
from pathlib import Path

import numpy as np


LEARNER_CLASS = "dcrl.agents.my_expo_learner:MyEXPOLearner"


def _resolve_class(path: str):
    module_name, separator, class_name = path.partition(":")
    if not separator:
        module_name, class_name = path.rsplit(".", 1)
    return getattr(importlib.import_module(module_name), class_name)


def build_learner(*, args, output_dir: Path, instruction: str, stats_dir: Path,
                  mesh, data_sharding, replicated_sharding):
    """Build the configured learner and its shared Pi0.5 data transforms."""
    import jax
    import jax.numpy as jnp
    from openpi import transforms
    from openpi.shared import normalize
    from openpi.training import config as openpi_configs, optimizer, weight_loaders
    from dcrl.expo.agents.vla.pi05 import Pi05Agent
    from configs.sft import pi05_base_libero

    learner_cls = _resolve_class(LEARNER_CLASS)
    stats = normalize.load(stats_dir)

    base = pi05_base_libero.get_config()
    model = dataclasses.replace(base.model, action_horizon=args.action_horizon)
    if args.finetune_mode == "lora":
        model = dataclasses.replace(
            model,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        )
    train_config = dataclasses.replace(
        base,
        model=model,
        freeze_filter=model.get_freeze_filter(),
        ema_decay=None if args.finetune_mode == "lora" else base.ema_decay,
        data=dataclasses.replace(
            base.data,
            assets=openpi_configs.AssetsConfig(
                assets_dir=str(output_dir / "student_assets"),
                asset_id="physical-intelligence/libero",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            args.student_checkpoint.rstrip("/") + "/params"
        ),
        lr_schedule=optimizer.CosineDecaySchedule(
            warmup_steps=args.warmup_steps,
            peak_lr=args.learning_rate,
            decay_steps=args.lr_decay_steps,
            decay_lr=args.learning_rate,
        ),
        batch_size=args.batch_size,
        num_train_steps=args.num_sft_steps,
    )

    actor, actor_train_state, _ = Pi05Agent.initialize(
        train_config,
        mesh,
        jax.random.PRNGKey(args.seed),
        resume=False,
        data_sharding=data_sharding,
        replicated_sharding=replicated_sharding,
        default_prompt=instruction,
        freeze_pi05_encoder=False,
    )
    actor.action_dim = 7
    actor.state_dim = 8
    target_actor_params = actor.get_params(actor_train_state)

    # MyEXPOLearner uses these only to initialize its EXPO critic/editor
    # structures. SFT gradients still flow through the base actor only.
    example_observation = jnp.zeros(
        (args.image_size, args.image_size, 6), dtype=jnp.float32
    )
    example_action = np.zeros((7,), dtype=np.float32)
    example_state = np.zeros((8,), dtype=np.float32)
    learner = learner_cls.create(
        seed=args.seed,
        observation_space=example_observation,
        action_space=example_action,
        states=example_state,
        actor=actor,
        actor_train_state=actor_train_state,
        target_actor_params=target_actor_params,
        action_horizon=args.action_horizon,
        mesh=mesh,
        freeze_encoder=True,
        replan_steps=args.replan_steps,
        data_sharding=data_sharding,
        replicated_sharding=replicated_sharding,
        default_prompt=instruction,
        resize_size=args.image_size,
        actor_success_only=True,
        use_full_augmentation=True,
        resume=args.resume,
        freeze_critic_encoder=False,
        num_qs=10,
        num_min_qs=2,
        N=8,
        n_edit_samples=8,
        edit_scale=0.2,
        latent_dim_image=512,
        latent_dim_state=64,
        encoder_stage_sizes=(3, 4, 6, 3),
        encoder_num_filters=64,
        hidden_dims=(256, 256, 256),
        critic_layer_norm=True,
    )

    dc = actor.data_config
    transform = transforms.compose([
        *dc.data_transforms.inputs,
        transforms.Normalize(stats, use_quantiles=dc.use_quantile_norm),
        *dc.model_transforms.inputs,
    ])
    output_transform = transforms.compose([
        *dc.model_transforms.outputs,
        transforms.Unnormalize(stats, use_quantiles=dc.use_quantile_norm),
        *dc.data_transforms.outputs,
    ])
    return learner, transform, output_transform

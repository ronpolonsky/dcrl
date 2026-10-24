"""Small EXPO learner with the original RL rules and compact trainable models."""

from functools import partial
from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from dcrl.agents.pi_expo_learner import PiExpoLearner
from dcrl.agents.small_actor import SmallActor
from dcrl.expo.agents.alg.batch_utils import CRITIC_CAMERA_KEYS
from dcrl.expo.networks.tiny_vit import TinyViTEncoder


class SmallExpoLearner(PiExpoLearner):
    """EXPO updates with a small flow actor and one shared ~1M ViT encoder.

    The critic ensemble and edit policy use the *same* heads and losses as
    PiExpoLearner. Their shared image encoder is trained by the critic loss.
    The action actor has its own compact vision encoder and generates complete
    action chunks with independent noise for candidate sampling.
    """

    @classmethod
    def from_examples(
        cls,
        *,
        seed: int,
        example_observation: Mapping[str, Any],
        example_action: Any,
        example_state: Any,
        mesh: Any,
        data_sharding: Any,
        replicated_sharding: Any,
        action_horizon: int = 10,
        replan_steps: int = 5,
        padded_action_dim: int = 32,
        image_size: int = 224,
        camera_keys: Sequence[str] = CRITIC_CAMERA_KEYS,
        actor_lr: float = 3e-4,
        action_mean: Any = None,
        action_std: Any = None,
        state_mean: Any = None,
        state_std: Any = None,
        norm_stats: Any = None,
        pi05_train_config: Any = None,
        **expo_kwargs,
    ) -> 'SmallExpoLearner':
        """Initialize all three small components; no Pi0.5 checkpoint is loaded.

        `example_observation` has an `image` dict and `state`; raw camera keys
        at its top level are also accepted. Replay batches retain EXPO's padded
        state/action format, reward aggregation and replan-step discounting.
        """
        action_dim = int(np.asarray(example_action).shape[-1])
        state_dim = int(np.asarray(example_state).shape[-1])
        if not (1 <= replan_steps <= action_horizon):
            raise ValueError('replan_steps must be between 1 and action_horizon')
        if padded_action_dim < max(action_dim, state_dim):
            raise ValueError('padded_action_dim must cover action and state dimensions')
        # Match configs/model/expo_ft_pi_config.py, except for the two compact encoders.
        preset = dict(num_qs=10, num_min_qs=2, critic_layer_norm=True,
                      N=8, n_edit_samples=8, edit_scale=0.2,
                      latent_dim_image=512, latent_dim_state=64,
                      hidden_dims=(256, 256, 256), actor_success_only=True)
        for name, value in preset.items():
            expo_kwargs.setdefault(name, value)
        if expo_kwargs['n_edit_samples'] > expo_kwargs['N']:
            raise ValueError('n_edit_samples cannot exceed N')
        if 'encoder_cls' in expo_kwargs:
            raise ValueError('SmallExpoLearner fixes the critic/edit encoder to TinyViTEncoder')
        infer_sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])
        actor = SmallActor(
            action_dim=action_dim, action_horizon=action_horizon,
            camera_keys=tuple(camera_keys), mesh=mesh,
            data_sharding=data_sharding, infer_sharding=infer_sharding,
            image_size=image_size, padded_action_dim=padded_action_dim,
            action_mean=action_mean, action_std=action_std,
            state_mean=state_mean, state_std=state_std,
            norm_stats=norm_stats, pi05_train_config=pi05_train_config,
        )
        transformed = actor.process_raw_inputs(example_observation, action_dim, image_size)
        actor_state = actor.initialize(seed, transformed, actor_lr)
        critic_image = jnp.concatenate([transformed['image'][k] for k in camera_keys], axis=-1)
        if critic_image.shape[0] != 1:
            raise ValueError('example_observation must contain one example')
        expo_kwargs.setdefault('freeze_encoder', True)
        return cls.create(
            seed=seed,
            observation_space=critic_image[0],
            action_space=np.asarray(example_action),
            states=np.asarray(example_state),
            actor=actor,
            actor_train_state=actor_state,
            target_actor_params=actor.get_params(actor_state),
            action_horizon=action_horizon,
            replan_steps=replan_steps,
            actor_lr=actor_lr,
            mesh=mesh,
            data_sharding=data_sharding,
            replicated_sharding=replicated_sharding,
            encoder_cls=partial(TinyViTEncoder, width=144, depth=4, heads=4,
                                mlp_width=288, patch_size=16),
            critic_camera_keys=tuple(camera_keys),
            **expo_kwargs,
        )

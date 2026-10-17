"""Small flow-matching action-chunk actor with the PiExpo actor interface."""

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import struct
from flax.training.train_state import TrainState

from dcrl.expo.networks.tiny_vit import TinyViTEncoder


@dataclass(frozen=True)
class SmallActorConfig:
    action_dim: int
    action_horizon: int


@struct.dataclass
class SmallActorTrainState(TrainState):
    # Keep the same checkpoint-facing fields used by PiExpoLearner.
    ema_params: Any = None


class _FlowBlock(nn.Module):
    width: int

    @nn.compact
    def __call__(self, x):
        residual = x
        x = nn.LayerNorm()(x)
        x = nn.Dense(self.width)(x)
        x = nn.gelu(x)
        x = nn.Dense(self.width)(x)
        return residual + x


class SmallFlowActor(nn.Module):
    """Chunk-level conditional flow model with image, state, and prompt inputs."""

    action_horizon: int
    action_dim: int
    camera_keys: tuple[str, ...]
    width: int = 384
    flow_blocks: int = 2
    denoise_steps: int = 10
    vocab_size: int = 8192
    prompt_dim: int = 64

    @nn.compact
    def __call__(self, observation, noisy_actions, time):
        images = jnp.concatenate([observation['image'][k] for k in self.camera_keys], axis=-1)
        vision = TinyViTEncoder(name='vision')(images)
        state = jnp.asarray(observation['state'], jnp.float32)
        tokens = observation.get('tokenized_prompt')
        token_mask = observation.get('tokenized_prompt_mask')
        if tokens is None:
            prompt = jnp.zeros((state.shape[0], self.prompt_dim), jnp.float32)
        else:
            ids = jnp.mod(jnp.asarray(tokens, jnp.int32), self.vocab_size)
            embedded = nn.Embed(self.vocab_size, self.prompt_dim, name='prompt_embedding')(ids)
            mask = jnp.ones_like(ids, jnp.float32) if token_mask is None else jnp.asarray(token_mask, jnp.float32)
            prompt = (embedded * mask[..., None]).sum(axis=1) / jnp.maximum(mask.sum(axis=1, keepdims=True), 1)
        time = jnp.broadcast_to(jnp.asarray(time, jnp.float32).reshape(-1, 1), (state.shape[0], 1))
        x = jnp.concatenate([vision, state, prompt, noisy_actions.reshape(state.shape[0], -1), time], axis=-1)
        x = nn.Dense(self.width)(x)
        for i in range(self.flow_blocks):
            x = _FlowBlock(self.width, name=f'flow_block_{i}')(x)
        return nn.Dense(self.action_horizon * self.action_dim,
                        kernel_init=nn.initializers.normal(0.01))(nn.gelu(x)).reshape(
                            state.shape[0], self.action_horizon, self.action_dim)


class SmallActor:
    """Adapter used by the unchanged PiExpoLearner actor sampling/update paths."""

    def __init__(self, *, action_dim: int, action_horizon: int, camera_keys: Sequence[str],
                 mesh: Any, data_sharding: Any, infer_sharding: Any,
                 image_size: int = 224, denoise_steps: int = 10, padded_action_dim: int = 32,
                 action_mean=None, action_std=None, state_mean=None, state_std=None,
                 norm_stats=None, pi05_train_config=None):
        self.action_dim = action_dim
        if padded_action_dim < action_dim:
            raise ValueError("padded_action_dim must cover the environment action dimension")
        self.model_config = SmallActorConfig(padded_action_dim, action_horizon)
        self.camera_keys = tuple(camera_keys)
        self.mesh = mesh
        self.data_sharding = data_sharding
        self.infer_sharding = infer_sharding
        self.image_size = image_size
        self.denoise_steps = denoise_steps
        self.action_mean = jnp.zeros((action_dim,)) if action_mean is None else jnp.asarray(action_mean)
        self.action_std = jnp.ones((action_dim,)) if action_std is None else jnp.asarray(action_std)
        if np.any(np.asarray(self.action_std) <= 0):
            raise ValueError('action_std must be positive')
        self.state_mean = None if state_mean is None else jnp.asarray(state_mean)
        self.state_std = None if state_std is None else jnp.asarray(state_std)
        if self.state_std is not None and np.any(np.asarray(self.state_std) <= 0):
            raise ValueError('state_std must be positive')
        self.input_transforms = None
        self.input_model_transforms = None
        self.output_transforms = None
        self.state_normalize = None
        self.action_unnormalize = None
        if norm_stats is not None:
            if pi05_train_config is not None:
                raise ValueError('Provide either norm_stats or pi05_train_config')
            from openpi import transforms as _transforms
            self.state_normalize = _transforms.Normalize(
                {'state': norm_stats['state']}, use_quantiles=True)
            self.action_unnormalize = _transforms.Unnormalize(
                {'actions': norm_stats['actions']}, use_quantiles=True)
        if pi05_train_config is not None:
            from openpi import transforms as _transforms
            model_config = pi05_train_config.model
            if model_config.action_horizon != action_horizon or model_config.action_dim != padded_action_dim:
                raise ValueError('action_horizon and padded_action_dim must match pi05_train_config')
            data_config = pi05_train_config.data.create(pi05_train_config.assets_dirs, model_config)
            self.input_transforms = _transforms.compose([
                *data_config.repack_transforms.inputs,
                *data_config.data_transforms.inputs,
                _transforms.Normalize(data_config.norm_stats,
                                      use_quantiles=data_config.use_quantile_norm),
            ])
            self.input_model_transforms = _transforms.compose(data_config.model_transforms.inputs)
            self.output_transforms = _transforms.compose([
                *data_config.model_transforms.outputs,
                _transforms.Unnormalize(data_config.norm_stats,
                                        use_quantiles=data_config.use_quantile_norm),
                *data_config.data_transforms.outputs,
            ])
        self.model = SmallFlowActor(action_horizon, action_dim, self.camera_keys, denoise_steps=denoise_steps)

    def initialize(self, seed: int, example_observation: Mapping, lr: float = 3e-4):
        key = jax.random.PRNGKey(seed)
        batch = self._model_inputs(example_observation)
        dummy = jnp.zeros((batch['state'].shape[0], self.model_config.action_horizon, self.action_dim))
        params = self.model.init(key, batch, dummy, jnp.zeros((batch['state'].shape[0],)))['params']
        return SmallActorTrainState.create(apply_fn=self.model.apply, params=params,
                                           tx=optax.adam(lr))

    def _model_inputs(self, observations):
        inputs = {k: observations[k] for k in ('image', 'state', 'tokenized_prompt', 'tokenized_prompt_mask')
                  if k in observations and observations[k] is not None}
        state = jnp.asarray(inputs['state'], jnp.float32)
        if state.shape[-1] > self.model_config.action_dim:
            raise ValueError('state dimension exceeds padded_action_dim')
        inputs['state'] = jnp.pad(state, ((0, 0), (0, self.model_config.action_dim - state.shape[-1])))
        return inputs

    def get_params(self, train_state):
        return train_state.params if train_state.ema_params is None else train_state.ema_params

    def process_raw_inputs(self, raw_observations, action_dim, resize_size):
        if self.input_transforms is not None:
            raw = dict(raw_observations)
            raw['actions'] = np.zeros(action_dim, dtype=np.float32)
            for key, value in list(raw.items()):
                raw[key] = np.asarray(value)
                if 'image' in key:
                    raw[key] = raw[key].astype(np.uint8)
            transformed = self.input_model_transforms(self.input_transforms(raw))
            packed = jax.tree.map(lambda x: jnp.asarray(x)[None, ...], transformed)
            packed['image'] = {
                k: v.astype(jnp.float32) / 255.0 * 2.0 - 1.0 if v.dtype == jnp.uint8 else v
                for k, v in packed['image'].items()
            }
            return {k: packed.get(k) for k in ('image', 'image_mask', 'state',
                    'tokenized_prompt', 'tokenized_prompt_mask', 'token_ar_mask', 'token_loss_mask')}
        if 'prompt' in raw_observations and 'tokenized_prompt' not in raw_observations:
            raise ValueError('provide tokenized_prompt and tokenized_prompt_mask for language-conditioned rollout')
        images = raw_observations.get('image', raw_observations)
        processed = {}
        for name in self.camera_keys:
            x = jnp.asarray(images[name])
            if x.ndim == 3:
                x = x[None]
            if x.shape[1:3] != (self.image_size, self.image_size):
                x = jax.image.resize(x, (x.shape[0], self.image_size, self.image_size, x.shape[-1]), method='bilinear')
            if np.issubdtype(np.asarray(images[name]).dtype, np.integer) or np.asarray(images[name]).max() > 1.0:
                x = x.astype(jnp.float32) / 127.5 - 1.0
            processed[name] = x
        state = jnp.asarray(raw_observations['state'], jnp.float32)
        if self.state_normalize is not None:
            state = jnp.asarray(self.state_normalize({'state': np.asarray(state)})['state'])
        elif self.state_mean is not None:
            state = (state - self.state_mean) / self.state_std
        if state.ndim == 1:
            state = state[None]
        if state.shape[-1] > self.model_config.action_dim:
            raise ValueError('state dimension exceeds padded_action_dim')
        state = jnp.pad(state, ((0, 0), (0, self.model_config.action_dim - state.shape[-1])))
        tokens = raw_observations.get('tokenized_prompt')
        if tokens is None:
            tokens = jnp.zeros((state.shape[0], 1), jnp.int32)
            token_mask = jnp.zeros_like(tokens, jnp.bool_)
        else:
            tokens = jnp.asarray(tokens, jnp.int32)
            if tokens.ndim == 1:
                tokens = tokens[None]
            token_mask = raw_observations.get('tokenized_prompt_mask')
            if token_mask is None:
                token_mask = jnp.ones_like(tokens, jnp.bool_)
            else:
                token_mask = jnp.asarray(token_mask, jnp.bool_)
                if token_mask.ndim == 1:
                    token_mask = token_mask[None]
        return {'image': processed,
                'image_mask': {k: jnp.ones((state.shape[0],), jnp.bool_) for k in processed},
                'state': state,
                'tokenized_prompt': tokens,
                'tokenized_prompt_mask': token_mask}

    def process_transformed_outputs(self, actions):
        if self.output_transforms is not None:
            n = actions.shape[0]
            padded = jnp.pad(actions, ((0, 0), (0, 0),
                                       (0, self.model_config.action_dim - self.action_dim)))
            dummy_state = np.zeros((n, self.model_config.action_dim), np.float32)
            result = [self.output_transforms({'state': dummy_state[i],
                                              'actions': np.asarray(padded[i])})['actions']
                      for i in range(n)]
            return jnp.asarray(np.stack(result))
        if self.action_unnormalize is not None:
            raw = self.action_unnormalize({'actions': np.asarray(actions)})['actions']
            return jnp.asarray(raw)
        return actions * self.action_std + self.action_mean

    def prepare_batch_for_actor(self, batch):
        actions = batch['full_actions'].reshape(-1, self.model_config.action_horizon, self.action_dim)
        return self._model_inputs(batch), actions

    def train_step(self, rng, state, batch):
        observations, actions = batch
        noise_key, time_key = jax.random.split(rng)
        noise = jax.random.normal(noise_key, actions.shape)
        time = jax.random.beta(time_key, 1.5, 1.0, (actions.shape[0],)) * 0.999 + 0.001
        noisy = time[:, None, None] * noise + (1 - time[:, None, None]) * actions

        def loss_fn(params):
            velocity = state.apply_fn({'params': params}, observations, noisy, time)
            return jnp.mean((velocity - (noise - actions)) ** 2)

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, {'actor_loss': loss, 'actor_grad_norm': optax.global_norm(grads),
                       'actor_param_norm': optax.global_norm(state.params), 'actor_state_step': state.step}

    def _sample(self, observations, train_state, rng, num_samples):
        obs = self._model_inputs(observations)
        obs = jax.tree_util.tree_map(lambda x: jnp.repeat(x, num_samples, axis=0), obs)
        shape = (obs['state'].shape[0], self.model_config.action_horizon, self.action_dim)
        x = jax.random.normal(rng, shape)
        for step in range(self.denoise_steps, 0, -1):
            t = jnp.full((shape[0],), step / self.denoise_steps)
            velocity = train_state.apply_fn({'params': self.get_params(train_state)}, obs, x, t)
            x = x - velocity / self.denoise_steps
        return x

    def sample_actions(self, transformed_inputs, train_state, rng, train=False, num_samples=1):
        return self._sample(transformed_inputs, train_state, rng, num_samples), 0.0

    def sample_training_actions(self, transformed_inputs, train_state, rng, train=False, num_samples=1):
        return self._sample(transformed_inputs, train_state, rng, num_samples), 0.0

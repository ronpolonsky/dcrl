"""Wrapper for pi05 agent to make it compatible with expo training framework."""

import dataclasses
import functools
from functools import partial
from typing import Any, Dict, Optional, Tuple, Union

import flax.nnx as nnx
import flax.traverse_util as traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import struct
import orbax.checkpoint as ocp


import openpi.models.model as _model
import openpi.models.pi0 as _pi0  # for make_attn_mask
import openpi.policies.policy as _policy
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.config as _config
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders
import openpi.transforms as _transforms

from openpi_client import image_tools

from dcrl.expo.agents.vla.vla_base import Model
from dcrl.expo.data.dataset import DatasetDict


def _load_weights_and_validate(
    loader: _weight_loaders.WeightLoader, params_shape: at.Params
) -> at.Params:
    """Load and validate weights.
    
    Args:
        loader: Weight loader instance.
        params_shape: Expected parameter shape.
        
    Returns:
        Loaded subset of the weights.
    """
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(
        expected=params_shape,
        got=loaded_params,
        check_shapes=True,
        check_dtypes=True,
    )

    return traverse_util.unflatten_dict(
        {
            k: v
            for k, v in traverse_util.flatten_dict(loaded_params).items()
            if not isinstance(v, jax.ShapeDtypeStruct)
        }
    )

def pi05_get_params(state: training_utils.TrainState) -> tuple[training_utils.TrainState, at.Params]:
    if state.ema_params is not None:
        params = state.ema_params
    else:
        params = state.params
    return params


def pi05_init_train_state(
    config: _config.TrainConfig,
    init_rng: at.KeyArrayLike,
    mesh: jax.sharding.Mesh,
    *,
    resume: bool,
    is_target: bool = False,
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(
        config.optimizer, config.lr_schedule, weight_decay_mask=None
    )

    def init(
        rng: at.KeyArrayLike, partial_params: at.Params | None = None
    ) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        model = config.model.create(model_rng)

        if partial_params is not None:
            graphdef, state = nnx.split(model)
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        params = nnx_utils.state_map(
            params,
            config.freeze_filter,
            lambda p: p.replace(p.value.astype(jnp.bfloat16)),
        )

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        # Attach the current-topology sharding to every abstract leaf so orbax
        # can restore (and reshard) the checkpoint even when the number of
        # available devices differs from the one used to save it. `eval_shape`
        # yields ShapeDtypeStructs with `sharding=None`; without an explicit
        # sharding orbax falls back to the sharding stored in the checkpoint
        # metadata, which is only valid when the exact same devices are present.
        # On a different topology that fallback resolves to None and
        # deserialization fails ("sharding ... Got None"). `state_sharding` has
        # the same pytree structure as `train_state_shape` (both come from the
        # same tree_map), so we can zip them leaf-wise.
        train_state_shape = jax.tree_util.tree_map(
            lambda x, s: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=s),
            train_state_shape,
            state_sharding,
        )
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(
        config.weight_loader, train_state_shape.params.to_pure_dict()
    )
    replicated_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec()
    )

    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    if is_target:
        model_params = pi05_get_params(train_state)
        return model_params
    return train_state, state_sharding


def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss)

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions)

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    # Get current learning rate from the schedule
    lr_schedule_fn = config.lr_schedule.create()
    current_lr = lr_schedule_fn(state.step)
    info = {
        "actor_loss": loss,
        "actor_grad_norm": optax.global_norm(grads),
        "actor_param_norm": optax.global_norm(kernel_params),
        "actor_lr": current_lr,
        "actor_state_step": state.step,
    }
    return new_state, info


def _bind_model(train_state: training_utils.TrainState, train: bool = False):
    """Bind model from train state."""
    model = nnx.merge(train_state.model_def, train_state.params)
    if train:
        model.train()
    return model


@functools.partial(jax.jit, static_argnames=['train', 'num_samples'])
def _jitted_infer(transformed_inputs, train_state, rng, policy_metadata, train, num_samples, noise=None):
    """`noise`: optional seeds (b, num_samples, H, D_pad) replacing the internal draw."""
    model = _bind_model(train_state, train=False)
    sample_policy = _policy.Policy(
        model=model,
        rng=rng,
        transforms=[],
        output_transforms=[],
        sample_kwargs=dict(train=train, num_samples=num_samples),
        metadata=policy_metadata,
        is_pytorch=False,
        pytorch_device=None,
    )
    sample_info = sample_policy.infer(transformed_inputs, noise=noise, is_batch=True, for_training=True)
    return sample_info


# =========================================================================
# Prefix-conditioned training step (RTC-SFT) and prefix-clamped inference.
# =========================================================================

def _forward_get_velocity(model, observation, x_t, time):
    """v_t for a pre-processed observation: the velocity half of Pi0.compute_loss."""
    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)
    suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = model.embed_suffix(observation, x_t, time)
    input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
    ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
    attn_mask = _pi0.make_attn_mask(input_mask, ar_mask)
    positions = jnp.cumsum(input_mask, axis=1) - 1
    (_, suffix_out), _ = model.PaliGemma.llm(
        [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions,
        adarms_cond=[None, adarms_cond],
    )
    return model.action_out_proj(suffix_out[:, -model.action_horizon:])

def train_step_p1_prefix(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple,
    delay_spec: jnp.ndarray,
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    """Prefix-conditioned flow-matching step (RTC-SFT, arXiv 2512.05964).

    `batch = (observation, padded_actions)` with padded_actions (B, H, D_pad). `delay_spec`
    sets the per-example prefix length d: a scalar draws d ~ Unif{0..delay_spec} (offline
    RTC-SFT); a (B,) array is used as d directly (online fine-tuning passes the deployed
    delay, 0 for the boot chunk). The first d positions of x_t are the clean actions
    (per-token t=0) and only the remaining H-d positions are noised and supervised. d=0 is
    vanilla pi05 BC; d=replan_steps is the prefix-inpainted regime.
    """
    model = nnx.merge(state.model_def, state.params)
    model.train()

    def loss_fn(model, rng, observation, actions):
        preprocess_rng, noise_rng, time_rng, delay_rng = jax.random.split(rng, 4)
        obs = _model.preprocess_observation(preprocess_rng, observation, train=True)
        B, H, _ = actions.shape
        noise = jax.random.normal(noise_rng, actions.shape)
        # Same τ distribution as pi05's default compute_loss.
        time_b = jax.random.beta(time_rng, 1.5, 1, (B,)) * 0.999 + 0.001  # (B,)
        if delay_spec.ndim == 0:
            d = jax.random.randint(delay_rng, (B,), 0, delay_spec + 1)    # (B,)
        else:
            d = delay_spec                                                # (B,)
        # postfix_mask: (B, H) — 0 on first d_i positions (prefix), 1 on the rest.
        postfix_mask = (jnp.arange(H)[None, :] >= d[:, None]).astype(jnp.float32)
        # Per-token τ: prefix = 0 (clean), postfix = per-batch scalar.
        time_per_pos = time_b[:, None] * postfix_mask                     # (B, H)
        t_exp = time_per_pos[..., None]                                   # (B, H, 1)
        x_t = t_exp * noise + (1 - t_exp) * actions                       # prefix slots -> actions
        # u_t on prefix slots is unused: the loss is masked to the postfix.
        u_t = noise - actions
        v_t = _forward_get_velocity(model, obs, x_t, time_per_pos)        # (B, H, D)
        per_pos = jnp.mean(jnp.square(v_t - u_t), axis=-1)                # (B, H)
        masked = per_pos * postfix_mask                                   # (B, H)
        denom = jnp.maximum(postfix_mask.sum(), 1.0)
        return masked.sum() / denom

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(
        model, train_rng, observation, actions
    )

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new,
                state.ema_params, new_params,
            ),
        )

    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    lr_schedule_fn = config.lr_schedule.create()
    current_lr = lr_schedule_fn(state.step)
    info = {
        "actor_loss": loss,
        "actor_grad_norm": optax.global_norm(grads),
        "actor_param_norm": optax.global_norm(kernel_params),
        "actor_lr": current_lr,
        "actor_state_step": state.step,
    }
    return new_state, info


@functools.partial(jax.jit, static_argnames=['num_samples'])
def _jitted_sample_with_prefix(transformed_inputs, train_state, rng, prefix, num_samples, noise=None):
    """KV-cached prefix-inpainted sampler: `prefix` (b, delay, D_pad) stays pinned through
    the denoising steps. `noise`: optional seeds (b, num_samples, H, D_pad)."""
    model = _bind_model(train_state, train=False)
    observation = _model.Observation.from_dict(transformed_inputs)
    return model.sample_actions_with_prefix(
        rng, observation, prefix=prefix, num_samples=num_samples, noise=noise,
    )


@jax.jit
def _jitted_pack_observation(transformed_inputs):
    """Add the batch dim and scale uint8 images to [-1, 1] in one dispatch."""
    packed = jax.tree.map(lambda x: jnp.asarray(x)[jnp.newaxis, ...], transformed_inputs)
    packed["image"] = {
        k: v.astype(jnp.float32) / 255.0 * 2.0 - 1.0 if v.dtype == jnp.uint8 else v
        for k, v in packed["image"].items()
    }
    return packed


def build_pi05(config, seed, mesh, data_sharding, replicated_sharding,
               resume, default_prompt):
    """Build Pi05 actor, train state, target params, and metadata from agent config.

    Returns (actor, actor_train_state, target_actor_params, agent_kwargs, metadata)
    where metadata is a dict with action_horizon, resize_size, freeze_encoder
    ready to pass into PiExpoLearner.create().
    """
    from dcrl.expo.utils.train_utils import build_pi05_config
    agent_kwargs, pi05_train_config, pi05_resize_size, _ = build_pi05_config(config)
    freeze_encoder = agent_kwargs.pop("freeze_pi05_encoder", False)

    rng = jax.random.PRNGKey(seed)
    init_rng, rng = jax.random.split(rng)
    target_rng, rng = jax.random.split(rng)

    actor, actor_train_state, _ = Pi05Agent.initialize(
        pi05_train_config,
        mesh,
        init_rng,
        resume=resume,
        default_prompt=default_prompt,
        data_sharding=data_sharding,
        replicated_sharding=replicated_sharding,
        freeze_pi05_encoder=freeze_encoder,
        infer_device=jax.devices()[0],
    )
    if resume:
        target_actor_params = actor.get_params(actor_train_state)
    else:
        target_actor_params = actor.init_target_params(target_rng, resume=resume)

    metadata = dict(
        action_horizon=pi05_train_config.model.action_horizon,
        resize_size=pi05_resize_size,
        freeze_encoder=freeze_encoder,
    )
    return actor, actor_train_state, target_actor_params, agent_kwargs, metadata


class Pi05Agent(Model):
    """Wrapper for pi05 model to make it compatible with expo training framework."""

    def __init__(
        self,
        *,
        train_config: Any,
        mesh: jax.sharding.Mesh,
        train_state_sharding: jax.sharding.NamedSharding,
        data_sharding: jax.sharding.NamedSharding,
        replicated_sharding: jax.sharding.NamedSharding,
        default_prompt: str,
        freeze_pi05_encoder: bool = False,
        infer_device: Optional[jax.Device] = None,
        action_dim: Optional[int] = None,
        state_dim: Optional[int] = None,
    ):

        self.action_dim = action_dim
        self.state_dim = state_dim
        self.train_config = train_config
        self.data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
        self.model_config = train_config.model

        self.default_prompt = default_prompt
        self.freeze_pi05_encoder = freeze_pi05_encoder

        self.mesh = mesh
        self.train_state_sharding = train_state_sharding
        self.data_sharding = data_sharding
        self.replicated_sharding = replicated_sharding
        self.infer_device = infer_device or jax.devices()[0]
        self.infer_sharding = jax.sharding.SingleDeviceSharding(self.infer_device)
        self.input_transforms, self.input_model_transforms = self._build_input_transform_pipeline()
        self.output_transforms = self._build_output_transform_pipeline()

        self.train_step = jax.jit(
            functools.partial(train_step, self.train_config),
            in_shardings=(self.replicated_sharding, self.train_state_sharding, self.data_sharding),
            out_shardings=(self.train_state_sharding, self.replicated_sharding),
            donate_argnums=(1,),
        )

        # Prefix-conditioned train step (RTC-SFT); `delay_spec` rides replicated sharding.
        self.train_step_p1_prefix = jax.jit(
            functools.partial(train_step_p1_prefix, self.train_config),
            in_shardings=(self.replicated_sharding, self.train_state_sharding, self.data_sharding,
                          self.replicated_sharding),
            out_shardings=(self.train_state_sharding, self.replicated_sharding),
            donate_argnums=(1,),
        )

    def _build_input_transform_pipeline(self, normalize: bool = True):
        """(pre-model, model) halves; the seam is where extra state dims are appended."""
        transforms = [
            *self.data_config.repack_transforms.inputs,
            *self.data_config.data_transforms.inputs,
        ]
        if normalize:
            transforms.append(_transforms.Normalize(
                self.data_config.norm_stats, use_quantiles=self.data_config.use_quantile_norm
            ))
        return (_transforms.compose(transforms),
                _transforms.compose(self.data_config.model_transforms.inputs))
    
    def _build_output_transform_pipeline(self, unnormalize: bool = True):
        """Compose model, unnormalize, and data transforms for model outputs."""
        if unnormalize:
            transforms = [
                *self.data_config.model_transforms.outputs,
                _transforms.Unnormalize(self.data_config.norm_stats, use_quantiles=self.data_config.use_quantile_norm),
                *self.data_config.data_transforms.outputs,
                # *repack_transforms.outputs,
            ]
        else:
            transforms = [
                *self.data_config.model_transforms.outputs,
                *self.data_config.data_transforms.outputs,
            ]
        return _transforms.compose(transforms)

    @classmethod
    def load_pi05_config(cls, config_name: str):
        return _config.get_config(config_name)

    @classmethod
    def initialize(
        cls,
        train_config: _config.TrainConfig,
        mesh: jax.sharding.Mesh,
        init_rng: jax.random.PRNGKey,
        *,
        resume: bool = False,
        data_sharding: Optional[jax.sharding.NamedSharding] = None,
        replicated_sharding: Optional[jax.sharding.NamedSharding] = None,
        default_prompt: Optional[str] = None,
        freeze_pi05_encoder: bool = False,
        infer_device: Optional[jax.Device] = None,
    ) -> tuple["Pi05Agent", Any]:
        """Initialize a Pi05Agent instance using init_train_state."""
        train_state, train_state_sharding = pi05_init_train_state(
            train_config,
            init_rng,
            mesh,
            resume=resume,
            is_target=False,
        )

        agent = cls(
            train_config=train_config,
            mesh=mesh,
            train_state_sharding=train_state_sharding,
            data_sharding=data_sharding,
            replicated_sharding=replicated_sharding,
            default_prompt=default_prompt,
            freeze_pi05_encoder=freeze_pi05_encoder,
            infer_device=infer_device,
        )
        
        return agent, train_state, train_state_sharding

    def get_params(self, train_state):
        return pi05_get_params(train_state)

    def init_target_params(self, rng, *, resume=False):
        return pi05_init_train_state(
            self.train_config, rng, self.mesh, resume=resume, is_target=True
        )

    def process_raw_inputs(self, raw_observations, action_dim, resize_size, normalize=True):
        """Convert raw env observations into a batched model-ready Observation dict."""
        # create a dummy actions
        raw_observations["actions"] = np.zeros(action_dim)
        for key, value in raw_observations.items():
            raw_observations[key] = np.asarray(value)
            if "image" in key:
                # make image uint8
                raw_observations[key] = raw_observations[key].astype(np.uint8)
                assert np.max(raw_observations[key]) > 1

        transformed_inputs = self.input_transforms(raw_observations)
        transformed_inputs = self.input_model_transforms(transformed_inputs)
        packed = _jitted_pack_observation(transformed_inputs)
        # Same key structure Observation.from_dict().to_dict() produced (extra
        # keys like the dummy "actions" are dropped; absent ones default None).
        return {
            "image": packed["image"],
            "image_mask": packed["image_mask"],
            "state": packed["state"],
            "tokenized_prompt": packed.get("tokenized_prompt"),
            "tokenized_prompt_mask": packed.get("tokenized_prompt_mask"),
            "token_ar_mask": packed.get("token_ar_mask"),
            "token_loss_mask": packed.get("token_loss_mask"),
        }

    def process_transformed_outputs(self, transformed_actions, unnormalize=True):
        """Unnormalize unpadded actions back to the environment action space."""
        n = transformed_actions.shape[0]
        padded = self._pad_actions(transformed_actions.reshape(n, -1))
        dummy_state = np.zeros((n, self.model_config.action_dim), dtype=np.float32)
        output_dict = {
            "state": dummy_state,
            "actions": np.array(padded),
        }
        processed = [
            self.output_transforms(jax.tree.map(lambda x: x[i], output_dict))
            for i in range(n)
        ]
        return jax.tree.map(lambda *xs: np.stack(xs, axis=0), *processed)["actions"]

    def prepare_batch_for_actor(self, batch):
        """Build (observation, padded actions) tuple for actor loss computation."""
        obs = _model.Observation.from_dict(batch.copy())
        actions = self._pad_actions(batch["full_actions"])
        return (obs, actions)

    def _unpad_actions(self, actions):
        """Reshape padded model actions to (batch, horizon, env action_dim)."""
        padded_dim = self.model_config.action_dim
        action_horizon = self.model_config.action_horizon
        actions = actions.reshape(actions.shape[0], action_horizon, padded_dim)
        return actions[..., :self.action_dim]

    def _pad_actions(self, actions):
        """Zero-pad env actions to the model's expected action dimension."""
        padded_dim = self.model_config.action_dim
        action_horizon = self.model_config.action_horizon
        actions = actions.reshape(actions.shape[0], action_horizon, self.action_dim)
        return jnp.concatenate([
            actions,
            jnp.zeros((actions.shape[0], action_horizon, padded_dim - self.action_dim))
        ], axis=-1)

    def sample_actions(
        self,
        transformed_inputs: Dict,
        train_state: training_utils.TrainState,
        rng: jax.random.PRNGKey,
        train: Optional[bool] = False,
        num_samples: Optional[int] = 1,
    ) -> tuple:
        """Sample actions from pi05 model using already-transformed inputs.

        """
        infer_sharding = self.infer_sharding
        transformed_inputs = jax.tree.map(
            lambda x: jax.device_put(x, infer_sharding) if isinstance(x, (jnp.ndarray, np.ndarray)) else x,
            transformed_inputs,
        )
        if num_samples > 1 and not self.freeze_pi05_encoder:
            def repeat_value(v, n):
                if isinstance(v, dict):
                    return {k: repeat_value(vv, n) for k, vv in v.items()}
                if isinstance(v, (np.ndarray, jnp.ndarray)):
                    return jnp.repeat(v, n, axis=0)
                return v
            transformed_inputs = repeat_value(transformed_inputs, num_samples)

        key, rng = jax.random.split(rng)
        key = jax.device_put(key, infer_sharding)
        infer_train_state = jax.device_put(train_state, infer_sharding)
        noise_samples = 1 if not self.freeze_pi05_encoder else num_samples
        sample_info = _jitted_infer(transformed_inputs, infer_train_state, key, self.train_config.policy_metadata, train, noise_samples)

        actions = self._unpad_actions(sample_info["actions"])
        return actions, sample_info["policy_timing"]["infer_ms"]

    def sample_actions_with_prefix(
        self,
        transformed_inputs: Dict,
        train_state: training_utils.TrainState,
        rng: jax.random.PRNGKey,
        prefix_padded: jnp.ndarray,
        num_samples: int = 1,
    ) -> jnp.ndarray:
        """Rollout sampler with the clean prefix `prefix_padded` (r, D_pad) pinned; the
        single observation is replicated to `num_samples` candidate chunks.
        Returns (num_samples, H, env_action_dim)."""
        infer_sharding = self.infer_sharding
        transformed_inputs = jax.tree.map(
            lambda x: jax.device_put(x, infer_sharding) if isinstance(x, (jnp.ndarray, np.ndarray)) else x,
            transformed_inputs,
        )
        key, _ = jax.random.split(rng)
        key = jax.device_put(key, infer_sharding)
        infer_train_state = jax.device_put(train_state, infer_sharding)
        # (1, r, D_pad): single-obs clean prefix; the model replicates obs+prefix
        # to `num_samples` candidates internally.
        prefix = jax.device_put(jnp.asarray(prefix_padded)[None], infer_sharding)
        x_clean = _jitted_sample_with_prefix(
            transformed_inputs, infer_train_state, key, prefix, num_samples,
        )
        return self._unpad_actions(x_clean)

    def sample_training_actions(
        self,
        transformed_inputs,
        train_state: training_utils.TrainState,
        rng: jax.random.PRNGKey,
        train: Optional[bool] = True,
        num_samples: Optional[int] = 1,
        noise: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """Batched sampler for training; `noise`: optional seeds (B, num_samples, H, D_pad)."""
        key, rng = jax.random.split(rng)
        sample_info = _jitted_infer(transformed_inputs, train_state, key, self.train_config.policy_metadata, train, num_samples, noise=noise)
        actions = self._unpad_actions(sample_info["actions"])
        return actions, sample_info["policy_timing"]["infer_ms"]

    def sample_training_actions_with_prefix(
        self,
        transformed_inputs,
        train_state: training_utils.TrainState,
        rng: jax.random.PRNGKey,
        prefix_padded: jnp.ndarray,
        num_samples: int = 1,
        noise: Optional[jnp.ndarray] = None,
    ) -> tuple:
        """Batched prefix-conditioned sampler for the critic backup: `prefix_padded` is
        (B, delay, D_pad) aligned with the batch, train_state stays on its training
        sharding, `noise` optionally seeds each row. Returns (actions, None) with
        actions (B * num_samples, H, action_dim)."""
        prefix_padded = jnp.asarray(prefix_padded)
        key, _ = jax.random.split(rng)
        x_clean = _jitted_sample_with_prefix(
            transformed_inputs, train_state, key, prefix_padded, num_samples, noise=noise,
        )
        return self._unpad_actions(x_clean), None


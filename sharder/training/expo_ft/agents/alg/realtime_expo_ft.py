from functools import partial
from typing import Any, Callable, Dict, Optional, Sequence, Tuple
import dataclasses

import logging

import flax
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
from flax import struct
from flax.training.train_state import TrainState

import numpy as np

from expo_ft.agents.alg.agent import AgentLearner, initialize_checkpoint_dir
from expo_ft.agents.alg.batch_utils import prepare_critic_batch, prepare_actor_sampling_batch, prepare_actor_sampling_batch_delayed, stack_critic_cameras, CRITIC_CAMERA_KEYS
from expo_ft.networks.temperature import Temperature
from expo_ft.data.dataset import DatasetDict
from expo_ft.distributions import TanhNormal
from expo_ft.networks import (
    MLP,
    Ensemble,
    MLPResNetV2,
    StateActionValue,
    subsample_image_ensemble,
    PixelMultiplexer,
    PixelEditMultiplexer,
    BatchEncoder,
)
from expo_ft.networks.encoders import ResNetV2Encoder

from expo_ft.utils.augmentation import make_data_augmentation_fn

import openpi.shared.array_typing as at
import openpi.training.sharding as _sharding
import openpi.training.utils as training_utils


def _split_params(agent: Any) -> tuple[Any, dict[str, at.Params]]:
    batch_encoder_params = agent.batch_encoder.params
    edit_actor_params = agent.edit_actor.params
    temp_params = agent.temp.params
    critic_params = agent.critic.params
    filter_critic_params = agent.filter_critic.params

    with at.disable_typechecking():
        if agent.actor_train_state.ema_params is not None:
            actor_params = agent.actor_train_state.ema_params
            actor_train_state = dataclasses.replace(agent.actor_train_state, ema_params=None)
        else:
            actor_params = agent.actor_train_state.params
            actor_train_state = dataclasses.replace(agent.actor_train_state, params={})

    agent = dataclasses.replace(
        agent, 
        batch_encoder=dataclasses.replace(agent.batch_encoder, params={}),
        edit_actor=dataclasses.replace(agent.edit_actor, params={}), 
        temp=dataclasses.replace(agent.temp, params={}),
        critic=dataclasses.replace(agent.critic, params={}),
        filter_critic=dataclasses.replace(agent.filter_critic, params={}),
        actor_train_state=actor_train_state
    )

    params = {
        "batch_encoder_params": batch_encoder_params,
        "edit_actor_params": edit_actor_params,
        "temp_params": temp_params,
        "critic_params": critic_params,
        "filter_critic_params": filter_critic_params,
        "actor_params": actor_params
    }
    return agent, params


def _merge_params(agent: Any, params: dict[str, at.Params]) -> Any:
    batch_encoder = dataclasses.replace(agent.batch_encoder, params=params["batch_encoder_params"])
    edit_actor = dataclasses.replace(agent.edit_actor, params=params["edit_actor_params"])
    temp = dataclasses.replace(agent.temp, params=params["temp_params"])
    critic = dataclasses.replace(agent.critic, params=params["critic_params"])
    filter_critic = dataclasses.replace(agent.filter_critic, params=params["filter_critic_params"])

    with at.disable_typechecking():
        if agent.actor_train_state.params:
            actor_train_state = dataclasses.replace(agent.actor_train_state, ema_params=params["actor_params"])
        else:
            actor_train_state = dataclasses.replace(agent.actor_train_state, params=params["actor_params"])

    agent = dataclasses.replace(
        agent,
        batch_encoder=batch_encoder,
        edit_actor=edit_actor,
        temp=temp,
        critic=critic,
        filter_critic=filter_critic,
        actor_train_state=actor_train_state,
    )
    return agent


def restore_checkpoint(checkpoint_manager, agent, step: int | None = None):
    agent, params = _split_params(agent)
    agent_restore_args = ocp.checkpoint_utils.construct_restore_args(agent)
    params_restore_args = ocp.checkpoint_utils.construct_restore_args(params)
    restored = checkpoint_manager.restore(
        step,
        items={
            "agent": agent,
            "params": params,
        },
        restore_kwargs={
            "agent": {"restore_args": agent_restore_args},
            "params": {"restore_args": params_restore_args},
        },
    )
    return _merge_params(restored["agent"], restored["params"])

def save_checkpoint(
    checkpoint_manager: ocp.CheckpointManager,
    agent: Any,
    step: int,
):
    agent, params = _split_params(agent)
    items = {
        "agent": agent,
        "params": params,
    }
    checkpoint_manager.save(step, items)

def load_agent(seed, example_observation, example_action, example_state,
               actor, actor_train_state, target_actor_params, agent_kwargs, metadata,
               mesh, data_sharding, replicated_sharding, resume, replan_steps,
               default_prompt, edit_action_xyzg):
    """Create a RealTimeEXPOFTLearner from a pre-built VLA actor and remaining config kwargs."""
    agent_kwargs.update(
        actor=actor,
        actor_train_state=actor_train_state,
        target_actor_params=target_actor_params,
        mesh=mesh,
        resume=resume,
        replan_steps=replan_steps,
        data_sharding=data_sharding,
        replicated_sharding=replicated_sharding,
        default_prompt=default_prompt,
        edit_action_xyzg=edit_action_xyzg,
        **metadata,
    )
    return RealTimeEXPOFTLearner.create(seed, example_observation, example_action, example_state, **agent_kwargs)

def decay_mask_fn(params):
    flat_params = flax.traverse_util.flatten_dict(params)
    flat_mask = {path: path[-1] != "bias" for path in flat_params}
    return flax.core.FrozenDict(flax.traverse_util.unflatten_dict(flat_mask))

@partial(jax.jit, static_argnames=('critic_fn', 'num_min_qs'))
def compute_q(critic_fn, critic_params, observations, actions, states, num_min_qs=None):
    q_values = critic_fn({'params': critic_params}, observations, actions, p=states, sample_num=num_min_qs)
    q_values = q_values.min(axis=0)
    return q_values


@partial(jax.jit, static_argnames=('encoder_fn', 'stop_gradient'))
def batch_encode(encoder_fn, encoder_params, observations, stop_gradient=False):
    encoded = encoder_fn({'params': encoder_params}, observations, stop_gradient=stop_gradient)
    return encoded


@partial(jax.jit, static_argnames="apply_fn")
def _sample_actions(rng, apply_fn, params, observations: jnp.ndarray, states, actions) -> jnp.ndarray:
    key, rng = jax.random.split(rng)
    dist = apply_fn({"params": params}, observations, actions=actions, p=states)
    return dist.sample(seed=key), rng


@partial(
    jax.jit,
    static_argnames=(
        "encoder_fn", "edit_fn", "critic_fn", "N", "n_edit_samples",
        "num_qs", "num_min_qs", "replan_steps", "action_dim", "state_dim",
        "padded_action_dim", "delay", "edit_scale", "edit_action_xyzg",
        "freeze_encoder", "only_base_actions", "critic_camera_keys",
    ),
)
def _jitted_fast_select(
    rng, precached, images, state,
    encoder_params, edit_params, target_critic_params,
    *, encoder_fn, edit_fn, critic_fn, N, n_edit_samples, num_qs,
    num_min_qs, replan_steps, action_dim, state_dim, padded_action_dim,
    delay, edit_scale, edit_action_xyzg, freeze_encoder,
    only_base_actions, critic_camera_keys,
):
    """Fast phase of sample_actions in one dispatch: window slice, critic encode,
    edit candidates, target-Q argmax, executed_padded."""
    r = replan_steps
    window = precached[:, delay:delay + r, :]  # (N, r, action_dim) normalized

    # Channel-stack the critic cameras; slice the padded VLA state down to the critic state.
    critic_obs = jnp.concatenate([images[k] for k in critic_camera_keys], axis=-1)
    critic_states = state.reshape(state.shape[0], padded_action_dim)[..., :state_dim]

    if only_base_actions or N <= 1:
        selected = window[0]
        out = {"selected": selected, "qs": None, "idx": None, "cand": None,
               "enc_all": None, "q_all": None, "states_all": None}
    else:
        cand = window
        q_actions = window.reshape(N, r * action_dim)

        if freeze_encoder:
            critic_obs = critic_obs.repeat(N, axis=0)
            critic_states = critic_states.repeat(N, axis=0)
        enc = encoder_fn({"params": encoder_params}, critic_obs[:1], stop_gradient=True)
        enc = enc.repeat(N, axis=0)

        key, rng = jax.random.split(rng)
        target_params = subsample_image_ensemble(key, target_critic_params, num_min_qs, num_qs)

        enc_all, states_all = enc, critic_states
        if n_edit_samples > 0:
            key, rng = jax.random.split(rng, 2)
            enc_all = jnp.concatenate(
                [enc, jnp.expand_dims(enc[0], axis=0).repeat(n_edit_samples, axis=0)], axis=0)
            states_all = jnp.concatenate(
                [critic_states, jnp.expand_dims(critic_states[0], axis=0).repeat(n_edit_samples, axis=0)], axis=0)
            r_observations = jnp.repeat(jnp.expand_dims(enc_all[0], axis=0), n_edit_samples, axis=0)
            r_states = jnp.repeat(jnp.expand_dims(states_all[0], axis=0), n_edit_samples, axis=0)
            base_actions = q_actions[:n_edit_samples]
            # Edits of the first n_edit_samples base candidates.
            key2, rng = jax.random.split(key)
            dist = edit_fn({"params": edit_params}, r_observations, actions=base_actions, p=r_states)
            edit_scaled = dist.sample(seed=key2) * edit_scale
            if edit_action_xyzg:
                # mask: 1 for xyz (0,1,2) and gripper (last dim), 0 for rotation (3,4,5)
                mask = jnp.ones(action_dim).at[3:6].set(0.0)
                edit_scaled = edit_scaled * jnp.tile(mask, r)
            combined = edit_scaled + base_actions
            q_actions = jnp.concatenate([q_actions, combined], axis=0)
            cand = jnp.concatenate([cand, combined.reshape(n_edit_samples, r, action_dim)], axis=0)

        qs = critic_fn({"params": target_params}, enc_all, q_actions, p=states_all,
                       sample_num=num_min_qs).min(axis=0)
        idx = jnp.argmax(qs)
        out = {"selected": cand[idx], "qs": qs, "idx": idx, "cand": cand,
               "enc_all": enc_all, "q_all": q_actions, "states_all": states_all}

    out["executed_padded"] = jnp.concatenate(
        [out["selected"], jnp.zeros((r, padded_action_dim - action_dim), dtype=out["selected"].dtype)], axis=-1)
    out["rng"] = rng
    return out


class RealTimeEXPOFTLearner(AgentLearner, struct.PyTreeNode):
    rng: jax.random.PRNGKey
    data_augmentation_fn: Callable = struct.field(pytree_node=False)
    critic: TrainState
    batch_encoder: TrainState
    target_critic: TrainState
    filter_critic: TrainState  # Q_f(s, noise seed); no target network
    actor: Any = struct.field(pytree_node=False)
    actor_train_state: training_utils.TrainState
    target_actor_params: at.Params
    edit_actor: TrainState
    temp: TrainState
    N: int = struct.field(pytree_node=False)
    n_edit_samples: int = struct.field(pytree_node=False)
    edit_scale: float = struct.field(pytree_node=False)
    edit_action_xyzg: bool = struct.field(pytree_node=False)
    batch_split: int = struct.field(pytree_node=False)
    encode_batch_split: int = struct.field(pytree_node=False)
    actor_tau: float
    tau: float
    discount: float
    target_entropy: float
    entropy_scale: float
    num_qs: int = struct.field(pytree_node=False)
    num_min_qs: Optional[int] = struct.field(
        pytree_node=False
    )  # See M in RedQ https://arxiv.org/abs/2101.05982
    action_dim: int = struct.field(pytree_node=False)
    state_dim: int = struct.field(pytree_node=False)
    full_action_dim: int = struct.field(pytree_node=False)
    replan_steps: int = struct.field(pytree_node=False)
    action_horizon: int = struct.field(pytree_node=False)
    resize_size: Optional[int] = struct.field(pytree_node=False)
    default_prompt: Optional[str] = struct.field(pytree_node=False)
    data_sharding: Optional[jax.sharding.NamedSharding] = struct.field(pytree_node=False)
    batch_encoder_sharding: Optional[jax.sharding.NamedSharding] = struct.field(pytree_node=False)
    freeze_encoder: Optional[bool] = struct.field(pytree_node=False)
    freeze_critic_encoder: bool = struct.field(pytree_node=False)
    actor_success_only: bool = struct.field(pytree_node=False)
    _infer_cache: Optional[dict] = struct.field(pytree_node=False, default=None)
    delay: int = struct.field(pytree_node=False, default=0)
    p1_use_prefix_conditioning: bool = struct.field(pytree_node=False, default=True)
    q_edit_use_main_obs: bool = struct.field(pytree_node=False, default=False)  # backup edit/Q-select on the delayed obs
    train_base_actor: bool = struct.field(pytree_node=False, default=True)  # False = freeze the pi0.5 actor
    critic_camera_keys: Tuple[str, ...] = struct.field(pytree_node=False, default=CRITIC_CAMERA_KEYS)
    # Noise-Q pre-filter (critic backup only)
    filter_N: int = struct.field(pytree_node=False, default=8)
    filter_temperature: float = struct.field(pytree_node=False, default=0.0)  # 0 = argmax over Q_f
    filter_num_qs: int = struct.field(pytree_node=False, default=2)
    filter_n_edit: int = struct.field(pytree_node=False, default=-1)  # <0 = n_edit_samples
    filter_add_delayed_obs: bool = struct.field(pytree_node=False, default=False)  # Q_f also sees the delayed obs

    @classmethod
    def create(
        cls,
        seed: int,
        observation_space,
        action_space,
        states,
        # Pre-built VLA actor (constructed by build_pi05 or similar factory)
        actor: Any = None,
        actor_train_state: Any = None,
        target_actor_params: Any = None,
        # VLA metadata (extracted by factory from backbone config)
        action_horizon: int = 1,
        mesh: Optional[Any] = None,
        freeze_encoder: bool = False,
        # EXPO-specific params
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        temp_lr: float = 3e-4,
        hidden_dims: Sequence[int] = (256, 256),
        discount: float = 0.99,
        tau: float = 0.005,
        num_qs: int = 2,
        num_min_qs: Optional[int] = None,
        critic_dropout_rate: Optional[float] = None,
        critic_weight_decay: Optional[float] = None,
        critic_layer_norm: bool = False,
        target_entropy: Optional[float] = None,
        adjust_target_entropy: Optional[bool] = False,
        entropy_scale: float = 1.0,
        init_temperature: float = 1.0,
        use_pnorm: bool = False,
        use_critic_resnet: bool = False,
        actor_drop: Optional[float] = None,
        N: int = 32,
        batch_split: int = 1,
        encode_batch_split: int = 1,
        n_edit_samples: int = 0,
        edit_scale: float = 1.0,
        edit_action_xyzg: bool = False,
        actor_tau: float = 0.001,
        include_state: bool = True,
        latent_dim_image: int = 50,
        latent_dim_state: int = 50,
        encoder_stage_sizes: Tuple[int, int, int, int] = (2, 2, 2, 2),
        encoder_num_filters: int = 64,
        pixel_keys: Tuple[str, ...] = ("pixels",),
        depth_keys: Tuple[str, ...] = (),
        resume: bool = False,
        replan_steps: int = 1,
        freeze_critic_encoder: bool = False,
        data_sharding: Optional[jax.sharding.NamedSharding] = None,
        replicated_sharding: Optional[jax.sharding.NamedSharding] = None,
        default_prompt: Optional[str] = None,
        resize_size: Optional[int] = None,
        actor_success_only: bool = False,
        use_full_augmentation: bool = True,
        delay: int = 0,
        p1_use_prefix_conditioning: bool = True,
        q_edit_use_main_obs: bool = False,
        train_base_actor: bool = True,
        critic_camera_keys: Tuple[str, ...] = CRITIC_CAMERA_KEYS,
        filter_N: int = 8,
        filter_temperature: float = 0.0,
        filter_num_qs: int = 2,
        filter_n_edit: int = -1,
        filter_add_delayed_obs: bool = False,
        **kwargs,
    ):
        action_dim = action_space.shape[-1]
        state_dim = states.shape[-1]
        full_action_dim = replan_steps * action_dim
        observations = observation_space
        actions = jnp.zeros((full_action_dim,))
        print("observation shape: ", observations.shape)
        print("action shape: ", actions.shape, "action horizon: ", action_horizon, "action dim: ", action_dim)
        print("edit actor output size / q function input size: ", full_action_dim, " (replan_steps=", replan_steps, "* action_dim=", action_dim, ")")
        print("states shape: ", states.shape)

        if target_entropy is None:
            if adjust_target_entropy:
                target_entropy = -full_action_dim / 2 + full_action_dim * jnp.log(edit_scale)
            else:
                target_entropy = -full_action_dim / 2

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, temp_key = jax.random.split(rng, 4)
        rng, encoder_key = jax.random.split(rng, 2)

        encoder_cls = partial(
            ResNetV2Encoder,
            stage_sizes=encoder_stage_sizes,
            num_filters=encoder_num_filters,
        )

        batch_encoder_def = BatchEncoder(
            encoder_cls=encoder_cls,
            latent_dim=latent_dim_image,
            pixel_keys=pixel_keys,
            depth_keys=depth_keys,
        )

        batch_encoder_params = batch_encoder_def.init(encoder_key, observations)["params"]
        batch_encoder = TrainState.create(
            apply_fn=batch_encoder_def.apply,
            params=batch_encoder_params,
            tx=optax.adam(learning_rate=critic_lr),
        )

        batch_encoder_shape = jax.eval_shape(lambda: batch_encoder)
        batch_encoder_sharding = _sharding.fsdp_sharding(batch_encoder_shape, mesh, log=True)
        batch_encoder = jax.jit(
            lambda x: x,
            in_shardings=replicated_sharding,
            out_shardings=batch_encoder_sharding,
        )(batch_encoder)

        critic_observations = jnp.ones((1, latent_dim_image))
        critic_actions = jnp.expand_dims(actions, axis = 0)
        critic_states = jnp.expand_dims(states, axis = 0)
        critic_states_ext = critic_states
        print("critic actions shape: ", critic_actions.shape)
        print("critic states shape: ", critic_states.shape)

        edit_actor_base_cls = partial(
            MLP, hidden_dims=hidden_dims, dropout_rate=actor_drop, activate_final=True, use_pnorm=use_pnorm
        )
        edit_actor_cls= TanhNormal(edit_actor_base_cls, full_action_dim)
        edit_actor_def = PixelEditMultiplexer(
            network_cls=edit_actor_cls,
            latent_dim=latent_dim_state,
            include_state=include_state,
        )

        edit_actor_params = edit_actor_def.init(actor_key, jnp.ones((1, latent_dim_image)), actions=jnp.ones((1, full_action_dim)), p=critic_states)["params"]
        edit_actor = TrainState.create(
            apply_fn=edit_actor_def.apply, 
            params=edit_actor_params, 
            tx=optax.adam(learning_rate=actor_lr),
        )

        edit_actor_shape = jax.eval_shape(lambda: edit_actor)
        edit_actor_sharding = _sharding.fsdp_sharding(edit_actor_shape, mesh, log=True)
        edit_actor = jax.jit(
            lambda x: x,
            in_shardings=replicated_sharding,
            out_shardings=edit_actor_sharding,
        )(edit_actor)

        if use_critic_resnet:
            critic_base_cls = partial(
                MLPResNetV2,
                num_blocks=1,
            )
        else:
            critic_base_cls = partial(
                MLP,
                hidden_dims=hidden_dims,
                activate_final=True,
                dropout_rate=critic_dropout_rate, 
                use_layer_norm=critic_layer_norm,
                use_pnorm=use_pnorm,
            )

        critic_cls = partial(StateActionValue, base_cls=critic_base_cls)
        critic_cls = partial(Ensemble, net_cls=critic_cls, num=num_qs)
        critic_def = PixelMultiplexer(
            network_cls=critic_cls,
            latent_dim=latent_dim_state,
            include_state=include_state,
        )
        critic_params = critic_def.init(critic_key, critic_observations, critic_actions, p=critic_states_ext)["params"]
        if critic_weight_decay is not None:
            tx = optax.adamw(
                learning_rate=critic_lr,
                weight_decay=critic_weight_decay,
                mask=decay_mask_fn,
            )
        else:
            tx = optax.adam(learning_rate=critic_lr)
            
        critic = TrainState.create(
            apply_fn=critic_def.apply,
            params=critic_params,
            tx=tx,
        )

        critic_shape = jax.eval_shape(lambda: critic)
        critic_sharding = _sharding.fsdp_sharding(critic_shape, mesh, log=True)
        critic = jax.jit(
            lambda x: x,
            in_shardings=replicated_sharding,
            out_shardings=critic_sharding,
        )(critic)

        target_critic = TrainState.create(
            apply_fn=critic_def.apply,
            params=critic.params,
            tx=optax.GradientTransformation(lambda _: None, lambda _: None),
        )

        # Filter critic: same stack as the critic, but its action input is the raw noise seed (H*D_pad).
        rng, filter_key = jax.random.split(rng, 2)
        padded_action_dim = actor.model_config.action_dim
        filter_cls = partial(StateActionValue, base_cls=critic_base_cls)
        filter_cls = partial(Ensemble, net_cls=filter_cls, num=filter_num_qs)
        filter_critic_def = PixelMultiplexer(
            network_cls=filter_cls,
            latent_dim=latent_dim_state,
            include_state=include_state,
        )
        filter_noise = jnp.zeros((1, action_horizon * padded_action_dim))
        # With filter_add_delayed_obs the delayed obs is concatenated onto both inputs.
        if filter_add_delayed_obs:
            filter_observations = jnp.ones((1, 2 * latent_dim_image))
            filter_states_ext = jnp.concatenate([critic_states_ext, critic_states_ext], axis=-1)
        else:
            filter_observations = critic_observations
            filter_states_ext = critic_states_ext
        print("filter critic noise shape: ", filter_noise.shape, " states shape: ", filter_states_ext.shape)
        filter_critic_params = filter_critic_def.init(
            filter_key, filter_observations, filter_noise, p=filter_states_ext
        )["params"]
        filter_critic = TrainState.create(
            apply_fn=filter_critic_def.apply,
            params=filter_critic_params,
            tx=optax.adam(learning_rate=critic_lr),
        )

        filter_critic_shape = jax.eval_shape(lambda: filter_critic)
        filter_critic_sharding = _sharding.fsdp_sharding(filter_critic_shape, mesh, log=True)
        filter_critic = jax.jit(
            lambda x: x,
            in_shardings=replicated_sharding,
            out_shardings=filter_critic_sharding,
        )(filter_critic)

        temp_def = Temperature(init_temperature)
        temp_params = temp_def.init(temp_key)["params"]
        temp = TrainState.create(
            apply_fn=temp_def.apply,
            params=temp_params,
            tx=optax.adam(learning_rate=temp_lr),
        )

        temp_shape = jax.eval_shape(lambda: temp)
        temp_sharding = _sharding.fsdp_sharding(temp_shape, mesh, log=True)
        temp = jax.jit(
            lambda x: x,
            in_shardings=replicated_sharding,
            out_shardings=temp_sharding,
        )(temp)


        agent = cls(
            rng=rng,
            actor=actor,
            actor_train_state=actor_train_state,
            target_actor_params=target_actor_params,
            edit_actor=edit_actor,
            N=N,
            n_edit_samples=n_edit_samples,
            encode_batch_split=encode_batch_split,
            edit_scale=edit_scale,
            edit_action_xyzg=edit_action_xyzg,
            batch_split=batch_split,
            actor_tau=actor_tau,
            critic=critic,
            target_critic=target_critic,
            filter_critic=filter_critic,
            batch_encoder=batch_encoder,
            temp=temp,
            target_entropy=target_entropy,
            entropy_scale=entropy_scale,
            tau=tau,
            discount=discount,
            num_qs=num_qs,
            num_min_qs=num_min_qs,
            data_augmentation_fn=make_data_augmentation_fn(use_full_augmentation),

            action_dim=action_dim,
            state_dim=state_dim,
            full_action_dim=full_action_dim,
            action_horizon=action_horizon,
            replan_steps=replan_steps,
            resize_size=resize_size,
            default_prompt=default_prompt,
            data_sharding=data_sharding,
            batch_encoder_sharding=batch_encoder_sharding,
            freeze_encoder=freeze_encoder,
            freeze_critic_encoder=freeze_critic_encoder,
            actor_success_only=actor_success_only,
            delay=delay,
            p1_use_prefix_conditioning=p1_use_prefix_conditioning,
            q_edit_use_main_obs=q_edit_use_main_obs,
            train_base_actor=train_base_actor,
            critic_camera_keys=tuple(critic_camera_keys),
            filter_N=filter_N,
            filter_temperature=filter_temperature,
            filter_num_qs=filter_num_qs,
            filter_n_edit=filter_n_edit,
            filter_add_delayed_obs=filter_add_delayed_obs,
        )
        if not resume:
            agent = agent.cache_infer_params()
        return agent

    def _apply_edit_xyzg_mask(self, edit: jnp.ndarray) -> jnp.ndarray:
        """Zero out rotation dims (3,4,5) when edit_action_xyzg is True. Keeps xyz (0,1,2) and gripper (6)."""
        if not self.edit_action_xyzg:
            return edit
        # mask: 1 for xyz (0,1,2) and gripper (last dim), 0 for rotation (3,4,5)
        mask = jnp.ones(self.action_dim).at[3:6].set(0.0)
        full_mask = jnp.tile(mask, self.replan_steps)
        return edit * full_mask

    def cache_infer_params(self):
        """Copy params onto infer_sharding for rollout sampling.

        sample_actions reads _infer_cache to avoid device_put on every env step.
        Call again after update() so rollouts use the latest weights.
        """
        s = self.actor.infer_sharding
        return self.replace(_infer_cache={
            "actor_train_state": jax.device_put(self.actor_train_state, s),
            "batch_encoder_params": jax.device_put(self.batch_encoder.params, s),
            "edit_actor_params": jax.device_put(self.edit_actor.params, s),
            "target_critic_params": jax.device_put(self.target_critic.params, s),
        })

    def _sample_edit(self, key, edit_params, encoded_obs, states, base_actions):
        r_samples, rng = _sample_actions(
            key, self.edit_actor.apply_fn, edit_params, encoded_obs, states, base_actions
        )
        edit_scaled = self._apply_edit_xyzg_mask(r_samples * self.edit_scale)
        combined = edit_scaled + base_actions
        return combined, edit_scaled, rng

    def _encode_observations(self, observations, encoder_params=None, stop_gradient=True):
        params = encoder_params or self.batch_encoder.params
        if self.encode_batch_split > 1:
            one_call = observations.shape[0] // self.encode_batch_split
            chunks = [
                batch_encode(self.batch_encoder.apply_fn, params, observations[i * one_call:(i + 1) * one_call], stop_gradient=stop_gradient)
                for i in range(self.encode_batch_split)
            ]
            return jnp.concatenate(chunks)
        return batch_encode(self.batch_encoder.apply_fn, params, observations, stop_gradient=stop_gradient)

    def _compute_q_split(self, critic_fn, critic_params, obs, actions, states):
        if self.batch_split > 1:
            total = obs.shape[0]
            one_call = total // self.batch_split
            q_list = [
                compute_q(critic_fn, critic_params, obs[i * one_call:(i + 1) * one_call], actions[i * one_call:(i + 1) * one_call], states[i * one_call:(i + 1) * one_call], self.num_min_qs)
                for i in range(self.batch_split)
            ]
            return jnp.concatenate(q_list)
        return compute_q(critic_fn, critic_params, obs, actions, states, self.num_min_qs)

    def sample_pre_cache(self, observations, prefix_padded=None):
        """Slow phase of real-time chunking: N candidate chunks from the Pi0.5 actor, inpainting
        `prefix_padded` (the in-flight actions) when given. Returns (precached, new_agent) with
        precached (N, H, action_dim) normalized; pass it to sample_actions with the current obs
        and delay = len(prefix_padded)."""
        infer_sharding = self.actor.infer_sharding
        rng = jax.device_put(self.rng, infer_sharding)
        c = self._infer_cache or {}
        _actor_train_state = c.get("actor_train_state") or self.actor_train_state

        p1_transformed_inputs = self.actor.process_raw_inputs(observations, self.action_dim, self.resize_size)
        key, rng = jax.random.split(rng)
        if prefix_padded is None:
            precached, _ = self.actor.sample_actions(
                p1_transformed_inputs, train_state=_actor_train_state, rng=key,
                train=False, num_samples=self.N,
            )
        else:
            prefix_padded = jax.device_put(jnp.asarray(prefix_padded), infer_sharding)
            d = prefix_padded.shape[0]
            if d < 0 or d > self.replan_steps:
                raise ValueError(
                    f"sample_pre_cache: prefix length must be in [0, replan_steps]"
                    f"={[0, self.replan_steps]}; got {d}."
                )
            precached = self.actor.sample_actions_with_prefix(
                p1_transformed_inputs, train_state=_actor_train_state, rng=key,
                prefix_padded=prefix_padded,
                num_samples=self.N,
            )
        precached = jnp.asarray(precached)  # (N, H, action_dim) normalized
        return precached, self.replace(rng=rng)

    def sample_actions(self, observations, precached=None, delay=0, only_base_actions=False):
        """Fast phase of real-time chunking: edit+critic correction and Q-selection on the
        execution window [delay : delay+r] of `precached`, conditioned on the current obs.
        With precached=None the actor is sampled inline at delay 0 (the EXPOLearner path).
        Returns (output_chunk, new_agent, sample_info); sample_info["executed_padded"] is the
        corrected window, normalized+padded, whose tail [r-delay:r] is the next prefix."""
        r = self.replan_steps
        H = self.action_horizon
        if 2 * r > H:
            raise ValueError(
                f"sample_actions requires action_horizon >= 2*replan_steps; got H={H}, r={r}."
            )

        # Keep inference-time randomness on the same single device as inference params
        # to avoid cross-device gather/indexing mismatches.
        infer_sharding = self.actor.infer_sharding
        rng = jax.device_put(self.rng, infer_sharding)
        c = self._infer_cache or {}
        _actor_train_state = c.get("actor_train_state") or self.actor_train_state
        _batch_encoder_params = c.get("batch_encoder_params") or jax.device_put(self.batch_encoder.params, infer_sharding)
        _edit_actor_params = c.get("edit_actor_params") or jax.device_put(self.edit_actor.params, infer_sharding)
        _target_critic_params = c.get("target_critic_params") or jax.device_put(self.target_critic.params, infer_sharding)

        sample_time = None
        if precached is None:
            # Non-overlapped / boot: sample the actor inline on the current obs (delay=0).
            delay = 0
            key, rng = jax.random.split(rng)
            precached, sample_time = self.actor.sample_actions(
                self.actor.process_raw_inputs(observations, self.action_dim, self.resize_size),
                train_state=_actor_train_state, rng=key, train=False, num_samples=self.N,
            )
        if delay < 0 or delay > r:
            raise ValueError(
                f"sample_actions: delay must be in [0, replan_steps]={[0, r]}; got {delay}."
            )

        D_pad = self.actor.model_config.action_dim

        def _unnorm(win_norm):
            # Pad the window to a full chunk for process_transformed_outputs, then slice [:r] back.
            n = win_norm.shape[0]
            win_full = jnp.concatenate(
                [win_norm, jnp.zeros((n, H - r, self.action_dim), dtype=win_norm.dtype)], axis=1,
            )
            return jnp.asarray(self.actor.process_transformed_outputs(win_full))[:, :r, :]

        # Critic / edit correction conditions on the CURRENT obs.
        transformed_inputs_cur = self.actor.process_raw_inputs(observations, self.action_dim, self.resize_size)

        # Window slice -> critic encode -> edit candidates -> Q-select, as one jitted call.
        sel = _jitted_fast_select(
            rng, jnp.asarray(precached), transformed_inputs_cur["image"], transformed_inputs_cur["state"],
            _batch_encoder_params, _edit_actor_params, _target_critic_params,
            encoder_fn=self.batch_encoder.apply_fn,
            edit_fn=self.edit_actor.apply_fn,
            critic_fn=self.target_critic.apply_fn,
            N=self.N, n_edit_samples=self.n_edit_samples,
            num_qs=self.num_qs, num_min_qs=self.num_min_qs,
            replan_steps=r, action_dim=self.action_dim,
            state_dim=self.state_dim, padded_action_dim=D_pad,
            delay=delay, edit_scale=self.edit_scale,
            edit_action_xyzg=self.edit_action_xyzg,
            freeze_encoder=self.freeze_encoder,
            only_base_actions=only_base_actions,
            critic_camera_keys=self.critic_camera_keys,
        )
        selected = sel["selected"]  # (r, action_dim)
        executed_padded = sel["executed_padded"]  # (r, D_pad) → next prefix = [r-delay:r]
        rng = sel["rng"]

        # Output in env action space.
        output_chunk = _unnorm(selected[None])[0].reshape(r, self.action_dim)  # driver executes [:r]

        rng, _ = jax.random.split(rng, 2)
        sample_info = {
            "sample_time": sample_time,
            "delay": int(delay),
            "executed_padded": executed_padded,
        }
        return jnp.array(output_chunk), self.replace(rng=rng), sample_info

    def sample_batch_actions(self, batch):
        if self.q_edit_use_main_obs and self.delay > 0:
            # Edit + Q-selection see the main actor's delayed obs.
            critic_obs = jnp.squeeze(stack_critic_cameras(batch["next_delayed_image"], self.critic_camera_keys))
            states = jnp.squeeze(batch["next_delayed_critic_states"])
        else:
            critic_obs = jnp.squeeze(batch["next_observations"])
            states = jnp.squeeze(batch["next_critic_states"])
        critic_obs = jax.device_put(critic_obs)
        batch_size = critic_obs.shape[0]
        states = jax.device_put(states, self.data_sharding)
        rng = self.rng

        # Prepare VLA inputs; with delay>0 the backup samples the actor on the obs `delay`
        # steps before next_state and takes the window [delay : delay+r], like the rollout.
        delay = self.delay
        r = self.replan_steps
        if delay > 0:
            transformed_inputs = prepare_actor_sampling_batch_delayed(batch)
        else:
            transformed_inputs = prepare_actor_sampling_batch(batch)

        # Encode observations (true next state s' for edit/critic).
        encoded_obs = self._encode_observations(critic_obs, stop_gradient=True)
        encoded_obs = jax.device_put(encoded_obs, self.data_sharding)

        # Noise-Q pre-filter: score filter_N raw seeds with Q_f and denoise only the best one.
        H = self.action_horizon
        D_pad = self.actor.model_config.action_dim
        key, rng = jax.random.split(rng)
        noise = jax.random.normal(key, (batch_size, self.filter_N, H, D_pad))
        noise_flat = jax.device_put(
            noise.reshape(batch_size * self.filter_N, H * D_pad), self.data_sharding
        )
        if self.filter_add_delayed_obs:
            # Also condition Q_f on the actor's (delayed) obs: image latent + proprio.
            if delay > 0:
                old_critic_obs = jax.device_put(
                    jnp.squeeze(stack_critic_cameras(batch["next_delayed_image"], self.critic_camera_keys))
                )
                old_encoded_obs = jax.device_put(
                    self._encode_observations(old_critic_obs, stop_gradient=True), self.data_sharding
                )
                old_states = jax.device_put(jnp.squeeze(batch["next_delayed_critic_states"]), self.data_sharding)
            else:
                old_encoded_obs, old_states = encoded_obs, states  # actor obs == next obs
            filter_encoded_obs = jax.device_put(
                jnp.concatenate([encoded_obs, old_encoded_obs], axis=-1), self.data_sharding
            )
            filter_states = jnp.concatenate([states, old_states], axis=-1)
        else:
            filter_encoded_obs = encoded_obs
            filter_states = states
        f_obs = jax.device_put(jnp.repeat(filter_encoded_obs, self.filter_N, axis=0), self.data_sharding)
        f_states = jax.device_put(jnp.repeat(filter_states, self.filter_N, axis=0), self.data_sharding)
        # Min over the full filter ensemble; it is small, so no REDQ subsampling.
        q_f = compute_q(
            self.filter_critic.apply_fn, self.filter_critic.params, f_obs, noise_flat, f_states
        ).reshape(batch_size, self.filter_N)
        if self.filter_temperature > 0:
            # FASTER-style stochastic selection: categorical over zscore(Q_f)/temperature.
            z = (q_f - q_f.mean(axis=1, keepdims=True)) / (q_f.std(axis=1, keepdims=True) + 1e-8)
            key, rng = jax.random.split(rng)
            seed_idx = jax.random.categorical(key, z / self.filter_temperature, axis=1)
        else:
            seed_idx = jnp.argmax(q_f, axis=1)
        kept_noise = jnp.take_along_axis(
            noise, seed_idx[:, None, None, None], axis=1
        ).squeeze(1)  # (B, H, D_pad)
        filter_q_selected = jnp.take_along_axis(q_f, seed_idx[:, None], axis=1).squeeze(1)

        # Denoise ONLY the kept seed.
        key, rng = jax.random.split(rng)
        if delay > 0:
            # In-flight prefix = last `delay` actions of the executed chunk, at positions [0:delay].
            full_actions = batch["full_actions"].reshape(batch_size, self.action_horizon, self.action_dim)
            prefix = full_actions[:, r - delay:r, :]  # (B, delay, action_dim)
            prefix = jnp.concatenate(
                [prefix, jnp.zeros((batch_size, delay, D_pad - self.action_dim), dtype=prefix.dtype)],
                axis=-1,
            )  # (B, delay, D_pad)
            prefix = jax.device_put(prefix, self.data_sharding)
            actor_actions, sample_time = self.actor.sample_training_actions_with_prefix(
                transformed_inputs=transformed_inputs,
                train_state=self.actor_train_state,
                rng=key,
                prefix_padded=prefix,
                num_samples=1,
                noise=kept_noise[:, None],
            )
            actor_actions = actor_actions[:, delay:delay + r, :]  # execution window
        else:
            actor_actions, sample_time = self.actor.sample_training_actions(
                transformed_inputs=transformed_inputs,
                train_state=self.actor_train_state,
                rng=key,
                train=False,
                num_samples=1,
                noise=kept_noise[:, None],
            )
            actor_actions = actor_actions[:, :r, :]
        actions = actor_actions.reshape(batch_size, 1, self.full_action_dim)

        # Sample edit actions (bases = the single filtered action, repeated)
        n_edit = self.n_edit_samples if self.filter_n_edit < 0 else self.filter_n_edit
        total_candidates = 1 + n_edit

        if n_edit > 0:
            key, rng = jax.random.split(rng, 2)
            r_observations = jax.device_put(jnp.repeat(encoded_obs, n_edit, axis=0), self.data_sharding)
            r_states = jax.device_put(jnp.repeat(states, n_edit, axis=0), self.data_sharding)
            d_actions = jnp.repeat(actions[:, :1], n_edit, axis=1).reshape(-1, actions.shape[-1])

            r_samples, edit_scaled, rng = self._sample_edit(key, self.edit_actor.params, r_observations, r_states, d_actions)
            mean_d_actions_norm = jnp.mean(jnp.linalg.norm(d_actions, axis=1))
            mean_edit_scaled_norm = jnp.mean(jnp.linalg.norm(edit_scaled, axis=1))

            actions = jnp.concatenate([actions, r_samples.reshape(batch_size, n_edit, -1)], axis=1)
        
        else:
            mean_d_actions_norm = jnp.array(0.0, dtype=jnp.float32)
            mean_edit_scaled_norm = jnp.array(0.0, dtype=jnp.float32)

        # Select best actions via Q-values; qs[:, 0] is also the filter critic's regression target.
        key, rng = jax.random.split(rng)
        target_params = subsample_image_ensemble(
            key, self.target_critic.params, self.num_min_qs, self.num_qs
        )

        obs_flat = jax.device_put(jnp.repeat(encoded_obs, total_candidates, axis=0), self.data_sharding)
        states_flat = jax.device_put(jnp.repeat(states, total_candidates, axis=0), self.data_sharding)
        actions_flat = jax.device_put(actions.reshape(-1, actions.shape[-1]), self.data_sharding)

        qs = self._compute_q_split(self.target_critic.apply_fn, target_params, obs_flat, actions_flat, states_flat)
        qs = qs.reshape(batch_size, total_candidates)

        best_indices = jnp.argmax(qs, axis=1)

        batch_indices = jnp.arange(batch_size)
        best_actions = actions[batch_indices, best_indices]

        without_edit_mask = best_indices < 1
        with_edit_mask = (best_indices >= 1) & (best_indices < total_candidates)
        vf_select_ratio_without_edit = jnp.mean(without_edit_mask.astype(jnp.float32))
        vf_select_ratio_with_edit = jnp.mean(with_edit_mask.astype(jnp.float32))

        sample_info_extra = {
            "select_ratio_without_edit": vf_select_ratio_without_edit,
            "select_ratio_with_edit": vf_select_ratio_with_edit,
            "mean_d_actions_norm": mean_d_actions_norm,
            "mean_edit_scaled_norm": mean_edit_scaled_norm,
            "filter_q_selected_mean": filter_q_selected.mean(),
            "filter_q_pool_mean": q_f.mean(),
            "filter_q_pool_std": q_f.std(axis=1).mean(),
            # Popped by update_critic for the filter-critic regression; never logged.
            "_filter_batch": {
                "encoded_obs": filter_encoded_obs,
                "states": filter_states,
                "noise_flat": kept_noise.reshape(batch_size, H * D_pad),
                "q_target": jax.lax.stop_gradient(qs[:, 0]),
            },
        }

        rng, _ = jax.random.split(rng, 2)
        return jnp.array(best_actions.squeeze()), sample_info_extra, rng
         
    def update_edit_actor(self, batch: DatasetDict) -> Tuple[AgentLearner, Dict[str, float]]:
        key, rng = jax.random.split(self.rng)
        key2, rng = jax.random.split(rng)
        dropout_key, rng = jax.random.split(rng)

        def edit_actor_loss_fn(actor_params) -> Tuple[jnp.ndarray, Dict[str, float]]:

            observations = batch_encode(self.batch_encoder.apply_fn, self.batch_encoder.params, batch["observations"], stop_gradient=True)
            
            # Apply sharding constraint to encoded observations
            observations = jax.lax.with_sharding_constraint(observations, self.data_sharding)
            dist = self.edit_actor.apply_fn({"params": actor_params}, observations, actions=batch["actions"], training=True, p=batch['critic_states'], rngs={"dropout": dropout_key},)
            actions = dist.sample(seed=key)

            log_probs = dist.log_prob(actions)
            edit_scaled = self._apply_edit_xyzg_mask(actions * self.edit_scale)
            # Subtract log of action scale for each action dimension
            log_probs -= actions.shape[-1] * jnp.log(self.edit_scale)

            actions = edit_scaled + batch["actions"]

            qs = self.critic.apply_fn(
                {"params": self.critic.params},
                observations,
                actions,
                True,
                p=batch['critic_states'], 
                rngs={"dropout": key2},
            )  # training=True
            q = qs.mean(axis=0)
            edit_actor_loss = (
                self.entropy_scale * log_probs * self.temp.apply_fn({"params": self.temp.params}) - q
            ).mean()
            return edit_actor_loss, {"edit_q": q.mean(), "edit_actor_loss": edit_actor_loss, "entropy": -log_probs.mean()}

        grads, actor_info = jax.grad(edit_actor_loss_fn, has_aux=True)(self.edit_actor.params)
        edit_actor = self.edit_actor.apply_gradients(grads=grads)

        return self.replace(edit_actor=edit_actor, rng=rng), actor_info
    
    def update_actor(self, batch: DatasetDict) -> Tuple[AgentLearner, Dict[str, float]]:
        actor_batch = self.actor.prepare_batch_for_actor(batch)
        
        # Ensure actor_batch has correct sharding
        def ensure_sharding(x):
            if isinstance(x, jnp.ndarray):
                return jax.device_put(x, self.data_sharding)
            return x
        actor_batch = jax.tree_util.tree_map(ensure_sharding, actor_batch)

        rng = self.rng
        key, rng = jax.random.split(rng, 2)

        with _sharding.set_mesh(self.actor.mesh):
            if self.p1_use_prefix_conditioning:
                d = jnp.where(
                    batch["episode_step"] < self.replan_steps, 0, self.delay
                ).astype(jnp.int32)
                d = jax.device_put(d, self.actor.replicated_sharding)
                new_train_state, info = self.actor.train_step_p1_prefix(
                    key, self.actor_train_state, actor_batch, d,
                )
            else:
                new_train_state, info = self.actor.train_step(key, self.actor_train_state, actor_batch)

        new_train_state_params = self.actor.get_params(new_train_state)
        target_score_params = optax.incremental_update(
            new_train_state_params, self.target_actor_params, self.actor_tau
        )

        new_agent = self.replace(actor_train_state=new_train_state, target_actor_params=target_score_params, rng=rng)
        
        return new_agent, info


    def update_temperature(self, entropy: float) -> Tuple[AgentLearner, Dict[str, float]]:
        def temperature_loss_fn(temp_params):
            temperature = self.temp.apply_fn({"params": temp_params})
            temp_loss = temperature * (entropy - self.target_entropy).mean()
            return temp_loss, {
                "temperature": temperature,
                "temperature_loss": temp_loss,
            }

        grads, temp_info = jax.grad(temperature_loss_fn, has_aux=True)(self.temp.params)
        temp = self.temp.apply_gradients(grads=grads)

        return self.replace(temp=temp), temp_info


    def update_critic(self, batch: DatasetDict) -> Tuple[TrainState, Dict[str, float]]:
        next_actions, sample_info, rng = self.sample_batch_actions(batch)
        # Filter-critic regression inputs; popped so they are not logged.
        filter_batch = sample_info.pop("_filter_batch")
        next_actions = jax.device_put(next_actions, self.data_sharding)

        # Used only for REDQ.
        key, rng = jax.random.split(rng)
        target_params = subsample_image_ensemble(
            key, self.target_critic.params, self.num_min_qs, self.num_qs
        )

        key, rng = jax.random.split(rng)

        next_observations = batch_encode(self.batch_encoder.apply_fn, self.batch_encoder.params, 
                                        batch["next_observations"], stop_gradient=True)
        next_observations = jax.device_put(next_observations, self.data_sharding)  

        next_qs = self.target_critic.apply_fn(
            {"params": target_params},
            next_observations,
            next_actions,
            False,  # training=False: target must be deterministic (no dropout)
            p=batch['next_critic_states'],
            sample_num=self.num_min_qs,
        )

        next_q_nan_mask = jnp.isnan(next_qs)
        next_q_nan_ratio = jnp.mean(next_q_nan_mask)
        next_qs = jnp.where(next_q_nan_mask, 0.0, next_qs)
        next_q = next_qs.min(axis=0)
        target_q = batch["rewards"] + (self.discount ** self.replan_steps) * batch["masks"] * next_q

        key, rng = jax.random.split(rng)

        params_dict = {"critic": self.critic.params}
        if not self.freeze_critic_encoder:
            params_dict["batch_encoder"] = self.batch_encoder.params

        def critic_loss_fn(params_dict) -> Tuple[jnp.ndarray, Dict[str, float]]:
            if self.freeze_critic_encoder:
                observations = batch_encode(
                    self.batch_encoder.apply_fn,
                    self.batch_encoder.params,
                    batch["observations"],
                    stop_gradient=True,
                )
            else:
                observations = batch_encode(
                    self.batch_encoder.apply_fn, params_dict["batch_encoder"], batch["observations"]
                )
            # Apply sharding constraint to encoded observations (works inside grad)
            observations = jax.lax.with_sharding_constraint(observations, self.data_sharding)
            qs = self.critic.apply_fn(
                {"params": params_dict['critic']},
                observations,
                batch["actions"],
                True,
                p=batch['critic_states'], 
                rngs={"dropout": key},
            )
            critic_loss = (((qs - target_q) ** 2) * batch["valids"]).mean()
            return critic_loss, {
                "critic_loss": critic_loss,
                "q": qs.mean(),
                "q_min": qs.min(),
                "q_max": qs.max(),
                "target_q_min": target_q.min(),
                "target_q_max": target_q.max(),
                "target_q_mean": target_q.mean(),
            }

        grads, info = jax.grad(critic_loss_fn, has_aux=True)(params_dict)

        critic = self.critic.apply_gradients(grads=grads["critic"])

        if self.freeze_critic_encoder:
            batch_encoder = self.batch_encoder
        else:
            batch_encoder = self.batch_encoder.apply_gradients(grads=grads["batch_encoder"]) 

        critic_grad_norm = optax.global_norm(grads["critic"])
        info["critic_grad_norm"] = critic_grad_norm
        info["critic_param_norm"] = optax.global_norm(critic.params)
        info["next_q_nan_ratio"] = next_q_nan_ratio
        
        target_critic_params = optax.incremental_update(
            critic.params, self.target_critic.params, self.tau
        )
        target_critic = self.target_critic.replace(params=target_critic_params)
        info["target_critic_param_norm"] = optax.global_norm(target_critic_params)

        # Filter-critic regression: Q_f(s', seed) -> stop_grad(Q_target(s', denoise(seed))).
        key, rng = jax.random.split(rng)

        def filter_critic_loss_fn(filter_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            qfs = self.filter_critic.apply_fn(
                {"params": filter_params},
                filter_batch["encoded_obs"],
                filter_batch["noise_flat"],
                True,
                p=filter_batch["states"],
                rngs={"dropout": key},
            )
            filter_loss = ((qfs - filter_batch["q_target"][None, :]) ** 2).mean()
            return filter_loss, {
                "filter_critic_loss": filter_loss,
                "filter_q_train_mean": qfs.mean(),
                "filter_q_target_mean": filter_batch["q_target"].mean(),
            }

        filter_grads, filter_info = jax.grad(filter_critic_loss_fn, has_aux=True)(self.filter_critic.params)
        filter_critic = self.filter_critic.apply_gradients(grads=filter_grads)
        filter_info["filter_critic_grad_norm"] = optax.global_norm(filter_grads)
        info.update(filter_info)

        info.update(sample_info)

        return self.replace(critic=critic, target_critic=target_critic, filter_critic=filter_critic, batch_encoder=batch_encoder, rng=rng), info


    def update(self, agent, batch: DatasetDict, utd_ratio: int, actor_batch: DatasetDict = None):
        # Drop stale inference copies before JIT; rebuild after so rollouts use new weights.
        train_actor = bool(self.train_base_actor)  # jit static: False freezes the pi0.5 actor
        new_agent, info = self.replace(_infer_cache=None)._update_jit(
            agent.replace(_infer_cache=None), batch, utd_ratio, actor_batch,
            train_actor=train_actor,
        )
        return new_agent.cache_infer_params(), info


    @partial(jax.jit, static_argnames=("utd_ratio", "train_actor"))
    def _update_jit(self, agent, batch: DatasetDict, utd_ratio: int, actor_batch: DatasetDict = None,
                    train_actor: bool = True):
        batch = batch.copy()
        rng, key1 = jax.random.split(agent.rng)
        rng, key2 = jax.random.split(rng)
        batch["image"] = self.data_augmentation_fn(key1, batch["image"])
        batch["next_image"] = self.data_augmentation_fn(key2, batch["next_image"])
        # The delayed next obs (critic backup at delay>0) is augmented like next_image.
        if self.delay > 0 and "next_delayed_image" in batch:
            rng, key3 = jax.random.split(rng)
            batch["next_delayed_image"] = self.data_augmentation_fn(key3, batch["next_delayed_image"])
        batch = prepare_critic_batch(batch, self.actor.model_config.action_dim, self.action_dim, self.state_dim, self.action_horizon, self.replan_steps, self.critic_camera_keys)
        new_agent = agent.replace(rng=rng)

        total_bs = batch["actions"].shape[0]
        assert total_bs % utd_ratio == 0, (
            f"Batch size ({total_bs}) must be a multiple of utd_ratio ({utd_ratio})"
        )
        minibatch_size = total_bs // utd_ratio

        def reshape_minibatch(x):
            return x.reshape((utd_ratio, minibatch_size) + x.shape[1:])

        minibatches = jax.tree_util.tree_map(reshape_minibatch, batch)

        def create_minibatch_sharding(x):
            # Create sharding spec: (None, DATA_AXIS, ...)
            # None for utd_ratio dimension, DATA_AXIS for minibatch dimension
            ndim = len(x.shape)
            spec_tuple = (None,) + (_sharding.DATA_AXIS,) + (None,) * (ndim - 2)
            new_sharding = jax.sharding.NamedSharding(
                self.data_sharding.mesh,
                jax.sharding.PartitionSpec(*spec_tuple)
            )
            return jax.device_put(x, new_sharding)

        minibatches = jax.tree_util.tree_map(create_minibatch_sharding, minibatches)

        def critic_update_step(carry, mb):
            (agent,) = carry
            agent, info = agent.update_critic(mb)
            return (agent,), info

        (new_agent,), critic_infos = jax.lax.scan(critic_update_step, (new_agent,), minibatches)

        # Use last minibatch for actor updates
        last_minibatch = jax.tree_util.tree_map(lambda x: x[-1] if x is not None and hasattr(x, "shape") else x, minibatches)

        # When actor_success_only, use the dedicated success-episode batch for
        # the Pi05 actor update; otherwise use the last critic minibatch.
        if not train_actor:
            actor_info = {}
        elif self.actor_success_only:
            actor_batch = actor_batch.copy()
            rng, key = jax.random.split(new_agent.rng)
            actor_batch["image"] = self.data_augmentation_fn(key, actor_batch["image"])
            new_agent = new_agent.replace(rng=rng)
            actor_batch = prepare_critic_batch(actor_batch, self.actor.model_config.action_dim, self.action_dim, self.state_dim, self.action_horizon, self.replan_steps, self.critic_camera_keys)
            new_agent, actor_info = new_agent.update_actor(actor_batch)
        else:
            new_agent, actor_info = new_agent.update_actor(last_minibatch)

        actor_info = dict(actor_info)

        if self.n_edit_samples > 0:
            new_agent, r_actor_info = new_agent.update_edit_actor(last_minibatch)
            new_agent, temp_info = new_agent.update_temperature(r_actor_info["entropy"])
            actor_info = {**actor_info, **r_actor_info, **temp_info}

        critic_info = jax.tree_util.tree_map(lambda x: x[-1], critic_infos)
        return new_agent, {**actor_info, **critic_info}

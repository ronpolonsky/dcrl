"""RTCLearner: pi0.5 BC with action-prefix conditioning (RTC-SFT), the base policy
for the real-time chunking learner. Trains the actor with ``train_step_p1_prefix`` and can
sample with a caller-provided clean prefix.
"""

import dataclasses
from typing import Any, Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from flax import struct

import openpi.shared.array_typing as at
import openpi.training.sharding as _sharding
import openpi.training.utils as training_utils

from expo_ft.agents.alg.agent import AgentLearner
from expo_ft.agents.alg.batch_utils import prepare_critic_batch
from expo_ft.data.dataset import DatasetDict
from expo_ft.utils.augmentation import make_data_augmentation_fn


def _split_params_rtc(agent: Any) -> tuple[Any, dict[str, at.Params]]:
    """Pull the actor params out of the agent for serialization."""
    with at.disable_typechecking():
        if agent.actor_train_state.ema_params is not None:
            actor_params = agent.actor_train_state.ema_params
            actor_train_state = dataclasses.replace(agent.actor_train_state, ema_params=None)
        else:
            actor_params = agent.actor_train_state.params
            actor_train_state = dataclasses.replace(agent.actor_train_state, params={})
    agent = dataclasses.replace(agent, actor_train_state=actor_train_state)
    return agent, {"actor_params": actor_params}


def _merge_params_rtc(agent: Any, params: dict[str, at.Params]) -> Any:
    """Put actor params back into the agent after restore."""
    with at.disable_typechecking():
        if agent.actor_train_state.params:
            actor_train_state = dataclasses.replace(
                agent.actor_train_state, ema_params=params["actor_params"],
            )
        else:
            actor_train_state = dataclasses.replace(
                agent.actor_train_state, params=params["actor_params"],
            )
    return dataclasses.replace(agent, actor_train_state=actor_train_state)


def restore_checkpoint(checkpoint_manager, agent, step: int | None = None):
    """Restore actor params, allowing topology changes between save and load."""
    agent, params = _split_params_rtc(agent)
    agent_restore_args = ocp.checkpoint_utils.construct_restore_args(agent)
    params_restore_args = ocp.checkpoint_utils.construct_restore_args(params)
    restored = checkpoint_manager.restore(
        step,
        items={"agent": agent, "params": params},
        restore_kwargs={
            "agent": {"restore_args": agent_restore_args},
            "params": {"restore_args": params_restore_args},
        },
    )
    return _merge_params_rtc(restored["agent"], restored["params"])


def save_checkpoint(checkpoint_manager: ocp.CheckpointManager, agent: Any, step: int):
    agent, params = _split_params_rtc(agent)
    checkpoint_manager.save(step, {"agent": agent, "params": params})


def load_agent(seed, example_observation, example_action, example_state,
               actor, actor_train_state, target_actor_params, agent_kwargs, metadata,
               mesh, data_sharding, replicated_sharding, resume, replan_steps,
               default_prompt, **kwargs):
    """Create an RTCLearner from the pre-built VLA actor.

    ``target_actor_params`` is accepted for loader API compatibility but not used.
    """
    p1_use_prefix_conditioning = agent_kwargs.pop("p1_use_prefix_conditioning", True)
    p1_max_delay = agent_kwargs.pop("p1_max_delay", -1)
    agent_kwargs.update(
        actor=actor,
        actor_train_state=actor_train_state,
        resume=resume,
        replan_steps=replan_steps,
        data_sharding=data_sharding,
        replicated_sharding=replicated_sharding,
        default_prompt=default_prompt,
        p1_use_prefix_conditioning=p1_use_prefix_conditioning,
        p1_max_delay=p1_max_delay,
        **metadata,
    )
    return RTCLearner.create(seed, example_observation, example_action, example_state, **agent_kwargs)


class RTCLearner(AgentLearner, struct.PyTreeNode):
    """Prefix-conditioned BC learner. Inference is vanilla pi0.5 sampling, or
    prefix-inpainted sampling from a delayed observation and a caller-provided prefix.
    """

    rng: jax.random.PRNGKey
    data_augmentation_fn: Callable = struct.field(pytree_node=False)
    actor: Any = struct.field(pytree_node=False)
    actor_train_state: training_utils.TrainState
    action_dim: int = struct.field(pytree_node=False)
    state_dim: int = struct.field(pytree_node=False)
    full_action_dim: int = struct.field(pytree_node=False)
    replan_steps: int = struct.field(pytree_node=False)
    action_horizon: int = struct.field(pytree_node=False)
    resize_size: Optional[int] = struct.field(pytree_node=False)
    default_prompt: Optional[str] = struct.field(pytree_node=False)
    data_sharding: Optional[jax.sharding.NamedSharding] = struct.field(pytree_node=False)
    replicated_sharding: Optional[jax.sharding.NamedSharding] = struct.field(pytree_node=False)
    freeze_encoder: Optional[bool] = struct.field(pytree_node=False)
    p1_use_prefix_conditioning: bool = struct.field(pytree_node=False, default=True)
    p1_max_delay: int = struct.field(pytree_node=False, default=-1)
    _infer_cache: Optional[dict] = struct.field(pytree_node=False, default=None)

    @classmethod
    def create(
        cls,
        seed: int,
        observation_space,
        action_space,
        states,
        actor: Any = None,
        actor_train_state: Any = None,
        action_horizon: int = 1,
        freeze_encoder: bool = False,
        replan_steps: int = 1,
        data_sharding: Optional[jax.sharding.NamedSharding] = None,
        replicated_sharding: Optional[jax.sharding.NamedSharding] = None,
        default_prompt: Optional[str] = None,
        resize_size: Optional[int] = None,
        use_full_augmentation: bool = True,
        p1_use_prefix_conditioning: bool = True,
        p1_max_delay: int = -1,
        **kwargs,
    ):
        action_dim = action_space.shape[-1]
        state_dim = states.shape[-1]
        full_action_dim = replan_steps * action_dim

        if p1_max_delay is None or p1_max_delay < 0:
            p1_max_delay = replan_steps
        assert 0 <= p1_max_delay < action_horizon, (
            f"p1_max_delay must be in [0, action_horizon); got {p1_max_delay}, "
            f"action_horizon={action_horizon}."
        )

        return cls(
            rng=jax.random.PRNGKey(seed),
            data_augmentation_fn=make_data_augmentation_fn(use_full_augmentation),
            actor=actor,
            actor_train_state=actor_train_state,
            action_dim=action_dim,
            state_dim=state_dim,
            full_action_dim=full_action_dim,
            replan_steps=replan_steps,
            action_horizon=action_horizon,
            resize_size=resize_size,
            default_prompt=default_prompt,
            data_sharding=data_sharding,
            replicated_sharding=replicated_sharding,
            freeze_encoder=freeze_encoder,
            p1_use_prefix_conditioning=p1_use_prefix_conditioning,
            p1_max_delay=p1_max_delay,
        ).cache_infer_params()

    def cache_infer_params(self):
        leaves = jax.tree_util.tree_leaves(self.actor_train_state.params)
        if any(isinstance(x, jax.ShapeDtypeStruct) for x in leaves):
            return self
        s = self.actor.infer_sharding
        return self.replace(_infer_cache={
            "actor_train_state": jax.device_put(self.actor_train_state, s),
        })

    def _place_aug_key(self, key):
        if self.replicated_sharding is not None:
            return jax.device_put(key, self.replicated_sharding)
        if self.data_sharding is not None:
            replicated = jax.sharding.NamedSharding(
                self.data_sharding.mesh, jax.sharding.PartitionSpec()
            )
            return jax.device_put(key, replicated)
        return key

    def update_actor(self, batch: DatasetDict) -> Tuple[AgentLearner, Dict[str, float]]:
        obs, padded_actions = self.actor.prepare_batch_for_actor(batch)
        actor_batch = (obs, padded_actions)

        def ensure_sharding(x):
            if isinstance(x, jnp.ndarray):
                return jax.device_put(x, self.data_sharding)
            return x

        actor_batch = jax.tree_util.tree_map(ensure_sharding, actor_batch)
        rng = self.rng
        key, rng = jax.random.split(rng, 2)
        with _sharding.set_mesh(self.actor.mesh):
            if self.p1_use_prefix_conditioning:
                max_delay = jax.device_put(
                    jnp.asarray(self.p1_max_delay, jnp.int32),
                    self.replicated_sharding,
                )
                new_train_state, info = self.actor.train_step_p1_prefix(
                    key, self.actor_train_state, actor_batch, max_delay,
                )
            else:
                new_train_state, info = self.actor.train_step(
                    key, self.actor_train_state, actor_batch,
                )
        return self.replace(actor_train_state=new_train_state, rng=rng), info

    def update(self, agent, batch: DatasetDict, utd_ratio: int, actor_batch: DatasetDict = None):
        if actor_batch is None:
            raise ValueError("RTCLearner.update expected actor_batch but got None.")
        actor_batch = actor_batch.copy()
        rng, key = jax.random.split(agent.rng)
        key = self._place_aug_key(key)
        actor_batch["image"] = self.data_augmentation_fn(key, actor_batch["image"])
        agent = agent.replace(rng=rng)
        actor_batch = prepare_critic_batch(
            actor_batch,
            self.actor.model_config.action_dim,
            self.action_dim,
            self.state_dim,
            self.action_horizon,
            self.replan_steps,
        )
        agent, info = agent.update_actor(actor_batch)
        return agent.cache_infer_params(), info

    def sample_actions(
        self,
        observations,
        only_base_actions=False,
        delay=0,
        p1_observations=None,
        obs_env_time=None,
        prefix_padded=None,
    ):
        """Sample from the prefix-conditioned actor.

        With ``delay == 0`` this is the vanilla actor sampler. With ``delay > 0``
        it runs the RTC sampler, using ``p1_observations`` and ``prefix_padded``
        from the caller's delayed rollout loop.
        """
        if delay <= 0:
            rng = self.rng
            key, rng = jax.random.split(rng)
            c = self._infer_cache or {}
            train_state = c.get("actor_train_state") or self.actor_train_state
            transformed_inputs = self.actor.process_raw_inputs(
                observations, self.action_dim, self.resize_size,
            )
            transformed_actions, sample_time = self.actor.sample_actions(
                transformed_inputs,
                train_state=train_state,
                rng=key,
                train=False,
                num_samples=1,
            )
            raw_actions = self.actor.process_transformed_outputs(transformed_actions)
            executed_padded = self.actor._pad_actions(
                jnp.asarray(transformed_actions).reshape(1, self.action_horizon, self.action_dim)
            )[0, :self.replan_steps]
            sample_info = {
                "sample_time": sample_time,
                "delay": int(delay),
                "executed_padded": executed_padded,
            }
            return jnp.array(raw_actions[0]), self.replace(rng=rng), sample_info

        rtc_observations = p1_observations if p1_observations is not None else observations
        return self.sample_actions_p1_rtc(
            rtc_observations,
            delay=delay,
            obs_env_time=obs_env_time,
            only_base_actions=only_base_actions,
            prefix_padded=prefix_padded,
        )

    def sample_actions_p1_rtc(
        self,
        observations,
        delay,
        obs_env_time=0,
        only_base_actions=False,
        prefix_padded=None,
    ):
        """RTC-style inference using only the actor with a caller-owned prefix."""
        if delay < 0:
            raise ValueError(f"delay must be >= 0; got {delay}.")
        if delay + self.replan_steps > self.action_horizon:
            raise ValueError(
                f"delay + replan_steps must be <= action_horizon; got "
                f"delay={delay}, replan_steps={self.replan_steps}, "
                f"action_horizon={self.action_horizon}."
            )

        c = self._infer_cache or {}
        train_state = c.get("actor_train_state") or self.actor_train_state
        rng = self.rng
        key, rng = jax.random.split(rng, 2)
        H = self.action_horizon

        inputs = self.actor.process_raw_inputs(
            observations, self.action_dim, self.resize_size,
        )

        boot = prefix_padded is None or only_base_actions or delay == 0

        if boot:
            transformed_actions, sample_time = self.actor.sample_actions(
                inputs, train_state=train_state, rng=key,
                train=False, num_samples=1,
            )
            chunk_padded = self.actor._pad_actions(
                jnp.asarray(transformed_actions).reshape(1, H, self.action_dim)
            )[0]
            raw_actions = self.actor.process_transformed_outputs(transformed_actions)[0]
            output_chunk = raw_actions
            executed_padded = chunk_padded[:self.replan_steps]
            sample_info = {
                "sample_time": sample_time,
                "delay": int(delay),
                "obs_env_time": None if obs_env_time is None else int(obs_env_time),
                "executed_padded": executed_padded,
            }
        else:
            prefix_padded = jnp.asarray(prefix_padded)
            if prefix_padded.shape[0] != delay:
                raise ValueError(
                    f"RTC prefix length must equal delay; got prefix length "
                    f"{prefix_padded.shape[0]} and delay={delay}."
                )
            prefix_padded = jax.device_put(
                prefix_padded,
                self.actor.infer_sharding,
            )
            prefixed_actions = self.actor.sample_actions_with_prefix(
                inputs,
                train_state=train_state,
                rng=key,
                prefix_padded=prefix_padded,
            )
            prefixed_padded = self.actor._pad_actions(
                jnp.asarray(prefixed_actions).reshape(1, H, self.action_dim)
            )[0]
            raw_actions = self.actor.process_transformed_outputs(prefixed_actions)
            output_chunk = jnp.asarray(raw_actions)[0, delay:, :]
            executed_padded = prefixed_padded[delay:delay + self.replan_steps]
            sample_info = {
                "sample_time": None,
                "delay": int(delay),
                "obs_env_time": None if obs_env_time is None else int(obs_env_time),
                "executed_padded": executed_padded,
            }

        new_agent = self.replace(
            rng=rng,
        )
        return jnp.array(output_chunk), new_agent, sample_info

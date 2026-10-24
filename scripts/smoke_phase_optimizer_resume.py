"""Small end-to-end checkpoint/resume smoke test for the 3-phase actor state.

This uses the real SmallExpoLearner, its real actor update methods, and Orbax
checkpoint save/restore.  It intentionally runs on a tiny CPU-compatible input
and does not require LIBERO or rollout data.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

from openpi.training import sharding as openpi_sharding

from dcrl.agents.pi_expo_learner import restore_checkpoint, save_checkpoint
from dcrl.agents.small_actor import SmallActorTrainState
from dcrl.agents.pi_expo_learner import PiExpoLearner
from dcrl.expo.agents.alg.agent import initialize_checkpoint_dir


def _tree_max_abs_diff(left, right) -> float:
    left_leaves, left_def = jax.tree_util.tree_flatten(left)
    right_leaves, right_def = jax.tree_util.tree_flatten(right)
    assert left_def == right_def
    diffs = []
    for a, b in zip(left_leaves, right_leaves):
        if a is None or b is None:
            assert a is b
            continue
        aa = np.asarray(jax.device_get(a))
        bb = np.asarray(jax.device_get(b))
        assert aa.shape == bb.shape
        diffs.append(float(np.max(np.abs(aa.astype(np.float64) - bb.astype(np.float64)))))
    return max(diffs, default=0.0)


def _assert_tree_equal(left, right, name: str, atol: float = 1e-7):
    diff = _tree_max_abs_diff(left, right)
    assert diff <= atol, f"{name} differs by {diff}"


class _TinyEncoder(nn.Module):
    stage_sizes: tuple[int, ...] = (1,)
    num_filters: int = 2

    @nn.compact
    def __call__(self, x):
        batched = x if x.ndim == 4 else x[None]
        y = nn.Conv(self.num_filters, (3, 3), padding="SAME")(batched)
        y = nn.relu(y).mean(axis=(-3, -2))
        return y if x.ndim == 4 else y[0]


class _TinyActor:
    """Minimal actor adapter exercising the real PiExpoLearner update path."""

    class _Config:
        action_dim = 1

    def __init__(self, mesh, data_sharding, seed):
        self.action_dim = 1
        self.mesh = mesh
        self.data_sharding = data_sharding
        self.infer_sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])
        self.model_config = self._Config()
        del seed

        def apply_fn(_variables, observations):
            return observations

        self._apply_fn = apply_fn

    def prepare_batch_for_actor(self, batch):
        return batch["observations"], batch["full_actions"]

    def train_step(self, rng, state, batch):
        del rng
        observations, actions = batch

        def loss_fn(params):
            prediction = observations * params["w"]
            return jnp.mean((prediction - actions) ** 2)

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, {"actor_loss": loss, "actor_state_step": state.step}

    def get_params(self, state):
        ema_params = getattr(state, "ema_params", None)
        return state.params if ema_params is None else ema_params


def _make_agent(seed: int):
    mesh = openpi_sharding.make_mesh(1)
    data_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec(openpi_sharding.DATA_AXIS)
    )
    replicated_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec()
    )
    actor = _TinyActor(mesh, data_sharding, seed)
    params = {"w": jnp.asarray([0.25], jnp.float32)}
    actor_state = SmallActorTrainState.create(
        apply_fn=actor._apply_fn,
        params=params,
        tx=optax.adam(0.1),
    )
    observation = np.zeros((1, 8, 8, 6), np.float32)
    agent = PiExpoLearner.create(
        seed=seed,
        observation_space=jnp.ones((8, 8, 6), jnp.float32),
        action_space=np.zeros((1,), np.float32),
        states=np.zeros((1,), np.float32),
        actor=actor,
        actor_train_state=actor_state,
        target_actor_params=params,
        action_horizon=1,
        replan_steps=1,
        mesh=mesh,
        data_sharding=data_sharding,
        replicated_sharding=replicated_sharding,
        hidden_dims=(4,),
        latent_dim_image=4,
        latent_dim_state=4,
        encoder_cls=_TinyEncoder,
        encoder_stage_sizes=(1,),
        encoder_num_filters=2,
        N=1,
        n_edit_samples=0,
        freeze_encoder=True,
        resume=True,
    )
    return agent, observation


def _make_batch(observation):
    return {
        "observations": observation,
        "full_actions": np.zeros((1, 1), np.float32),
    }


def _save(manager, agent, step: int):
    save_checkpoint(manager, agent, step)
    manager.wait_until_finished()


def main():
    root = Path(tempfile.mkdtemp(prefix="dcrl_phase_optimizer_smoke_"))
    try:
        # Build a deterministic uninterrupted reference and a source that will
        # be checkpointed after two distillation updates.
        reference, observation = _make_agent(11)
        source, _ = _make_agent(11)
        batch = _make_batch(observation)
        reference = reference.activate_actor_optimizer("distill")
        source = source.activate_actor_optimizer("distill")

        assert reference.distill_actor_step == 0
        assert reference.rl_actor_step == 0
        assert reference.temp.step == 0

        for _ in range(3):
            reference, _ = reference.bc_update(batch)
        for _ in range(2):
            source, _ = source.bc_update(batch)

        # Distillation advances only its actor optimizer.  RL and temperature
        # states must still be exactly fresh.
        assert source.distill_actor_step == 2
        assert source.rl_actor_step == 0
        assert source.temp.step == 0
        fresh, _ = _make_agent(11)
        _assert_tree_equal(
            source.rl_actor_opt_state,
            fresh.rl_actor_opt_state,
            "fresh RL optimizer after distillation",
        )

        manager, _ = initialize_checkpoint_dir(
            root / "distill_checkpoint",
            keep_period=None,
            overwrite=True,
            resume=False,
        )
        _save(manager, source, 2)

        resumed, _ = _make_agent(11)
        resumed = restore_checkpoint(manager, resumed, 2)
        assert resumed.distill_actor_step == 2
        assert resumed.rl_actor_step == 0
        resumed = resumed.activate_actor_optimizer("distill")
        resumed, _ = resumed.bc_update(batch)

        # Resuming from the distillation checkpoint must reproduce the third
        # uninterrupted update, including its Adam moments.
        _assert_tree_equal(resumed.actor_train_state.params, reference.actor_train_state.params, "resumed distill params")
        _assert_tree_equal(resumed.distill_actor_opt_state, reference.distill_actor_opt_state, "resumed distill optimizer")
        assert resumed.distill_actor_step == reference.distill_actor_step == 3
        assert resumed.rl_actor_step == 0

        # Skipping the remaining distillation phase activates the untouched RL
        # state.  The first RL update must start at RL step one, not BC step
        # three, while retaining the actor weights produced by distillation.
        skipped = restore_checkpoint(manager, _make_agent(11)[0], 2)
        skipped = skipped.activate_actor_optimizer("rl")
        assert skipped.actor_train_state.step == 0
        skipped, _ = skipped.update_actor(batch)
        assert skipped.rl_actor_step == 1
        assert skipped.distill_actor_step == 2
        assert skipped.temp.step == 0

        # Save after one RL update and verify that a resumed RL update matches
        # an uninterrupted RL optimizer continuation.
        rl_reference = restore_checkpoint(manager, _make_agent(11)[0], 2)
        rl_reference = rl_reference.activate_actor_optimizer("rl")
        rl_reference, _ = rl_reference.update_actor(batch)
        rl_reference, _ = rl_reference.update_temperature(jax.numpy.asarray(-0.5))

        rl_manager, _ = initialize_checkpoint_dir(
            root / "rl_checkpoint",
            keep_period=None,
            overwrite=True,
            resume=False,
        )
        _save(rl_manager, rl_reference, 3)

        rl_reference, _ = rl_reference.update_actor(batch)
        rl_reference, _ = rl_reference.update_temperature(jax.numpy.asarray(-0.5))

        rl_resumed, _ = _make_agent(11)
        rl_resumed = restore_checkpoint(rl_manager, rl_resumed, 3)
        rl_resumed = rl_resumed.activate_actor_optimizer("rl")
        assert rl_resumed.rl_actor_step == 1
        assert rl_resumed.temp.step == 1
        rl_resumed, _ = rl_resumed.update_actor(batch)
        rl_resumed, _ = rl_resumed.update_temperature(jax.numpy.asarray(-0.5))

        _assert_tree_equal(rl_resumed.actor_train_state.params, rl_reference.actor_train_state.params, "resumed RL params")
        _assert_tree_equal(rl_resumed.rl_actor_opt_state, rl_reference.rl_actor_opt_state, "resumed RL optimizer")
        _assert_tree_equal(rl_resumed.temp.params, rl_reference.temp.params, "resumed temperature params")
        _assert_tree_equal(rl_resumed.temp.opt_state, rl_reference.temp.opt_state, "resumed temperature optimizer")
        assert rl_resumed.rl_actor_step == rl_reference.rl_actor_step == 2
        assert rl_resumed.temp.step == rl_reference.temp.step == 2

        print("phase optimizer/resumption smoke test: PASS")
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()

import functools
from typing import Optional, Type

import tensorflow_probability.substrates.jax as tfp

from expo_ft.distributions.tanh_transformed import TanhTransformedDistribution
tfd = tfp.distributions

import flax.linen as nn
import jax.numpy as jnp

from expo_ft.networks import default_init


class Normal(nn.Module):
    base_cls: Type[nn.Module]
    action_dim: int
    log_std_min: Optional[float] = -20
    log_std_max: Optional[float] = 2
    state_dependent_std: bool = True
    squash_tanh: bool = False
    # Scales Gaussian before tanh (dsrl_pi0 action_magnitude: Tanh(M*u), u ~ N(mu,sigma)).
    pre_tanh_scale: float = 1.0

    @nn.compact
    def __call__(self, inputs, *args, **kwargs) -> tfd.Distribution:
        x = self.base_cls()(inputs, *args, **kwargs)

        means = nn.Dense(
            self.action_dim, kernel_init=default_init(), name="OutputDenseMean"
        )(x)
        if self.state_dependent_std:
            log_stds = nn.Dense(
                self.action_dim, kernel_init=default_init(), name="OutputDenseLogStd"
            )(x)
        else:
            log_stds = self.param(
                "OutpuLogStd", nn.initializers.zeros, (self.action_dim,), jnp.float32
            )

        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        m = jnp.asarray(self.pre_tanh_scale, dtype=means.dtype)
        distribution = tfd.MultivariateNormalDiag(
            loc=means * m, scale_diag=jnp.exp(log_stds) * m
        )

        if self.squash_tanh:
            return TanhTransformedDistribution(distribution)
        else:
            return distribution


TanhNormal = functools.partial(Normal, squash_tanh=True)

from functools import partial
from typing import Type

import flax.linen as nn
import jax
import jax.numpy as jnp


class Ensemble(nn.Module):
    net_cls: Type[nn.Module]
    num: int = 2

    @nn.compact
    def __call__(self, *args, sample_num=None):
        axis_size = sample_num if sample_num is not None else self.num
        
        ensemble = nn.vmap(
            self.net_cls,
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            in_axes=None,
            out_axes=0,
            axis_size=axis_size,
        )
        return ensemble()(*args)


@partial(jax.jit, static_argnames=("num_sample", "num_qs"))
def subsample_image_ensemble(key: jax.random.PRNGKey, params, num_sample: int, num_qs: int):
    if num_sample is not None:
        all_indx = jnp.arange(0, num_qs)
        indx = jax.random.choice(key, a=all_indx, shape=(num_sample,), replace=False)
        if "Ensemble_0" in params:
            ens_params = jax.tree_util.tree_map(
                lambda param: param[indx], params["Ensemble_0"]
            )
            # For regular dict, use copy() and update, or dict unpacking
            params = params.copy()
            params["Ensemble_0"] = ens_params
        else:
            params = jax.tree_util.tree_map(lambda param: param[indx], params)
    return params
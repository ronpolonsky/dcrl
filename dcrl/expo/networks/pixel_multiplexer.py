from typing import Dict, Optional, Tuple, Type, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict


default_init = nn.initializers.xavier_uniform


class BatchSeparateEncoder(nn.Module):
    encoder_cls: Type[nn.Module]
    latent_dim: int
    pixel_keys: Tuple[str, ...] = ("pixels",)
    depth_keys: Tuple[str, ...] = ()


    @nn.compact
    def __call__(
        self,
        observations: Union[FrozenDict, Dict],
        stop_gradient: bool = False, 
    ) -> jnp.ndarray:
        x = observations.astype(jnp.float32)

        # If inputs are stacked cameras (B, K, H, W, C): encode each camera with a separate encoder module
        b, k, h, w, c = x.shape
        latents = []
        for i in range(k):
            xi = x[:, i]
            hi = self.encoder_cls(name=f"encoder_{i}")(xi)
            if stop_gradient:
                hi = jax.lax.stop_gradient(hi)
            hi = nn.Dense(self.latent_dim, kernel_init=default_init())(hi)
            hi = nn.LayerNorm()(hi)
            latents.append(hi)
        x = jnp.concatenate(latents, axis=-1)
        # Project concatenated camera latents back to latent_dim
        x = nn.Dense(self.latent_dim, kernel_init=default_init())(x)
        x = nn.LayerNorm()(x)
        x = nn.tanh(x)
        return x

class BatchEncoder(nn.Module):
    encoder_cls: Type[nn.Module]
    latent_dim: int
    pixel_keys: Tuple[str, ...] = ("pixels",)
    depth_keys: Tuple[str, ...] = ()


    @nn.compact
    def __call__(
        self,
        observations: Union[FrozenDict, Dict],
        stop_gradient: bool = False, 
    ) -> jnp.ndarray:
        # observations = FrozenDict(observations)
        if len(self.depth_keys) == 0:
            depth_keys = [None] * len(self.pixel_keys)
        else:
            depth_keys = self.depth_keys

        xs = []
        for i, (pixel_key, depth_key) in enumerate(zip(self.pixel_keys, depth_keys)):
            x = observations.astype(jnp.float32)

            if depth_key is not None:
                # The last dim is always for stacking, even if it's 1.
                x = jnp.concatenate([x, observations[depth_key]], axis=-2)

            # x = jnp.reshape(x, (*x.shape[:-2], -1))

            x = self.encoder_cls(name=f"encoder_{i}")(x)

            if stop_gradient:
                # We do not update conv layers with policy gradients.
                x = jax.lax.stop_gradient(x)

            x = nn.Dense(self.latent_dim, kernel_init=default_init())(x)
            x = nn.LayerNorm()(x)
            x = nn.tanh(x)
            xs.append(x)

        x = jnp.concatenate(xs, axis=-1)

        if len(x.shape) == 1:
            x = jnp.expand_dims(x, axis=0)

        return x 



class PixelMultiplexer(nn.Module):
    network_cls: Type[nn.Module]
    latent_dim: int
    include_state: bool = False

    @nn.compact
    def __call__(
        self,
        observations: Union[FrozenDict, Dict],
        actions: Optional[jnp.ndarray] = None,
        training: bool = False,
        p: Optional[jnp.array] = None,
        sample_num: Optional[int] = None,
    ) -> jnp.ndarray:
        # if len(self.depth_keys) == 0:
        #     depth_keys = [None] * len(self.pixel_keys)
        # else:
        #     depth_keys = self.depth_keys

        # xs = []
        # for i, (pixel_key, depth_key) in enumerate(zip(self.pixel_keys, depth_keys)):
        #     x = observations.astype(jnp.float32) / 255.0

        #     if depth_key is not None:
        #         # The last dim is always for stacking, even if it's 1.
        #         x = jnp.concatenate([x, observations[depth_key]], axis=-2)

        #     x = self.encoder_cls(name=f"encoder_{i}")(x)

        #     if self.stop_gradient:
        #         # We do not update conv layers with policy gradients.
        #         x = jax.lax.stop_gradient(x)

        #     x = nn.Dense(self.latent_dim, kernel_init=default_init())(x)
        #     x = nn.LayerNorm()(x)
        #     x = nn.tanh(x)
        #     xs.append(x)

        # x = jnp.concatenate(xs, axis=-1)

        # if len(x.shape) == 1:
        #     x = jnp.expand_dims(x, axis=0)

        x = observations
        

        if self.include_state:
            y = nn.Dense(self.latent_dim, kernel_init=default_init())(p)
            y = nn.LayerNorm()(y)
            y = nn.tanh(y)

            x = jnp.concatenate([x, y], axis=-1)



        if actions is None:
            return self.network_cls()(x, training, sample_num=sample_num)
        else:
            return self.network_cls()(x, actions, training, sample_num=sample_num)


class PixelEditMultiplexer(nn.Module):
    network_cls: Type[nn.Module]
    latent_dim: int
    include_state: bool = False

    @nn.compact
    def __call__(
        self,
        observations: Union[FrozenDict, Dict],
        actions: Optional[jnp.ndarray] = None,
        training: bool = False,
        p: Optional[jnp.array] = None, 
    ) -> jnp.ndarray:
        # if len(self.depth_keys) == 0:
        #     depth_keys = [None] * len(self.pixel_keys)
        # else:
        #     depth_keys = self.depth_keys

        # xs = []
        # for i, (pixel_key, depth_key) in enumerate(zip(self.pixel_keys, depth_keys)):
        #     x = observations.astype(jnp.float32) / 255.0
        #     if depth_key is not None:
        #         # The last dim is always for stacking, even if it's 1.
        #         x = jnp.concatenate([x, observations[depth_key]], axis=-2)

        #     # x = jnp.reshape(x, (*x.shape[:-2], -1))

        #     x = self.encoder_cls(name=f"encoder_{i}")(x)

        #     if self.stop_gradient:
        #         # We do not update conv layers with policy gradients.
        #         x = jax.lax.stop_gradient(x)

        #     x = nn.Dense(self.latent_dim, kernel_init=default_init())(x)
        #     x = nn.LayerNorm()(x)
        #     x = nn.tanh(x)
        #     xs.append(x)

        # x = jnp.concatenate(xs, axis=-1)

        # if len(x.shape) == 1:
        #     x = jnp.expand_dims(x, axis=0)


        x = observations
        

        if self.include_state:
            y = nn.Dense(self.latent_dim, kernel_init=default_init())(p)
            y = nn.LayerNorm()(y)
            y = nn.tanh(y)

            x = jnp.concatenate([x, y], axis=-1)


        if actions is None:
            return self.network_cls(x, training)
        else:
            x = jnp.concatenate([x, actions], axis=-1)
            return self.network_cls(x, training)


class PixelActorMultiplexer(nn.Module):
    network_cls: Type[nn.Module]
    latent_dim: int
    include_state: bool = False

    @nn.compact
    def __call__(
        self,
        s: Union[FrozenDict, Dict],
        a: Optional[jnp.ndarray] = None,
        time: any = None,
        training: bool = False,
        p: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:

        # if len(self.depth_keys) == 0:
        #     depth_keys = [None] * len(self.pixel_keys)
        # else:
        #     depth_keys = self.depth_keys

        # xs = []
        # for i, (pixel_key, depth_key) in enumerate(zip(self.pixel_keys, depth_keys)):
        #     x = s.astype(jnp.float32) / 255.0
        #     if depth_key is not None:
        #         # The last dim is always for stacking, even if it's 1.
        #         x = jnp.concatenate([x, s[depth_key]], axis=-2)


        #     x = self.encoder_cls(name=f"encoder_{i}")(x)



        #     if self.stop_gradient:
        #         # We do not update conv layers with policy gradients.
        #         x = jax.lax.stop_gradient(x)

        #     x = nn.Dense(self.latent_dim, kernel_init=default_init())(x)
        #     x = nn.LayerNorm()(x)
        #     x = nn.tanh(x)
        #     xs.append(x)

        # x = jnp.concatenate(xs, axis=-1)

        # if len(x.shape) == 1:
        #     x = jnp.expand_dims(x, axis=0)


        x = s



        if self.include_state:
            y = nn.Dense(self.latent_dim, kernel_init=default_init())(p)
            y = nn.LayerNorm()(y)
            y = nn.tanh(y)

            x = jnp.concatenate([x, y], axis=-1)


        return self.network_cls(s=x, a=a, time=time, training=training)



class PixelActorDebugMultiplexer(nn.Module):
    encoder_cls: Type[nn.Module]
    network_cls: Type[nn.Module]
    latent_dim: int
    stop_gradient: bool = False
    pixel_keys: Tuple[str, ...] = ("pixels",)
    depth_keys: Tuple[str, ...] = ()
    include_state: bool = False,

    @nn.compact
    def __call__(
        self,
        s: Union[FrozenDict, Dict],
        a: Optional[jnp.ndarray] = None,
        time: any = None,
        training: bool = False,
        p: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:

        if len(self.depth_keys) == 0:
            depth_keys = [None] * len(self.pixel_keys)
        else:
            depth_keys = self.depth_keys

        xs = []
        for i, (pixel_key, depth_key) in enumerate(zip(self.pixel_keys, depth_keys)):
            x = s.astype(jnp.float32)
            if depth_key is not None:
                # The last dim is always for stacking, even if it's 1.
                x = jnp.concatenate([x, s[depth_key]], axis=-2)


            x = self.encoder_cls(name=f"encoder_{i}")(x)



            if self.stop_gradient:
                # We do not update conv layers with policy gradients.
                x = jax.lax.stop_gradient(x)

            x = nn.Dense(self.latent_dim, kernel_init=default_init())(x)
            x = nn.LayerNorm()(x)
            x = nn.tanh(x)
            xs.append(x)

        x = jnp.concatenate(xs, axis=-1)

        if len(x.shape) == 1:
            x = jnp.expand_dims(x, axis=0)


        # x = s



        if self.include_state:
            y = nn.Dense(self.latent_dim, kernel_init=default_init())(p)
            y = nn.LayerNorm()(y)
            y = nn.tanh(y)

            x = jnp.concatenate([x, y], axis=-1)


        return self.network_cls(s=x, a=a, time=time, training=training)



class PixelTanhNormalMultiplexer(nn.Module):
    """Multiplexer specifically designed for TanhNormal networks."""
    network_cls: Type[nn.Module]
    latent_dim: int
    include_state: bool = False
    state_latent_dim: int = 50

    @nn.compact
    def __call__(
        self,
        observations: jnp.ndarray,
        actions: Optional[jnp.ndarray] = None,
        training: bool = False,
        p: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """Apply the network with optional state concatenation.
        
        Args:
            observations: Encoded observations of shape (batch_size, latent_dim)
            actions: Optional actions (not used for TanhNormal, but kept for compatibility)
            training: Training flag
            p: Optional states of shape (batch_size, state_dim)
            
        Returns:
            Network output
        """
        x = observations
        
        if self.include_state and p is not None:
            # Process states through a dense layer and concatenate
            y = nn.Dense(self.state_latent_dim, kernel_init=default_init())(p)
            y = nn.LayerNorm()(y)
            y = nn.tanh(y)
            
            # Concatenate observations and processed states
            x = jnp.concatenate([x, y], axis=-1)
            
            # Add another layer to compress concatenated features back to latent_dim
            x = nn.Dense(self.latent_dim, kernel_init=default_init())(x)
            x = nn.LayerNorm()(x)
            x = nn.tanh(x)
        
        # Apply the network (TanhNormal) - now always receives latent_dim input
        return self.network_cls(x, training=training)


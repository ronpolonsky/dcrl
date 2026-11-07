"""Compact vision transformer for the shared EXPO critic/edit observation encoder."""

import flax.linen as nn
import jax.numpy as jnp


class _TransformerBlock(nn.Module):
    width: int
    heads: int
    mlp_width: int

    @nn.compact
    def __call__(self, x):
        y = nn.LayerNorm()(x)
        y = nn.SelfAttention(num_heads=self.heads, qkv_features=self.width, out_features=self.width)(y)
        x = x + y
        y = nn.LayerNorm()(x)
        y = nn.Dense(self.mlp_width)(y)
        y = nn.gelu(y)
        y = nn.Dense(self.width)(y)
        return x + y


class TinyViTEncoder(nn.Module):
    """About 1M parameters at width=144, depth=4, patch_size=16, six input channels.

    The encoder consumes the same channel-stacked camera tensor as ResNetV2Encoder
    and returns a flat vector, so EXPO's BatchEncoder projection is unchanged.
    """

    width: int = 144
    depth: int = 4
    heads: int = 4
    mlp_width: int = 288
    patch_size: int = 16

    @nn.compact
    def __call__(self, x):
        x = jnp.asarray(x, jnp.float32)
        if x.ndim == 3:
            x = x[None]
        x = nn.Conv(self.width, (self.patch_size, self.patch_size),
                    strides=(self.patch_size, self.patch_size), padding='SAME')(x)
        x = x.reshape(x.shape[0], -1, self.width)
        pos = self.param('position_embedding', nn.initializers.normal(0.02),
                         (1, x.shape[1], self.width))
        x = x + pos
        for i in range(self.depth):
            x = _TransformerBlock(self.width, self.heads, self.mlp_width,
                                  name=f'block_{i}')(x)
        return nn.LayerNorm()(x).mean(axis=1)

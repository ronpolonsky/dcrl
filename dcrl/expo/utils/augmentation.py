"""Image augmentation utilities for Pi training."""

from typing import Callable, Dict

import augmax
import jax
import jax.numpy as jnp


def batched_openpi_augmentation(rng, image_dict: Dict[str, jnp.ndarray]) -> Dict[str, jnp.ndarray]:
    """Apply same image augmentation as openpi preprocess_observation.

    image_dict has base_0_rgb, left_wrist_0_rgb in [-1, 1] float32.
    Base and wrist each get: RandomCrop(95%), Resize, Rotate(-5,5), ColorJitter.
    Returns augmented image dict.
    """
    base = image_dict["base_0_rgb"]
    wrist = image_dict["left_wrist_0_rgb"]
    height, width = base.shape[1], base.shape[2]

    base = base / 2.0 + 0.5
    wrist = wrist / 2.0 + 0.5

    base_transforms = [
        augmax.RandomCrop(int(width * 0.95), int(height * 0.95)),
        augmax.Resize(width, height),
        augmax.Rotate((-5, 5)),
        augmax.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    ]
    wrist_transforms = [
        augmax.RandomCrop(int(width * 0.95), int(height * 0.95)),
        augmax.Resize(width, height),
        augmax.Rotate((-5, 5)),
        augmax.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    ]

    sub_rngs = jax.random.split(rng, 2 * base.shape[0])
    base_rngs = sub_rngs[0::2]
    wrist_rngs = sub_rngs[1::2]

    base = jax.vmap(augmax.Chain(*base_transforms))(base_rngs, base)
    wrist = jax.vmap(augmax.Chain(*wrist_transforms))(wrist_rngs, wrist)

    base = base * 2.0 - 1.0
    wrist = wrist * 2.0 - 1.0

    return dict(
        image_dict,
        base_0_rgb=base,
        left_wrist_0_rgb=wrist,
    )


def random_crop(key, img: jnp.ndarray, padding: int) -> jnp.ndarray:
    crop_from = jax.random.randint(key, (2,), 0, 2 * padding + 1)
    crop_from = jnp.concatenate([crop_from, jnp.zeros((1,), dtype=jnp.int32)])
    padded_img = jnp.pad(
        img, ((padding, padding), (padding, padding), (0, 0)), mode="edge"
    )
    return jax.lax.dynamic_slice(padded_img, crop_from, img.shape)


def batched_random_crop_per_image(key, obs: jnp.ndarray, padding: int = 4) -> jnp.ndarray:
    """Apply random crop independently to each image in obs.

    obs is (B, H, W, 6) = [base_0_rgb (3 ch), left_wrist_0_rgb (3 ch)].
    Base and wrist get independent crop offsets.
    """
    base = obs[..., :3]
    wrist = obs[..., 3:6]
    keys = jax.random.split(key, 2 * obs.shape[0])
    keys_base = keys[0::2]
    keys_wrist = keys[1::2]
    base = jax.vmap(random_crop, (0, 0, None))(keys_base, base, padding)
    wrist = jax.vmap(random_crop, (0, 0, None))(keys_wrist, wrist, padding)
    return jnp.concatenate([base, wrist], axis=-1)


def batched_crop_only_augmentation(rng, image_dict: Dict[str, jnp.ndarray]) -> Dict[str, jnp.ndarray]:
    """Apply only random crop (padding=12) independently to base and wrist."""
    base = image_dict["base_0_rgb"]
    wrist = image_dict["left_wrist_0_rgb"]
    obs = jnp.concatenate([base, wrist], axis=-1)
    obs = batched_random_crop_per_image(rng, obs, padding=12)
    return dict(
        image_dict,
        base_0_rgb=obs[..., :3],
        left_wrist_0_rgb=obs[..., 3:6],
    )


def make_data_augmentation_fn(
    use_full_augmentation: bool = True,
) -> Callable[[jax.Array, Dict[str, jnp.ndarray]], Dict[str, jnp.ndarray]]:
    """Return rng-keyed augmentation fn for learner create()."""

    def data_augmentation_fn(rng, image_dict):
        if use_full_augmentation:
            return batched_openpi_augmentation(rng, image_dict)
        return batched_crop_only_augmentation(rng, image_dict)

    return data_augmentation_fn

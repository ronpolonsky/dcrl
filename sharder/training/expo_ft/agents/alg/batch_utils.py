"""Batch preparation utilities for critic and actor training."""

import jax.numpy as jnp

# Cameras the critic encoder sees, channel-stacked in this (sorted) order. The
# critic obs is a single tensor, so this set + order MUST be identical across
# every site that builds it (prepare_critic_batch, extract_critic_fields, and
# the buffer's convert_to_critic_format init example).
#
# Default is upstream's set (side + wrist); a task can opt into all three slots via
# `config.critic_camera_keys`.
CRITIC_CAMERA_KEYS = ("base_0_rgb", "left_wrist_0_rgb")


def stack_critic_cameras(image_dict, camera_keys=CRITIC_CAMERA_KEYS):
    """Channel-stack the critic's camera views into one tensor (order = camera_keys)."""
    return jnp.concatenate([image_dict[k] for k in camera_keys], axis=-1)


def prepare_critic_batch(batch, padded_dim, action_dim, state_dim, action_horizon, replan_steps,
                         critic_camera_keys=CRITIC_CAMERA_KEYS):
    """Add critic observations, states, and truncated action targets to a training batch."""
    batch_size = batch["state"].shape[0]

    batch['observations'] = stack_critic_cameras(batch["image"], critic_camera_keys)
    batch['next_observations'] = stack_critic_cameras(batch["next_image"], critic_camera_keys)

    batch['states'] = batch["state"].reshape(batch_size, padded_dim)[..., :state_dim].reshape(batch_size, state_dim)
    batch['next_states'] = batch["next_state"].reshape(batch_size, padded_dim)[..., :state_dim].reshape(batch_size, state_dim)
    batch['critic_states'] = batch['states']
    batch['next_critic_states'] = batch['next_states']

    # State `delay` env-steps before next_state (env time t' - delay). Used as the
    # VLA `state` input for the real-time chunking critic backup's delayed next-action
    # inference. When the buffer's delay==0 this equals next_states.
    if "next_delayed_state" in batch:
        batch['next_delayed_states'] = batch["next_delayed_state"].reshape(batch_size, padded_dim)[..., :state_dim].reshape(batch_size, state_dim)
        batch['next_delayed_critic_states'] = batch['next_delayed_states']

    actions_unpadded = batch["actions"].reshape(batch_size, action_horizon, padded_dim)[..., :action_dim]
    batch['full_actions'] = actions_unpadded.reshape(batch_size, action_horizon * action_dim)
    batch['actions'] = actions_unpadded[:, :replan_steps, :].reshape(batch_size, replan_steps * action_dim)

    return batch


def prepare_actor_sampling_batch(batch):
    """Select next-state fields from a batch for actor action sampling."""
    return {
        "image": batch["next_image"],
        "image_mask": batch["next_image_mask"],
        "state": batch["next_states"],
        "tokenized_prompt": batch['tokenized_prompt'],
        "tokenized_prompt_mask": batch['tokenized_prompt_mask'],
        "token_ar_mask": batch.get('token_ar_mask', None),
        "token_loss_mask": batch.get('token_loss_mask', None),
    }


def prepare_actor_sampling_batch_delayed(batch):
    """Select the obs `delay` steps before next_state for the real-time chunking critic
    backup's delayed next-action inference (env time t' - delay)."""
    return {
        "image": batch["next_delayed_image"],
        "image_mask": batch["next_delayed_image_mask"],
        "state": batch["next_delayed_states"],
        "tokenized_prompt": batch['tokenized_prompt'],
        "tokenized_prompt_mask": batch['tokenized_prompt_mask'],
        "token_ar_mask": batch.get('token_ar_mask', None),
        "token_loss_mask": batch.get('token_loss_mask', None),
    }

def extract_critic_fields(processed_inputs, padded_dim, state_dim,
                          critic_camera_keys=CRITIC_CAMERA_KEYS):
    """Add critic_obs and critic_states keys to a processed inputs dict."""
    processed_inputs["critic_obs"] = stack_critic_cameras(processed_inputs["image"], critic_camera_keys)
    processed_inputs["critic_states"] = processed_inputs["state"].reshape(
        processed_inputs["state"].shape[0], padded_dim
    )[..., :state_dim]
    return processed_inputs

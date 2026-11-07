"""Shared utilities for Pi robot training scripts."""

import dataclasses
import logging
import pathlib
from typing import Any, Dict

import etils.epath as epath
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import wandb
from flax import traverse_util

from openpi.training import config as openpi_config
from openpi.training import weight_loaders as openpi_weight_loaders


def _restore_pi05_params(params_path: str) -> Any:
    """Restore a pi05 param tree, tolerating any of the known top-level wrapper keys.

    openpi train.py saves the tree under ``{"params": ...}``; train_offline_rtc.py
    saves the actor (pi05) weights under ``{"actor_params": ...}``. Mirrors
    openpi's ``restore_params`` but picks whichever key is present.
    """
    path = pathlib.Path(params_path).resolve()
    with ocp.PyTreeCheckpointer() as ckptr:
        metadata = ckptr.metadata(path)
        top = next((k for k in ("params", "actor_params") if k in metadata), None)
        if top is None:
            raise KeyError(
                f"No 'params'/'actor_params' key in checkpoint {path}; "
                f"got {list(metadata)}"
            )
        # orbax requires the restore item to match the full on-disk structure, so
        # restore every top-level key and keep only the one we want.
        item = {k: metadata[k] for k in metadata}
        restored = ckptr.restore(
            path,
            ocp.args.PyTreeRestore(
                item=item,
                restore_args=jax.tree.map(
                    lambda _: ocp.ArrayRestoreArgs(restore_type=np.ndarray), item
                ),
            ),
        )[top]
    # nnx.State leaves end with a "value" key; strip it to match the pure dict
    # the reference params use (same convention as openpi's restore_params).
    flat = traverse_util.flatten_dict(restored)
    if flat and all(kp[-1] == "value" for kp in flat):
        flat = {kp[:-1]: v for kp, v in flat.items()}
    return traverse_util.unflatten_dict(flat)


@dataclasses.dataclass(frozen=True)
class FlexibleCheckpointWeightLoader:
    """Drop-in for openpi's ``CheckpointWeightLoader`` that also accepts the
    ``actor_params``-wrapped trees saved by train_offline_rtc.py."""

    params_path: str

    def load(self, params: Any) -> Any:
        loaded = _restore_pi05_params(self.params_path)
        # Merge onto the reference tree, filling missing LoRA weights from init
        # (identical policy to openpi's CheckpointWeightLoader).
        return openpi_weight_loaders._merge_params(loaded, params, missing_regex=".*lora.*")


def init_logging() -> None:
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(checkpoint_dir: epath.Path, resuming: bool, project: str, name: str) -> None:
    wandb_id_file = checkpoint_dir / "wandb_id.txt"
    settings = wandb.Settings(init_timeout=300)  # loaded shared nodes are slow to start wandb-core
    if resuming and wandb_id_file.exists():
        run_id = wandb_id_file.read_text().strip()
        wandb.init(id=run_id, resume="must", project=project, name=name, settings=settings)
    else:
        wandb.init(project=project, name=name, settings=settings)
        wandb_id_file.write_text(wandb.run.id)


def set_compilation_cache_dir(tag: str) -> str:
    """Point JAX's persistent compilation cache at a device-layout-scoped subdir.

    JAX strips the device assignment from its GPU cache key, so a single-device
    executable cached for cuda:1 is replayed with buffers on cuda:0 and fails with
    "replica is assigned to device cuda:1". Entries are only safe to share between
    runs that place the same computation on the same device, so scope the dir by
    driver, learner, and device count (all three decide those placements).
    """
    cache_dir = epath.Path("~/.cache/jax").expanduser() / f"{tag}-n{jax.device_count()}"
    jax.config.update("jax_compilation_cache_dir", str(cache_dir))
    logging.info("JAX compilation cache: %s", cache_dir)
    return str(cache_dir)


def clear_batch(batch: Dict[str, Any]) -> None:
    """Recursively clear a batch dictionary to free memory."""
    if isinstance(batch, dict):
        for v in batch.values():
            if isinstance(v, dict):
                clear_batch(v)
        batch.clear()


def get_batch_info(batch: Dict[str, Any]) -> Dict[str, float]:
    """Extract basic statistics from a batch dictionary for logging."""
    return {
        "rewards_mean": float(np.mean(batch["rewards"])),
        "rewards_std": float(np.std(batch["rewards"])),
        "rewards_max": float(np.max(batch["rewards"])),
        "rewards_min": float(np.min(batch["rewards"])),
        "masks_mean": float(np.mean(batch["masks"])),
        "masks_std": float(np.std(batch["masks"])),
        "valids_mean": float(np.mean(batch["valids"])),
        "valids_std": float(np.std(batch["valids"])),
        "actions_mean": float(np.mean(batch["actions"])),
        "actions_std": float(np.std(batch["actions"])),
        "actions_max": float(np.max(batch["actions"])),
        "actions_min": float(np.min(batch["actions"])),
        "states_mean": float(np.mean(batch["state"])),
        "states_std": float(np.std(batch["state"])),
        "states_max": float(np.max(batch["state"])),
        "states_min": float(np.min(batch["state"])),
        "base_image_max": float(np.max(batch["image"]["base_0_rgb"])),
        "base_image_min": float(np.min(batch["image"]["base_0_rgb"])),
        "base_image_std": float(np.std(batch["image"]["base_0_rgb"])),
    }


def build_pi05_config(config):
    """Extract pi05 settings from agent config and build the openpi train config.

    Returns (agent_kwargs, pi05_train_config, pi05_resize_size, model_cls).
    ``agent_kwargs`` is a plain dict with pi05-specific keys removed.
    """
    agent_kwargs = dict(config)
    pi05_config_name = agent_kwargs.pop("pi05_config_name")
    pi05_resize_size = agent_kwargs.pop("pi05_resize_size")
    pi05_weight_loader_path = agent_kwargs.pop("pi05_weight_loader_path", "") or None
    pi05_assets_dir = agent_kwargs.pop("pi05_assets_dir", "") or None
    pi05_asset_id = agent_kwargs.pop("pi05_asset_id", "") or None
    model_cls = agent_kwargs.pop("model_cls")

    pi05_train_config = openpi_config.get_config(
        pi05_config_name, weight_loader_path=pi05_weight_loader_path
    )
    if pi05_weight_loader_path:
        # Use a loader that tolerates both openpi-style ("params") and
        # train_offline_rtc.py-style ("actor_params") checkpoint trees.
        pi05_train_config = dataclasses.replace(
            pi05_train_config,
            weight_loader=FlexibleCheckpointWeightLoader(pi05_weight_loader_path),
        )
    if pi05_assets_dir or pi05_asset_id:
        from openpi.training.config import AssetsConfig
        new_assets = AssetsConfig(
            assets_dir=pi05_assets_dir or pi05_train_config.data.assets.assets_dir,
            asset_id=pi05_asset_id or pi05_train_config.data.assets.asset_id,
        )
        pi05_train_config = dataclasses.replace(
            pi05_train_config,
            data=dataclasses.replace(pi05_train_config.data, assets=new_assets),
        )
    return agent_kwargs, pi05_train_config, pi05_resize_size, model_cls


def _concat_leaves(x, y):
    """tree_map'd concatenation along axis 0 that tolerates None leaves."""
    if x is None:
        return y
    if y is None:
        return x
    return jnp.concatenate([jnp.asarray(x), jnp.asarray(y)], axis=0)


def _shuffle_batch(key, batch):
    """Shuffle a (possibly nested) batch along axis=0 with one shared permutation."""
    leaves, treedef = jax.tree_util.tree_flatten(batch)
    n = leaves[0].shape[0]
    perm = jax.random.permutation(key, n)
    shuffled = [jnp.asarray(x)[perm] for x in leaves]
    return jax.tree_util.tree_unflatten(treedef, shuffled)


def combine_batches(online_batch, offline_batch, rng):
    """Combine online and offline batches: concatenate, then shuffle to mix.

    Batches must already have the desired sizes (proportional to offline_ratio).
    """
    combined = jax.tree_util.tree_map(_concat_leaves, online_batch, offline_batch)
    return _shuffle_batch(rng, combined)

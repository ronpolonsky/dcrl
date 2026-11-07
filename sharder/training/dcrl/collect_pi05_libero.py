"""Collect official Pi0.5 LIBERO rollouts as full action-chunk examples.

This is the frozen base actor used by PiExpoLearner.  It follows OpenPI's
LIBERO interaction loop: 224px policy inputs, fixed benchmark initial states,
ten settling actions, ten-action predictions, and five executed actions before
replanning.  Each HDF5 row stores the observation and the complete predicted
chunk at that replanning point.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import jax
import numpy as np
from tqdm import tqdm


def _raw_input(obs: dict, instruction: str) -> dict:
    return {
        "observation/image": obs["image"]["base_0_rgb"],
        "observation/wrist_image": obs["image"]["left_wrist_0_rgb"],
        "observation/state": obs["state"],
        "prompt": instruction,
    }


def _partition(source: Path, root: Path) -> None:
    with h5py.File(source, "r") as src:
        if not bool(src.attrs.get("complete", False)):
            raise ValueError(f"incomplete collection: {source}")
        metadata = src.attrs["metadata"]
        # Keep completed episodes easy to consume downstream: partition the
        # mixed source file into explicit successes/failures directories while
        # retaining the mixed source file for resumability and auditing.
        for label, wanted in (("successes", True), ("failures", False)):
            destination = root / label / source.name
            if destination.exists():
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            partial = destination.with_suffix(".partial.hdf5")
            with h5py.File(partial, "w") as dst:
                for key, value in src.attrs.items():
                    dst.attrs[key] = value
                dst.attrs["partition"] = label
                for name, group in src.items():
                    if bool(group.attrs.get("success", False)) == wanted:
                        src.copy(name, dst, name=name)
                dst.attrs["complete"] = True
            partial.replace(destination)


def collect(
    *,
    checkpoint: str,
    norm_stats: str,
    suite: str,
    task_id: int,
    episodes: int | None,
    seed: int,
    output: Path,
    successes: int | None = None,
    failures: int | None = None,
) -> None:
    import dataclasses
    from openpi import transforms
    from openpi.models import model
    from openpi.shared import download, normalize
    from openpi.training import config as configs, optimizer, sharding, weight_loaders
    from dcrl.expo.agents.vla.pi05 import Pi05Agent
    from dcrl.libero_env import LIBERO_SUITE_MAX_STEPS, LiberoTaskEnv

    if episodes is not None and episodes <= 0:
        raise ValueError("episodes must be positive")
    if episodes is not None:
        successes = failures = None
    if (successes is None) != (failures is None):
        raise ValueError("provide both successes and failures, or neither")
    if successes is not None and (successes <= 0 or failures <= 0):
        raise ValueError("successes and failures must be positive")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(".partial.hdf5")

    # Official pi05_libero is action_horizon=10 and action_dim=7.
    config = dataclasses.replace(
        configs.get_config("pi05_libero"),
        weight_loader=weight_loaders.CheckpointWeightLoader(checkpoint.rstrip("/") + "/params"),
        optimizer=optimizer.SGD(momentum=0.0),
    )
    stats_source = download.maybe_download(norm_stats)
    stats_dir = output.parent / "pi05_assets" / "physical-intelligence" / "libero"
    stats_dir.mkdir(parents=True, exist_ok=True)
    stats_path = stats_dir / "norm_stats.json"
    if not stats_path.exists():
        stats_path.write_bytes(Path(stats_source).read_bytes())
    stats = normalize.load(stats_dir)
    mesh = sharding.make_mesh(1)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    actor, state, _ = Pi05Agent.initialize(
        config, mesh, jax.random.PRNGKey(seed), resume=False,
        data_sharding=data_sharding, replicated_sharding=replicated_sharding,
        default_prompt="", freeze_pi05_encoder=False,
    )
    print(f"Initialized Pi0.5 base actor from {checkpoint}", flush=True)
    actor.action_dim, actor.state_dim = 7, 8
    dc = actor.data_config
    transform = transforms.compose([
        *dc.data_transforms.inputs,
        transforms.Normalize(stats, use_quantiles=dc.use_quantile_norm),
        *dc.model_transforms.inputs,
    ])
    output_transform = transforms.compose([
        *dc.model_transforms.outputs,
        transforms.Unnormalize(stats, use_quantiles=dc.use_quantile_norm),
        *dc.data_transforms.outputs,
    ])

    env = LiberoTaskEnv(suite, task_id, seed=seed, image_size=224, benchmark_eval=True)
    max_steps = LIBERO_SUITE_MAX_STEPS[suite]
    instruction = env.instruction
    norm_bytes = stats_path.read_bytes()
    metadata = {
        "source": "pi05_base_actor",
        "checkpoint": checkpoint,
        "suite": suite,
        "task_id": int(task_id),
        "seed": int(seed),
        "episodes": None if episodes is None else int(episodes),
        "target_successes": None if successes is None else int(successes),
        "target_failures": None if failures is None else int(failures),
        "image_size": 224,
        "render_resolution": 256,
        "action_horizon": 10,
        "replan_steps": 5,
        "max_steps": max_steps,
        "official_pi05_libero": True,
        "norm_sha256": hashlib.sha256(norm_bytes).hexdigest(),
        "format_version": 1,
    }

    try:
        if output.exists():
            with h5py.File(output, "r") as existing:
                if json.loads(existing.attrs["metadata"]) != metadata:
                    raise ValueError("existing collection does not match this request")
            _partition(output, output.parent)
            return
        if partial.exists():
            try:
                with h5py.File(partial, "r"):
                    pass
            except OSError:
                partial.unlink()

        rng = jax.random.PRNGKey(seed)
        with h5py.File(partial, "a") as file:
            if "metadata" in file.attrs and json.loads(file.attrs["metadata"]) != metadata:
                raise ValueError("partial collection does not match this request")
            file.attrs["metadata"] = json.dumps(metadata, sort_keys=True)
            file.attrs["instruction"] = instruction
            file.attrs["normalization_stats"] = norm_bytes.decode()
            episode = 0
            while True:
                if successes is None:
                    if episode >= episodes:
                        break
                else:
                    success_count = sum(bool(file[name].attrs.get("success", False)) for name in file)
                    failure_count = sum(not bool(file[name].attrs.get("success", False)) for name in file)
                    if success_count >= successes and failure_count >= failures:
                        break
                    if episode >= max(episodes or 0, successes + failures) * 4:
                        raise RuntimeError(
                            f"reached collection attempt limit with {success_count} successes and "
                            f"{failure_count} failures; requested {successes}/{failures}"
                        )
                name = f"episode_{episode:06d}"
                if name in file and bool(file[name].attrs.get("complete", False)):
                    episode += 1
                    continue
                if name in file:
                    del file[name]
                # Repeat the fixed official benchmark starts if more than the
                # suite's 50 standard evaluation states are requested.
                env.init_state_offset = episode % len(env.init_states)
                env.episode_index = 0
                obs = env.reset()
                rows = []
                success = False
                step = 0
                while step < max_steps and not success:
                    start = obs
                    inputs = transform(_raw_input(obs, instruction))
                    inputs = jax.tree.map(lambda value: np.asarray(value)[None], inputs)
                    packed = model.Observation.from_dict(inputs).to_dict()
                    rng, key = jax.random.split(rng)
                    actions, _ = actor.sample_actions(packed, state, key, num_samples=1)
                    actions = np.asarray(jax.device_get(actions))[0]
                    actions = output_transform({
                        "actions": actions,
                        "state": np.asarray(inputs["state"])[0],
                    })["actions"]
                    actions = np.asarray(actions, dtype=np.float32)
                    if actions.shape != (10, 7) or not np.isfinite(actions).all():
                        raise ValueError(f"invalid Pi0.5 action chunk shape={actions.shape}")
                    executed = min(5, max_steps - step)
                    for action in actions[:executed]:
                        obs, _, success, _ = env.step(action)
                        step += 1
                        if success or step >= max_steps:
                            break
                    rows.append((start, actions, executed))
                # Keep exactly the requested number in each partition. Extra
                # outcomes are still executed but omitted from this dataset.
                if successes is not None:
                    success_count = sum(bool(file[n].attrs.get("success", False)) for n in file)
                    failure_count = sum(not bool(file[n].attrs.get("success", False)) for n in file)
                    if (success and success_count >= successes) or (not success and failure_count >= failures):
                        episode += 1
                        continue
                group = file.create_group(name)
                group.attrs.update(success=bool(success), steps=int(step),
                                   initial_state_index=episode % len(env.init_states), complete=True)
                for key in ("base_0_rgb", "left_wrist_0_rgb"):
                    group.create_dataset(key, data=np.stack([r[0]["image"][key] for r in rows]), compression="gzip")
                group.create_dataset("state", data=np.stack([r[0]["state"] for r in rows]))
                group.create_dataset("actions", data=np.stack([r[1] for r in rows]))
                group.create_dataset("executed_prefix", data=[r[2] for r in rows])
                file.flush()
                episode += 1
            file.attrs["complete"] = True
        partial.replace(output)
        _partition(output, output.parent)
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="gs://openpi-assets/checkpoints/pi05_base")
    parser.add_argument("--norm-stats", default="gs://openpi-assets/checkpoints/pi05_libero/assets/physical-intelligence/libero/norm_stats.json")
    parser.add_argument("--suite", default="libero_90")
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=None,
                        help="Collect a fixed number of attempts instead of outcome quotas.")
    parser.add_argument("--successes", type=int, default=100)
    parser.add_argument("--failures", type=int, default=25)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    collect(**vars(args))


if __name__ == "__main__":
    main()

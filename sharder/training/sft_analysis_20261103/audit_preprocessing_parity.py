"""Audit SFT training/inference preprocessing parity on real collected rows.

This is deliberately a CPU-only, no-model-weights check.  It applies the
exact OpenPI LIBERO transform objects used by the SFT learner to rows from the
saved teacher collections, then applies the same transform chain used by the
Pi05 runtime after its inference-side repack.  The comparison is made after
``Observation.from_dict`` so uint8 image conversion is included.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np


OUT = Path(__file__).resolve().parent / "preprocessing_parity_audit.json"
NORM_PATH = Path(
    "/projects/bgqe/rpolonsky/dcrlcheckpoints/openpi_cache/openpi-assets/"
    "checkpoints/pi05_libero/assets/physical-intelligence/libero/norm_stats.json"
)
DATA_ROOT = Path("/work/hdd/bgqe/rpolonsky/dcrlcheckpoints/sft_agnostic_small_expo")
RUNS = {
    "task_02": DATA_ROOT / "task_02_teacher_375398_success150_sft200k" / "pi_base_data/success/teacher_chunks.hdf5",
    "task_28": DATA_ROOT / "task_28_teacher_275001_success150_sft200k" / "pi_base_data/success/teacher_chunks.hdf5",
    "task_29": DATA_ROOT / "task_29_teacher_375380_success150_sft200k" / "pi_base_data/success/teacher_chunks.hdf5",
    "task_60": DATA_ROOT / "task_60_teacher_450223_success150_sft200k" / "pi_base_data/success/teacher_chunks.hdf5",
}


def _copy(value: Any) -> Any:
    return copy.deepcopy(value)


def _leaves(value: Any, path: tuple[str, ...] = ()):
    if isinstance(value, dict):
        for key in sorted(value):
            yield from _leaves(value[key], path + (str(key),))
    else:
        yield "/".join(path), value


def _leaf_summary(value: Any) -> dict[str, Any]:
    arr = np.asarray(value)
    result: dict[str, Any] = {"shape": list(arr.shape), "dtype": str(arr.dtype)}
    if arr.dtype.kind in "biuf":
        finite = np.isfinite(arr)
        result.update(
            finite=bool(finite.all()),
            min=float(arr[finite].min()) if finite.any() else None,
            max=float(arr[finite].max()) if finite.any() else None,
        )
    return result


def _compare_tree(left: Any, right: Any) -> dict[str, Any]:
    left_leaves = dict(_leaves(left))
    right_leaves = dict(_leaves(right))
    all_paths = sorted(set(left_leaves) | set(right_leaves))
    rows = []
    max_abs = 0.0
    max_rel = 0.0
    exact = True
    for path in all_paths:
        if path not in left_leaves or path not in right_leaves:
            rows.append({"path": path, "status": "missing"})
            exact = False
            continue
        a = np.asarray(left_leaves[path])
        b = np.asarray(right_leaves[path])
        if a.shape != b.shape:
            rows.append({"path": path, "status": "shape", "left": list(a.shape), "right": list(b.shape)})
            exact = False
            continue
        if a.dtype.kind in "biuf" and b.dtype.kind in "biuf":
            delta = np.abs(a.astype(np.float64) - b.astype(np.float64))
            abs_err = float(delta.max()) if delta.size else 0.0
            denom = np.maximum(np.abs(a.astype(np.float64)), 1e-12)
            rel_err = float((delta / denom).max()) if delta.size else 0.0
            equal = bool(np.array_equal(a, b))
            rows.append({"path": path, "status": "ok", "exact": equal, "max_abs": abs_err, "max_rel": rel_err})
            max_abs = max(max_abs, abs_err)
            max_rel = max(max_rel, rel_err)
            exact &= equal
        else:
            equal = bool(np.array_equal(a, b))
            rows.append({"path": path, "status": "ok", "exact": equal})
            exact &= equal
    return {"exact": exact, "max_abs": max_abs, "max_rel": max_rel, "leaves": rows}


def _sample_row(path: Path, row_index: int = 0) -> dict[str, Any]:
    with h5py.File(path, "r") as file:
        names = sorted(file.keys())
        if not names:
            raise ValueError(f"empty collection: {path}")
        group = file[names[0]]
        row_index = min(row_index, len(group["actions"]) - 1)
        return {
            "observation/image": np.asarray(group["base_0_rgb"][row_index]),
            "observation/wrist_image": np.asarray(group["left_wrist_0_rgb"][row_index]),
            "observation/state": np.asarray(group["state"][row_index]),
            "actions": np.asarray(group["actions"][row_index]),
            "prompt": str(file.attrs["instruction"]),
            "episode": names[0],
            "row": row_index,
        }


def main() -> None:
    if not NORM_PATH.is_file():
        raise FileNotFoundError(NORM_PATH)
    for path in RUNS.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    # Import the project's exact OpenPI/config copies, not a system install.
    from openpi import transforms
    from openpi.models import model as model_lib
    from openpi.shared import normalize
    from openpi.training import config as openpi_config
    from configs.sft import pi05_base_libero

    base = pi05_base_libero.get_config()
    stats = normalize.load(NORM_PATH.parent)
    # Passing the checkpoint's real assets root makes create() construct the
    # same official transforms and also verifies that the loaded stats match.
    assets_root = NORM_PATH.parents[2]
    data_config = base.data.create(assets_root, base.model)
    if data_config.norm_stats is None:
        raise AssertionError("official norm stats were not loaded by DataConfig.create")
    if json.dumps({k: str(v) for k, v in data_config.norm_stats.items()}, sort_keys=True) != json.dumps(
        {k: str(v) for k, v in stats.items()}, sort_keys=True
    ):
        raise AssertionError("DataConfig and explicit norm-stat loads differ")

    # These are exactly Pi05Agent._build_input_transform_pipeline() and the
    # SFT config.build_learner() transform, respectively.
    inference_transform = transforms.compose([
        *data_config.repack_transforms.inputs,
        *data_config.data_transforms.inputs,
        transforms.Normalize(stats, use_quantiles=data_config.use_quantile_norm),
        *data_config.model_transforms.inputs,
    ])
    training_transform = transforms.compose([
        *data_config.data_transforms.inputs,
        transforms.Normalize(stats, use_quantiles=data_config.use_quantile_norm),
        *data_config.model_transforms.inputs,
    ])

    # Observation.from_dict is the training-side conversion in Pi05Agent, and
    # _jitted_pack_observation performs the equivalent inference-side uint8
    # conversion.  Keep this CPU implementation explicit and inspectable.
    def finalize_training(transformed):
        batched = {
            key: {
                subkey: np.asarray(value)[None]
                for subkey, value in subvalue.items()
            } if isinstance(subvalue, dict) else np.asarray(subvalue)[None]
            for key, subvalue in transformed.items()
        }
        observation = model_lib.Observation.from_dict(batched)
        return observation.to_dict()

    def finalize_inference(transformed):
        batched = {
            key: {
                subkey: np.asarray(value)[None]
                for subkey, value in subvalue.items()
            } if isinstance(subvalue, dict) else np.asarray(subvalue)[None]
            for key, subvalue in transformed.items()
        }
        batched["image"] = {
            key: value.astype(np.float32) / 255.0 * 2.0 - 1.0
            if value.dtype == np.uint8 else value
            for key, value in batched["image"].items()
        }
        # process_raw_inputs() returns the optional FAST fields with ``None``
        # via packed.get(), so retain those explicit None leaves as well.
        return {
            key: batched.get(key)
            for key in (
                "image", "image_mask", "state", "tokenized_prompt",
                "tokenized_prompt_mask", "token_ar_mask", "token_loss_mask",
            )
        }

    result: dict[str, Any] = {
        "script": str(Path(__file__).resolve()),
        "norm_stats": str(NORM_PATH),
        "norm_stats_sha256": hashlib.sha256(NORM_PATH.read_bytes()).hexdigest(),
        "model_type": base.model.model_type.value,
        "discrete_state_input": bool(base.model.discrete_state_input),
        "use_quantile_norm": bool(data_config.use_quantile_norm),
        "action_horizon": int(base.model.action_horizon),
        "model_action_dim": int(base.model.action_dim),
        "cases": [],
    }

    for name, path in RUNS.items():
        row = _sample_row(path)
        train_input = {
            key: value for key, value in row.items()
            if key in ("observation/image", "observation/wrist_image", "observation/state", "actions", "prompt")
        }
        infer_input = {
            "image": row["observation/image"],
            "wrist_image": row["observation/wrist_image"],
            "state": row["observation/state"],
            # Pi05Agent.process_raw_inputs adds this dummy 7D action before
            # its repack transform, because the shared transform group also
            # maps the action field even though inference later discards it.
            "actions": np.zeros((7,), dtype=np.float32),
            "prompt": row["prompt"],
        }
        train_transformed = training_transform(_copy(train_input))
        infer_transformed = inference_transform(_copy(infer_input))
        train_observation = finalize_training(train_transformed)
        infer_observation = finalize_inference(infer_transformed)
        observation_comparison = _compare_tree(
            {key: value for key, value in train_observation.items() if key != "actions"},
            infer_observation,
        )

        # Verify the exact normalized full chunk the learner receives and its
        # inverse back to raw teacher actions.  Padding is intentionally not
        # included in this comparison because the learner truncates to 7D
        # before flattening and then pads internally.
        normalized_action = np.asarray(train_transformed["actions"])[..., :7]
        unnormalized = transforms.Unnormalize(
            stats, use_quantiles=data_config.use_quantile_norm
        )({
            "actions": normalized_action,
            # Unnormalize is intentionally strict about the complete stats
            # tree, so include the state leaf while checking the action leaf.
            "state": np.asarray(train_transformed["state"])[..., :8],
        })["actions"]
        action_error = np.abs(np.asarray(unnormalized) - row["actions"])
        result["cases"].append({
            "name": name,
            "path": str(path),
            "episode": row["episode"],
            "row": row["row"],
            "raw": {
                key: _leaf_summary(row[key])
                for key in ("observation/image", "observation/wrist_image", "observation/state", "actions")
            },
            "training_transform_leaves": {
                key: _leaf_summary(value) for key, value in _leaves(train_transformed)
            },
            "inference_transform_leaves": {
                key: _leaf_summary(value) for key, value in _leaves(infer_transformed)
            },
            "observation_parity": observation_comparison,
            "action_roundtrip": {
                "max_abs": float(action_error.max()),
                "mean_abs": float(action_error.mean()),
                "finite": bool(np.isfinite(action_error).all()),
                "within_1e-5": bool(action_error.max() <= 1e-5),
            },
        })

    result["all_observation_parity_exact"] = all(
        case["observation_parity"]["exact"] for case in result["cases"]
    )
    result["all_observation_parity_within_1e-6"] = all(
        case["observation_parity"]["max_abs"] <= 1e-6 for case in result["cases"]
    )
    result["all_action_roundtrips_within_1e-5"] = all(
        case["action_roundtrip"]["within_1e-5"] for case in result["cases"]
    )
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "output": str(OUT),
        "all_observation_parity_exact": result["all_observation_parity_exact"],
        "all_observation_parity_within_1e-6": result["all_observation_parity_within_1e-6"],
        "all_action_roundtrips_within_1e-5": result["all_action_roundtrips_within_1e-5"],
        "cases": [
            {
                "name": case["name"],
                "max_observation_abs": case["observation_parity"]["max_abs"],
                "max_action_roundtrip_abs": case["action_roundtrip"]["max_abs"],
            }
            for case in result["cases"]
        ],
    }, indent=2))


if __name__ == "__main__":
    main()

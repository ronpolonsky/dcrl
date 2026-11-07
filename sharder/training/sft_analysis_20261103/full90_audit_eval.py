"""Run the existing LIBERO-90 evaluator with the audit SFT factory enabled.

This wrapper changes only checkpoint-factory dispatch for the isolated audit
checkpoints. Rollout, task selection, seeds, and actor-only evaluation remain
implemented by ``full_90_libero_evals_inference``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import full_90_libero_evals_inference as evaluator


def _audit_sft_checkpoint_config(checkpoint: Path, assets_dir: Path | None,
                                 replan_steps: int, image_size: int):
    root, step_hint = evaluator._resolve_full_root(checkpoint)
    manifest_path = root.parent / "sft_config.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"sft_pi requires {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("learner_config") not in (
        "configs.sft.my_expo_libero", "audit_sft_config"
    ):
        raise ValueError(
            "unsupported SFT learner_config: "
            f"{manifest.get('learner_config')!r}"
        )
    for key, expected in (("action_horizon", 10),
                          ("replan_steps", replan_steps),
                          ("image_size", image_size)):
        if manifest.get(key) != expected:
            raise ValueError(
                f"Saved {key}={manifest.get(key)} differs from evaluation {expected}"
            )
    assets = assets_dir or root.parent / "student_assets"
    if not (assets / "physical-intelligence/libero/norm_stats.json").is_file():
        raise FileNotFoundError(f"Missing SFT normalization statistics below {assets}")
    args = argparse.Namespace(**{**manifest, "resume": True, "num_sft_steps": 1})
    return root, step_hint, args, assets


def _audit_build_sft_agent(checkpoint, assets_dir, instruction, mesh,
                           data_sharding, replicated_sharding, replan_steps,
                           image_size):
    root, step_hint, args, assets = _audit_sft_checkpoint_config(
        checkpoint, assets_dir, replan_steps, image_size
    )
    if args.learner_config == "audit_sft_config":
        import audit_sft_config as sft_config
    else:
        from configs.sft import my_expo_libero as sft_config
    from dcrl.agents.my_expo_learner import restore_checkpoint
    from dcrl.expo.agents.alg.agent import initialize_checkpoint_dir

    manager, _ = initialize_checkpoint_dir(
        root, keep_period=None, overwrite=False, resume=True
    )
    try:
        steps = tuple(int(step) for step in manager.all_steps())
        if not steps:
            raise FileNotFoundError(f"No SFT learner checkpoints below {root}")
        step = step_hint if step_hint is not None else max(steps)
        if step not in steps:
            raise ValueError(f"Checkpoint step {step} is not present below {root}")
        agent, _, _ = sft_config.build_learner(
            args=args, output_dir=root.parent, instruction=instruction,
            stats_dir=assets / "physical-intelligence/libero", assets_dir=assets,
            restore_only=True, mesh=mesh, data_sharding=data_sharding,
            replicated_sharding=replicated_sharding,
        )
        agent = restore_checkpoint(manager, agent, step)
        evaluator.LOGGER.info(
            "Restored audit SFT learner step %d from %s", step, root
        )
        return agent, manager
    except Exception:
        manager.close()
        raise


evaluator._sft_checkpoint_config = _audit_sft_checkpoint_config
evaluator._build_sft_agent = _audit_build_sft_agent


if __name__ == "__main__":
    raise SystemExit(evaluator.main())

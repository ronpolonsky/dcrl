"""Isolated balanced multi-task wrapper around the audit SFT trainer.

It reuses the audited learner/update/evaluation implementation but injects a
prompt-preserving, balanced multi-task dataset.  No production trainer is
modified.  Collection paths are supplied through SFT_MULTITASK_SUCCESS_PATHS.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path


AUDIT_DIR = Path(__file__).resolve().parent
ACTION_DIR = AUDIT_DIR.parent / "sft_actionbounds_20261103"
PROJECT = Path("/projects/bgqe/rpolonsky/dcrl")
def main():
    if str(AUDIT_DIR) not in sys.path:
        sys.path.insert(0, str(AUDIT_DIR))
    if str(ACTION_DIR) not in sys.path:
        sys.path.insert(0, str(ACTION_DIR))
    if str(PROJECT) not in sys.path:
        sys.path.insert(0, str(PROJECT))
    data_module = importlib.import_module("multitask_scope_data")
    sys.modules["scope_data"] = data_module
    trainer = importlib.import_module("train_sft_retention_actionbounds")

    args = trainer.parse_args()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.collection_source = "small_expo"
    args.official_demo_ratio = 0.0
    args.multitask_success_paths = [str(path) for path in data_module.configured_paths()]

    # The collection metadata records the original frozen Small EXPO teacher,
    # which is distinct from the directory holding the SFT HDF5 partition.
    # Use that authoritative identity for the reused trainer's validation.
    manifest = data_module.audit_paths([Path(path) for path in args.multitask_success_paths])
    first_metadata = manifest["tasks"][0]["metadata"]
    args.teacher_dir = Path(first_metadata["teacher_dir"]).resolve()
    args.teacher_step = int(first_metadata["checkpoint_step"])
    args.source_task_id = int(manifest["tasks"][0]["task_id"])

    # The reused trainer checks for the canonical partition before constructing
    # the injected dataset.  A symlink records the first audited collection
    # without copying or rewriting any HDF5 data.
    success_link = args.output_dir / "pi_base_data/success/teacher_chunks.hdf5"
    success_link.parent.mkdir(parents=True, exist_ok=True)
    first_path = Path(args.multitask_success_paths[0])
    if not success_link.exists():
        success_link.symlink_to(first_path)

    manifest["target_mode"] = args.target_mode
    (args.output_dir / "multitask_dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    wandb_run = trainer._start_wandb(args)
    try:
        trainer.train(args, wandb_run)
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()

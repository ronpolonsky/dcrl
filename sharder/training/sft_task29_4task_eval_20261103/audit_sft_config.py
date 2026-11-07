"""Evaluation-only adapter for the isolated task-29 SFT checkpoints.

The training manifest records ``zero_lora_b=true`` because the training run
zeroed LoRA-B at initialization.  During restore, the checkpoint already
contains the resulting parameters, so repeating that mutation is unnecessary
and trips a strict jaxtyping check on the placeholder TrainState step.  Keep
the original audit factory unchanged and suppress only that redundant restore
time mutation.
"""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path


_ORIGINAL = Path(
    "/projects/bgqe/rpolonsky/dcrl/sharder/training/sft_analysis_20261103/audit_sft_config.py"
)
_SPEC = importlib.util.spec_from_file_location("_audit_sft_config_original", _ORIGINAL)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Could not load original audit config from {_ORIGINAL}")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def build_learner(*, args, restore_only: bool = False, **kwargs):
    if restore_only and getattr(args, "zero_lora_b", False):
        args = copy.copy(args)
        args.zero_lora_b = False
    return _MODULE.build_learner(
        args=args,
        restore_only=restore_only,
        **kwargs,
    )

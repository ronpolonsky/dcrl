#!/usr/bin/env python3
"""Validate and summarize actor-only LIBERO-90 task-result directories.

The evaluator may write task files directly in one directory or below shard
subdirectories.  This utility is intentionally read-only: it refuses to infer
a full-suite score from incomplete, duplicated, malformed, or protocol-
mismatched task records.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


EXPECTED_TASKS = set(range(90))
EXPECTED = {
    "suite": "libero_90",
    "episodes": 20,
    "max_steps": 400,
    "eval_seed": 7,
    "action_horizon": 10,
    "replan_steps": 5,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    files = sorted(args.directory.rglob("task_*.json"))
    records: dict[int, tuple[Path, dict]] = {}
    errors: list[str] = []
    for path in files:
        try:
            record = json.loads(path.read_text())
        except Exception as exc:  # noqa: BLE001 - report malformed artifacts
            errors.append(f"{path}: invalid JSON ({exc})")
            continue
        task_id = record.get("task_id")
        if not isinstance(task_id, int) or task_id not in EXPECTED_TASKS:
            errors.append(f"{path}: invalid task_id={task_id!r}")
            continue
        if task_id in records:
            errors.append(
                f"duplicate task_id={task_id}: {records[task_id][0]} and {path}"
            )
            continue
        for key, expected in EXPECTED.items():
            if record.get(key) != expected:
                errors.append(
                    f"{path}: {key}={record.get(key)!r}, expected {expected!r}"
                )
        successes = record.get("successes")
        if not isinstance(successes, int) or not 0 <= successes <= 20:
            errors.append(f"{path}: invalid successes={successes!r}")
        trials = record.get("trials")
        if not isinstance(trials, list) or len(trials) != 20:
            errors.append(f"{path}: expected 20 trials, got {len(trials) if isinstance(trials, list) else trials!r}")
        records[task_id] = (path, record)

    missing = sorted(EXPECTED_TASKS - records.keys())
    total_successes = sum(record.get("successes", 0) for _, record in records.values())
    complete = not missing and not errors and len(records) == 90
    print(f"directory={args.directory}")
    print(f"files={len(files)} unique_tasks={len(records)}")
    print(f"missing_tasks={','.join(map(str, missing)) if missing else '-'}")
    print(f"errors={len(errors)}")
    print(f"status={'COMPLETE' if complete else 'INCOMPLETE/INVALID'}")
    if complete:
        print(f"successes={total_successes}/1800")
        print(f"success_rate={total_successes / 18:.2f}%")
    for error in errors[:20]:
        print(f"ERROR: {error}")
    if len(errors) > 20:
        print(f"ERROR: ... {len(errors) - 20} more")
    return 0 if complete else 2


if __name__ == "__main__":
    raise SystemExit(main())

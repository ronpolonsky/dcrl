#!/usr/bin/env python3
"""Merge per-task LIBERO-90 evaluator JSON files into one summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    files = sorted(args.results_dir.glob("task_[0-8][0-9].json"))
    if not files:
        raise SystemExit(f"No task_00.json .. task_89.json files under {args.results_dir}")
    results = []
    failures = []
    for path in files:
        item = json.loads(path.read_text())
        if "eval/success_rate" not in item:
            failures.append({"path": str(path), **item})
        else:
            results.append(item)
    ids = sorted(int(item["task_id"]) for item in results)
    expected = set(range(90))
    missing = sorted(expected - set(ids))
    duplicate = sorted({task_id for task_id in ids if ids.count(task_id) > 1})
    summary = {
        "suite": "libero_90",
        "results_dir": str(args.results_dir.resolve()),
        "completed_tasks": len(results),
        "missing_tasks": missing,
        "duplicate_tasks": duplicate,
        "failed_tasks": failures,
        "mean_task_success_rate": (
            float(np.mean([item["eval/success_rate"] for item in results]))
            if results else None
        ),
        "overall_episode_success_rate": (
            float(sum(int(item["successes"]) for item in results) /
                  sum(int(item["episodes"]) for item in results))
            if results else None
        ),
        "results": [
            {
                "task_id": int(item["task_id"]),
                "instruction": item.get("instruction"),
                "successes": int(item["successes"]),
                "episodes": int(item["episodes"]),
                "eval/success_rate": float(item["eval/success_rate"]),
                "eval/mean_steps": float(item["eval/mean_steps"]),
            }
            for item in sorted(results, key=lambda value: int(value["task_id"]))
        ],
    }
    output = args.output or args.results_dir / "summary.json"
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if missing or duplicate or failures:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Audit and merge one LIBERO-90 checkpoint across multiple result roots.

Result roots are ordered by priority.  When the same completed task exists in
more than one root, the first root wins and the task is included exactly once.
This lets the sharded result tree remain canonical while older serial jobs can
still contribute tasks that the sharded jobs have not completed.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


EXPECTED_TASKS = set(range(90))


def _is_valid(item: dict[str, Any], *, episodes: int) -> tuple[bool, str | None]:
    if item.get("suite") != "libero_90":
        return False, f"suite={item.get('suite')!r}"
    task_id = item.get("task_id")
    if not isinstance(task_id, int) or task_id not in EXPECTED_TASKS:
        return False, f"task_id={task_id!r}"
    if item.get("episodes") != episodes:
        return False, f"episodes={item.get('episodes')!r}"
    trials = item.get("trials")
    if not isinstance(trials, list) or len(trials) != episodes:
        return False, f"trials={None if not isinstance(trials, list) else len(trials)}"
    indices = [trial.get("initial_state_index") for trial in trials]
    if indices != list(range(episodes)):
        return False, f"initial_state_indices={indices!r}"
    successes = item.get("successes")
    if not isinstance(successes, int) or not 0 <= successes <= episodes:
        return False, f"successes={successes!r}"
    if "eval/success_rate" not in item:
        return False, "missing eval/success_rate"
    if item.get("action_horizon") != 10 or item.get("replan_steps") != 5:
        return False, (
            f"action_horizon/replan_steps="
            f"{item.get('action_horizon')!r}/{item.get('replan_steps')!r}"
        )
    return True, None


def _result_files(root: Path, mode: str, name: str) -> list[Path]:
    directory = root / mode / name
    if not directory.is_dir():
        return []
    return sorted(directory.glob("**/task_[0-8][0-9].json"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", required=True, type=Path,
                        help="Result root, in descending canonical priority")
    parser.add_argument("--mode", choices=("actor", "compact", "full"), required=True)
    parser.add_argument("--name", required=True,
                        help="Checkpoint result directory name below MODE")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()

    candidates: dict[int, list[dict[str, Any]]] = defaultdict(list)
    invalid: list[dict[str, str]] = []
    physical_files = 0
    for priority, root in enumerate(args.root):
        for path in _result_files(root, args.mode, args.name):
            physical_files += 1
            try:
                item = json.loads(path.read_text())
            except Exception as error:
                invalid.append({"path": str(path), "reason": repr(error)})
                continue
            valid, reason = _is_valid(item, episodes=args.episodes)
            if not valid:
                invalid.append({"path": str(path), "reason": str(reason)})
                continue
            candidates[int(item["task_id"])].append(
                {"priority": priority, "path": path, "item": item}
            )

    selected: dict[int, dict[str, Any]] = {}
    duplicate_tasks: dict[int, list[str]] = {}
    conflicting_tasks: dict[int, dict[str, Any]] = {}
    for task_id, values in sorted(candidates.items()):
        values.sort(key=lambda value: (value["priority"], str(value["path"])))
        selected[task_id] = values[0]
        if len(values) > 1:
            duplicate_tasks[task_id] = [str(value["path"]) for value in values]
            outcomes = {
                (
                    int(value["item"]["successes"]),
                    float(value["item"]["eval/success_rate"]),
                )
                for value in values
            }
            if len(outcomes) > 1:
                conflicting_tasks[task_id] = {
                    str(value["path"]): {
                        "successes": int(value["item"]["successes"]),
                        "success_rate": float(value["item"]["eval/success_rate"]),
                    }
                    for value in values
                }

    missing = sorted(EXPECTED_TASKS - set(selected))
    results = []
    for task_id, value in sorted(selected.items()):
        item = value["item"]
        results.append({
            "task_id": task_id,
            "instruction": item.get("instruction"),
            "successes": int(item["successes"]),
            "episodes": int(item["episodes"]),
            "eval/success_rate": float(item["eval/success_rate"]),
            "eval/mean_steps": float(item["eval/mean_steps"]),
            "source": str(value["path"]),
        })

    total_episodes = sum(result["episodes"] for result in results)
    total_successes = sum(result["successes"] for result in results)
    summary = {
        "suite": "libero_90",
        "mode": args.mode,
        "name": args.name,
        "roots_in_priority_order": [str(root.resolve()) for root in args.root],
        "episodes_per_task": args.episodes,
        "physical_result_files": physical_files,
        "valid_unique_tasks": len(results),
        "missing_tasks": missing,
        "redundant_valid_files": sum(len(values) - 1 for values in candidates.values()),
        "duplicate_tasks": {str(key): value for key, value in duplicate_tasks.items()},
        "conflicting_duplicate_tasks": {
            str(key): value for key, value in conflicting_tasks.items()
        },
        "invalid_files": invalid,
        "mean_task_success_rate": (
            sum(result["eval/success_rate"] for result in results) / len(results)
            if results else None
        ),
        "overall_episode_success_rate": (
            total_successes / total_episodes if total_episodes else None
        ),
        "results": results,
    }

    rendered = json.dumps(summary, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(rendered)
        temporary.replace(args.output)
    print(rendered, end="")

    if invalid:
        return 2
    if args.require_complete and missing:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Summarize controlled-scope retention evals with paired initial-state deltas."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("/work/hdd/bgqe/rpolonsky/dcrlcheckpoints/sft_retention_scope_20261103")
OUT = Path(__file__).resolve().parent / "scope_curve_summary.json"
ARMS = ("existing", "lora_only")


def _trials(record: dict, task: str) -> dict[int, bool]:
    return {
        int(trial["initial_state_index"]): bool(trial["success"])
        for trial in record["tasks"][task]["trials"]
    }


def _paired(before: dict[int, bool], after: dict[int, bool]) -> dict[str, int]:
    common = sorted(set(before) & set(after))
    return {
        "common_cases": len(common),
        "retained_successes": sum(before[i] and after[i] for i in common),
        "lost_successes": sum(before[i] and not after[i] for i in common),
        "gained_successes": sum(not before[i] and after[i] for i in common),
        "still_failed": sum(not before[i] and not after[i] for i in common),
    }


def main() -> None:
    summary = {"root": str(ROOT), "arms": {}}
    for arm in ARMS:
        eval_dir = ROOT / f"task_28_{arm}" / "evals"
        records = []
        for path in sorted(eval_dir.glob("*.json")):
            record = json.loads(path.read_text())
            record["_path"] = str(path)
            records.append(record)
        if not records:
            summary["arms"][arm] = {"status": "no_evals"}
            continue
        baseline = next((record for record in records if int(record["step"]) == 0), None)
        arm_summary = {"status": "ok", "evals": []}
        baseline_trials = {
            task: _trials(baseline, task)
            for task in baseline["tasks"]
        } if baseline else None
        for record in records:
            row = {
                "step": int(record["step"]),
                "path": record["_path"],
                "macro_success_rate": record.get("eval/macro_success_rate"),
                "tasks": {
                    task: {
                        "success_rate": data.get("eval/success_rate"),
                        "mean_steps": data.get("eval/mean_steps"),
                    }
                    for task, data in record["tasks"].items()
                },
            }
            if baseline_trials is not None and int(record["step"]) != 0:
                row["paired_to_step0"] = {
                    task: _paired(baseline_trials[task], _trials(record, task))
                    for task in record["tasks"]
                    if task in baseline_trials
                }
            arm_summary["evals"].append(row)
        summary["arms"][arm] = arm_summary
    OUT.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

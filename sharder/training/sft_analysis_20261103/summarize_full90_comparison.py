#!/usr/bin/env python3
"""Validate and summarize matched LIBERO-90 evaluations for Pi0.5, SFT and EXPO."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path("/work/hdd/bgqe/rpolonsky")
BASE = Path("/projects/bgqe/rpolonsky/dcrl/results/official_pi0.5_libero_chkpt")
SFT = {
    "sft_task02_20k_actor": ROOT / "dcrlcheckpoints/sft_full90_actor_only_eval_20261103/task_02_step_20000",
    "sft_task28_40k_actor": ROOT / "dcrlcheckpoints/sft_full90_actor_only_eval_20261103/task_28_step_40000",
    "sft_task29_20k_actor": ROOT / "dcrlcheckpoints/sft_full90_actor_only_eval_20261103/task_29_step_20000",
}
EXPO = {
    "expo_task02_60k_full_system": ROOT / "dcrlcheckpoints/pi_robo_libero_unsharded_h200_b16_200k/full90_eval_direct_rl_pi_20261102/task_2",
    "expo_task28_40k_full_system": ROOT / "dcrlcheckpoints/pi_robo_libero_unsharded_h200_b16_200k/full90_eval_direct_rl_pi_20261102/task_28",
    "expo_task29_60k_full_system": ROOT / "dcrlcheckpoints/pi_robo_libero_unsharded_h200_b16_200k/full90_eval_direct_rl_pi_20261102/task_29",
    "expo_task60_60k_full_system": ROOT / "dcrlcheckpoints/pi_robo_libero_unsharded_h200_b16_200k/full90_eval_direct_rl_pi_20261102/task_60",
    "expo_task79_60k_full_system": ROOT / "dcrlcheckpoints/pi_robo_libero_unsharded_h200_b16_200k/full90_eval_direct_rl_pi_20261102/task_79",
}
OUT = ROOT / "sharder/training/sft_analysis_20261103"


def load_tree(path: Path) -> dict[int, dict]:
    records = {}
    for file in path.rglob("task_*.json"):
        relative_parent = file.parent.relative_to(path)
        # Ignore task-level smoke probes and accept only suite-root files or
        # explicitly named evaluation shards.
        if any("smoke" in part.lower() for part in relative_parent.parts):
            continue
        if relative_parent.parts and not all(part.startswith("shard_") for part in relative_parent.parts):
            continue
        try:
            row = json.loads(file.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if "task_id" in row and "trials" in row:
            task = int(row["task_id"])
            if task in records:
                raise ValueError(f"duplicate result for task {task} under {path}")
            records[task] = row
    return records


def validate(name: str, records: dict[int, dict], baseline: dict[int, dict]) -> None:
    expected = set(range(90))
    if set(records) != expected:
        raise ValueError(f"{name}: expected all 90 tasks; missing={sorted(expected-set(records))}, extra={sorted(set(records)-expected)}")
    for task in range(90):
        row, base = records[task], baseline[task]
        if row.get("episodes") != 20 or len(row["trials"]) != 20:
            raise ValueError(f"{name} task {task}: expected 20 episode results")
        if row.get("eval_seed") != 7 or base.get("eval_seed") != 7:
            raise ValueError(f"{name} task {task}: unexpected eval seed")
        if [x.get("initial_state_index") for x in row["trials"]] != [x.get("initial_state_index") for x in base["trials"]]:
            raise ValueError(f"{name} task {task}: initial-state indices differ from baseline")
        if row.get("instruction") != base.get("instruction"):
            raise ValueError(f"{name} task {task}: instruction differs from baseline")


def summarize(name: str, records: dict[int, dict], baseline: dict[int, dict]) -> dict:
    paired = {"both": 0, "base_only": 0, "model_only": 0, "neither": 0}
    for task in range(90):
        for model, base in zip(records[task]["trials"], baseline[task]["trials"]):
            m, b = bool(model["success"]), bool(base["success"])
            paired["both" if m and b else "base_only" if b else "model_only" if m else "neither"] += 1
    successes = sum(int(records[t]["successes"]) for t in range(90))
    return {
        "name": name,
        "tasks": 90,
        "episodes": 1800,
        "successes": successes,
        "success_rate": successes / 1800,
        "paired_vs_official_base": paired,
        "target_task_counts": {
            str(t): {"successes": records[t]["successes"], "episodes": records[t]["episodes"], "rate": records[t]["eval/success_rate"]}
            for t in (2, 28, 29, 60, 79)
        },
    }


def main() -> None:
    base = load_tree(BASE)
    if set(base) != set(range(90)):
        raise ValueError(f"baseline incomplete: found {len(base)} tasks")
    models = {}
    for name, path in {**SFT, **EXPO}.items():
        models[name] = load_tree(path)
        validate(name, models[name], base)

    columns = ["task_id", "instruction", "official_successes", "official_rate"]
    for name in models:
        columns += [f"{name}_successes", f"{name}_rate"]
    with (OUT / "full90_comparison_20261103.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for task in range(90):
            row = {"task_id": task, "instruction": base[task]["instruction"],
                   "official_successes": base[task]["successes"],
                   "official_rate": base[task]["eval/success_rate"]}
            for name, records in models.items():
                row[f"{name}_successes"] = records[task]["successes"]
                row[f"{name}_rate"] = records[task]["eval/success_rate"]
            writer.writerow(row)

    summaries = [summarize("official_pi05_actor", base, base)]
    summaries.extend(summarize(name, records, base) for name, records in models.items())
    payload = {
        "protocol": {"suite": "libero_90", "episodes_per_task": 20, "total_episodes": 1800,
                     "eval_seed": 7, "max_steps": 400, "action_horizon": 10,
                     "replan_steps": 5, "paired_initial_state_indices": True},
        "interpretation": {
            "sft": "actor-only policy evaluations",
            "expo": "full EXPO deployed-system evaluations with editor/critic candidate selection; not actor-only causal comparisons",
        },
        "results": summaries,
    }
    (OUT / "full90_comparison_20261103.json").write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "# Matched LIBERO-90 SFT and Full EXPO comparison",
        "",
        "All entries use 20 episodes per task (1,800 total), evaluation seed 7, 400 control steps, action horizon 10 and replan interval 5. The validation requires all 90 tasks and identical task instructions and initial-state indices versus official Pi0.5.",
        "",
        "SFT evaluations run the actor only. Full EXPO evaluations include its editor/critic candidate selection, so those are deployed-system comparisons rather than actor-only causal comparisons.",
        "",
        "| Policy/checkpoint | Successes | Rate | Paired both | Base-only | Policy-only | Neither | Target task rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summaries:
        target = "—"
        if item["name"] != "official_pi05_actor":
            run_task = int(item["name"].split("_task")[1].split("_")[0])
            target = f"{item['target_task_counts'][str(run_task)]['successes']}/20 ({item['target_task_counts'][str(run_task)]['rate']*100:.1f}%)"
        p = item["paired_vs_official_base"]
        lines.append(f"| {item['name']} | {item['successes']}/1800 | {item['success_rate']*100:.2f}% | {p['both']} | {p['base_only']} | {p['model_only']} | {p['neither']} | {target} |")
    lines += ["", "The 90-task, per-task data are in `full90_comparison_20261103.csv`; validated source totals and paired outcome counts are in `full90_comparison_20261103.json`.", ""]
    (OUT / "full90_comparison_20261103.md").write_text("\n".join(lines))
    print("Wrote full90_comparison_20261103.{csv,json,md}")
    for item in summaries:
        print(item["name"], item["successes"], f"{item['success_rate']*100:.2f}%", item["paired_vs_official_base"])


if __name__ == "__main__":
    main()

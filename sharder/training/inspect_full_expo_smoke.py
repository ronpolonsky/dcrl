"""Read-only checks for the isolated full-EXPO smoke outputs."""

import argparse
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--max-steps", type=int, default=4500)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    rows = []
    # A live writer may be in the middle of the last JSON line.
    for line in (args.run_dir / "metrics.jsonl").read_text().splitlines(keepends=True):
        if line.endswith("\n"):
            rows.append(json.loads(line))
    updates = [r for r in rows if "training/actor_loss" in r]
    keys = (
        "training/actor_loss", "training/actor_grad_norm",
        "training/critic_loss", "training/critic_grad_norm",
        "training/edit_actor_loss", "training/temperature_loss",
        "training/temperature", "training/next_q_nan_ratio",
    )
    finite = bool(updates) and all(
        k in r and math.isfinite(r[k]) for r in updates for k in keys
    )
    no_q_nans = bool(updates) and all(r["training/next_q_nan_ratio"] == 0 for r in updates)
    evals = [json.loads(p.read_text()) for p in sorted((args.run_dir / "evals").glob("*.json"))]
    after_learning = [r for r in evals if updates and r["step"] >= updates[0]["step"]]
    latest = max((r["step"] for r in rows), default=-1)
    result = {
        "last_step": latest,
        "expected_final_step": args.max_steps,
        "completed_loop": latest >= args.max_steps,
        "completed_update_cycles": len(updates),
        "finite_update_metrics": finite,
        "zero_q_nan_ratio": no_q_nans,
        "last_actor_state_step": updates[-1].get("training/actor_state_step") if updates else None,
        "post_update_evaluations": [{"step": r["step"], "success_rate": r.get("eval/success_rate")} for r in after_learning],
    }
    print(json.dumps(result, sort_keys=True))
    if not finite or not no_q_nans:
        raise SystemExit(1)
    if args.require_complete and not (result["completed_loop"] and after_learning):
        raise SystemExit(2)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import json
import subprocess
import time
from pathlib import Path

ROOT = Path("/work/hdd/bgqe/rpolonsky/dcrlcheckpoints/pi_robo_libero_unsharded_h200_b16_200k")
LOG_PATH = Path("/projects/bgqe/rpolonsky/dcrl/sharder/training/logs/stop_full_expo_after_60k.log")
TARGET = 60000
PAIR_TOLERANCE = 1000
POLL_SECONDS = 10
JOBS = {
    "29": ("23324737_5", "pi05-unsharded-b16-task29-h200"),
    "60": ("23324737_3", "pi05-unsharded-b16-task60-h200"),
    "79": ("23324737_4", "pi05-unsharded-b16-task79-h200"),
}

def say(message):
    stamp = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    line = f"{stamp} {message}"
    with LOG_PATH.open("a") as f:
        f.write(line + chr(10))
        f.flush()
    print(line, flush=True)

def evals_for(task):
    d = ROOT / f"pi_robo_libero_unsharded_h200_b16_task{task}" / "evals"
    rows = []
    for p in d.glob("*.json"):
        try:
            x = json.loads(p.read_text())
            rows.append((int(x["step"]), float(x["eval/success_rate"])))
        except Exception:
            pass
    return rows

def complete_checkpoints_for(task):
    d = ROOT / f"pi_robo_libero_unsharded_h200_b16_task{task}" / "checkpoints"
    rows = []
    for p in d.iterdir():
        if p.is_dir() and p.name.isdigit() and (p / "_CHECKPOINT_METADATA").is_file():
            rows.append(int(p.name))
    return rows

def matching_60k_pair(task):
    evals = [(s, r) for s, r in evals_for(task) if s >= TARGET]
    cps = [s for s in complete_checkpoints_for(task) if s >= TARGET]
    pairs = [(max(es, cs), abs(es - cs), es, cs, rate)
             for es, rate in evals for cs in cps
             if abs(es - cs) <= PAIR_TOLERANCE]
    return min(pairs, default=None, key=lambda x: (x[0], x[1]))

def job_state(job):
    r = subprocess.run(["squeue", "-h", "-j", job, "-o", "%T"], capture_output=True, text=True)
    return r.stdout.strip().splitlines()[0] if r.stdout.strip() else "NOT_IN_QUEUE"

def main():
    done = set()
    say(f"Watcher active: stop each task only when a finalized checkpoint and eval are both >= {TARGET}, within {PAIR_TOLERANCE} steps.")
    while len(done) < len(JOBS):
        for task, (job, session) in JOBS.items():
            if task in done:
                continue
            state = job_state(job)
            pair = matching_60k_pair(task)
            if pair is not None:
                _, diff, eval_step, cp_step, rate = pair
                if state == "RUNNING":
                    subprocess.run(["scancel", job], check=True)
                    say(f"Task {task}: confirmed eval step {eval_step} ({rate:.0%}) and complete checkpoint {cp_step} (delta {diff}); canceled only {job}.")
                else:
                    say(f"Task {task}: confirmed eval step {eval_step} ({rate:.0%}) and complete checkpoint {cp_step}; allocation state={state}, no cancel needed.")
                subprocess.run(["tmux", "kill-session", "-t", session], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                done.add(task)
            elif state not in ("RUNNING", "PENDING"):
                say(f"Task {task}: allocation {job} is {state} before a matching 60k checkpoint/eval pair; no cancellation issued.")
                done.add(task)
            else:
                eval_steps = [s for s, _ in evals_for(task)]
                cp_steps = complete_checkpoints_for(task)
                say(f"Task {task}: state={state}, latest_eval={max(eval_steps, default=0)}, latest_complete_checkpoint={max(cp_steps, default=0)}; awaiting matching pair.")
        if len(done) < len(JOBS):
            time.sleep(POLL_SECONDS)
    say("All watched tasks reached the checkpoint/eval condition or ended; watcher exiting.")

if __name__ == "__main__":
    main()

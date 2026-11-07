# Task-specific SFT diagnosis run plan — 2026-11-04

## Purpose

Test whether the remaining task-specific SFT factors can improve retention on
LIBERO-90 while preserving the task-specific data rule. The mixed four-task
SFT arms are excluded from this diagnosis.

## Shared invariants

- Training source: the corresponding task's Small EXPO successful collection
  only; never mix target chunks from another task.
- Teacher/checkpoint and collection are task-specific: Task 28 uses teacher
  step 275001 and 760 chunks; Task 29 uses teacher step 375380 and 2,150
  chunks; Task 60 uses teacher step 450223 and 2,226 chunks. Task 2 uses its
  own teacher collection and remains a side-panel run here rather than a
  selected diagnosis result.
- Student: official `pi05_libero` checkpoint.
- Trainable scope: LoRA-only gradients with zero LoRA-B initialization.
- Official-demo ratio: 0.
- Batch size: 16; learning rate: 5e-5; warmup: 10,000 updates.
- Action interface: horizon 10, execute/replan 5; image size 224.
- Seed 42; evaluation seed 7; development tasks 2, 28, 29 and 60.
- 500 optimizer updates, evaluation and learner checkpoint every 100 updates.

## Arms

1. `full_proposal`: existing ten-action teacher proposal targets.
2. `executed_backfill`: same Task 28 rows, with the proposal tail replaced by
   the subsequent executed prefix and terminal padding.
3. `outer_aug`: full proposal targets with the outer EXPO image augmentation
   applied before the Pi0.5 train step.

The first two arms isolate target construction. The third isolates the
augmentation path identified as missing from the SFT `bc_update` path.

For the later Task-29 and Task-60 full-proposal arms, the same arm name means
the same target construction but with that task's own Small EXPO collection;
the task-28 collection is not reused. Runtime manifests must show the matching
`source_task_id`, `small_expo_chunks`, and zero official-demo chunks before a
result is admitted to the task-specific comparison.

## GPU and session assignments

- `full_proposal`: allocation `23416001_2`, `gpua082`, tmux
  `sft_diag_target_full_task28` (historical; stopped after update 300).
- `executed_backfill`: allocation `23416001_4`, `gpua077`, tmux
  `sft_diag_target_exec_task28`.
- `outer_aug`: allocation `23433381_1`, `gpua087`, tmux
  `sft_diag_outer_aug_task28` (historical; stopped after update 300).

The Task 2 full-suite side evaluation on `gpua070` and the Small EXPO teacher
full-suite evaluation on `gpua044` remain untouched.

Current scheduling note: the Task-28 full-proposal and outer-augmentation
trainers were stopped after their complete update-300 artifacts were written
so their two allocations could be used for the two-shard Task-60 update-200
actor-only LIBERO-90 evaluation. Their artifacts remain valid and unchanged.
The Task-29 full-proposal update-200 full-suite evaluator remains active on
`gpua077`; the Task-2 side evaluator remains out of scope.

## Evidence gates

Before using a run as evidence, verify `small_expo_chunks=760`,
`official_demo_chunks=0`, the initialization audit, step-0 retention, every
100-update checkpoint/evaluation, and the final paired four-task curve. A
candidate must not be selected from the four-task macro alone: source-task
retention is an explicit gate, and held-out-task performance is the ranking
criterion. Target-task success or training loss alone is not a selection
criterion.

## Protocol revision — 2026-11-05

Routine LIBERO-90 evaluation is removed from the inner-loop diagnosis because
it is too slow for screening. Existing full-90 artifacts remain preserved, but
new candidates are screened using the four-task retention panel
`{2, 28, 29, 60}` (or an analogous source-plus-three-held-out panel when a
different source task is selected). The source task is reported separately;
candidates that materially regress on it are rejected before ranking by
held-out macro success and margin over matched Full EXPO. Full LIBERO-90 is
reserved for final confirmation of the best Pareto candidates.

The first revised arm is same-task-only full-proposal SFT with `zero_lora_b`
disabled for source Tasks 28 and 60. It preserves the other invariants above
and runs in separate tmux sessions on idle gpua002/gpua087. The prior Task-60
outer-augmentation full-90 evaluator sessions were stopped after preserving
their partial outputs; training and production allocations were not changed.

## Closure and next method (2026-11-05)

The no-`zero_lora_b` and warmup-100 sessions were stopped before any committed
training evaluation/checkpoint. They are retained only as failed/incomplete
diagnostic artifacts. Do not relaunch them merely to complete the factorial:
the existing validated full-90 and four-task curves already establish the main
failure mode.

The diagnosis is now a constrained distillation problem, not a search over
full-90 checkpoints. The next candidate must use the true `lora_only` mask,
zero-B initialization, the source task's Small EXPO targets, and early
retention evaluations. A candidate is admissible only if source-task success
does not materially regress from the official-base/step-0 control; among
admissible candidates, rank the three held-out tasks and the margin over
matched deployed Full EXPO. Same-task official demonstrations may be added as
a small controlled mixture only when needed for source retention; other-task
data is forbidden for this diagnosis. Full LIBERO-90 is reserved for the final
shortlist.

The existing strongest practical recipe is task-specific `full_proposal` SFT
at roughly 200--500 updates: Task 29 reaches 29.06--29.11% on full LIBERO-90
with 75% source retention, and Task 2 reaches 28.67% with 70% source
retention. Task 28 and Task 60 have larger aggregate margins in some cases but
only 60%/55% source retention, so they are not acceptable under the revised
constraint.

## Same-task demo screen (2026-11-05)

The next controlled factor is a 25% same-task official-demonstration fraction,
with the remaining 75% from the same task's Small EXPO targets. This is not a
cross-task mixture: Task 2/28/29 each use only its own teacher collection and
its own official LIBERO demonstration file. The screen uses true `lora_only`,
zero-B initialization, full-proposal targets, 500 updates, and a retention
evaluation every 100 updates. It writes no learner checkpoints because one
learner checkpoint consumes about 12--13 GiB and the current allocation quota
has only about 40 GiB of headroom; the screen persists all eval JSONs, metrics,
manifests and W&B logs. A surviving Pareto candidate must be rerun with one
checkpoint before any full-90 confirmation.

The Task-60 arm failed before training on gpua051's known CUDA visibility
fault. Do not retry that arm on gpua051; use a newly allocated/verified GPU or
the gpua038 slot only after its separate full-90 evaluator finishes.

## Next controlled factor after demo25 (prepared, not yet launched)

The 25% same-task official-data screen is still running through its 100-update
retention panels. If its complete curve confirms the preliminary negative
result, the next isolated factor is `official_demo_ratio=0.0625` (exactly one
official row in a batch of 16), with all other settings unchanged: the same
task-pure Small EXPO collection, official `pi05_libero` initialization,
true `lora_only`/zero-B scope, full-proposal 10/5 targets, batch 16,
`5e-5` peak LR, 10k warmup, 500 updates, and no checkpoints. The launcher is
`run_same_task_demo625_screen_20261106.sh`; it will only be started on a
verified free diagnostic allocation after the current arms finish. This
isolates mixture strength instead of changing architecture, target semantics,
or the comparison protocol.

## Priority revision after the Task-28 update-400 screen

The Task-28 demo25 arm produced the first source-valid screening Pareto point
at update 400: source Task 28 rose from 50% to 65%, while held-out macro rose
from 63.33% to 65.00%. The next healthy diagnostic slot should therefore
reproduce this exact task-pure recipe with checkpointing and a larger paired
retention panel before launching the prepared 6.25% mixture arm. The
checkpointed reproduction must use a fresh output directory, retain only the
selected learner state, and remain isolated from production runs. The 6.25%
arm is deferred rather than discarded; it becomes the next factor only if the
Task-28 point fails the larger paired evaluation or full-90 confirmation.

## Checkpointed shortlist validation (2026-11-05)

After the demo25 screen completed, three task-pure shortlist validations were
started on the freed diagnostic A100 slots. Each uses official `pi05_libero`
initialization, true `lora_only`/zero-B scope, full-proposal 10/5 targets,
batch 16, LR `5e-5`, and 50 episodes per task (the maximum valid LIBERO
initial-state panel). Full learner checkpoints are written only to the
node-local `/tmp/usertmp/rpolonsky` tree because the shared filesystems are
full; no production checkpoint tree is touched.

| Slot | Source task | Recipe | Updates | Official-data ratio | Status |
|---|---:|---|---:|---:|---|
| gpua085 / `23433381_2` | 28 | demo25 screen reproduction | 400 | 25% same-task | step-0 panel running |
| gpua002 / `23433381_3` | 60 | pure full-proposal shortlist | 200 | 0% | step-0 panel running |
| gpua087 / `23433381_1` | 29 | pure full-proposal shortlist | 200 | 0% | step-0 panel running |

The first 100-episode attempt was invalid because each task has only 50
official initial states; it exited before training and is excluded from all
results. The corrected 50-episode runs have passed setup and loaded the
expected task-specific data counts: Task 28 = 760 Small EXPO chunks plus 760
official chunks; Task 60 = 2,226 Small EXPO chunks; Task 29 = 2,150 Small
EXPO chunks.

# Task 29 SFT target-semantics comparison

> **Stopped before training (2026-11-03).** Follow-up source inspection
> confirmed that the stored 10-action teacher proposal matches the actor's
> 10-action output horizon and that collection, SFT, and rollout all use the
> same 10/5 predict/execute/replan interface. The proposal-versus-backfill
> pair was not a valid test of a stack inconsistency. Both processes were
> stopped before step-0 evaluation or optimizer updates; there are no results.

## Question

Does replacing the unused tail of each Small EXPO ten-action proposal with the
subsequent five-step actions actually issued by the same Task 29 teacher improve
SFT retention, when the rest of the recipe is fixed?

## Task-specific data constraint

Both arms use only the Task 29 Small EXPO collection (teacher checkpoint
375380, 150 successful episodes, 2,150 chunks) and 50% official Task 29 demo
chunks (1,231 chunks from the matching Kitchen Scene 5 HDF5, 50 episodes).
The Small EXPO teacher was itself trained on Task 29 official replay plus Task
29 online rollouts. Distillation initialization was disabled. Tasks 2, 28 and
60 are evaluation-only; they are never sampled for training.

## Matched settings

Both arms start from official `pi05_libero`, use fresh zero-B LoRA initialization
and adapter-only gradients, batch 16, seed 42, 10/5 chunking, 1,500 optimizer
updates, 10,000-step warmup to 5e-5, and the same 50/50 Task 29 teacher/demo
sampler. Both evaluate on the same 20 LIBERO initial states for tasks 2, 28, 29
and 60 at updates 0, 500, 1,000 and 1,500. Only the Small EXPO chunk target
construction differs: the complete teacher proposal or backfilled actions
from subsequent executed prefixes. Official-demo rows are identical in both.

The actor-only four-task panel is a development signal. The decisive retention
comparison is a matched actor-only LIBERO-90 evaluation of the final checkpoints
against official Pi0.5 and Full EXPO on the same task states. Full EXPO's editor
and critic deployment evaluation remains a separate system-level comparison.

## Data integrity and isolation

Each output manifest records the teacher, target mode, task ID, demo ratio,
normalization asset, seed and optimizer settings. The training log must report
`small_expo_chunks=2150`, `official_demo_chunks=1231`, and
`official_demo_episodes=50` before the run is considered data-ready. All new
scripts and experiment notes are isolated in this directory; existing project
trainer source and other running allocation elements are left untouched.

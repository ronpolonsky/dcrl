# Matched LIBERO-90 SFT and Full EXPO comparison

All entries use 20 episodes per task (1,800 total), evaluation seed 7, 400 control steps, action horizon 10 and replan interval 5. The validation requires all 90 tasks and identical task instructions and initial-state indices versus official Pi0.5.

SFT evaluations run the actor only. Full EXPO evaluations include its editor/critic candidate selection, so those are deployed-system comparisons rather than actor-only causal comparisons.

| Policy/checkpoint | Successes | Rate | Paired both | Base-only | Policy-only | Neither | Target task rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| official_pi05_actor | 512/1800 | 28.44% | 512 | 0 | 0 | 1288 | — |
| sft_task02_20k_actor | 112/1800 | 6.22% | 81 | 431 | 31 | 1257 | 19/20 (95.0%) |
| sft_task28_40k_actor | 46/1800 | 2.56% | 16 | 496 | 30 | 1258 | 20/20 (100.0%) |
| sft_task29_20k_actor | 219/1800 | 12.17% | 161 | 351 | 58 | 1230 | 20/20 (100.0%) |
| expo_task02_60k_full_system | 447/1800 | 24.83% | 351 | 161 | 96 | 1192 | 19/20 (95.0%) |
| expo_task28_40k_full_system | 451/1800 | 25.06% | 350 | 162 | 101 | 1187 | 20/20 (100.0%) |
| expo_task29_60k_full_system | 441/1800 | 24.50% | 347 | 165 | 94 | 1194 | 16/20 (80.0%) |
| expo_task60_60k_full_system | 368/1800 | 20.44% | 276 | 236 | 92 | 1196 | 20/20 (100.0%) |
| expo_task79_60k_full_system | 483/1800 | 26.83% | 373 | 139 | 110 | 1178 | 20/20 (100.0%) |

The 90-task, per-task data are in `full90_comparison_20261103.csv`; validated source totals and paired outcome counts are in `full90_comparison_20261103.json`.

## Controlled task-specific diagnosis candidate (2026-11-04)

The Task-60 full-proposal SFT checkpoint at update 200 passed the strict
actor-only LIBERO-90 validator: all 90 task files are present, with 20 trials
per task and the fixed seed/horizon/replanning protocol.

| Policy/checkpoint | Training data | Successes | Rate | Matched Full EXPO | Margin vs matched EXPO | Official base |
|---|---|---:|---:|---:|---:|---:|
| task-60 SFT full proposal, update 200 | Task-60 Small EXPO only; 0 official demos | 519/1800 | **28.83%** | 368/1800 (20.44%) | **+8.39 pp** | 512/1800 (28.44%) |
| task-60 SFT executed backfill, update 200 | Task-60 Small EXPO only; 0 official demos | 517/1800 | **28.72%** | 368/1800 (20.44%) | **+8.28 pp** | 512/1800 (28.44%) |

The full-proposal update-200 row remains the strongest validated task-specific
SFT result. The executed-backfill row is independently complete and confirms
that the result is not dependent on mixing tasks or official demonstrations.
The remaining 2026-11-05 controlled candidates are still incomplete and must
pass the same validator before entering the ranking.

## Earlier completed Task-28 controls recovered from the audit tree

The recursive artifact audit found three additional complete actor-only
LIBERO-90 evaluations that were not included in the later controlled-candidate
table. Their launch/config evidence verifies the same-task-only invariant:
official `pi05_libero` initialization, Task-28 Small EXPO chunks only (760
chunks), zero official-demo ratio, zero-B LoRA, and the unchanged 10/5 action
interface.

| Policy/checkpoint | Successes | Rate | Matched Full EXPO | Margin vs matched EXPO |
|---|---:|---:|---:|---:|
| Task-28 SFT LoRA-only, update 2,000 | 471/1800 | 26.17% | 451/1800 (25.06%) | +1.11 pp |
| Task-28 SFT raw action bounds, update 1,500 | 464/1800 | 25.78% | 451/1800 (25.06%) | +0.72 pp |
| Task-28 SFT clipped action bounds, update 1,500 | 470/1800 | 26.11% | 451/1800 (25.06%) | +1.05 pp |

These are valid same-task controls, but the newer Task-28 full-proposal
update-300 arm remains stronger at 518/1800 (28.78%, +3.72 pp).

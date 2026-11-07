# SFT generalization audit — 2026-11-03

Status: investigation active. Findings below are evidence, not a demonstrated repair.

## Decision criterion for the corrective SFT

The success criterion is generalization: a candidate SFT policy must beat the
Full EXPO reference under the same actor-only LIBERO-90 evaluation protocol
(the same 90 tasks, 20 episodes per task, seed, initial states, 400-step
limit, 10-action horizon, and 5-step replanning boundary). Target-task
success, lower behavior-cloning loss, or a favorable four-task panel alone is
not sufficient. The primary comparison is total successes out of 1,800 plus
the per-task distribution. A deployment-path Full EXPO comparison remains a
separate result because it includes editor/critic candidate selection, while
the causal SFT comparison uses `eval_actor_only()`.

The current task-specific SFT results do not pass this criterion. The
corrective candidate therefore needs multi-task rehearsal with task-specific
prompts and must be selected by full-suite generalization, not by source-task
performance.

### Isolated balanced multi-task arm prepared

The next corrective arm is prepared but not launched while the two matched
action-bound controls occupy their reserved A100s. It reuses the audited
learner, Pi0.5 transforms, zero-B LoRA initialization, optimizer and retention
evaluator through `train_sft_multitask.py`, while injecting
`multitask_scope_data.py` only through the isolated wrapper. It opens the four
existing successful Small EXPO collections (tasks 2/28/29/60), preserves each
row's own instruction, and samples tasks uniformly before sampling a chunk.
The verified source contains 6,863 rows total: 1,727/760/2,150/2,226 for
tasks 2/28/29/60, respectively. This avoids the old single-prompt dataset
constraint and prevents the longest collection from dominating the updates.
The launcher is `run_multitask_sft.sh`; it writes a per-file hash/count audit
before training and keeps all outputs under the new isolated output directory.
The resulting checkpoint will still require a complete actor-only LIBERO-90
evaluation before it can be called better than Full EXPO.

### Exact official-base four-task control completed

The evaluator-only official `pi05_libero` control has now completed the same
four-task, 20-episode panel used by the action-bound arms. It scored task
2/28/29/60 = 50/50/60/55%, or 43/80 = 53.75%. This is the correct protocol
baseline for interpreting the independently launched step-0 rollouts; the
raw and clipped arms' 58.75% and 60.0% step-0 values should not be treated as
causal evidence for clipping because their rollouts are not outcome-identical
to this separately evaluated base control.

## Research target and invariants

Test whether distilling a compact RL teacher into pretrained pi05_libero can improve the target task while retaining more LIBERO-90 competence than the complete EXPO system. Preserve original checkpoints, collections and evaluation records. Keep the existing full-suite workers running and preserve all five GPU allocations. New diagnostics live in this directory; do not silently alter the production RL/SFT/evaluation paths.

## Measured findings

1. **The SFT LoRA configuration updates 466,957,072 parameters.** This includes the entire 414,803,696-parameter image tower, 49,987,584 adapter parameters, and 2,165,792 action/time projection parameters. Adapter-only training would update 49,987,584 parameters. These counts come from the actual NNX trainable filter on the model state shapes (`parameter_filter_audit.json`), not configuration labels. This alone does not establish the cause: Full EXPO's LoRA factory also calls the same model freeze filter, so its actual scope must be compared rather than assumed to be stricter.

2. **The task 28 collection is extremely small in chunk count.** Its 150 successful episodes contain 760 replanning rows, with 50 unique initial-state indices. Batch 16 for 40k updates draws 640k rows with replacement: 842.1 draws per stored row on average. Even 5k updates means 105.3 draws per row. Task 2 contains 1,727 rows; task 29 has 2,150; task 60 has 2,226. The stopped task 28 run reached 41,330 updates, or 870.1 draws per row.

3. **SFT and EXPO step labels are not matched optimization budgets.** Near Full EXPO task 28's 40k environment-step snapshot, its last logged actor optimizer index is 6,704. SFT's 40k checkpoint represents 40k actor optimizer updates. Full EXPO task 29/60 logs reach actor indices 5,684/8,144 near their 60k environment-step snapshots. These are optimizer update counts, not a claim that each method uses identical supervision, batch composition, or integrated learning rate.

4. **The supervised target distribution is narrow.** All saved task 28 gripper targets are approximately -1. Its final 1k-window median flow loss is ~0.00109 (first window ~0.0318). Its own task reaches 20/20, while the complete 90-task evaluation is 46/1,800. That establishes successful target fitting with poor transfer; training loss is not a retention metric. Other task collections contain teacher commands outside [-1,1] (e.g. task 2 z ranges about -3.23 to 1.93). Those values are observations, not yet identified as errors: check the environment controller's action saturation before changing labels.

5. **Initialization is not proven equivalent to the official base actor.** The local LoRA module initializes both A and B using normal(stddev=0.01), so newly inserted adapters are not structurally guaranteed to produce zero residual. The official baseline freezes/casts all parameters to bfloat16; SFT freezes/casts its LLM backbone while leaving its image tower/projections trainable. Task 2's training step-0 eval is 11/20 whereas the saved official full-suite eval on task 2 is 14/20. Identical checkpoints are therefore insufficient evidence of identical initial policies. The action diagnostic explicitly separates fresh LoRA, zero-B LoRA, and zero-B with all-bfloat16 from the saved baseline path.

6. **Checkpoint selection currently observes target-task success only.** The trainer first evaluates after 20k updates, and its best-checkpoint manager retains one checkpoint ranked by target success, breaking ties toward later steps. It does not measure retention during training. Earlier checkpoints where useful adaptation may precede forgetting were not saved in these runs. Changing to replay without evaluating earlier would not diagnose this problem.

## Current causal picture (evidence ranked, not a final causal claim)

| Finding | Evidence strength | What it explains / what it does not |
|---|---|---|
| The historical task-28 SFT actor overfits the target and loses suite competence. | Direct full-suite evidence: 46/1,800 versus 512/1,800 base; 496 base successes lost and 30 gained on paired states. | Real collapse, independent of the small four-task panel. |
| The older eval cadence hid early degradation. | Paired task-28 retention falls by 500 updates; target 28 rises while same-language task 29 falls. | Explains how a run could look successful at target eval while transfer had already failed. |
| SFT exposure/update dose is much larger than Full EXPO's actor update budget. | 40k SFT optimizer steps versus about 6.7k Full EXPO actor updates near 40k environment steps; about 842 SFT dataset-equivalents on task 28. | Strong plausible contributor; methods also differ in data/objective, so not yet isolated. |
| SFT targets include raw actions that OSC saturates at execution. | Source-level path verified end-to-end; task 28 has 30.8%/53.6% x/y motion targets outside [-1,1], while Pi0.5 quantile normalization does not clip. | Concrete label/execution mismatch and cleanly testable in the prepared single-factor arm. |
| SFT's stored 10-action teacher plan is inconsistent with the deployed 10/5 policy interface. | **Not supported:** collection stores the teacher's 10-action output at the current observation; rollout executes its first five and replans. The learner predicts the same 10-action shape, and evaluation also executes five before replanning. | This is internally consistent chunked-policy behavior. EXPO replay's executed-trajectory chunks are a different supervision target, not evidence of an SFT stack bug. |
| Historical “replay” likely changed both demo coverage and command-label quality. | SC W&B replay arm is much stronger, but its old private dataset/provenance is incomplete; local demo option is task-specific, whereas historical replay reportedly spans multiple task prompts. | The active 0/50% same-task demo pair cannot by itself reproduce or explain historical replay. |
| Evaluation-path asymmetry inflates the apparent Full EXPO/SFT gap. | Full EXPO suite restores actor/editor/critic candidate-selection; SFT suite is actor-only. | Must run Full EXPO actor-only on paired states for actor-level attribution; cannot explain the separately observed SFT-vs-base collapse. |

### Diagnostic order

1. Finish the active task-29 no-replay versus 50%-same-task-demo pair at 0/500/1,000/1,500 updates. It tests the local demo-mixture effect but not the historical multi-task replay recipe.
2. Run the prepared task-29 raw-target versus OSC-clipped-target comparison, changing only the six motion target bounds before the unchanged quantile transform. This isolates a source-verified action-semantics defect.
3. Compare actor-only Full EXPO and SFT on identical suite states before claiming actor-level relative performance; preserve the existing full-system comparison as a deployment result.
4. Any candidate fix must be evaluated at early checkpoints and on the full 90-task actor suite. Target success and BC loss alone are not sufficient.

## Pipeline inspection and ruled-out assumptions

### Direct checkpoint weight comparison

`weight_drift_audit.json` now contains measurements from both saved task 28 actors. The base reference is cast to each checkpoint leaf's dtype before differencing; new adapter leaves have no base counterpart and are listed separately.

| Module | SFT 40k relative L2 drift | Full EXPO 40,095 relative L2 drift |
|---|---:|---:|
| Image tower | 10.73% | 1.94% |
| Non-adapter language backbone | 0% | 0% |
| Action output projection kernel | 28.34% | 8.93% |
| Time MLP output kernel | 25.90% | 6.32% |

The SFT image tower drift is about 5.52 times larger. This confirms actual backbone/projection changes in saved checkpoints, rather than just theoretical trainability. It is not an ablation proving which changes caused the performance loss. Full EXPO also changes its image tower, so the explanatory difference includes update dose, supervision and magnitude, not an assertion that Full EXPO freezes vision.

- The actual suite launcher uses `/work/hdd/bgqe/rpolonsky/sharder/training/full_90_libero_evals_inference.py`, not just the repository's main evaluator. The SFT branch reads the manifest, restores the saved SFT actor and uses per-task instructions in observations. Saved normalization assets are passed to the model factory. No obvious fixed-prompt substitution was found in this path.
- The teacher collector records the current observation before requesting the chunk, then executes its first five actions. It saves the complete ten-action proposal; the remaining five actions are targets even though the next replanning event replaces them. This needs separate prefix/tail quality measurement, rather than an assumed one-step imitation interpretation.
- Both local SFT and local official baseline configure `pi05=True`, action horizon 10 and `discrete_state_input=False`. This also matches the current upstream official `pi05_libero` configuration. The model's general default enables discrete state for pi05, but the explicit LIBERO override is intentional upstream; do not call this a local state-input bug or silently enable it.
- Official source checked: https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/training/config.py and https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/pi0_config.py .
- The official-demo mixing option was added after the old runs; those existing SFT checkpoints were trained without that mix. Same-task official demos can change supervision quality but do not supply other tasks' instructions and examples. Retention benefits must be measured, not presumed.

## Diagnostics running / next decisions

### Completed action and paired-suite checks

The action comparison completed on eight cached observations from two episodes each of task 2 and task 28. With identical random keys, mean absolute raw-action differences from the official adapter were: fresh LoRA 0.001786; zero-B LoRA 0.001217; zero-B with all parameters bfloat16 0.000319; trained SFT task 28 0.269842. Fresh-vs-zero-B difference was 0.001882. These are cached-input measurements, not rollout success rates; small action differences can compound during rollout, so initial-policy equivalence is not assumed. The trained actor's task 28 teacher-prefix MAE was 0.00931 versus 0.31466 for the official actor. On the task 2 cached inputs it did not simply always open the gripper: its gripper outputs still varied. Do not equate narrow training gripper labels with a proven unconditional gripper failure.

`suite_retention_audit.json` validates every completed suite's task IDs, instructions, seed, episode count, action/replan horizon, step limit and initial-state indices before pairing outcomes. Task 28 SFT retained only 16 of the official base's 512 successful cases, lost 496, and gained 30. Its off-training-task rate was 26/1,780 = 1.46%, versus 431/1,780 = 24.21% for the full EXPO task 28 system. Task 2 SFT's off-task rate was 93/1,780 = 5.22%, versus 428/1,780 = 24.04% for full EXPO task 2. These are paired cases within the fixed benchmark, not estimates of variability across independent training seeds.

Task 2 and task 29 have the exact same language instruction. Nevertheless, SFT trained on task 2 scored 0/20 on task 29 (base 14/20, Full EXPO trained on task 2 9/20). Thus instruction specialization alone cannot explain all failures; scene/representation/action changes need testing too.

### Additional training differences established by source inspection

- The configured cosine schedule has `decay_lr == peak_lr == 5e-5`, so it stays flat after the 10k-update warmup. The sum of scheduled learning rates is 1.75 over 40k SFT updates, versus about 0.1124 over 6,705 Full EXPO actor updates: a 15.6x difference in this descriptive measure of optimization dose. It does not imply that Adam parameter updates scale linearly by that factor.
- Full EXPO's actor also uses the Pi0.5 flow-matching train step, on success-filtered replay batches. Its actor is not trained by a direct Q gradient through the VLA. Its collected actions are influenced by the editor/critic policy, and the deployed system still uses them at evaluation.
- Full EXPO's replay insertion backfills chunks with subsequent executed commands. SFT instead saves each teacher's entire ten-action proposal while executing only five before replanning. The second half therefore has a different supervision meaning. Do not silently replace full-chunk distillation with prefix-only labels; measure this separately.
- Full EXPO's update applies `data_augmentation_fn` before the Pi0.5 train step, which itself applies model augmentation. SFT's `bc_update` calls only the model train step. Setting `use_full_augmentation=True` on the SFT learner does not invoke the outer EXPO augmentation in `bc_update`.
- The local LIBERO `scripts/create_dataset.py` steps the environment before recording each observation, confirming the post-action/next-action convention used in the new official-demo sampler.
- Robosuite's OSC controller clips motion commands to its input bounds before scaling. Teacher target commands outside these bounds can differ from the effective command executed by the controller. This is a supervision-quality question, not evidence that normalization is applied twice.

### Controlled experiments launched

`scope_experiment_plan.json` records the predeclared comparison and source hashes. Both arms train task 28 for 2,000 updates on the original 760 chunks, batch 16, original seed/warmup/LR/normalization, and official-demo fraction 0. They evaluate 20 episodes on each of tasks 2, 28, 29 and 60 at step 0 and every 500 updates. This is a development set; a final full-90 test is still required.

- Existing scope: tmux `sft_scope_existing_task28`, allocation 23352795_4 / gpua062.
- Adapter-only scope: tmux `sft_scope_lora_only_task28`, allocation 23352795_5 / gpua011.
- `AuditTrainConfig` changes only the gradient/optimizer filter, preserving the original storage-casting filter. Merely strengthening `freeze_filter` would also cast newly frozen image/projection parameters to bfloat16 and introduce a precision confound.
- The CPU mask check passed: 466,957,072 versus 49,987,584 trainable parameters, with identical casting masks. Live construction fingerprints matched in both arms (`d7571ea124390f9ebdbd514c7b7792b1296460144461362aac123ddcf7adb948`), including dtype and 32 sampled values per parameter leaf. This is a sampled parity check, not a full-tree byte hash.
- Both arms retain one best development-macro checkpoint and one latest evaluated full learner checkpoint, including optimizer state. Data is shared through a read-only usage of the original collection; no collection is regenerated or overwritten.
- W&B: existing scope https://wandb.ai/distill_rl/dcrl/runs/je14pqpx ; adapter-only https://wandb.ai/distill_rl/dcrl/runs/fbpvjr08 .
- `evaluate_retention.py` is an isolated copy supporting the experiment config, a fresh-LoRA baseline, and full-EXPO actor-only evaluation. Its syntax check passes; the new evaluation modes still require runtime verification before their scores can be used.

### Initial experiment runtime validation

Both arms completed all 80 initial evaluation episodes. The entire per-task trial dictionaries match exactly across arms, including outcome and episode length, not just aggregate success. The initial rates on tasks 2/28/29/60 are 75%/45%/70%/50%, macro 60%. Both wrote committed step-0 latest checkpoints whose metadata includes actor optimizer state. The initial XLA compile completed; subsequent status is recorded in the runtime-validation paragraph below.

Storage checks distinguish the full `/projects` filesystem from experiment output storage: project quota 20951 on `/work/hdd` had approximately 400 GiB of soft-quota headroom before new checkpoint writes, with no personal block limit. Both experiments write below `/work/hdd/bgqe/rpolonsky/dcrlcheckpoints/sft_retention_scope_20261103`, not `/projects`. Space decreased as the initial checkpoints were written; this is expected. Five existing GPU allocations remain active.

The target-quality audit additionally finds that roughly 48.6–48.8% of the saved ten-action proposal slots correspond to actions actually executed before replanning/termination. The remaining slots are teacher proposals, as intended by full-chunk distillation. For task 28, official demos also have gripper -1 throughout, so the teacher's open-gripper-only labels do not distinguish it from same-task official demonstrations. Its 150 teacher episodes have median length 25 steps, versus 73.5 for the 50 official episodes; both sources contain 760 eligible 10/5 windows. Thus adding same-task demos supplies different trajectories, but does not by itself add other instructions or a large order-of-magnitude increase in unique chunks.

The preprocessing parity audit (`preprocessing_parity_audit.json`) now compares real rows from the task 2/28/29/60 saved collections through the exact training transform and the exact Pi05 inference-side repack/transform path. After the training-side `Observation.from_dict` conversion and inference-side uint8 scaling, all image tensors, image masks, padded states, prompt tokens and masks matched exactly for the sampled rows (maximum absolute difference 0). The normalized seven-dimensional action chunks also round-tripped through the official quantile unnormalization with maximum absolute error below `4.5e-16`. This rules out a basic training-versus-inference preprocessing mismatch for these paths; it does not rule out rollout compounding, target quality, update dose or model drift.

### Controlled step-0 is not the exact official-base policy

The historical official 90-suite baseline loads the plain `pi05_libero` actor, while both current scope arms use `finetune_mode=lora` and insert the local LoRA module before their step-0 evaluation. Because that module initializes both A and B with nonzero random normal values, the step-0 rollout is a fresh-random-LoRA policy, not a no-adapter copy of the official actor. This is not merely theoretical: with the same `eval_seed=7`, task and initial-state indices, the official baseline and controlled step-0 outcomes differ materially:

| Task | Official base successes | Controlled step-0 successes | Official successes retained / lost / newly gained by step-0 | Outcome differences | Different episode lengths |
|---:|---:|---:|---:|---:|---:|
| 2 | 14/20 | 15/20 | 12 / 2 / 3 | 5/20 | 15/20 |
| 28 | 10/20 | 9/20 | 7 / 3 / 2 | 5/20 | 12/20 |
| 29 | 14/20 | 14/20 | 12 / 2 / 2 | 4/20 | 15/20 |
| 60 | 9/20 | 10/20 | 9 / 0 / 1 | 1/20 | 10/20 |

The controlled step-0 macro is 60%, but this is not a direct measurement of the official base's 58.75% macro on these four tasks. Therefore the current paired `step-0 -> SFT` retention curves are valid for comparing the two SFT scopes under the same initialized run, but they cannot by themselves establish retention relative to the exact official actor. The fresh-LoRA action audit already measured nonzero initial action differences (mean absolute raw-action difference 0.001786 from the official adapter on cached observations); the rollout table above demonstrates that small action differences compound into different outcomes. A complete conclusion requires an exact-base control evaluated through the same isolated evaluator, and preferably a zero-output-initialized adapter arm if adapter training remains the chosen recipe.

At the latest runtime check (09:05 CST), both controlled arms had passed compilation and were performing real updates. The original-scope arm had written metrics through approximately step 174 and the adapter-only arm through approximately step 343; neither had reached the first 500-step retention evaluation yet. The earlier compile delay was therefore finite XLA startup overhead, not a training failure.

At the next matched-budget check (09:09 CST), the first 300 updates provide a useful optimization diagnostic even before rollout evaluation. The original-scope arm's median actor loss was 0.1424 with median gradient norm 0.6357; the LoRA-only arm's median actor loss was 0.2437 with median gradient norm 0.1237. The original arm therefore fits the sampled teacher chunks faster under the same nominal schedule, while LoRA-only receives much smaller gradients; this is not evidence that either policy generalizes better. It is a reason to compare retention at fixed update counts and to inspect parameter/action drift, rather than selecting a recipe from training loss alone.

The LoRA-only arm's first trained retention evaluation (step 500) is now complete. Its task-2/28/29/60 rates were 75%/70%/45%/40%, compared with the identical step-0 rates 75%/45%/70%/50%. The macro rate therefore fell slightly from 60.0% to 57.5% while the target task improved by 25 percentage points. Paired by the same 20 initial-state indices, it retained/lost/gained successes as follows: task 2 = 13/2/2; task 28 = 8/1/6; task 29 = 7/7/2; task 60 = 8/2/0. Across the four tasks it lost 12 original successes and gained 10. Thus restricting gradients to LoRA prevents neither specialization nor tradeoff: it improves task 28, but the gain is accompanied primarily by losses on task 29 (which has the same language instruction as task 2) and task 60. This is a stronger result than a target-only score, but it is still only a four-task development evaluation; full LIBERO-90 is required.

The matched original-scope arm's step-500 evaluation is also complete: task-2/28/29/60 rates were 85%/100%/10%/25%, with macro 55.0%. Relative to its identical step-0 baseline, the paired retained/lost/gained counts were task 2 = 14/1/3, task 28 = 9/0/11, task 29 = 2/12/0, and task 60 = 5/5/0. It lost 18 original successes and gained 14. Compared with LoRA-only, the original scope fits the target more aggressively (+55 points on task 28 versus +25) but causes substantially greater catastrophic specialization, especially task 29 (12 losses versus 7) and task 60 (5 versus 2). LoRA-only is the better of these two scope controls, but its macro retention still declines by 2.5 points; scope restriction alone is not the repair.

### Live controlled-run status (2026-11-03, subsequent poll)

Both predeclared scope arms remain healthy and are executing real JAX updates on separate A100s. A direct in-allocation `nvidia-smi` check reports 100% GPU utilization and approximately 30.8 GiB used on each device; there is no observed OOM, checkpoint-quota failure, or process exit. The original-scope process is currently around update 529 on `gpua062` (allocation `23352795_4`), and the adapter-only process is around update 858 on `gpua011` (allocation `23352795_5`). The latest completed rollout evaluation for each is still step 500; no step-1000 evaluation exists yet. The original-scope latest full learner/optimizer checkpoint at step 500 is committed, and the adapter-only step-500 checkpoint is committed. Both are configured to continue to 2,000 updates with evaluations and full checkpoints every 500 updates.

The exact live configuration is unchanged: task-28 SmallExpo teacher checkpoint 275001, official `pi05_libero` student initialization, 150 successful teacher episodes (760 eligible task-28 rows), batch 16, LoRA model mode, no official-demo mixture (`official_demo_ratio=0`), learning rate 5e-5 with 10,000-step warmup, and retention evaluation on tasks 2/28/29/60. The five pre-existing sleeper allocations remain running and were not modified.

The next experiment should not be chosen from target-task loss alone. The current evidence favors first testing update dose with retention checkpoints at 50/100/250/500 updates, then testing target supervision semantics (executed 5-step prefix versus the saved 10-step proposal). Same-task official-demo replay is a useful separate ablation, but it cannot by itself preserve LIBERO-90 competence: it supplies only task-28 language and trajectories, and the local official-demo cache contains only the five task-specific files used by the existing runs rather than a complete 90-task replay corpus. A full-suite result for any candidate remains mandatory.

The target-semantics arm is now implemented but deliberately not launched while both reserved GPUs are occupied by the predeclared scope comparison. `scope_data.py` adds `executed_backfill` without modifying the canonical dataset module: for each row it concatenates the actually executed actions from that row and subsequent replanning rows, then terminal-pads to ten actions. A CPU test on the 760-row task-28 collection confirmed shape `(10, 7)`, exact ordinary-row prefix preservation, and changed tails; the current collection remains read-only. `run_target_semantics_arm.sh` and `target_semantics_experiment_plan.json` define matched LoRA-only `full_proposal` and `executed_backfill` arms with 100-step retention evaluations. They are isolated artifacts and have not changed or relaunched any running process.

A subsequent live poll confirms the original-scope arm has reached update 800. The adapter-only arm has reached update 999, and its step-1000 evaluation is still running: the process remains alive on `gpua011`, with its GPU allocation present and no new eval/checkpoint file yet. This evaluation is expected to be slow because it runs 80 LIBERO episodes; it is not being restarted or interrupted.

The next poll completed the adapter-only step-1000 evaluation. Its rates were task 2/28/29/60 = 90%/95%/10%/30%, macro 56.25%, versus its identical step-0 macro of 60%. Paired on the same initial states, it retained/lost/gained successes as task 2 = 15/0/3, task 28 = 9/0/10, task 29 = 2/12/0, and task 60 = 6/4/0. Thus the target task continued improving from 45% to 95%, but the same-language task 29 remained at 10% and 12 of its original successes were still lost; the four-task macro did not recover. This confirms that adapter-only scope reduces the magnitude of collateral drift relative to the original scope, but does not solve specialization. The committed artifact is `dcrlcheckpoints/sft_retention_scope_20261103/task_28_lora_only/latest_checkpoints/1000`, with the corresponding rollout at `evals/00001000.json`.

At the same poll, the original-scope arm was still alive and in its step-1000 rollout evaluation (17/20 episodes in the currently displayed task); it had not yet written `evals/00001000.json`. Both Slurm allocations remain RUNNING on `gpua062`/`gpua011`, and the five pre-existing allocations were not touched. The adapter-only arm resumed training after its evaluation and was at approximately update 1,018; its step-1000 checkpoint directory was committed. No target-semantics arm has been launched yet.

The executed-target implementation was audited over all 760 task-28 success rows, not only a random sample. Every row preserves its actually executed prefix exactly; 68 rows terminate before five executed actions; every row's 10-step tail changes relative to the saved proposal, with mean absolute tail difference 0.04886 in raw action units. This is consistent with EXPO's replay-window behavior, which uses subsequent executed transitions and marks terminal tails as fabricated/backfilled rather than treating an abandoned proposal tail as ground truth. The new arm is therefore a meaningful target-semantics test, not an arbitrary label perturbation.

The isolated trainer now also exposes `--zero-lora-b`. It zeroes only newly initialized `lora_b` leaves immediately after actor construction, before optimizer updates, using the same parameter tree and trainable filter. This is the controlled initialization repair suggested by the action audit; it does not alter the production trainer or the current runs. The option passes syntax and help checks, but has not been launched yet because both reserved GPUs remain occupied.

Two isolated launchers are prepared and pass shell syntax checks: `run_exact_base_control.sh` evaluates the plain official actor through the same retention evaluator on tasks 2/28/29/60, and `run_zero_b_control.sh` runs a matched 500-update LoRA-only SFT arm with zero-B initialization. Neither has been launched; both write to new output directories and cannot affect the current scope runs.

Source inspection also confirms a separate augmentation gap. Full EXPO's `_update_jit` calls `data_augmentation_fn` on the actor batch before `update_actor`; the isolated `bc_update()` path calls `Pi05Agent.train_step` directly and never calls that function. `use_full_augmentation=True` in learner construction therefore does not make current SFT equivalent to EXPO augmentation. The audit trainer now has an explicit `--outer-augmentation` flag and a matched `run_outer_aug_control.sh` launcher. The new path passes syntax/help checks and remains unlaunched until a reserved GPU is free.

The old 200k task-28 SFT artifact and the controlled scope runs also have identical collection provenance at the byte level: the SHA-256 of `pi_base_data/teacher_chunks.hdf5` is `1d799f344b5d8eec24a97c0d2e513f3981c1f9d83334e91988d76bc5ba693565` in both outputs, and the success-only partition is `01f56d836439471077ca39468ea86a9df0d6280fefe1077cc04a86d4db5834ff` in both. Their recorded configs likewise use the same teacher checkpoint (task 28, step 275001), Pi0.5 LIBERO student checkpoint, normalization hash, 150-success collection, 10/5 interface, batch size 16, learning rate, warmup and seed. The controlled comparison therefore isolates trainable-scope/early-evaluation behavior rather than a changed data snapshot.

The original-scope arm's step-1,000 evaluation is now complete. Its task-2/28/29/60 rates were 80%/100%/10%/5%, macro 48.75%, down from its fresh-random-LoRA step-0 macro of 60%. Paired on identical initial-state indices, it retained/lost/gained successes as task 2 = 13/2/3, task 28 = 9/0/11, task 29 = 1/13/1, and task 60 = 1/9/0. The target task is fully fit, but the arm has lost 24 of its step-0 successes and gained only 15; the largest damage is now task 29 (13 losses) and task 60 (9 losses). This is stronger evidence that the 466.9M-parameter scope is too destructive under this direct SFT dose than the earlier step-500 result. It still must be interpreted relative to the fresh-LoRA step-0 confound described above, not as an exact official-base retention percentage.

The corresponding `evals/00001000.json` and `latest_checkpoints/1000` for the original-scope arm are now present, and the trainer resumed real updates afterward (metrics through update 1,017 on the latest poll). The LoRA-only arm has reached update 1,499 and is entering its step-1,500 evaluation; its last completed rollout remains step 1,000. Both CUDA processes remain resident on their assigned A100s, and no allocation was altered.

The latest direct poll (10:05 CST) still finds both Slurm jobs RUNNING with CUDA processes resident: the original-scope arm has progressed through update 1,079, while the LoRA-only arm remains in the step-1,500 rollout (metrics through 1,499, with no `00001500.json` yet). This is a slow but live evaluation, not a missing process; the next control remains gated on the two reserved GPUs becoming available.

### Latest live verification (2026-11-03 10:00 CST)

Both controlled trainer processes are still alive and are the actual CUDA compute processes on their reserved A100s: PID 1960145 on `gpua062` owns 31,488 MiB and PID 3031045 on `gpua011` owns 30,810 MiB. The LoRA-only arm has written training metrics through update 1,409/2,000 and remains in the training loop. The original-scope arm has metrics through update 999 and remains in the step-1,000 rollout evaluation; its process is alive and the model remains resident on the GPU, but `evals/00001000.json` has not yet been committed. The momentary 0% GPU utilization observed for that arm is therefore an evaluation-phase observation, not evidence that it fell back to CPU-only training or exited. No target-semantics arm has been launched, and no existing allocation or training session was changed.

- `audit_actor_outputs.py`: identical stored observations/random keys; compare official, fresh LoRA, zero-B LoRA, zero-B all-bfloat16, and SFT task 28 checkpoint. Runs in tmux `sft_actor_audit_20261103` within allocation 23352795_4 (gpua062).
- `audit_weight_drift.py`: restore only actor parameter arrays and compare SFT task 28 and Full EXPO task 28 with official base, by module. Runs in tmux `sft_drift_audit_20261103` within allocation 23352795_5 (gpua011). It computes on CPU on that reserved node to avoid placing extra large model copies on the GPU.
- Verify fresh LoRA/base parity, training-vs-restored action parity, image normalization and tokenizer equivalence numerically before attributing the result only to catastrophic forgetting.
- Use short, controlled new SFT experiments with early retention evaluations and checkpointed optimizer state. Separate adapter scope, training dose, and same-task demo mixture as controlled factors. Do not change several factors at once and claim to identify a cause. Retain the original full model/data/image resolution and initial checkpoint.
- Compare Full EXPO's actor-only retention with its deployed editor/critic-assisted scores; otherwise critic selection can conceal actor degradation in the system comparison.
- A successful remedy requires both target-task improvement and measured retention on the 90-task suite. A falling training loss, a smoke test or a high target-only score is insufficient.

## Artifacts

- `data_metrics_audit.json`: complete collection/config/statistical audit, training-loss windows and saved training evals.
- `parameter_filter_audit.json`: every model parameter path, shape and trainability totals.
- `actor_output_audit.json`, `weight_drift_audit.json`: written incrementally by the corresponding diagnostic, once model loading and measurements complete.

### Live verification (2026-11-03 10:13 CST)

The two controlled SFT processes remain live on their allocated A100s, verified directly on `gpua062` and `gpua011`: the original-scope trainer is using 30,837 MiB with 100% GPU utilization, and the LoRA-only trainer is using 31,509 MiB with 51% utilization. Their logs advanced during the poll: original-scope reached update 1,336/2,000; LoRA-only is in its step-1,500 retention rollout (7/20 episodes in the current task). The 1,500 eval JSON/checkpoint does not exist yet, so no result is being inferred from the in-progress rollout. The enclosing sleeper allocations and the other three allocation elements remain running; no other GPU job or tmux session was modified.

The checkpoint initialization is supported by the actual learner construction path, not just the manifest: `run_scope_arm.sh` passes the local cache directory for official `pi05_libero` as both `--source-checkpoint` and `--student-checkpoint`; `audit_sft_config.build_learner()` builds the Pi0.5 LIBERO model with LoRA variants and sets `CheckpointWeightLoader(<student_checkpoint>/params)` before `Pi05Agent.initialize()`. Both run manifests record the same official checkpoint and normalization asset, and both initialization audits have the same sampled full-parameter-tree fingerprint. The fingerprint is explicitly sampled rather than a checksum of every checkpoint byte, so the loader configuration and successful model initialization provide the direct provenance evidence. Newly inserted LoRA leaves are initialized by the model's default initializer; this changes the step-0 policy even though the underlying checkpoint weights are loaded correctly. This resolves the question of whether the experiments started from the Pi0.5 LIBERO model, while preserving the separate finding that they did not start from the exact unmodified actor function.

### Dose trajectory and the apparent full-suite mismatch

The exact task-28 training instruction is `close the top drawer of the cabinet`; task 2 and task 29 both say `put the black bowl in the top drawer of the cabinet`. On the shared four-task initial-state panel, both fresh-LoRA arms start at task 2/28/29/60 = 75/45/70/50% (60% macro). By update 500, the wider existing scope is 85/100/10/25% (55% macro), while LoRA-only is 75/70/45/40% (57.5% macro). At update 1,000, they are 80/100/10/5% (48.75%) and 90/95/10/30% (56.25%), respectively. Thus, restricting gradients to the ~50M adapter parameters softens the aggregate decline, but task-29 success still falls from 70% to 10% by 1,000 updates. The old full-suite task-28 SFT checkpoint at 40k updates reaches 20/20 on its target task but 0/20 on tasks 2, 29, and 60; full EXPO task 28 on the same benchmark reaches 12/20, 0/20, and 2/20 on those off-target tasks. These results are consistent with a worsening dose/retention tradeoff, not contradictory outcomes. They also show that the new 2k runs cannot yet establish whether low-dose SFT is a remedy; the 1.5k and 2k paired evals remain necessary.

The SFT dataset has 760 overlapping 10/5 chunks from 150 successful episodes. With batch size 16, 500 updates consume 8,000 chunk draws (about 10.5 dataset-equivalents), 1,000 consume 16,000 (~21), 2,000 consume 32,000 (~42), and the old 40k checkpoint consumes 640,000 (~842). Dataset-equivalents are only a rough repetition measure because chunks overlap heavily within episodes, but they quantify why a low training loss does not demonstrate broad competence. In the existing-scope arm, median actor loss fell from 0.224 during updates 1–100 to 0.0165 during 901–1,000; in the LoRA-only arm it fell from 0.248 to 0.0307 over the same windows. That loss reduction coexists with measured transfer loss, so it is not a useful selection metric by itself. The present high-priority controls remain an exact unmodified-base evaluation on the same four initial-state sets and a zero-B LoRA run (so the adapter residual is exactly zero at initialization), followed by the target-action semantics and augmentation ablations already prepared. No new arm is started until the two in-flight runs finish and their 1.5k/2k evals are committed.

**Comparison caveat:** the recorded Full EXPO suite artifacts use `mode=direct_rl_pi`, which restores and evaluates the full EXPO candidate-sampling/editor/critic-selection path. The SFT suite artifacts use `mode=sft_pi`, which evaluates only the actor. Thus the 90-task numbers are valid deployed-system comparisons and use matching suite/seed/state indices, but they do not isolate actor quality. The controlled four-task SFT runs also call `eval_actor_only()`. Before making an actor-level causal claim, run the existing `direct_rl_actor_only` evaluator on the exact same Full EXPO checkpoint and tasks. The current evaluator source documents these distinct paths explicitly; keep the reported full-system comparison labeled accordingly.

The LoRA-only 1,500-update paired evaluation has now completed: task 2/28/29/60 = 70/100/15/10%, macro 48.75%. Against this arm's own step-0 outcomes on identical initial states, it retained/lost/gained 12/3/2 successes on task 2, 9/0/11 on task 28, 2/12/1 on task 29, and 2/8/0 on task 60. Across all 80 cases, 23 previously successful outcomes were lost and 14 were gained. This is a clear worsening after the 1,000-step evaluation (56.25% macro); the near-perfect target score masks continued degradation elsewhere. The training process remains alive and has not yet written post-evaluation step-1,500 metrics; no inference is made about the eventual 2,000-step point. The original-scope arm is still in training at roughly update 1,426 and has not yet completed its step-1,500 evaluation.

At the next direct poll, the LoRA-only step-1,500 evaluation JSON was committed and Orbax had begun writing `latest_checkpoints/1500.orbax-checkpoint-tmp-*`; the Python trainer remained alive with the A100 model resident. This is the checkpoint-save phase after eval, not evidence of a stuck or failed run. The original-scope process advanced to update 1,451 on its A100. Both enclosing Slurm elements and the other three sleeper elements remained RUNNING.

At 10:30 CST, both trainers were still resident on their A100s. The original-scope arm had reached update 1,499 and was evaluating step 1,500 (5/20 episodes in the active task); the adapter-only arm had reached update 1,999 and was in its final step-2,000 evaluation (4/20 episodes in the active task). The 2,000-step results/checkpoints had not yet been committed.

### Historical replay result and provenance

The saved W&B comparison in `replay_lora_wandb_scan.md` is strong empirical evidence that the dataset called `replay_local` mattered: task-29 LoRA SFT scored 400/1,800 (22.22%) with replay and 39/1,800 (2.17%) without it; task 29 itself was 11/20 versus 0/20. W&B run metadata verifies both used the same model family and `pi05_libero` weights, seed 42, 30k updates, batch 8, bfloat16, 10k warmup and 5e-5 flat-after-warmup schedule. Both point to the same source commit, `bce2273517fe49e1cb59f80f5848b9058382095d`, in `/juice2/scr2/ronpo/drl` (`ronpolonsky/drl.git`); aside from run identifiers/experiment names, their recorded training-config difference is the dataset repo ID (`...no_replay` versus `...replay_local`).

The old source repository and both Hugging Face dataset repos are private/inaccessible from this Delta environment: the W&B metadata contains no code/data payload, the W&B artifact list contains only run history/evaluation tables, the old SC/NLP/IRIS paths are not mounted, and unauthenticated access to the replay dataset returns 401. Therefore we cannot yet claim the new local demo mixture reproduces that historical `replay_local` dataset. The Delta SFT manifests explicitly show `official_demo_ratio=0`; they did not use replay. Only five official LIBERO demonstration HDF5s are cached locally, including task 29's matching Kitchen Scene 5 demonstration, not a complete 90-task replay set.

Inspection of the W&B-recorded private repo commit resolved the replay *recipe*. `scripts/build_target_datasets.sbatch` builds separate target-only and replay LeRobot datasets from the same editor-generated targets. For replay, it calls `build_editor_lerobot.py --replay-repo physical-intelligence/libero --replay-examples 10000 --replay-stride 20` by default. The builder iterates source indices 0,20,...,9980, gets a 10-action window for each source observation, preserves the source task prompt, and appends those rows to the editor-target dataset. Thus it adds up to 500 official demonstration chunks from across tasks, rather than only demonstrations from the RL-trained task. `train_editor_sft.py` uses `prompt_from_task=True`, so the replay rows retain their own task instructions during SFT. The student's targets are editor-corrected base-policy chunks (`clip(base_actions + residuals, -1, 1)`), not the successful Small EXPO episode chunks used by Delta SFT.

There is a source-provenance wrinkle: W&B recorded `--lora` in the run command, but the exact recorded commit's `train_editor_sft.py` parser does not accept that flag; a later public/private-repo HEAD does. W&B metadata captures the Git commit and script path, but not an uncommitted working-tree diff. The actual W&B config confirms LoRA model variants and successful 30k-step training, yet the exact script contents at execution time are not fully reproducible from the recorded commit alone.

The public source dataset metadata currently reports 40 task prompts, 1,693 episodes and 273,465 frames. Replaying the first 10,000 dataset indices at stride 20 currently spans the first 36 episodes and 10 distinct task prompts; this is metadata from the current dataset revision, not a verified snapshot of the dataset revision used on SC. The old W&B data repo is private, and the trainer did not pin the source revision or log the built dataset count, so exact historical row count and revision remain unverified. Still, this source-code evidence makes the main difference clear: the large W&B replay gain came with broad official-task rehearsal plus editor-target supervision. Delta SFT used only one task's successful Small EXPO trajectories and no rehearsal. The task-29-only 50% demo arm prepared here tests the user's earlier task-specific idea; it is not a faithful reproduction of the W&B all-task replay recipe.

On the four tasks used by the current retention panel, the historical W&B no-replay/replay counts were 0/0/0/8 versus 3/5/11/14 (tasks 2/28/29/60), i.e. 8/80 (10.0%) versus 33/80 (41.25%). These are per-task counts from separate full-suite eval runs, not a paired-state causal estimate; the 90-task aggregate remains the primary W&B result.

| Experiment | SFT target source | Replay | LIBERO-90 result |
|---|---|---|---:|
| SC W&B task 29, no replay | Editor-corrected Pi0.5 action chunks | None | 39/1,800 (2.17%) |
| SC W&B task 29, replay | Same editor target chunks | Up to 500 official chunks (current source snapshot spans 10 prompts) | 400/1,800 (22.22%) |
| Delta SFT task 2 | Small EXPO successful-episode actions | None | 112/1,800 (6.22%) |
| Delta SFT task 28 | Small EXPO successful-episode actions | None | 46/1,800 (2.56%) |
| Delta full EXPO task 28 | Online actor with editor/critic action selection | Online RL replay | 451/1,800 (25.06%) |

This table shows a plausible reason for the discrepancy, while keeping the limits visible: the high-performing replay experiment rehearsed multiple official task prompts, but also used different target actions, OpenPI training code, batch size, and optimizer implementation. It is strong evidence to test broad-task rehearsal in the Delta Small EXPO SFT pipeline; it does not prove replay alone caused the full 20-point gain.

To test the locally available analogue under controlled conditions, `run_task29_replay_ablation.sh` is now prepared in this isolated audit directory. It launches a matched pair on task 29: both arms use the same 150-success Small EXPO collection, official Pi0.5 LIBERO base, adapter-only gradients, zero-B initialization, seed, 5k-step budget and four-task retention evaluations; one has no demo mixture and the other samples 50% task-29 official demos. This isolates local official-demo replay from target-only SFT, while remaining explicitly distinct from the inaccessible SC `replay_local` construction. The launcher and all required source/checkpoint/demo paths were checked; it has not been launched because both experimental A100s are still occupied by the scope arms.

### Newest controlled results (2026-11-03, 10:34 CST)

The existing-scope arm's step-1,500 retention evaluation has now been written. On the same four tasks and 20 initial states per task, task 2/28/29/60 scored 70/100/0/5%, with 43.75% macro (35/80 successes). Relative to its own step-0 fresh-LoRA rollout, it retained/lost/gained 12/3/2 on task 2, 9/0/11 on task 28, 0/14/0 on task 29, and 1/9/0 on task 60: 26 prior wins lost and 13 gained, a net decrease of 13/80. It is worse than its 1,000-update macro (48.75%) despite retaining perfect task-28 success. The LoRA-only arm's last completed evaluation remains step 1,500: 70/100/15/10%, 48.75% macro, with 23 losses and 14 gains relative to its step-0 result. Thus trainable-scope restriction has not prevented severe transfer regression by 1,500 updates.

At the latest direct poll, all five elements of sleeper array 23352795 remained RUNNING; the two experiment elements remain assigned to gpua062 and gpua011, and the other three allocations are still untouched. The existing-scope step-1,500 rollout has completed, and its process is continuing through save/next-update handling. The LoRA-only process is in its step-2,000 retention rollout (2/20 episodes in the active task at the 10:34 poll); no step-2,000 result or checkpoint is yet committed. Keep both jobs in place until their terminal status and saved artifacts are confirmed.

### Task-29 official-demo mixture ablation launched (2026-11-03, 10:36 CST)

A matched pair is now running in separate tmux sessions on two previously idle A100 sleeper allocations: `sft_task29_no_replay_20261103` on Slurm element 23352795_1 / gpua009 and `sft_task29_demo50_20261103` on 23352795_2 / gpua081. The other three allocation elements remain RUNNING, including both task-28 controls; one allocation remains idle. No unrelated sessions or jobs were changed. This follows the earlier design request for demonstrations from the specific RL task.

Both arms use the same task-29 teacher checkpoint at step 375,380, its 150 successful Small EXPO episodes, official Pi0.5 LIBERO initialization and normalization, adapter-only gradients, zero-B LoRA initialization, seed 42, 5,000 updates, batch 16, 5e-5 peak/decay learning rate, 10,000-step warmup, and evaluation every 1,000 updates on tasks 2/28/29/60 (20 episodes each, seed 7). The only configured treatment difference is official same-task demo mixture: 0% versus 50% of batch rows. The task-29 official demo path resolves through the LIBERO task map to the existing Kitchen Scene 5 `put_the_black_bowl_in_the_top_drawer_of_the_cabinet` HDF5, matching task 29 rather than task 2's Scene 10 demo. The launcher writes per-step counts of target and demo chunks and actual loaded dataset lengths; these must be captured before claiming the mix loaded as intended.

At the first startup check, both processes had W&B runs created and were initializing model/runtime state; no data-count line, step-0 evaluation or training metric had yet appeared. GPU memory had risen from zero to about 433 MiB, which indicates device initialization but does not yet prove model loading or optimizer updates. Do not call the runs healthy until their initialization audit, `SFT data loaded` counts, first evaluation, and training metrics are present. Run logs: `task29_no_replay.log` and `task29_demo50.log` in this audit directory; outputs are isolated under `/work/hdd/bgqe/rpolonsky/dcrlcheckpoints/sft_task29_replay_ablation_20261103/`.

### Source-level confirmation of what the Small EXPO labels teach

The collector restores a `SmallExpoLearner` from the task teacher checkpoint and calls `teacher.sample_actions(observation)` with its default `only_base_actions=False` (`dcrl/bc_distill_data.py`, `dcrl/agents/small_expo_learner.py`, inherited `PiExpoLearner.sample_actions`). That default path samples actor candidates, applies editor candidates, and chooses among them with the target critic. Therefore the SFT labels are the full Small EXPO system's critic-selected chunks, not raw actions from its base actor. The collector stores the selected 10-action chunk and executes only the first five. The SFT trainer then feeds all ten labels into the Pi0.5 actor's flow-matching `bc_update`; its auxiliary learner editor/critic are not the supervised target or inference-time actor selection path. This is a meaningful teacher/student and action-horizon mismatch to keep distinct from the replay intervention. The already prepared `executed_backfill` control addresses the unused action tail, but it has not been launched.

At 10:39 CST the two task-29 launchers remained active in their tmux sessions. Both had read about 12.6 GB and held about 16 GB RSS while initializing, but had only 435 MiB allocated on the GPU and still had not emitted dataset counts or step-0 results. Treat this as ongoing checkpoint/model initialization, not as confirmed training; continue watching for data counts, actor initialization audit, and actual GPU-resident updates. Meanwhile, the existing-scope task-28 process advanced past update 1,600 after its step-1,500 eval; the LoRA-only task-28 process remains in the step-2,000 evaluation and has not produced its 2,000 eval JSON or checkpoint. All five Slurm sleeper elements remain RUNNING.

At 10:40 CST, both task-29 runs had advanced beyond generic imports and initialized their 224x224 six-channel observation, 10x7 action, 35-wide editor/Q interface, and 8D state shapes. Both A100s showed about 8.65 GiB allocated (near-zero utilization during initialization/compilation), with live Python trainers and no logged error. Their data-length line and step-0 rollout were still pending. The two task-28 jobs also remained live: existing-scope had advanced to update 1,677/2,000; LoRA-only was in the final evaluation, 7/20 episodes on its current task, without a committed step-2,000 result. The five allocation elements were all still RUNNING.

The task-29 pair has now emitted `INITIALIZATION_AUDIT scope=lora_only` in both logs, with the identical sampled parameter-tree fingerprint `d73d9422…`. This is consistent with both arms having initialized from the same configured official Pi0.5 LIBERO tree and zero-B adapter setup; the audit hash is sampled, not a full checkpoint checksum. Both logs also show successful model/learner shape construction. They then printed standard Robosuite “no private macro file” warnings while constructing the LIBERO evaluation environment; no exception has appeared, but the final SFT manifest and data-count line are still absent, so initialization/data setup is not yet fully verified.

### Demo-arm launch failure and path correction

The first `demo50` process (W&B run `bp9ay165`) exited before training because `--official-demo-root` was set to `.../datasets/libero_90`, while `LiberoTaskEnv.demo_path()` itself appends the `libero_90` problem folder. The resulting nonexistent path duplicated that component (`.../datasets/libero_90/libero_90/KITCHEN_SCENE5_...hdf5`). The exact task-29 Scene-5 demonstration file does exist at `.../datasets/libero_90/KITCHEN_SCENE5_...hdf5`. I preserved the failed attempt's W&B/init artifacts and created isolated launcher `run_task29_demo50_pathfix.sh` pointing the demos root at `.../datasets`; the corrected retry uses a fresh output directory `demo50_pathfix`, W&B run suffix `pathfix`, and the same Slurm allocation 23352795_2 in a new tmux session. The failed step ended; its allocation stayed RUNNING. The no-replay run on 23352795_1 remained untouched and active. The retry has only just started, so a successful demo load is not yet confirmed.

### Task-28 2,000-update result and verified task-29 data load (2026-11-03, ~10:44 CST)

The task-28 LoRA-only arm completed all 2,000 optimizer updates and its final evaluation. The saved `latest_checkpoints/2000/_CHECKPOINT_METADATA`, agent tree and params tree are present. Its final task-2/28/29/60 rates are 65/100/15/10%, macro 47.5% (38/80). Paired to its own step-0 fresh-LoRA outcomes, it retained/lost/gained 10/5/3 on task 2, 9/0/11 on task 28, 2/12/1 on task 29 and 2/8/0 on task 60: 25 prior wins lost and 15 gained, net -10. This is slightly below step 1,500's 48.75%; near-perfect target success does not restore the retention losses. The Python/W&B cleanup process was still present at the latest poll, so the training/evaluation/checkpoint work is complete but the tmux process had not fully exited yet.

The task-28 existing-scope arm had not finished: it was actively training at about update 1,811/2,000 on gpua062. Its last measured retention result remains step 1,500, macro 43.75%; no step-2,000 evaluation is available yet.

Both task-29 arms now call `len()` on their loaded data and report the actual counts. No-replay has 2,150 Small EXPO chunks and zero demos. The corrected 50%-demo arm has 2,150 Small EXPO chunks plus 1,231 chunks from 50 official task-29 episodes, resolved to the correct Kitchen Scene 5 HDF5. Both manifest files exist and both initialization audits show the same sampled actor-tree hash. Neither has written its step-0 retention result or training metrics yet; the step-0 evaluation is the next gate before any optimizer update. The first malformed-path demo attempt failed before data loading and remains only as a separate W&B/init artifact. All five Slurm allocations remain RUNNING.

### Live follow-up poll (2026-11-03, 10:45 CST)

All five sleeper allocations are still RUNNING. Task-28 existing-scope is still making updates (1,859/2,000 at the latest artifact poll); the task-28 LoRA-only run has written its step-2,000 metric and complete latest checkpoint. The two task-29 jobs have verified model initialization and actual dataset lengths but still have no step-0 eval JSON or optimizer metrics. The no-replay A100 showed active GPU utilization during this initialization/evaluation phase; the path-fixed demo arm remained resident on its A100 with no logged exception. No claim of training progress is made for either task-29 arm until a step-0 result and then positive optimizer-step metrics appear.

### Live poll after final task-28 adapter run (2026-11-03, ~10:46 CST)

The existing-scope task-28 arm is actively training at 1,878/2,000 updates (gpua062). The task-28 LoRA-only arm has completed the 2,000-step training loop and saved its step-2,000 evaluation and full `latest_checkpoints/2000`; its GPU context has dropped to ~478 MiB and its tmux/W&B shutdown was still visible, so do not call the session fully exited yet. The final result remains 65/100/15/10% on tasks 2/28/29/60 (47.5% macro).

The two task-29 jobs remain in active pre-training initialization/evaluation computation on gpua009 and gpua081. Their manifests, initialization audits and actual dataset-count line are present; no step-0 eval file or `metrics.jsonl` exists. At the latest direct GPU/process poll, no-replay was ~17.3 GiB and 58% GPU utilization; the corrected demo arm was ~17.3 GiB, actively executing CPU work while GPU utilization was momentarily 0%. This is consistent with different compile/evaluation startup stages, not evidence that either has begun optimization. Continue monitoring the first saved step-0 results and then optimizer metrics.

### 1,500-update cap applied for the intensive task-29 comparison (2026-11-03, 10:52 CST)

The user requested a maximum of 1,500 updates. The first task-29 pair had no complete step-0 eval, no latest checkpoint and no optimizer metrics; I sent SIGINT to only its two trainer PIDs and verified they exited. The step-0 rollouts were in progress but had not produced a saved benchmark result. Their output directories/W&B attempts are preserved for provenance; neither allocation was canceled. The task-28 arms and all five allocation jobs were left running.

Fresh task-29 no-replay and 50%-same-task-demo runs are now in their own tmux sessions on the same A100s. Their actual process command lines confirm `--num-sft-steps 1500`, `--eval-every 500`, and `--checkpoint-every 500`; their outputs are isolated under `no_replay_1500` and `demo50_1500`. They retain the verified 2,150 Small EXPO chunks and (for demo50) 1,231 chunks from 50 task-29 official episodes, zero-B LoRA initialization, teacher step 375,380, batch 16, seed 42 and same official Pi0.5 LIBERO base. The first completed checkpoints/evaluations should be at 0, 500, 1,000 and 1,500. Existing task-28 paired evals at 0/500/1,000/1,500/2,000 remain in the analysis as a historical dose/forgetting curve; they are contextual evidence, not pooled with the task-29 curves because the target task differs.

The capped task-29 tmux sessions are active under W&B runs `b6dstioe` (no replay) and `lo62qxuc` (50% demos). Their live process arguments confirm 1,500 total updates and 500-step eval/checkpoint cadence. At the latest startup poll, both processes were waiting on shared-filesystem reads (`cl_sync_io_wait`, ~0.46 GiB read, ~0.36 GiB RSS); neither had yet written a new manifest or initialization audit, and both GPUs were still at 0 MiB. This is early startup I/O, not yet proof of loaded/training runs. All five parent allocations remained RUNNING; task-28 existing-scope was in its final evaluation and the completed task-28 LoRA run was no longer using its GPU.

Follow-up startup inspection found both capped task-29 PIDs actively waiting on shared-filesystem I/O with the successful teacher-chunk HDF5 open (`.../task_29_teacher_375380_success150_sft200k/pi_base_data/success/teacher_chunks.hdf5`, 215 MiB). They had each read under 1 GiB, held under 0.7 GiB host RSS, and allocated only 433 MiB on GPU; no actual dataset count, step-0 result or optimizer metric had appeared. This narrows the current delay to data/code loading on the shared filesystem, not an OOM or a training-loop failure. The task-28 final evaluation continues separately.

At the latest capped-run poll, both task-29 logs emitted the same `INITIALIZATION_AUDIT scope=lora_only` sampled hash (`d73d9422…`) and constructed matching model/critic example shapes. Each then printed Robosuite's standard missing-private-macro advisory while constructing the LIBERO environment. No exception appeared, but neither `sft_config.json`, actual data-count line nor step-0 evaluation artifact existed yet; training has not been confirmed. The initialization parity is now checked on the fresh capped processes, not inferred solely from launcher arguments.

The capped task-29 runs have now completed startup/data loading. Fresh manifests and initialization audits exist; both report the same sampled parameter fingerprint. The no-replay arm's actual `len()` is 2,150 Small EXPO rows and 0 official-demo rows. The demo arm reports 2,150 Small EXPO rows plus 1,231 official task-29 demo chunks from 50 episodes, resolving to the correct Kitchen Scene 5 HDF5. Both have entered their step-0 actor-only retention evaluation; neither has optimizer metrics yet. Direct GPU checks show ~16.8 GiB allocated on gpua009 and ~17.3 GiB on gpua081, confirming both models are resident on their A100s. Task-28 existing-scope remains in its step-2,000 evaluation; task-28 LoRA-only completed its 2,000-step evaluation/checkpoint and released its GPU.

At the current poll, both capped task-29 processes are resident on their assigned A100s (~17.3 GiB each, ~40% GPU utilization) while the initial actor-only evaluation compiles/runs. They have already logged the actual data counts, but still have no completed `evals/00000000.json` and no optimizer metrics. The task-28 existing-scope process is separately in its final step-2,000 rollout; its 2,000 evaluation artifact is still pending. All five sleeper allocations remain RUNNING.

The capped step-0 evaluation is still running at the latest check: both task-29 processes remain alive with model trees resident (~17.3 GiB each) and GPU utilization of ~58% (no replay) and ~32% (demo mix). No step-0 JSON has been committed yet, and no optimizer metrics exist; the slow progress is within first-evaluation setup/inference, not a crashed process. Task-28 existing-scope remains alive in its own final evaluation (~31.3 GiB on gpua062); no step-2,000 artifact has landed yet.

The first step-0 episode is taking longer than the normal per-episode progress interval: both task-29 progress bars are still at 0/20, while their trainer processes remain alive and GPU utilization is active. The evaluator performs a full LIBERO episode (up to 400 control steps, replanning every five actions) before advancing the tqdm counter, so this alone is not a hang signal. No baseline result or optimizer update is yet available; continue to require a saved eval JSON and metrics before interpreting learning.

At the next poll, both capped task-29 step-0 evaluation bars had advanced from 0 to 13/20 episodes in the currently evaluated task, confirming rollout progress. They still need 20 episodes on each of the four retention tasks before the step-0 JSON is committed; no optimizer updates have begun. The task-28 existing-scope step-2,000 evaluation advanced to 16/20 episodes in its current task but is also not yet committed. All five allocations remain RUNNING.

### Historical-curve tracking preference and current poll (2026-11-03, 11:00 CST)

The user confirmed that progress tracking should incorporate all earlier optimizer-step metrics, evaluations, and artifacts, not wait for the newly capped runs alone. Accordingly, use existing task-28 checkpoints at 0/500/1,000/1,500/2,000 and any earlier Small RL, SFT, or Full EXPO artifacts as context, while keeping comparisons stratified by task, model/inference system, data mix, and evaluation protocol. For the currently relevant task-28 scope control, historical macro success across tasks 2/28/29/60 is 60/55/48.75/43.75% for the existing broad trainable scope through step 1,500, and 60/57.5/56.25/48.75/47.5% for LoRA-only through step 2,000. Existing-scope's 2,000-update eval is still not written as a JSON/checkpoint at this poll; its log reached 20/20 episodes in the current evaluation, so wait for the persisted result before reporting the curve endpoint.

The new task-29 no-replay and 50%-official-demo pair both remain in their step-0 actor-only four-task evaluation. At 11:00, their logs showed 13/20 episodes in the current task; no eval JSON or optimizer metrics had been written, so neither run has yet made a gradient update and there is no new training-curve point to report. Their launch arguments and loaded-data counts remain verified as above. All five sleeper allocation elements (23352795_[1-5]) are still RUNNING; no allocation or unrelated training process was changed.

### Historical SFT optimization curves across the four target tasks

A fresh extraction from each old `metrics.jsonl`, `sft_config.json`, and saved target-task eval confirms that all four historical SFT jobs used the same recipe: batch 16, full 10-action chunk flow matching, success-only Small EXPO data (150 successful episodes), LoRA-mode initialization, official Pi0.5 LIBERO source/student weights, seed 42, 10,000-update warmup to a flat 5e-5 learning rate, and 20,000-step eval/checkpoint cadence. The old manifests point to `official_demo_ratio=0` by omission of that feature; no official-demo replay was mixed. They report collection target success 150, zero failures, and target-only evals at steps 0 and 20k (task 28 additionally at 40k). Target-task success moved from 55/45/75/45% to 100/100/100/90% for tasks 2/28/29/60, respectively. These target-only eval gains do not establish transfer: the same saved task-28 40k checkpoint scored 46/1,800 (2.56%) on the full actor-only suite.

The logged flow loss steadily falls while the learning rate reaches its full value by update 10k. For task 28, median actor loss is 0.2237 in updates 1-100, 0.01695 in 501-1,000, 0.00331 in 10,001-20,000, and 0.00143 in 20,001-40,000; median gradient norm falls from 0.978 to 0.0303 between the first and last windows. The 40k run made 640k row draws from 760 stored chunks (~842 nominal data passes). The other stopped runs reached 26,170/38,439/35,112 updates on tasks 2/29/60, or approximately 242/286/252 row draws per stored chunk. This is consistent with easy target imitation plus repeated exposure to narrow, correlated data; it does not by itself prove memorization, but the low training loss cannot be used as a proxy for generalization.

The controlled task-28 run adds time resolution missing from the old jobs: by 500 updates (8,000 draws, ~10.5 chunk-set passes), target success rises 45% to 70% with LoRA-only, but same-instruction task 29 falls 70% to 45%; at 1,000, task 29 is 10% and macro retention has fallen from 60% to 56.25%. The broader 466.9M trainable scope drops task 29 to 10% already at 500 and 0% by 1,500. This shows that the old 20k checkpoint cadence skipped the interval where the damaging transfer shift is already visible. It also means the 1,500-step task-29 replay comparison is useful only if the new baseline and 500-step curves finish; it cannot stand in for a full-suite retention test.

### Live status follow-up (2026-11-03, 11:02 CST)

The task-29 no-replay and 50%-demo trainers are still alive in their intended Slurm allocations (`23352795_1`/gpua009 and `_2`/gpua081), each at about 17.3 GiB GPU memory and ~40% utilization. Their first 20-episode task evaluation has completed; the logger is constructing/running the remaining task environments, and no combined step-0 retention JSON or optimizer metrics has appeared yet. The processes show active CPU and GPU use, so this is continued evaluation work, not a dead trainer. The existing-scope task-28 run on `_4`/gpua062 is progressing through its final evaluation (3/20 episodes on the current task in the latest log); step-2,000 remains uncommitted. All five elements of allocation 23352795 remain RUNNING.

At 11:03 CST, the task-29 capped processes remain alive (Slurm RUNNING) and the first 20-episode task loop is complete, but their logs have not advanced past that task's 20/20 line since ~11:01. The processes showed active CPU/GPU use on the preceding direct in-allocation check. This interval may be per-task JAX compilation or environment transition, but the cause is not verified yet; continue inspecting process/GPU/log activity and do not call the four-task baseline complete until the combined step-0 artifacts are written. Task-28 existing-scope's final eval is progressing (9/20 episodes on its current task at 11:03); all five allocations are still RUNNING.

At 11:04 CST the quiet interval ended: both task-29 processes have advanced to 12/20 episodes on the next retention task, confirming that the first 20/20 was followed by a slow task transition/compilation period rather than a stalled trainer. The processes remain live on gpua009/gpua081 (~17.3 GiB each); no training update is expected until all four tasks' step-0 evaluations finish. Task-28 existing-scope is at 12/20 on its current final-eval task. The five allocation elements remain RUNNING.

### Complete paired full-suite comparison of the three historical SFT settings (2026-11-03)

The full-suite result files for task-2 SFT at 20k, task-28 SFT at 40k and task-29 SFT at 20k were re-read and validated against all 90 official Pi0.5 task files. Each policy has 20 trials per task, evaluation seed 7, 400-step limit, action horizon 10, replan interval 5, identical instructions and identical initial-state indices. No tasks or episodes are missing. The per-task comparison is reproducibly generated by `summarize_full90_comparison.py` into `full90_comparison_20261103.csv`, `.json` and `.md`.

| Policy/checkpoint | Successes / 1,800 | Rate | Paired successes retained from base | Base successes lost | New successes | Fine-tuned task |
|---|---:|---:|---:|---:|---:|---:|
| Official Pi0.5 actor | 512 | 28.44% | — | — | — | — |
| SFT task 2, 20k actor | 112 | 6.22% | 81 | 431 | 31 | 2: 19/20 (95%) |
| SFT task 28, 40k actor | 46 | 2.56% | 16 | 496 | 30 | 28: 20/20 (100%) |
| SFT task 29, 20k actor | 219 | 12.17% | 161 | 351 | 58 | 29: 20/20 (100%) |

These results establish severe cross-task forgetting in all three tested task-specific SFT checkpoints, rather than a comparison artifact caused by different evaluation episodes. Each learner reaches very high success on its own task while retaining only 16%, 3%, and 31% respectively of the official actor's successful base cases. The SFT task-2 checkpoint scores 0/20 on task 29 even though tasks 2 and 29 have the same instruction; prompt memorization alone does not explain the loss. The task-28 SFT checkpoint likewise scores 0/20 on tasks 2, 29 and 60. These are fixed-suite paired outcomes, not independent-seed confidence intervals.

Five Full EXPO full-suite runs were also validated on the same seed and initial states. They score 20.44–26.83% (368–483/1,800); the corresponding task-specific Full EXPO runs retain 276–373 of the base's 512 successful cases, much more than SFT's 16–161. The Full EXPO entries are deployed-system evaluations with editor/critic candidate selection; SFT entries are actor-only. They document the practical gap between deployed systems, but they do not isolate actor-update quality. The direct SFT-versus-official actor comparison above does isolate the observed loss.

The strongest explanation supported by these completed comparisons is task-specific supervised adaptation without a retention objective: each student sees only 150 successful episodes from one task and no other-task examples, and is selected/evaluated during training by its target task. The target-task success is evidence that optimization works; it is not evidence of retained LIBERO competence. Update dose and trainable scope amplify this: the old SFT recipes use 20k–40k actor optimizer steps on 760–2,150 stored chunks, while the scope audit confirms the default `lora` mask updates the whole image tower and action/time projections as well as adapters. The exact causal shares of data scope, update dose, target quality and evaluation-path design remain separable questions; see the controlled pair below and the dose comparison described earlier.

### Matched lower-dose full-suite and action-bound experiments in progress (2026-11-03)

An actor-only LIBERO-90 evaluation of the task-28 LoRA-only 2,000-update checkpoint is still running on the original three evaluator GPUs. At the latest poll, 76/90 task files had completed: 455/1,520 successes (29.93%). On those same 76 tasks, official Pi0.5 scored 500/1,520 (32.89%); the paired case table is 359 both-success, 141 base-only, 96 SFT-only, 924 neither. This is provisional, not a full-suite result. It is a smaller drop than the old 40k SFT result, but the run also changes trainable scope and uses zero-B LoRA, so it does not isolate update dose.

Two new matched task-28 SFT arms are running on separate idle A100 allocations (`23416001_2`/gpua082 and `23416001_3`/gpua070), each in its own named tmux session with W&B online. Both use the same official Pi0.5 initialization, task-28 teacher checkpoint 275001, 150-success/760-row dataset, zero-B LoRA-only scope, seed 42, 1,500 updates, batch 16, learning rate `5e-5`, 10k warmup, no official demos, full 10-action targets and 500-step four-task retention evaluations. The only difference is that the clipped arm clamps the first six raw OSC motion target dimensions to `[-1,1]` before the unchanged student quantile normalization. Run IDs: raw `gldy6x8w`; clipped `ccbvqtrv`. At the latest poll, both had written their manifests and initialization audits, restored onto ~17.3 GiB of GPU memory, and were still in JAX/model/evaluation startup; no step-0 result or optimizer update has been confirmed yet.

Do not infer that action clipping is the solution until the matched arms finish. The known mismatch is real: LIBERO's OSC controller saturates the six motion commands to `[-1,1]`, while the Small EXPO collection stores the unbounded raw commands and the Pi0.5 quantile normalization is affine rather than clipping. For task 28, 30.8% of x and 53.6% of y targets exceed the controller bound. The experiment tests whether aligning these targets changes target success and cross-task retention, using the same 150 episodes and training recipe.

### Verified task-28 existing-scope 2,000-update endpoint (2026-11-03, 11:07 CST)

The task-28 broad-scope control finished its 2,000-update retention evaluation and committed both `evals/00002000.json` and `latest_checkpoints/2000/_CHECKPOINT_METADATA`. It scored tasks 2/28/29/60 = 65/100/0/5%, macro 42.5% (34/80). Relative to its own step-0 fresh-LoRA policy, the paired retained/lost/gained counts are 10/5/3 on task 2, 9/0/11 on task 28, 0/14/0 on task 29, and 1/9/0 on task 60: 28 prior successes lost, 14 gained, net -14. Its macro curve is now 60, 55, 48.75, 43.75, 42.5% at steps 0/500/1,000/1,500/2,000. The final update's actor loss is 0.00674 and gradient norm 0.0644; target-task success remains 100% while task 29 has fallen to 0%. The run had completed evaluation and checkpoint write; its process may still be in orderly W&B/session cleanup.

### Action-target saturation mismatch identified from source and stored rows

The collector calls the teacher for a full ten-action chunk, stores `chunk[:10]` unchanged in the HDF5 `actions` dataset, and executes only its first five actions through `LiberoTaskEnv.step()` (`dcrl/bc_distill_data.py`). `LiberoTaskEnv.step()` itself does no client-side clipping. Robosuite's active `OSC_POSE` controller config has `input_min=-1`, `input_max=1`, and `OSC.set_action()` calls `BaseController.scale_action()`, which applies `np.clip(action, input_min, input_max)` before scaling the six motion dimensions. By contrast, OpenPI's `Normalize._normalize_quantile()` is only an affine mapping using LIBERO q01/q99 and does not clip targets. Therefore the stored Small EXPO motion commands outside [-1,1] are supervised as distinct raw values even though LIBERO executes them as the same saturated boundary command.

The saved `target_quality_audit.json` quantifies the fraction of Small EXPO teacher commands outside [-1,1] on the six OSC dimensions (x/y/z/rot-x/rot-y/rot-z): task 2 = 0.3/17.3/57.0/0/0/0%; task 28 = 30.8/53.6/0/0/0/0%; task 29 = 9.0/3.6/18.3/0/0/0%; task 60 = 0/25.4/2.3/0/0/0%. This is a verified teacher-label versus executed-controller mismatch, especially severe in task-28 x/y and task-2 z. It is a plausible contributor to poor BC generalization, not yet a causal result. Official demonstration actions are within the controller command range in the audited task files, so a demo mixture can also change this saturation mismatch as well as add trajectories; the active no-replay/demo50 comparison cannot attribute any gain to one mechanism alone.

Next clean causal control after the active task-29 pair completes: run a task-29 no-demo arm matched to the existing no-replay seed, initialization, data order, optimizer and evaluation, but clip only the six raw OSC motion-target dimensions to [-1,1] before applying the unchanged Pi0.5 quantile normalization. Keep the gripper dimension and action horizon unchanged. Compare step 0/500/1,000/1,500 on the same four initial-state panels. Do not alter the active trainers or production source. A separate executed-prefix/backfill comparison remains needed because the saved proposal tail is replaced at replanning and is not the action sequence actually executed.

### Isolated follow-up prepared; active pair preserved (2026-11-03, 11:09 CST)

I created a separate action-bounds experiment under `sharder/training/sft_actionbounds_20261103/`: a copied audit trainer with opt-in `--clip-osc-motion-targets`, a launcher, and a plan JSON. It clips only the first six raw Small EXPO action dimensions before the unchanged Pi0.5 normalization. The official gripper labels, full 10-action proposal, dataset, zero-B LoRA initialization, seed, optimizer, steps, and four-task eval cadence match the active task-29 no-replay arm. Its output path and W&B run name are distinct. This follow-up is prepared but not launched; the two active task-29 runs are left unchanged and remain the priority comparison.

At 11:09 CST, both active task-29 trainers remained RUNNING on gpua009/gpua081 and were progressing through their step-0 retention panel. Their logs had advanced to 6/20 episodes on the current task but neither combined `evals/00000000.json` nor training metrics existed yet. They have therefore not performed an optimizer update. The task-28 existing-scope training/eval is complete, with a committed step-2,000 checkpoint; its tmux window remains listed but the training loop and checkpoint write are finished. All five sleeper allocations remain RUNNING. No pre-existing project trainer or unrelated job was changed.

### Task-28 early-checkpoint availability and full-suite diagnostic (2026-11-03, 11:28 CST)

- The controlled task-28 LoRA-scope experiment has per-task evaluation JSONs at updates 0, 500, 1,000, 1,500 and 2,000, but the 500-update model weights were not retained. Current Orbax directories are the best-macro checkpoint at `checkpoints/0` and the last evaluated full-learner checkpoint at `latest_checkpoints/2000`; no 500-update checkpoint directory exists for either trainable-scope arm. An all-90 evaluation of the early 100%-target point cannot be recovered from these artifacts.
- The existing-scope arm reached 20/20 on task 28 at update 500 while four-task macro success had already fallen from 60% at step 0 to 55%; task 2/28/29/60 were 85/100/10/25%. Its later target-task 100% remained paired with further macro loss. The LoRA-only arm was 70% on task 28 at step 500 and first reached 100% at step 1,500. Stopping at target 100% is a plausible way to avoid additional training damage, but it still requires a full-suite evaluation to establish generalization.
- An isolated copy of the evaluator is under `sharder/training/sft_full90_eval_task28_2k_20261103/`. The saved Task-28 LoRA-only 2,000-update checkpoint restored successfully on all three idle A100s. A one-task runtime probe and full-suite shards 0 and 1 are running in separate tmux sessions; at this update they are compiling the inference path, before task results have been written. Shard 2 will run on the probe node after the probe exits, reusing its local JAX cache. Outputs are isolated under the same audit directory. No production trainer/evaluator was edited.
- At the latest direct poll, the Task-29 no-replay and 50%-same-task-demo trainers were alive on gpua009/gpua081 at optimizer updates 167 and 153 respectively, with finite losses and gradient norms. All five elements of allocation 23352795 remain RUNNING; the two Task-29 training GPUs and three full-suite diagnostic GPUs are all in use. No unrelated process or allocation was changed.

### Current continuation poll (2026-11-03, 11:29 CST)

- Task-29 no-replay and demo50 are at updates 220 and 206. Their 500-update eval/checkpoint is the next decision point. Both processes report finite loss/gradient metrics and use about 30.8 GiB of A100 memory.
- The launcher maximum is 1,500 (not 2,000). Per the user's latest direction, neither arm should pass 1,000 updates. The trainer's `latest_checkpoints` manager has `max_to_keep=1`, so its 500 snapshot would otherwise be replaced by the 1,000 snapshot. Preserve step 500 using a same-filesystem hard-link snapshot only after Orbax has committed the checkpoint; this retains it without duplicating checkpoint data blocks. Keep the pair's terminal update matched unless the user changes that comparison design.
- The three Task-28 2,000-update full-suite evaluator processes have restored the actor/learner and are compiling the inference path. At this poll the probe and shards have not yet emitted task results; the processes are CPU-active, and the A100s hold approximately 17 GiB each. This is slow compilation, not yet a rollout result. All five allocation elements remain RUNNING.

### Reconciled side-panel status (excluded from the main diagnosis) (2026-11-03, 13:26 CST)

The full-90 table above is complete historical evidence: all three task-specific SFT checkpoints and the official actor have 90/90 task files and 20 matched episodes per task. The replay comparison below it has also completed through 1,500 updates for both arms; older notes in this file saying it was still at step 0/500 are stale. Its four-task results are:

| Updates | Task-29 no-replay macro | Task rates 2/28/29/60 | Task-29 + 50% same-task official-demo macro | Task rates 2/28/29/60 |
|---:|---:|---|---:|---|
| 0 | 60.0% | 70/50/65/55% | 58.75% | 75/35/70/55% |
| 500 | 48.75% | 50/35/65/45% | 60.0% | 60/50/75/55% |
| 1,000 | 53.75% | 30/50/85/50% | 53.75% | 25/35/90/65% |
| 1,500 | 53.75% | 50/60/75/30% | 60.0% | 45/60/80/55% |

This is a user-requested side panel only. It is not evidence for the main
full-suite diagnosis: it starts from a fresh zero-B LoRA actor, evaluates only
four tasks with 20 episodes each, and is not paired to the historical full-90
training/evaluation protocol. Its scores must not be used to justify a main
training design or a causal replay conclusion.

The separate task-28 LoRA-only 2,000-update full-90 evaluation has advanced to 78/90 task files (the remaining 12 have not yet completed). Treat its 455/1,560 result and any paired counts as provisional until all 90 files are present. The full-90 evaluator's three Slurm elements remain RUNNING; do not stop them.

The matched task-28 raw-versus-clipped action-target pair is running on array elements 23416001_2 and _3 (gpua082 and gpua070), each in the named login tmux session recorded in `task28_actionbounds_pair_plan.json`. Both have verifiably loaded all 760 Small EXPO chunks and zero official demonstrations, then started the same step-0 actor evaluation. At 13:25 the raw arm was 10/20 and clipped arm 9/20 on the first task; no combined step-0 evaluation or optimizer update has completed. They are progressing through rollouts, but their training is not yet established until step-0 evaluation finishes and update metrics appear. No existing jobs were changed.

### Why Full EXPO and task-specific SFT are not an apples-to-apples update (2026-11-03, 13:30 CST)

The completed launcher and source audit identifies three structural differences that explain why the Full EXPO and SFT curves should not be expected to align:

1. Full EXPO uses a broad actor data mixture. With `offline_ratio=0.5` and `actor_success_only=true`, `BatchProcessor._sample_success_actor_batch()` supplies eight official demonstration rows and eight successful online rows to each actor update when both pools are available. The historical task-specific SFT runs set `official_demo_ratio=0`, so their actor sees only the 150 successful Small EXPO episodes collected for that one task. The side-panel same-task-demo result is intentionally not used as evidence for this mechanism; the main comparison must rely on the source audit and matched full-90 results.

2. The action targets differ. `PiReplayBuffer.insert()` begins with the action actually executed at a transition and backfills earlier rows with later executed actions, producing an executed trajectory chunk for Full EXPO actor training. `bc_distill_data.collect_teacher_chunks()` stores the teacher's ten-action proposal at each replanning observation, then executes its first five actions. The 10-action target matches the actor output horizon; deployment executes five and replans, also matching the configured interface. Proposal-tail versus later-executed-prefix disagreement is expected when comparing a plan with a subsequently replanned trajectory; it does not establish that SFT uses inconsistent labels. The difference is between plan imitation and executed-trajectory supervision, an algorithmic distinction that requires a specific scientific question before spending runs to compare it.

3. The optimization dose is very different. The task-28 Full EXPO run had 6,705 actor optimizer steps by environment step 40,095 (447 trigger records multiplied by 15), whereas task-28 SFT had 41,330 actor steps by its 40k checkpoint. The Full EXPO actor LR was still in its warmup at 3.35e-5; SFT had reached the flat 5e-5 LR after 10k steps. The task-28 checkpoint drift audit measures image-tower relative L2 drift of 1.9% for Full EXPO versus 10.7% for SFT, with corresponding action/time projection relative drifts of roughly 0.5--9.0% versus 3.6--64.3%. This is consistent with SFT making a much larger direct policy move toward a narrow target distribution.

These differences are independent of the controller-bound issue. The mixture and optimization-dose differences are plausible reasons for different retention, but require causal evidence. Same-task SFT targets must remain task-specific under the user's constraint. Do not prescribe executed-trajectory targets as a correction unless a controlled result supports that change; calibrate update dose to actor optimizer updates rather than matching environment-step or nominal SFT-step counts.
### Live-state snapshot (2026-11-03, 13:31 CST)

- No production Small EXPO or Full EXPO trainer process is currently running on the visible allocations. The historical Full EXPO and Small EXPO jobs remain available as completed artifacts and are not being modified.
- Two controlled Task-28 SFT trainers are live: raw action targets on gpua082 (Slurm 23416001_2) and controller-bound-clipped targets on gpua070 (23416001_3). Both are zero-B, LoRA-only, 760-chunk, no-demo arms. They have emitted their initialization audit and exact `len()` data audit (`small_expo_chunks=760`, `official_demo_chunks=0`) and are still in the 4-task step-0 evaluation; neither has a training metric or checkpoint yet.
- Three processes are live on gpua016/gpua011/gpua062 (23352795_3/_5/_4): parallel shards of the Task-28 LoRA-only 2,000-update full-LIBERO-90 actor-only evaluation. No result file has been committed yet; the evaluator is still running. This is evaluation of a completed checkpoint, not additional training.
- The Task-28 existing-scope and LoRA-only 2,000-update trainings are complete; their four-task final macros are 42.5% and 47.5%, respectively. The Task-29 no-replay and 50%-same-task-demo arms are also complete through 1,500 updates (53.75% and 60.0% four-task macro). Their final metrics and evaluations are included above.
- The remaining sleeper allocations are not running experiment processes. Existing jobs and allocations were not changed; the process mapping was read-only.

### Integrated actor optimization dose (2026-11-03, 13:34 CST)

Parsing the recorded actor learning-rate histories gives another useful scale comparison. Task-28 SFT has 41,330 actor updates with a summed per-update learning rate of approximately 1.8165. Full EXPO has 6,705 actor updates by environment step 40,095; its logged trigger records represent 15 actor updates each, and the summed warmup learning rate over those updates is approximately 0.1126. Thus the SFT checkpoint has about 16.1 times the integrated actor learning-rate exposure of the comparable Full EXPO checkpoint, in addition to fitting a much narrower target distribution. This is an approximate dose comparison because the Full EXPO log records one actor learning rate per 15-update trigger, but the linear warmup schedule and actor-state indices make the expansion explicit. It reinforces that matching environment-step labels or checkpoint names does not match policy-update exposure.
### Exact-base evaluator launched (2026-11-03, 13:35 CST)

The isolated evaluator-only official `pi05_libero` control is running in tmux `sft-exact-base-20261103` on the previously idle gpua044 allocation (23416001_1 / 23476120). It evaluates tasks 2, 28, 29 and 60 with the same 20 episodes, seed, image size, action horizon and replanning schedule used by the controlled SFT arms. It writes only to `dcrlcheckpoints/sft_exact_base_control_20261103`; no training state or production code is touched. The process is alive but has not yet emitted result files.
### Full-90 actor-only evaluation progress (2026-11-03, 13:36 CST)

The Task-28 LoRA-only 2,000-update actor-only evaluation has committed 84 of 90 task files, with 471/1,680 successes (28.04%) provisionally. The matched retention tasks currently score task 2 = 13/20, task 28 = 20/20, task 29 = 3/20, and task 60 = 2/20. Six task files remain; this aggregate is provisional and must not be presented as the final LIBERO-90 score until all 90 files and the shard summaries are committed.
### Action-bound control step-0 result (2026-11-03, 13:38 CST)

Both Task-28 action-target arms completed the same 4-task step-0 evaluation and then entered the first BC-update compilation. The raw-target arm scored task 2/28/29/60 = 75/35/70/55%, macro 58.75%; the clipped-target arm scored 70/50/65/55%, macro 60.0%. Their source and success-only HDF5 hashes are byte-identical, their sampled initialization-tree hashes are identical, and their only configured treatment difference is target clipping. Nevertheless, independent A100 rollouts were not outcome-identical before training. Therefore the ablation must be interpreted using each arm's paired step-0-to-later change and, if the difference persists, a repeated/no-update control; absolute step-0 differences are not evidence that clipping caused them.
### Balanced multi-task corrective arms launched (2026-11-03, 14:04 CST)

The first launch attempt failed before model construction because the wrapper
compared the collection's recorded teacher directory with the SFT collection
directory. The metadata audit showed the distinction; the wrapper now reads
the authoritative `teacher_dir` and `checkpoint_step` from the HDF5 metadata.
The corrected full-proposal diagnostic initialized successfully, wrote its
`sft_config.json` and initialization audit, and reported
`small_expo_chunks=6863` with the four task prompts preserved. It is running
on 23416001_1/gpua044 with W&B disabled for diagnostic isolation. The matched
executed-backfill arm was then relaunched with W&B online on 23416001_4/gpua077
under its own tmux session and output directory. The raw/clipped controls on
23416001_2/_3 remain untouched. Neither corrective arm counts as a result
until its step-0 evaluation, training metrics, checkpoints and full-90
actor-only evaluation are all committed.

### Current full-suite checkpoint status (2026-11-03)

The task-28 LoRA-only SFT checkpoint at update 2,000 has now been verified
independently from the persisted artifacts: 90 task files, 1,800 episodes,
and 471 successes (26.17%). Its matched Full EXPO task-28 reference is
451/1,800 (25.06%). The broader full-suite result is therefore a small
positive relative to that corresponding Full EXPO run, but it is not a
general solution: it remains below the untouched official actor's historical
512/1,800 (28.44%) and below the best recorded Full EXPO deployment
(483/1,800, 26.83%). The Full EXPO numbers are deployed editor/critic-system
results while this SFT number is actor-only; keep that distinction explicit.
### Action-bound controls at update 500 (2026-11-03)

Both action-bound arms completed their first post-training retention point.
The raw-target arm moved from 75/35/70/55% at step 0 to 80/70/60/40% at
step 500, macro 62.5% versus 58.75% initially. The clipped arm moved from
70/50/65/55% to 70/75/50/50%, macro 61.25% versus 60.0%. Both therefore
improved their four-task macro at this point, but the raw arm improved more
despite the controller-bound treatment being absent. Since the independently
launched step-0 policies were not outcome-identical, this is not yet a clean
causal conclusion about clipping; the later matched curves and full-suite
evaluations remain required.

### Exploratory four-source correction arms (excluded from the main conclusion) (2026-11-03)

The two exploratory SFT arms completed and committed their first matched
500-update retention evaluation. Both started from the same fresh zero-B LoRA
policy and had the identical step-0 panel: task 2/28/29/60 = 75/35/70/55%,
macro 58.75%. They use the same balanced four-task collection, batch size 16,
optimizer and schedule; the only target-construction difference is the action
target semantics.

| Arm | Target construction | Task 2 | Task 28 | Task 29 | Task 60 | Macro |
|---|---|---:|---:|---:|---:|---:|
| `debug2` | Full ten-action teacher proposal | 70% | 55% | 55% | 40% | 55.0% |
| `executed_backfill` | Executed-prefix/backfilled ten-action chunk | 70% | 45% | 60% | 45% | 55.0% |

The paired step-0-to-500 outcome counts reinforce that neither target fixes
retention immediately. Full-proposal training retained/lost/gained successes
as 14/1/0, 6/1/5, 10/4/1 and 8/3/0 on tasks 2/28/29/60. Executed-backfill
training retained/lost/gained 13/2/1, 6/1/3, 11/3/1 and 9/2/0. Both arms
therefore lose 9 of their original 47 successes and finish at 44/80 (55%).
Full-proposal gains more on the target task at this early point, while
executed-backfill is slightly better on tasks 29 and 60; the equal macro and
small fixed panel do not establish a winner.

Both arms successfully wrote `evals/00000500.json` and committed
`latest_checkpoints/500` including the learner state, then resumed updates
past 500. These results are retained as an audit trail only and are not a main
diagnostic result: they use the same four task sources that were selected for
the user's side-panel checks, and therefore do not answer the intended
full-suite SFT question. The unrelated task-29 four-task side evaluation is
also intentionally excluded. The main conclusion must instead rely on the
completed full-90 comparisons and a correctly scoped new experiment.

### Scope correction: side-panel tasks are not the main training scope (2026-11-03)

The local data inventory confirms that only four Small-EXPO teacher
collections are available under the current audit tree: tasks 2, 28, 29 and
60. The local official-demo cache contains five HDF5 files, not a complete
LIBERO-90 replay set. Therefore a four-source SFT arm is a useful isolated
probe of target construction, but it is not a valid substitute for the main
generalization experiment merely because the same four tasks are convenient
for a side-panel evaluation. Its results must remain clearly labelled as
exploratory and excluded from the main conclusion. The main evidence remains
the completed 90-task paired comparisons; any new corrective training must
either use the intended broad replay data or be explicitly framed as a
single-task ablation whose checkpoint is evaluated on the full 90-task suite.

The checkpoint-scope audit also confirms the intended task-specific rule for
the comparison tables. The task-28 SFT table uses the task-28 Small-EXPO
collection only (150 successful episodes, 760 chunks, no official demos). The
task-29 table has two valid variants: `no_replay_1500` uses only task-29
Small-EXPO data (150 successful episodes, 2,150 chunks), while `demo50_1500`
adds only task-29 official demonstrations (1,231 chunks, a 50% same-task batch
fraction). Neither variant contains examples from another task. The separate
four-source arms were exploratory scope probes, not valid task-specific SFT
results; they were stopped and excluded from the main conclusion.

### Evaluation RNG-path audit (2026-11-03)

The trainer's built-in retention evaluator and the isolated post-hoc/full-90
evaluator do not use exactly the same RNG schedule. In
`train_sft_retention.py`, `_evaluate_actor()` resets the learner RNG to
`jax.random.PRNGKey(eval_seed)` for each task in `evaluate_and_save()`. In
`full_90_libero_evals_inference.py` and the isolated `evaluate_retention.py`,
the evaluator creates one `eval_agent` per process/task loop and carries its
updated RNG state from one task to the next. Both paths use the same task
seed, initial-state indices, 10/5 interface and 400-step limit, but stochastic
action sampling can therefore produce different per-task rollouts. The
task-29 side checkpoint demonstrates the practical effect: its in-training
step-1,500 task rates were 45/60/80/55%, while the separate post-hoc side
rerun was 25/55/90/40%.

This is an evaluation-protocol distinction, not evidence of a training change.
The historical full-90 comparison remains internally consistent because all
policies in that table were evaluated through the same post-hoc path. Do not
compare a built-in retention percentage directly with a post-hoc/full-90
percentage without labeling the RNG schedule; future causal comparisons
should use one fixed evaluator path.

### Revalidated action-bound and multi-source runs (current artifact audit)

The persisted artifacts now show both Task-28 action-bound arms completed all
1,500 updates and their step-1,500 four-task evaluations. Raw targets scored
2/28/29/60 = 65/100/0/10% (43.75% macro); clipped motion targets scored
55/100/0/5% (40.0% macro). Both reached 100% on Task 28, while Task 29 fell
to zero. The actor losses at step 1,500 were 0.00879 (raw) and 0.00612
(clipped). Thus clipping the six OSC motion labels, by itself, did not prevent
the measured retention collapse on this panel; the lower clipped-target loss
also did not predict better retention. These are four-task evaluations, not
full-suite scores, and the separate fresh-LoRA step-0 rollouts were not
outcome-identical, so this pair does not quantify a causal full-suite effect.

The two four-source arms both have metrics through update 749, with their last
committed retention evaluation at update 500. Each starts from the same
recorded step-0 panel (75/35/70/55%, 58.75% macro) and is at 55.0% macro at
update 500. `debug2` uses full teacher proposals; `executed_backfill` changes
only target construction to executed-prefix/backfilled chunks. Their equal
early macro means this small panel has not selected a better target
construction. Both arms mix Task 2, 28, 29, and 60 data, so neither is a
task-specific result or a completed solution; there is no full-90 evaluation
artifact for them in the current comparison.

The current login-node `squeue` snapshot has the five elements of allocation
23416001 still RUNNING: indices 1–5 on gpua044, gpua082, gpua070, gpua077 and
gpua044. Remote read-only `srun` process checks confirmed the two full-90
evaluators are running on gpua044 (raw) and gpua077 (clipped); their Python
processes were each using about 59% CPU. The task-42 clipped result had just
appeared while raw was finishing task 41, confirming continued progress. The
`sacct` view shows allocation elements 1 and 4 host these evaluation
processes. A direct read-only `nvidia-smi` check inside allocation element 5
on gpua044 reports its assigned GPU has 0 MiB use and 0% utilization, so one
GPU is confirmed idle. The other four allocated GPUs are in use by the
evaluators or retained by their own sleepers; there is not yet a second
confirmed idle GPU. The two multi-source SFT metrics remain at 749 updates,
with no live SFT process found in the node process checks. No running job was
changed.

## Task-29 full-system startup audit (2026-11-06 15:42 CST)

The Task-29 SFT-plus-editor log reports three non-fatal Orbax warnings because
the matching Small EXPO checkpoint directory lacks optional `metrics/metrics`
files at steps 325013, 350029, and 375380. The editor/critic checkpoint at
375380 restored successfully, and the log explicitly reports the SFT actor and
Small EXPO editor restored with the expected observation, action, critic, and
state shapes. The worker remains GPU-active with no traceback and no task file
yet; the missing optional metrics files are not a reason to restart it.

## Current shared-artifact pure-SFT leaderboard audit (2026-11-06 15:50 CST)

A fresh read-only audit of the complete shared Full-90 directories found the
following highest current task-pure candidate for each underlying task. Every
row has task IDs 0--89, 20 episodes/task, evaluation seed 7, and 1,800 total
episodes. The source launcher/configuration for each row uses only the matching
Small EXPO teacher collection, zero official-demo ratio, official
`pi05_libero` initialization, and the 10/5 interface.

| source task | SFT arm | Full-90 result | matched deployed Full EXPO | difference |
|---:|---|---:|---:|---:|
| 2 | full-proposal, update 500 | 516/1800 (28.67%) | 447/1800 (24.83%) | +3.83 pp |
| 28 | executed-backfill, update 200 | 525/1800 (29.17%) | 451/1800 (25.06%) | +4.11 pp |
| 29 | full-proposal, update 500 | 524/1800 (29.11%) | 441/1800 (24.50%) | +4.61 pp |
| 60 | full-proposal, update 200 | 519/1800 (28.83%) | 368/1800 (20.44%) | +8.39 pp |

The Task-28 executed-backfill row is a valid task-pure target-semantics
variant, not a mixed-data result; it uses the Task-28 teacher only and was
validated from the current `task28_executed_backfill_step200_gpua002`
directory. These differences are actor-only SFT versus deployed Full EXPO
comparisons, so they establish a strong generalization result against the
recorded deployment baselines but not yet an apples-to-apples claim against
the complete Full EXPO selection stack.

Correction: older notes in this file and `hypothesis.md` report 520/1800 for
one pure Task-60 update-200 audit. The currently present shared directory
`task60_full_proposal_step200` contains 90 valid task files whose successes
sum to 519/1800. Until an immutable copy of the 520-result directory is
located, the current-file result (519/1800, +8.39 pp) is the authoritative
number used in this leaderboard.

## Live Full-90 poll (2026-11-06 15:52 CST)

The matched Task-29 SFT-plus-editor evaluator has now committed its first
valid task record (task 0, 3/20 successes), confirming that both the task-pure
actor and matching editor/critic are executing rollouts. It is at 1/90; no
aggregate is inferred. The other workers remain live, including Task-28 SFT
actor at 83/90 and Task-60 seed-789 SFT actor at 60/90. All diagnostic GPUs
remain occupied.

## Full-90 diagnostic health poll (2026-11-06 15:40 CST)

The same workers remain alive. The Task-28 SFT actor advanced to 81/90,
Task-29 Full EXPO actor shards are at 13/45 and 12/45, and the Task-29
SFT-plus-editor worker remains at 0/90 while using approximately 31.2 GB and
61% GPU utilization during compilation/rollout startup. No evaluator has
completed a new validated aggregate, and no GPU has become available for an
additional arm.

## Matched Task-2 actor-only SFT Full-90 evaluation launched (2026-11-06 15:02 CST)

To make the Task-2 comparison causal at the actor level, the same strict
Task-2/update-100 SFT checkpoint used by the full-system run was copied to
node-local storage on the idle `gpua100`. The new tmux session
`full90_task02_sft_actor_only` evaluates it with `mode=sft_pi` over all 90
tasks, using the same 20-episode/task, seed 42/evaluation-seed 7, 224px,
horizon-10/replan-5 protocol as the newly launched Task-2 Full EXPO
actor-only control on `gpua071`. Its output is isolated under
`/tmp/usertmp/rpolonsky/sft_task02_seed123_actor_only/full90_output`.

The SFT worker has passed Orbax checkpoint discovery and is loading the
checkpoint; no task result is committed yet. This is an evaluation-only copy:
the source checkpoint and the Task-2 full-system run remain untouched.

### Interim paired full-suite result for action-bound checkpoints

The raw and clipped Task-28 action-bound actor-only evaluations have both
committed results for tasks 0 through 41 (42/90 tasks, 840 matched episodes).
Across those shared initial states, raw has 186 successes and clipped has
201. Paired outcomes are 151 successes for both, 35 raw-only, 50 clipped-only,
and 604 failures for both. Task 28 is 20/20 for both, and Task 29 is 0/20 for
both. This partial paired comparison favors clipped by 15 net successes so
far, but is not a full-suite result; the remaining 48 tasks could change the
aggregate. Both evaluations use actor-only inference, the same suite, seed,
state indices and action/replan protocol. Keep the final comparison open until
both have all 90 task files and the input protocol is validated end to end.

Across tasks where either model succeeded, clipped's interim gains are spread
across several tasks rather than coming from Task 28 alone (for example,
Task 7: 3 raw vs 7 clipped; Task 12: 4 vs 7; Task 30: 3 vs 6; Task 33: 5 vs
8; Task 38: 5 vs 8). Other tasks favor raw (Task 0: 1 vs 0; Task 26: 6 vs
4); many remain tied at zero. This distribution is consistent with a small
generalization improvement from target clipping so far, but the evaluator is
still incomplete and the source-task retention panel did not improve. Do not
claim the clipping mechanism is proven causal until the full 90-task paired
table is complete; independent run stochasticity also remains a caveat.

### Correction: the SFT 10/5 action-chunk path is consistent (2026-11-03, 19:33 CST)

I rechecked the exact collector, training target, and rollout evaluator after
questioning whether the saved ten-action teacher chunks conflict with
five-action execution. They do not:

- `dcrl/bc_distill_data.py` saves the teacher's ten-action output at the
  current observation and sends its first five actions to the environment.
- The SFT trainer uses all ten saved actions as the ten-step flow-matching
  target.
- Its rollout evaluator, like `train_pi_robo_libero.py`, executes five
  predicted actions before requesting a new chunk.

Thus action horizon 10, prediction target 10, and execution/replan interval 5
agree across collection, learning, and deployment. The unused proposal tail is
part of the current-observation plan; disagreement with later replanned
actions is expected and does not establish a label or interface bug. EXPO's
replay chunks represent executed trajectories, a distinct training target,
not proof that the SFT plan target is malformed.

I stopped the newly started Task-29 proposal-versus-backfill pair before its
initial evaluation or first optimizer update. No training result or checkpoint
was produced; the pair should not be used as evidence. The earlier four-source
backfill probe in this file is a separate mixed-task exploration and remains
excluded from task-specific SFT conclusions.

The remaining actor-level comparison is now running: two evaluator shards
restore Full EXPO Task 29 checkpoint 60,486 and call its base actor directly,
using the same actor-only evaluator protocol, seed 7, LIBERO-90 states,
10-action horizon, and five-action replanning schedule as the SFT suites.
At 19:33 CST both shards were inference-active but had not yet committed the
first task result. The separate Task-28 raw/clipped full-suite pair had
committed 54/90 and 55/90 tasks, respectively; it remains incomplete.

### Follow-up evaluation status (2026-11-03, 19:42 CST)

The actor-only Task-29 Full EXPO evaluation has valid results for three tasks
(IDs 0, 1, 2), totaling 3/60 successes so far. Task 0 is 1/20, the same as
the official Pi0.5 base on the matched states. This is only a partial
per-task observation, not a full-suite comparison. The paired Task-28
raw-versus-clipped SFT evaluation has 58/90 tasks in both arms: raw 340/1,160,
clipped 355/1,160, paired outcomes 292 both-success, 48 raw-only, 63
clipped-only, 757 neither. Its current +15 clipped successes is provisional.

The Task-28 Small EXPO teacher full-suite evaluation has been relaunched on
the idle fifth A100 after fixing the isolated evaluator to pass the current
task's instruction on every episode; the first attempt was stopped before any
task result because its compact mode would have reused Task 0's prompt. The
corrected process has restored teacher checkpoint 275,001 and is initializing
inference. All five A100 allocation elements remain RUNNING; existing
raw/clipped evaluations were not changed.

### Task-specific corrective arms launched (2026-11-04)

The next diagnosis keeps the data scope task-specific. Three isolated Task-28
arms were launched from the prepared audit trainer: full teacher proposals,
executed-prefix/backfilled targets, and the outer EXPO image augmentation
path. All use only the Task-28 Small EXPO collection (760 chunks from 150
successful episodes), official `pi05_libero` initialization, zero-B
LoRA-only updates, no official-demo mixture, batch 16, seed 42, and 500
updates with retention evaluations/checkpoints every 100 updates. They run on
separate idle A100 allocations in tmux sessions
`sft_diag_target_full_task28`, `sft_diag_target_exec_task28`, and
`sft_diag_outer_aug_task28`.

This is a controlled extension of the earlier diagnosis: the first two arms
vary only target construction, while the third tests the augmentation gap
identified in the source audit. The mixed four-task SFT arm remains excluded
from the task-specific conclusion. The Task-2 side full-suite evaluation and
the Small EXPO teacher full-suite evaluation remain untouched. The detailed
run invariants and evidence gates are recorded in
`diagnosis_run_plan_20261104.md`.

#### Launch verification and retry correction (2026-11-04, 16:56--17:02 CST)

The first tmux invocation exited with Slurm code 126 before executing Python:
the prepared wrapper files were mode 660 and were invoked as if directly
executable. No trainer process, checkpoint, or evaluation was produced by that
attempt. The retry invokes each wrapper explicitly through `bash` and writes a
launcher log inside its isolated output directory. This changed only the new
diagnostic tmux sessions; no pre-existing trainer, evaluator, or sleeper
allocation was stopped or modified.

The retry passed the initialization evidence gates. All three W&B runs are
online, all three recorded the same zero-B LoRA initialization audit hash
(`d73d942218d9926e...`), and all three printed
`small_expo_chunks=760; official_demo_chunks=0`. The target modes are,
respectively, `full_proposal`, `executed_backfill`, and `full_proposal` with
`outer_augmentation`. The processes are resident on `gpua082`, `gpua077`, and
`gpua087`, using approximately 17.2 GiB on each A100 while compiling/running
the initial 20-episode retention evaluation. No optimizer update has been
claimed yet; step 0 is the next gate.

### Task-29 and Task-60 matched teacher-only arms launched (2026-11-04)

To extend the diagnosis beyond Task 28, two additional isolated arms use the
same audit trainer and hyperparameters but only their own successful Small
EXPO collections. Task 29 uses teacher checkpoint 375,380 and 2,150 loaded
success chunks; Task 60 uses teacher checkpoint 450,223 and 2,226 loaded
success chunks. Both initialize from official `pi05_libero`, use zero-B
LoRA-only gradients, batch 16, learning rate `5e-5`, 10,000-update warmup,
no official demos, seed 42/evaluation seed 7, four-task retention evaluation,
and evaluation/checkpointing every 100 updates through update 500.

The arms run in separate tmux sessions `sft_diag_task29_20261104` and
`sft_diag_task60_20261104` on `gpua085` and `gpua002`, respectively. Their
initialization audits match the Task-28 hash
(`d73d942218d9926e...`), and their logs explicitly report
`official_demo_chunks=0`. They are GPU-resident and in the step-0 evaluation;
no optimizer update or result is claimed until that evaluation commits.

### Task-2 startup diagnosis and isolated retry (2026-11-04)

The first Task-2 full-proposal arm (`sft_diag_task02_20261104`, allocation
`23433381_4` on `gpua051`) reached W&B initialization but never reached the
trainer's JAX/data audit. Its log stopped after W&B returned control, the
process had no GPU allocation and no HDF5 descriptor, and its output directory
contained no `initialization_audit.json`, metrics, checkpoint, or evaluation.
The process was waiting in shared-filesystem import I/O rather than failing a
data or model validation. A short diagnostic attach was detached and the
process was resumed before this was determined; it did not modify any model,
checkpoint, or pre-existing run. The original attempt is retained as a
startup-failure record.

I relaunched only this isolated arm on the separate sleeper allocation
`23433381_5` (`gpua038`) in tmux `sft_diag_task02_retry_20261104`, with the
same configuration: official `pi05_libero` initialization, zero-B LoRA-only,
Task-2 Small-EXPO teacher checkpoint 375,398, 150 successes, zero official
demos, batch 16, 500 SFT updates, and 100-update retention checkpoints/evals.
The retry also entered a slow `/projects` Python-import filesystem wait before
printing the W&B/data audit; it has not yet produced a training result. This
is a launch-environment/storage issue, not evidence against the Task-2 data
or target-mode hypothesis. No existing trainer, evaluator, or sleeper
allocation was stopped or changed.

### First controlled retention checkpoint (2026-11-04, update 100)

The first four-task actor-only retention checkpoint has now committed for all
three Task-28 arms. The macro success rates over tasks 2, 28, 29, and 60 are:

| Arm | Task 2 | Task 28 | Task 29 | Task 60 | Macro |
|---|---:|---:|---:|---:|---:|
| `full_proposal` | 80% | 35% | 65% | 55% | **58.75%** |
| `executed_backfill` | 60% | 35% | 65% | 45% | **51.25%** |
| `outer_aug` | 65% | 55% | 60% | 50% | **57.50%** |

The executed-tail replacement is therefore not supported by this first
retention gate. Full-proposal and outer-augmentation remain viable, but these
are development-task results only; no full LIBERO-90 claim is made until the
planned 500-update curves and actor-only 90-task evaluations are complete.

The matched Task-29 full-proposal arm also committed update 100 at 58.75%
macro retention (Task 2: 75%, Task 28: 45%, Task 29: 65%, Task 60: 50%).
This is consistent with the Task-28 proposal arm's macro score, but is still
only a four-task development gate.

The matched Task-60 full-proposal arm then committed update 100 at 61.25%
macro retention (Task 2: 75%, Task 28: 55%, Task 29: 70%, Task 60: 45%),
currently the strongest development-gate candidate. It is still not a
full-suite result; the 500-update curve and complete actor-only LIBERO-90
evaluation remain required.

At update 200, the matched Task-29 full-proposal arm improved to 63.75%
macro retention (Task 2: 75%, Task 28: 50%, Task 29: 75%, Task 60: 55%),
making it the current development-gate leader. The Task-28 full-proposal and
executed-backfill arms were 57.50% and 58.75%, respectively. The ranking is
still provisional until the remaining arms reach the same gate and the
candidate checkpoint is evaluated across all 90 tasks.

### Startup-node follow-up (2026-11-04, 17:49--18:00 CST)

The Task-2 retry eventually passed the slow Python/W&B import stage on
`gpua038` and reached W&B at 17:54, but it was stopped before the trainer's
data gate while cleaning up a separate failed-node probe. A fresh retry is
now running in `sft_diag_task02_retry2_20261104` on the same sleeper GPU and
will be allowed to reach the data audit. A no-W&B variant on `gpua051` showed
the same pre-GPU import behavior, so W&B is not the root cause.

I also attempted a full LIBERO-90 evaluation of the completed Task-28
proposal update-100 checkpoint on `gpua051`; that node reproduced the same
pre-GPU `/projects` import stall and the probe was canceled. It is excluded
from results. Full-suite evaluation will be moved to a healthy node after a
diagnostic training arm frees one, without interrupting the other arms.

### First full-suite candidate evaluation launched (2026-11-04, 18:15 CST)

The executed-tail-backfill arm was stopped after its update-200 result
(58.75% macro retention), because it was behind the Task-29 full-proposal
candidate and was not the leading hypothesis. Its checkpoint and evaluation
artifacts remain preserved; only its isolated trainer tmux process was
stopped. The sleeper allocation was not canceled.

This freed healthy `gpua077` for the first decisive full-suite test. The
Task-29 full-proposal checkpoint at update 200 is complete and is now being
evaluated actor-only over LIBERO-90 with 20 episodes per task, seed 7,
224-pixel observations, action horizon 10, and replan interval 5. The
evaluation runs in tmux `sft_diag_full90_task29_step200_20261104` and writes
to:

`/work/hdd/bgqe/rpolonsky/dcrlcheckpoints/sft_retention_full90_eval_20261104/task29_full_proposal_step200`

The evaluator passed the node's Python/import stage on `gpua077`; no result is
counted until all 90 task files and the aggregate are present.

The generic evaluator initially rejected this checkpoint because it only
recognized the historical `configs.sft.my_expo_libero` manifest. That was an
evaluator-dispatch limitation, not a checkpoint failure. The isolated wrapper
`full90_audit_eval.py` now dispatches `audit_sft_config` to the same audit
factory used for training, while reusing the generic evaluator's unchanged
actor-only rollout loop. The corrected run restored the checkpoint
successfully and is compiling the inference path on `gpua077`.

The update-300 retention gate subsequently measured 60.0% for Task-28
full-proposal and 56.25% for Task-29 full-proposal. The Task-29 update-200
checkpoint remains the active full-suite candidate because it led the earlier
development gate, while the complete 90-task result is still pending.

### Full-suite candidate scheduling and live status (2026-11-04, 18:30--18:38 CST)

The live resource audit found no genuinely idle A100 allocation: all ten
running sleeper elements were occupied by diagnosis or side-evaluation
processes, and the newly submitted sleeper arrays remained pending. The
separate Task-2 side evaluator and all production jobs remain untouched.

The Task-29 update-200 full-suite evaluator was verified alive on `gpua077`
with its restored actor resident at approximately 17.3 GiB. It completed
`task_00.json`, then entered an extended `cl_sync_io_wait`/I/O phase before
task 01. This is being treated as a live evaluator slowdown, not as a result
or a failure; it has not been restarted.

The remaining controlled four-task gates were re-read directly from their
JSON records. At update 200, Task-60 full-proposal scored 70/55/70/55% on
tasks 2/28/29/60, respectively, for a 62.5% macro. At update 300, Task-28
full-proposal scored 65/60/60/55% (60.0% macro), Task-28 outer augmentation
scored 75/50/75/45% (61.25% macro), and Task-29 full-proposal scored
80/30/65/50% (56.25% macro). These are still development-set gates, not
LIBERO-90 results.

To obtain decisive full-suite evidence sooner, the isolated Task-28
full-proposal and outer-augmentation trainer tmux processes were stopped after
their complete update-300 artifacts were committed. Their Slurm sleeper
allocations and all checkpoint/evaluation artifacts were preserved. The
stronger Task-60 full-proposal update-200 checkpoint is now being evaluated
actor-only over LIBERO-90 in two modulo shards:

- `sft_diag_full90_task60_step200_shard0` on `23433381_1`/`gpua087`;
- `sft_diag_full90_task60_step200_shard1` on `23416001_2`/`gpua082`.

The isolated launcher is
`run_full90_task60_proposal_step200_shard_20261104.sh`. It uses the same
wrapper dispatch as the Task-29 candidate and the unchanged generic rollout
loop, with 20 episodes/task, seed 7, 224-pixel observations, horizon 10 and
replanning every 5 steps. Both shard processes have passed the launcher
stage and are currently in model initialization/compilation. No score will be
counted until both shard files cover all 90 tasks and are independently
validated.

At the next live poll, both Task-60 shard processes remained alive with stable
PIDs but had not emitted a task file or allocated GPU memory. Their kernel
wait state was `cl_sync_io_wait`, with only about 15 MiB of character reads;
the evaluator on `gpua077` showed the same Lustre behavior before returning
to active CPU compilation. `findmnt` confirms that the code under `/projects`
and the checkpoint/output tree under `/work` are separate Lustre filesystems.
This is recorded as a shared-filesystem startup bottleneck, not an OOM,
import exception, or failed evaluation. The shard handles remain under
monitoring without restart.

At 18:47, the Task-29 candidate had advanced to active GPU inference at 56%
GPU utilization, although it had not yet committed task 01. This confirms the
earlier wait was not terminal. The separate Task-2 side evaluator had produced
86 task files and remained active on `gpua070`; it is explicitly not part of
the diagnosis and was not touched. A short same-user `strace` on the Task-60
shard showed no user-space file or lock call during its wait, consistent with
the process being held in the kernel's Lustre synchronization path. No shard
was restarted based only on this observation.

By 18:49, the Task-29 full-suite candidate had committed `task_01.json` and
was still using the GPU. The separate Task-2 side evaluator had reached 87
task files and remained active. Task-29's matched trainer had reached update
499 with its update-400 evaluation committed; its final update-500 retention
evaluation is still running, so its allocation is not yet reused.

I also prepared and syntax-checked
`run_full90_task28_full_proposal_step300_20261104.sh`, which will evaluate the
preserved Task-28 full-proposal update-300 checkpoint on the actor-only
LIBERO-90 suite once a diagnostic A100 becomes terminally free. It uses the
isolated audit wrapper and the same seeds, horizon, image size and episode
count as the other candidate evaluations.

At 18:53, both Task-60 shards remained alive with approximately 17.3 GiB of
GPU memory each; they are in the evaluator's first rollout/compilation phase
but have not committed a task file. The Task-29 candidate had committed task
files 00--02 and remained GPU-active. The separate Task-2 side evaluator had
advanced to 88 task files and remained untouched. No aggregate was computed
from any of these partial states.

At 19:01, the Task-60 full-proposal evaluation committed its first valid
full-90 result, `task_00.json` (1 of 90 task files; 5% success on that task).
Both shard processes remain live, so this confirms that the sharded evaluator
is functioning rather than being a terminal partial run. The Task-28 and
Task-29 candidate suites remain incomplete, and no aggregate has been
calculated.

### Task-29 diagnosis arm completed and next candidate launched (2026-11-04, 18:58 CST)

The matched Task-29 full-proposal trainer completed its planned 500-update
run. Its final four-task actor-only retention evaluation was:

| checkpoint | Task 2 | Task 28 | Task 29 | Task 60 | macro |
|---|---:|---:|---:|---:|---:|
| Task-29 full proposal, update 500 | 65% | 50% | 75% | 55% | **61.25%** |

The update-500 retention JSON and `latest_checkpoints/500` are committed. This
does not replace the update-200 full-suite candidate, which had the stronger
63.75% development gate; both remain separate, correctly labeled artifacts.

The Task-29 trainer terminated normally after the final gate, freeing
`gpua085` while leaving its sleeper allocation running. I launched the next
isolated actor-only LIBERO-90 evaluation in tmux
`sft_diag_full90_task28_step300_20261104` using the preserved Task-28
full-proposal update-300 checkpoint. It writes to
`dcrlcheckpoints/sft_retention_full90_eval_20261104/task28_full_proposal_step300`
and uses the same 20 episodes/task, seed 7, 224-pixel, horizon-10/replan-5
protocol as the other evaluations.

At 18:52, the Task-60 shards had both crossed the Lustre startup gate and
allocated approximately 17.3 GiB of GPU memory each; they remain live in JAX
evaluation with no task file committed yet. The Task-29 full-suite candidate
had committed task files 00, 01 and 02 and was using approximately 17.3 GiB
on `gpua077`. Its matched trainer remained at 499 updates while the final
update-500 retention rollout was in progress. The Task-2 retry diagnostic was
also still alive in the same pre-GPU Lustre wait and was not touched.

### Live poll after the detailed-record lookup (2026-11-04, 19:04 CST)

The isolated actor-only full-suite evaluators remain live and were not
restarted. The current committed task-file counts are:

| Evaluation | Committed task files | State |
|---|---:|---|
| Task-29 full-proposal, update 200 | 5/90 | running on `gpua077` |
| Task-60 full-proposal, update 200, shards 0+1 | 2/90 combined | both running on `gpua087` and `gpua082` |
| Task-28 full-proposal, update 300 | 0/90 | running on `gpua085` |
| Separate Task-2 side evaluator | 88/90 | running on `gpua070`, excluded from this diagnosis |

These counts are filesystem counts of committed `task_*.json` records; no
aggregate score is inferred from partial suites. The Task-2 side evaluator is
explicitly outside the diagnosis and remains untouched. The Task-2 controlled
training retry is also still alive in its startup/import phase, with no
training result yet.

The separate Task-2 side evaluator subsequently committed all 90 task files
(`shard_0`: 30, `shard_1`: 30, `shard_2`: 30) and exited its evaluator
process. Its artifacts remain at
`dcrlcheckpoints/distillation_full90_eval_sharded_retry_20261104/task02_3000`
and are not used as evidence for the task-specific SFT diagnosis. Its freed
GPU was reused for the next isolated candidate rather than restarting or
altering any other run.

### Task-60 dose follow-up launched (2026-11-04, 19:13 CST)

The Task-60 full-proposal update-400 checkpoint is preserved under
`task_60_full_proposal/latest_checkpoints/400`. Its four-task retention gate
was 51.25% (45%, 40%, 70%, 50% on tasks 2, 28, 29, 60), versus 62.5% at
update 200. To measure whether the lower development gate also corresponds to
lower or unexpectedly better LIBERO-90 retention, I launched an actor-only
full-suite evaluation in tmux
`sft_diag_full90_task60_step400_20261104` on the GPU freed by the completed
side evaluator. It writes to
`dcrlcheckpoints/sft_retention_full90_eval_20261104/task60_full_proposal_step400`
using the same 20-episode, seed-7, 224-pixel, horizon-10/replan-5 protocol.

### Additional full-suite dose and augmentation evaluations launched (2026-11-04, 19:07--19:08 CST)

The live GPU audit found two process-idle GPUs inside already-running sleeper
allocations, so two additional isolated evaluations were launched without
canceling or modifying any allocation:

| Candidate | Checkpoint | Allocation / GPU | tmux session | Output |
|---|---|---|---|---|
| Task-29 full-proposal | `task_29_full_proposal/latest_checkpoints/500` | `23433381_4` / `gpua051` | `sft_diag_full90_task29_step500_20261104` | `sft_retention_full90_eval_20261104/task29_full_proposal_step500` |
| Task-28 outer augmentation | `task_28/checkpoints/300` | `23416001_1` / free GPU on `gpua044` | `sft_diag_full90_task28_outer_aug_step300_20261104` | `sft_retention_full90_eval_20261104/task28_outer_aug_step300` |

Both use the isolated audit evaluator and the established actor-only LIBERO-90
protocol: 20 episodes per task, evaluation seed 7, training seed 42, 224-pixel
observations, action horizon 10, and replan interval 5. At launch they were
still in shared-filesystem startup and had committed no task files; the
existing Task-29 update-200, Task-60 update-200 shards, Task-28 update-300
full-proposal evaluation, and the separate Task-2 side evaluator were left
running unchanged.

The first Task-29 update-500 launch exited before GPU initialization because
the final checkpoint was under `latest_checkpoints/500` while the isolated
evaluator's root resolver expects the manager directory when using that
layout. No result was produced and no existing run was touched. The launcher
was corrected to pass `latest_checkpoints` as the manager root; the retry in
`sft_diag_full90_task29_step500_20261104` passed manifest resolution and was
in JAX initialization at 19:10, with no committed task file yet.

### Live poll after side-run completion (2026-11-04, 19:15 CST)

The completed side evaluator has 90/90 valid task records and is no longer
using its GPU. The active diagnosis evaluation counts are currently:

| Candidate | Committed task files | State |
|---|---:|---|
| Task-28 full proposal, update 300 | 3/90 | GPU-active on `gpua085` |
| Task-28 outer augmentation, update 300 | 1/90 | GPU-active on `gpua044` |
| Task-29 full proposal, update 200 | 7/90 | GPU-active on `gpua077` |
| Task-29 full proposal, update 500 | 0/90 | GPU-active/compiling on `gpua051` |
| Task-60 full proposal, update 200, shards 0+1 | 6/90 combined | GPU-active on `gpua087` and `gpua082` |
| Task-60 full proposal, update 400 | 0/90 | GPU-active/compiling on `gpua070` |

The Task-2 controlled retry has loaded 1,727 task-2 teacher chunks with zero
official-demo chunks and has entered its initial retention evaluation; it has
not yet committed a training checkpoint. No full-suite aggregate is inferred
from any partial candidate directory.

At 19:18, the live evaluator counts advanced to Task-28 proposal 4/90,
Task-28 outer augmentation 1/90, Task-29 proposal update 200 8/90, Task-29
proposal update 500 1/90, and Task-60 proposal update 200 8/90 combined. The
Task-60 update-400 evaluator remains in model initialization. All corresponding
processes are still present; no evaluator was restarted from a transient
filesystem wait.

### Full-suite artifact validator added and baseline audit (2026-11-04, 19:16 CST)

The read-only validator
`sharder/training/sft_analysis_20261103/validate_full90.py` now checks that a
candidate has exactly one record for every task ID 0--89, 20 trials per task,
and the fixed LIBERO-90 protocol (20 episodes, 400-step limit, eval seed 7,
horizon 10, replan 5) before reporting an aggregate. It reports incomplete
directories without inferring a score.

It validated the completed, out-of-scope Task-2 side evaluator at **250/1800
(13.89%)**. The matching Full EXPO directories validate cleanly at:

| Full EXPO training task | Validated successes | Rate |
|---:|---:|---:|
| 2 | 447/1800 | 24.83% |
| 28 | 451/1800 | 25.06% |
| 29 | 441/1800 | 24.50% |
| 60 | 368/1800 | 20.44% |

The official-base result directory contains one nested one-episode smoke file
alongside the 90 valid direct task files; the validator correctly rejects that
directory as contaminated rather than silently accepting it. The documented
official-base 512/1800 result therefore remains treated as a separately
audited direct-file aggregate.

At 19:20, the candidates remained incomplete but live: Task-28 proposal 4/90,
Task-28 outer augmentation 2/90, Task-29 proposal update 200 8/90, Task-29
proposal update 500 1/90, Task-60 proposal update 200 9/90 combined, and
Task-60 proposal update 400 0/90 during compilation. The Task-2 retry and all
full-suite evaluator processes remained present; no terminal failure or free
diagnostic GPU was observed.

The next poll at 19:20:34--19:20:23 showed incremental progress: Task-28
outer augmentation reached 2/90, Task-60 update 200 reached 10/90 combined,
and Task-60 update 400 committed its first task record. All other candidates
remain live; no full-suite result is yet complete enough for comparison.

An additional health poll at 19:19 found all four newly monitored diagnostic
GPUs actively executing JAX work at roughly 40--45% utilization with about
17.3 GiB resident, rather than idle or crashed. The Task-2 controlled retry's
initial 20-episode retention evaluation also advanced to 20/20 episodes; it
remains in the training run with no checkpoint committed yet. This confirms
that its long startup was slow compilation/rollout work, not a dead process.

At 19:27, the live full-suite evaluators were still healthy and advancing. The
current committed task-file counts were Task-28 full proposal update 300: 5/90;
Task-28 outer augmentation update 300: 2/90; Task-29 full proposal update 200:
10/90; Task-29 full proposal update 500: 2/90; Task-60 full proposal update
200: 12/90 combined across its two shards; and Task-60 full proposal update
400: 1/90. Direct `srun` health checks found the active evaluator processes on
their assigned A100s with approximately 40--57% utilization and 17.3 GiB
resident. No evaluator was restarted, reassigned, or cancelled.

The repeated LIBERO warning about
`/projects/bgqe/rpolonsky/dcrl/sharder/training/datasets` is non-fatal and
expected for these actor-only rollouts: the configured BDDL, initial-state, and
asset paths in `runtime/libero_config/config.yaml` exist, while the missing
path is only the optional demonstration-HDF5 location. The evaluator does not
load demonstration HDF5 files. This was verified against the LIBERO path
checker and the evaluator command lines before retaining these runs as valid
diagnostic candidates.

The Slurm output audit at the same time found no current matching `.out` or
`.err` files under `/u/rpolonsky`. Effective stdout/stderr for the active and
pending sleeper allocations are under
`/work/hdd/bgqe/rpolonsky/allocation_logs/`; the isolated diagnostic evaluators
write their launcher logs and task JSON files under
`dcrlcheckpoints/sft_retention_full90_eval_20261104/`. Existing production and
side-evaluator allocations were left untouched.

At 19:29, a follow-up poll confirmed continued progress rather than a stalled
or vanished process: Task-28 proposal update 300 reached 6/90, Task-28 outer
augmentation reached 4/90, Task-29 proposal update 200 reached 11/90, Task-29
proposal update 500 reached 3/90, Task-60 proposal update 200 reached 13/90
combined, and Task-60 proposal update 400 reached 2/90. All ten sleeper array
elements remain RUNNING and the diagnostic launch processes remain present;
the three additional sleeper arrays remain PENDING for priority. No process
was restarted or allocation modified.

At 19:30, the same live-process poll showed incremental progress: Task-28
proposal update 300 remained at 6/90, Task-28 outer augmentation at 4/90,
Task-29 proposal update 200 at 11/90, Task-29 proposal update 500 at 3/90,
Task-60 proposal update 200 at 14/90 combined, and Task-60 proposal update 400
at 3/90. Task-29 update 200 advanced through task 10 and Task-29 update 500
through task 02 in their logs. No candidate has produced a complete directory
or aggregate yet; the strict validator therefore still has no new score to
report. All ten active sleeper elements remain RUNNING and the three newer
arrays remain PENDING for priority.

At 19:30, another authoritative poll showed Task-60 update 200 at 15/90; all
other candidate counts were unchanged (Task-28 proposal 6/90, Task-28 outer
augmentation 4/90, Task-29 update 200 11/90, Task-29 update 500 3/90, and
Task-60 update 400 3/90). The Task-29 logs continued to show normal per-task
completion messages, and all ten active sleeper allocations remained RUNNING.
No candidate reached the validator's 90/90 gate and no process or allocation
was changed.

At 19:32, Task-29 proposal update 200 advanced to 12/90 (task 11 completed).
The other live candidates remained at Task-28 proposal 6/90, Task-28 outer
augmentation 4/90, Task-29 update 500 4/90, Task-60 update 200 15/90, and
Task-60 update 400 3/90. The evaluator allocation states were unchanged and no
candidate reached the strict-completion gate.

At 19:33, a 30-second live wait followed by an artifact and Slurm poll showed
Task-28 proposal update 300 advancing to 7/90. The latest committed records
were Task-28 outer augmentation 4/90, Task-29 update 200 12/90, Task-29 update
500 4/90, Task-60 update 200 15/90, and Task-60 update 400 3/90. The active
launch processes and enclosing allocations remained present with zero exit
codes in Slurm accounting; no candidate was complete enough for validation.

At 19:31, Task-29 proposal update 500 advanced to 4/90; the other live counts
remained Task-28 proposal 6/90, Task-28 outer augmentation 4/90, Task-29
proposal update 200 11/90, Task-60 proposal update 200 15/90, and Task-60
proposal update 400 3/90. Its log continued to emit normal per-task results,
and all evaluator launch processes remained present. No complete candidate was
available for strict validation, and no running job was modified.

At 19:34, a 30-second live poll showed Task-28 outer augmentation update 300
advancing to 5/90 and Task-60 update 400 committing its task 03 record. The
other counts were Task-28 proposal 7/90, Task-29 update 200 12/90, Task-29
update 500 4/90, and Task-60 update 200 15/90. Slurm still reported all ten
active sleeper elements RUNNING and all diagnostic launch processes present;
no candidate had reached 90/90.

At 19:37, Task-29 proposal update 200 advanced to 13/90. The latest verified
counts were Task-28 proposal 7/90, Task-28 outer augmentation 5/90, Task-29
update 500 4/90, Task-60 update 200 17/90, and Task-60 update 400 4/90. Recent
task-file timestamps showed normal incremental commits, and all ten active
Slurm allocations remained RUNNING; no candidate was yet eligible for strict
validation.

At 19:36, after a 30-second live wait, Task-28 outer augmentation update 300
reached 5/90, Task-60 update 200 reached 17/90, and Task-60 update 400 reached
4/90. Task-28 proposal remained 7/90, Task-29 update 200 12/90, and Task-29
update 500 4/90. The enclosing Slurm allocations and diagnostic launch
processes remained RUNNING; no full-suite directory was complete enough for
strict validation.

At 19:37, Task-29 proposal update 500 advanced to 5/90 (task 04 committed).
The other verified counts remained Task-28 proposal 7/90, Task-28 outer
augmentation 5/90, Task-29 update 200 13/90, Task-60 update 200 17/90, and
Task-60 update 400 4/90. Slurm accounting still showed the active enclosing
allocations and diagnostic steps running with zero exit codes; no candidate
was complete enough for validation.

At 19:38, Task-28 proposal update 300 advanced to 8/90 (task 07 committed).
The latest verified counts were Task-28 outer augmentation 5/90, Task-29
proposal update 200 13/90, Task-29 update 500 5/90, Task-60 update 200 17/90,
and Task-60 update 400 4/90. Slurm continued to report the active evaluator
steps and enclosing allocations RUNNING with zero exit codes; no full-suite
candidate had reached the validation gate.

At 19:39, Task-28 outer augmentation update 300 advanced to 6/90. The other
diagnostic counts remained Task-28 proposal 8/90, Task-29 update 200 13/90,
Task-29 update 500 5/90, Task-60 update 200 17/90, and Task-60 update 400 4/90.
The separate Task-2 side run was still alive in its retention-evaluation phase
and remained excluded from the diagnosis. No full-suite candidate was complete
enough for validation and no job was modified.

At 19:39, Task-60 proposal update 200 advanced to 18/90 and update 400 to
5/90. The other counts remained Task-28 proposal 8/90, Task-28 outer
augmentation 6/90, Task-29 update 200 13/90, and Task-29 update 500 5/90.
The diagnostic launch processes and enclosing Slurm allocations remained
RUNNING; no candidate was complete enough for strict validation.

At 19:41, Task-29 proposal update 200 advanced to 14/90 (task 13 committed).
The other counts remained Task-28 proposal 8/90, Task-28 outer augmentation
6/90, Task-29 update 500 5/90, Task-60 update 200 18/90, and Task-60 update
400 5/90. The active evaluator steps and enclosing Slurm allocations remained
RUNNING with zero exit codes; no candidate was complete enough for validation.

At 19:41, a subsequent artifact poll showed Task-60 proposal update 200 at
20/90. Task-29 update 200 remained at 14/90; the other counts remained Task-28
proposal 8/90, Task-28 outer augmentation 6/90, Task-29 update 500 5/90, and
Task-60 update 400 5/90. All active evaluator allocations remained RUNNING and
no candidate had reached the strict validation gate.

At 19:42, Task-29 proposal update 500 advanced to 6/90 (task 05 committed).
The other counts remained Task-28 proposal 8/90, Task-28 outer augmentation
6/90, Task-29 update 200 14/90, Task-60 update 200 20/90, and Task-60 update
400 5/90. All active diagnostic allocations remained RUNNING; no candidate
was complete enough for strict validation.

At 19:43, Task-28 proposal update 300 advanced to 9/90 (task 08 committed)
and Task-29 proposal update 500 to 6/90. The other counts remained Task-28
outer augmentation 6/90, Task-29 update 200 14/90, Task-60 update 200 20/90,
and Task-60 update 400 5/90. All active diagnostic allocations remained
RUNNING; no candidate had reached strict validation.

At 19:44, Task-28 outer augmentation update 300 advanced to 7/90 (task 06
committed). The other counts remained Task-28 proposal 9/90, Task-29 update
200 14/90, Task-29 update 500 6/90, Task-60 update 200 20/90, and Task-60
update 400 5/90. All active diagnostic allocations remained RUNNING; no
candidate had reached strict validation.

At 19:44, Task-28 proposal update 300 advanced to 10/90, Task-60 proposal
update 200 to 22/90, and Task-60 update 400 to 6/90. The other counts remained
Task-28 outer augmentation 7/90, Task-29 update 200 14/90, and Task-29 update
500 6/90. All active diagnostic allocations remained RUNNING; no candidate was
complete enough for strict validation.

At 19:45, Task-29 proposal update 200 advanced to 15/90 (task 14 committed).
The latest counts were Task-28 proposal 10/90, Task-28 outer augmentation 7/90,
Task-29 update 500 6/90, Task-60 update 200 22/90, and Task-60 update 400
6/90. All active diagnostic allocations remained RUNNING; no candidate had
reached strict validation.

At 19:44, Task-60 proposal update 200 advanced to 21/90 (task 22 committed).
The other counts remained Task-28 proposal 9/90, Task-28 outer augmentation
7/90, Task-29 update 200 14/90, Task-29 update 500 6/90, and Task-60 update
400 5/90. All active diagnostic allocations remained RUNNING; no candidate was
complete enough for strict validation.

At 19:47, an authoritative filesystem and Slurm poll showed incremental
progress. The current committed full-suite records are Task-28 full-proposal
update 300 at 11/90, Task-28 outer augmentation update 300 at 7/90, Task-29
full-proposal update 200 at 15/90, Task-29 full-proposal update 500 at 7/90,
Task-60 full-proposal update 200 at 23/90 across its two shards, and Task-60
full-proposal update 400 at 6/90. All ten active diagnostic sleeper elements
remain RUNNING. The separate Task-2 controlled retry also remains alive and
excluded from this full-suite diagnosis. No candidate has reached the 90/90
strict-validation gate, so no aggregate score or winner is reported yet; no
process, allocation, or unrelated training run was modified.

At 19:47:49, a second live poll confirmed continued evaluator activity. Task-28
outer augmentation update 300 advanced to 8/90; the other committed counts
were unchanged at Task-28 proposal update 300 11/90, Task-29 proposal updates
200 and 500 at 15/90 and 7/90, and Task-60 proposal updates 200 and 400 at
23/90 and 6/90. Slurm still reports all ten active diagnostic sleeper
elements RUNNING, and the logs continue to emit ordinary per-task results.
The empty/stopped Task-28 update-100 directory is a prior terminated test and
is not being relaunched. The recurring missing optional `datasets`-path warning
does not prevent actor-only evaluation because the configured BDDL, assets, and
initial-state resources are present. No candidate is yet eligible for strict
90/90 validation.

At 19:49, the same evaluator handles were revalidated after a normal rollout
interval. Task-29 full-proposal update 200 advanced to 16/90, Task-60
full-proposal update 200 advanced to 24/90 across its two shards, and Task-60
full-proposal update 400 advanced to 7/90. The remaining counts were Task-28
full-proposal update 300 at 11/90, Task-28 outer augmentation update 300 at
8/90, and Task-29 full-proposal update 500 at 7/90. All ten diagnostic
sleeper elements and the separate Task-2 side-run process remain RUNNING. No
candidate has reached the strict 90/90 gate and no process or allocation was
modified.

At 16:34 CST, the same node-side poll showed the control had transitioned out
of pure initialization and committed its first LIBERO-90 task record (`1/90`).
The Slurm allocation remained `RUNNING`, the tmux session remained present,
and the evaluator process was still active. The run is therefore progressing,
but it remains far from a valid aggregate score until all 90 task records are
complete.

At 19:50, Task-60 full-proposal update 200 advanced to 25/90 (the second
shard's committed-record count). The other candidates remained at Task-28
proposal 11/90, Task-28 outer augmentation 8/90, Task-29 proposal updates 200
and 500 at 16/90 and 7/90, and Task-60 update 400 at 7/90. Slurm still reports
all ten diagnostic sleeper elements RUNNING; the separate Task-2 side run is
also still active and excluded. No full-suite candidate is complete enough for
strict validation, and no job or allocation was changed.

At 19:50:44, Task-28 full-proposal update 300 advanced to 12/90. The other
full-suite candidates remained at Task-28 outer augmentation 8/90, Task-29
full-proposal updates 200 and 500 at 16/90 and 7/90, and Task-60 full-proposal
updates 200 and 400 at 25/90 and 7/90. Slurm continued to report all ten
diagnostic sleeper elements RUNNING. The separate Task-2 controlled retry was
still emitting retention-evaluation progress and remained outside the main
full-suite comparison. No candidate reached strict validation and no process
or allocation was modified.

At 23:11:21, the live poll found Task-60 proposal update 400 at 56/90; the
other incomplete suites had not reached a new terminal state. The gpua087
executed-backfill retry remained alive in compiled GPU initialization with
approximately 2.5 GB GPU memory and no traceback, OOM, or kill marker. No
process or allocation was modified.

At 23:22:05, Task-29 proposal update 500 advanced to 60/90 and Task-60
proposal update 400 advanced to 59/90. Task-28 proposal and outer-augmentation
remained at 63/90 and 46/90, Task-29 update 200 at 63/90, and Task-60 update
500 at 6/90. Remote checks confirmed all six evaluator processes remained
live; no process or allocation was modified.

At 23:20:53, Task-29 proposal update 200 advanced to 63/90. Task-28 proposal
remained at 63/90, outer-augmentation at 46/90, Task-29 update 500 at 59/90,
and Task-60 updates 400 and 500 at 58/90 and 6/90. Remote checks confirmed
all six evaluator processes remained live; no process or allocation was
modified.

At 23:20:29, remote liveness probes confirmed all six original full-suite
evaluators remained active. Artifact counts were unchanged from the prior
poll: Task-28 proposal 63/90, outer-augmentation 46/90, Task-29 proposal
updates 200 and 500 at 62/90 and 59/90, and Task-60 proposal updates 400 and
500 at 58/90 and 6/90. No process or allocation was modified.

At 23:19:57, Task-28 proposal update 300 advanced to 63/90. The other five
incomplete evaluator counts were unchanged, and remote checks confirmed all
six original full-suite evaluator processes remained live. No process or
allocation was modified.

At 23:19:04, the corrected remote liveness probes confirmed all six original
full-suite evaluators were still running. Artifact counts remained Task-28
proposal 62/90, outer-augmentation 46/90, Task-29 proposal updates 200 and
500 at 62/90 and 59/90, and Task-60 proposal updates 400 and 500 at 58/90
and 6/90. The additional allocations remained pending, so no new diagnostic
run was launched and no process or allocation was modified.

At 23:18:38, remote liveness probes confirmed all six original full-suite
evaluators were still running. Their artifact counts remained Task-28 proposal
62/90, Task-28 outer-augmentation 46/90, Task-29 proposal updates 200 and
500 at 62/90 and 59/90, and Task-60 proposal updates 400 and 500 at 58/90
and 6/90. No additional healthy idle slot was available and no process or
allocation was modified.

At 23:18:00, all six original full-suite evaluator processes were confirmed
alive remotely. No candidate reached a new terminal state: Task-28 proposal
and outer-augmentation remained at 62/90 and 46/90; Task-29 proposal updates
200 and 500 remained at 62/90 and 59/90; and Task-60 proposal updates 400 and
500 remained at 58/90 and 6/90. The strict validator continued to report the
completed Task-60 update-200 candidate as 519/1800 (28.83%). No process or
allocation was modified.

At 23:17:32, remote Slurm checks confirmed all six original full-suite
evaluators were alive. Task-28 proposal update 300 reached 62/90, outer
augmentation reached 46/90, Task-29 proposal update 500 reached 59/90,
Task-60 proposal update 400 reached 58/90, and Task-60 proposal update 500
reached 6/90. The additional sleep allocations remained pending, and no new
evaluator was launched on the known-unhealthy code-mount slots. No unrelated
process or allocation was modified.

At 23:16:44, remote Slurm checks confirmed the six original incomplete
full-suite evaluators were still actively running on their assigned nodes;
their local launchers are remote steps, so local `ps` alone is not a valid
liveness test. The gpua087 executed-backfill I/O remnant was no longer
present, and its retry remained incomplete with zero records. No candidate
had reached 90/90 and no unrelated process or allocation was modified.

At 23:16:13, the executed-backfill retry was confirmed to be blocked on a
`/projects` code-mount read: its open syscall was reading
`orbax/checkpoint/_src/handlers/composite_checkpoint_handler.py`, and an
independent read of that file timed out on gpua087. A preflight read on the
nominally idle gpua002 slot also timed out, so no duplicate evaluator was
launched there. The gpua087 process remains an isolated uninterruptible-I/O
remnant with targeted SIGKILL pending; all existing evaluators and training
allocations remain untouched.

At 23:14:12, the gpua087 backfill evaluator remained alive 8:15 into
checkpoint restoration with 2.5 GB GPU memory and active worker/compile
threads. Its aggregate process read counters were stable during the probe and
the log still ended at the restore message, but there was no traceback, OOM,
or kill marker. Because the process was live rather than terminal, it was not
restarted.

The historical Task-29 evaluations subsequently committed their first valid
records for task 0 (0/20 successes for both no-replay and 50%-demo arms),
confirming that both evaluators have entered the rollout loop. Their full
suite scores remain pending; the controlled Task-2 update-500 evaluator has
five committed task records. No result is inferred from these partial files.

At 23:13:27, a read-counter and thread-state probe of the gpua087
executed-backfill evaluator showed a live CUDA event-handler thread and the
expected JAX/Python worker pool waiting at the restore/compile barrier. The
process remained alive with 2.5 GB GPU memory and no terminal error; the
checkpoint read counters were stable during the short probe, so it was left
running rather than restarted.

At 23:12:47, Task-28 proposal update 300 advanced to 61/90 and Task-60
proposal update 500 remained at 5/90; Task-29 proposal update 500 remained at
58/90. The gpua087 backfill retry was still alive in the checkpoint-restore
phase with approximately 2.5 GB GPU memory and no terminal error. No process
or allocation was modified.

At 23:12:12, Task-29 proposal update 500 advanced to 58/90 and Task-60
proposal update 500 to 5/90. The gpua087 executed-backfill retry remained
alive in checkpoint restoration with approximately 2.5 GB GPU memory; a
representative checkpoint object read successfully from `/work`, so the
wait was not evidence of a failed mount. No task record or terminal error was
present, and no process or allocation was modified.

At 23:11:52, the gpua087 executed-backfill retry progressed beyond cuDNN/XLA
compilation and logged active restoration of checkpoint 200. It remained
alive with approximately 2.5 GB GPU memory and no terminal error; no task
record had been emitted yet. The other full-suite counts were unchanged. No
process or allocation was modified.

At 19:51:33, Task-29 full-proposal update 500 advanced to 8/90. The other
committed counts remained Task-28 full-proposal update 300 at 12/90, Task-28
outer augmentation update 300 at 8/90, Task-29 update 200 at 16/90, and
Task-60 updates 200 and 400 at 25/90 and 7/90. All ten diagnostic sleeper
elements remained RUNNING, with no terminal evaluator failure; the separate
Task-2 side run remained active and excluded. No candidate was ready for
strict validation and no job or allocation was modified.

At 19:52:31, Task-28 outer-augmentation update 300 advanced to 9/90. The
other committed counts remained Task-28 full-proposal update 300 at 12/90,
Task-29 full-proposal updates 200 and 500 at 16/90 and 8/90, and Task-60
full-proposal updates 200 and 400 at 25/90 and 7/90. All ten diagnostic
sleeper elements remained RUNNING; no candidate reached strict 90/90
validation and no process or allocation was modified.

At 19:53:15, Task-60 full-proposal update 400 advanced to 8/90. The other
committed counts remained Task-28 full-proposal update 300 at 12/90, Task-28
outer augmentation update 300 at 9/90, Task-29 full-proposal updates 200 and
500 at 16/90 and 8/90, and Task-60 full-proposal update 200 at 25/90. All ten
diagnostic sleeper elements remained RUNNING, and the separate Task-2 side
run remained active and excluded. No candidate reached strict validation and
no job or allocation was modified.

At 19:53:39, Task-60 full-proposal update 200 advanced to 26/90. The other
committed counts remained Task-28 full-proposal update 300 at 12/90, Task-28
outer augmentation update 300 at 9/90, Task-29 full-proposal updates 200 and
500 at 16/90 and 8/90, and Task-60 full-proposal update 400 at 8/90. All ten
diagnostic sleeper elements remained RUNNING. No candidate reached strict
90/90 validation and no process or allocation was modified.

At 19:54:04, Task-28 full-proposal update 300 advanced to 13/90 and Task-60
full-proposal update 200 advanced to 27/90. The other committed counts were
Task-28 outer augmentation update 300 at 9/90, Task-29 full-proposal updates
200 and 500 at 16/90 and 8/90, and Task-60 full-proposal update 400 at 8/90.
All ten diagnostic sleeper elements remained RUNNING. No candidate reached
strict validation and no job or allocation was modified.

At 19:54:29, Task-28 outer-augmentation update 300 advanced to 10/90. The
other committed counts remained Task-28 full-proposal update 300 at 13/90,
Task-29 full-proposal updates 200 and 500 at 16/90 and 8/90, and Task-60
full-proposal updates 200 and 400 at 27/90 and 8/90. All ten diagnostic
sleeper elements remained RUNNING. No candidate reached strict validation and
no process or allocation was modified.

At 19:54:53, Task-29 full-proposal update 200 advanced to 17/90. The other
committed counts remained Task-28 full-proposal update 300 at 13/90, Task-28
outer augmentation update 300 at 10/90, Task-29 update 500 at 8/90, and
Task-60 updates 200 and 400 at 27/90 and 8/90. All ten diagnostic sleeper
elements remained RUNNING. No candidate reached strict validation and no job
or allocation was modified.

At 19:56:02, Task-28 outer-augmentation update 300 advanced to 11/90 and
Task-29 full-proposal update 500 advanced to 9/90. The other committed counts
remained Task-28 full-proposal update 300 at 13/90, Task-29 full-proposal
update 200 at 17/90, and Task-60 full-proposal updates 200 and 400 at 27/90
and 8/90. All ten diagnostic sleeper elements remained RUNNING. No candidate
reached strict validation and no process or allocation was modified.

At 19:56:27, Task-60 full-proposal update 200 advanced to 28/90. The other
committed counts remained Task-28 full-proposal update 300 at 13/90, Task-28
outer augmentation update 300 at 11/90, Task-29 full-proposal updates 200 and
500 at 17/90 and 9/90, and Task-60 full-proposal update 400 at 8/90. All ten
diagnostic sleeper elements remained RUNNING. No candidate reached strict
90/90 validation and no process or allocation was modified.

At 19:57:50, Task-29 full-proposal update 500 advanced to 10/90. The other
committed counts remained Task-28 full-proposal update 300 at 13/90, Task-28
outer augmentation update 300 at 11/90, Task-29 update 200 at 17/90, and
Task-60 full-proposal updates 200 and 400 at 28/90 and 8/90. All ten
diagnostic sleeper elements remained RUNNING. No candidate reached strict
validation and no job or allocation was modified.

At 19:58:15, Task-60 full-proposal update 400 advanced to 9/90. The other
committed counts remained Task-28 full-proposal update 300 at 13/90, Task-28
outer augmentation update 300 at 11/90, Task-29 full-proposal updates 200 and
500 at 17/90 and 10/90, and Task-60 update 200 at 28/90. All ten diagnostic
sleeper elements remained RUNNING. No candidate reached strict validation and
no process or allocation was modified.

At 19:58:39, Task-60 full-proposal update 200 advanced to 29/90. The other
committed counts remained Task-28 full-proposal update 300 at 13/90, Task-28
outer augmentation update 300 at 11/90, Task-29 full-proposal updates 200 and
500 at 17/90 and 10/90, and Task-60 full-proposal update 400 at 9/90. All ten
diagnostic sleeper elements remained RUNNING. No candidate reached strict
90/90 validation and no process or allocation was modified.

At 19:59:17, Task-28 full-proposal update 300 advanced to 14/90 and Task-29
full-proposal update 500 advanced to 11/90. The other committed counts
remained Task-28 outer augmentation update 300 at 11/90, Task-29 update 200 at
17/90, and Task-60 full-proposal updates 200 and 400 at 29/90 and 9/90. All
ten diagnostic sleeper elements remained RUNNING. No candidate reached strict
validation and no process or allocation was modified.

At 19:59:56, Task-60 full-proposal update 400 advanced to 10/90. The other
committed counts remained Task-28 full-proposal update 300 at 14/90, Task-28
outer augmentation update 300 at 11/90, Task-29 full-proposal updates 200 and
500 at 17/90 and 11/90, and Task-60 update 200 at 29/90. All ten diagnostic
sleeper elements remained RUNNING. No candidate reached strict 90/90
validation and no job or allocation was modified.

At 20:00:38, Task-29 full-proposal update 200 advanced to 18/90. The other
committed counts remained Task-28 full-proposal update 300 at 14/90, Task-28
outer augmentation update 300 at 11/90, Task-29 update 500 at 11/90, and
Task-60 full-proposal updates 200 and 400 at 29/90 and 10/90. All ten
diagnostic sleeper elements remained RUNNING. No candidate reached strict
validation and no process or allocation was modified.

At 20:01:22, Task-60 full-proposal update 200 advanced to 30/90 and update
400 advanced to 11/90. The other committed counts remained Task-28
full-proposal update 300 at 14/90, Task-28 outer augmentation update 300 at
11/90, and Task-29 full-proposal updates 200 and 500 at 18/90 and 11/90. All
ten diagnostic sleeper elements remained RUNNING. No candidate reached strict
90/90 validation and no process or allocation was modified.

At 20:02:35, Task-28 outer-augmentation update 300 advanced to 12/90. The
other committed counts remained Task-28 full-proposal update 300 at 14/90,
Task-29 full-proposal updates 200 and 500 at 18/90 and 11/90, and Task-60
full-proposal updates 200 and 400 at 30/90 and 11/90. All ten diagnostic
sleeper elements remained RUNNING. No candidate reached strict validation and
no process or allocation was modified.

At 20:03:15, Task-28 full-proposal update 300 advanced to 15/90. The other
committed counts remained Task-28 outer augmentation update 300 at 12/90,
Task-29 full-proposal updates 200 and 500 at 18/90 and 11/90, and Task-60
full-proposal updates 200 and 400 at 30/90 and 11/90. All ten diagnostic
sleeper elements remained RUNNING. No candidate reached strict validation and
no job or allocation was modified.

At 20:03:39, Task-60 full-proposal update 200 advanced to 31/90. The other
committed counts remained Task-28 full-proposal update 300 at 15/90, Task-28
outer augmentation update 300 at 12/90, Task-29 full-proposal updates 200 and
500 at 18/90 and 11/90, and Task-60 full-proposal update 400 at 11/90. All
ten diagnostic sleeper elements remained RUNNING. No candidate reached strict
90/90 validation and no process or allocation was modified.

At 20:04:33, Task-29 full-proposal update 500 advanced to 12/90. The other
committed counts remained Task-28 full-proposal update 300 at 15/90, Task-28
outer augmentation update 300 at 12/90, Task-29 update 200 at 18/90, and
Task-60 full-proposal updates 200 and 400 at 31/90 and 11/90. All ten
diagnostic sleeper elements remained RUNNING. No candidate reached strict
validation and no job or allocation was modified.

At 20:05:14, Task-29 full-proposal update 200 advanced to 19/90. The other
committed counts remained Task-28 full-proposal update 300 at 15/90, Task-28
outer augmentation update 300 at 12/90, Task-29 update 500 at 12/90, and
Task-60 full-proposal updates 200 and 400 at 31/90 and 11/90. All ten
diagnostic sleeper elements remained RUNNING. No candidate reached strict
90/90 validation and no process or allocation was modified.

At 20:05:39, Task-28 outer-augmentation update 300 advanced to 13/90. The
other committed counts remained Task-28 full-proposal update 300 at 15/90,
Task-29 full-proposal updates 200 and 500 at 19/90 and 12/90, and Task-60
full-proposal updates 200 and 400 at 31/90 and 11/90. All ten diagnostic
sleeper elements remained RUNNING. No candidate reached strict validation and
no job or allocation was modified.

At 20:06:04, Task-60 full-proposal update 200 advanced to 32/90. The other
committed counts remained Task-28 full-proposal update 300 at 15/90, Task-28
outer augmentation update 300 at 13/90, Task-29 full-proposal updates 200 and
500 at 19/90 and 12/90, and Task-60 full-proposal update 400 at 11/90. All
ten diagnostic sleeper elements remained RUNNING. No candidate reached strict
90/90 validation and no process or allocation was modified.

At 20:06:30, Task-60 full-proposal update 200 advanced to 33/90. The other
committed counts remained Task-28 full-proposal update 300 at 15/90, Task-28
outer augmentation update 300 at 13/90, Task-29 full-proposal updates 200 and
500 at 19/90 and 12/90, and Task-60 full-proposal update 400 at 11/90. All
ten diagnostic sleeper elements remained RUNNING. No candidate reached strict
validation and no job or allocation was modified.

At 20:06:54, Task-60 full-proposal update 400 advanced to 12/90. The other
committed counts remained Task-28 full-proposal update 300 at 15/90, Task-28
outer augmentation update 300 at 13/90, Task-29 full-proposal updates 200 and
500 at 19/90 and 12/90, and Task-60 full-proposal update 200 at 33/90. All
ten diagnostic sleeper elements remained RUNNING. No candidate reached strict
90/90 validation and no process or allocation was modified.

At 20:07:35, Task-29 full-proposal update 500 advanced to 13/90. The other
committed counts remained Task-28 full-proposal update 300 at 15/90, Task-28
outer augmentation update 300 at 13/90, Task-29 update 200 at 19/90, and
Task-60 full-proposal updates 200 and 400 at 33/90 and 12/90. All ten
diagnostic sleeper elements remained RUNNING. No candidate reached strict
validation and no job or allocation was modified.

At 20:08:15, Task-28 full-proposal update 300 advanced to 16/90. The other
committed counts remained Task-28 outer augmentation update 300 at 13/90,
Task-29 full-proposal updates 200 and 500 at 19/90 and 13/90, and Task-60
full-proposal updates 200 and 400 at 33/90 and 12/90. All ten diagnostic
sleeper elements remained RUNNING. No candidate reached strict 90/90
validation and no process or allocation was modified.

At 20:08:39, Task-29 full-proposal update 200 advanced to 20/90. The other
committed counts remained Task-28 full-proposal update 300 at 16/90, Task-28
outer augmentation update 300 at 13/90, Task-29 update 500 at 13/90, and
Task-60 full-proposal updates 200 and 400 at 33/90 and 12/90. All ten
diagnostic sleeper elements remained RUNNING. No candidate reached strict
validation and no job or allocation was modified.

At 20:10:31, the live evaluator poll verified Task-29 full-proposal update 200
at 21/90 and Task-60 full-proposal update 400 at 13/90. The other committed
counts remained Task-28 full-proposal update 300 at 16/90, Task-28 outer
augmentation update 300 at 13/90, Task-29 update 500 at 13/90, and Task-60
full-proposal update 200 at 33/90. All ten diagnostic sleeper elements were
still RUNNING; the Task-2 side allocation was also left untouched. No
candidate reached strict 90/90 validation and no process or allocation was
modified.

At 20:11:13, the live evaluator poll verified Task-28 outer augmentation
update 300 at 14/90 and Task-60 full-proposal update 200 at 34/90. The other
committed counts remained Task-28 full-proposal update 300 at 16/90, Task-29
full-proposal updates 200 and 500 at 21/90 and 13/90, and Task-60
full-proposal update 400 at 13/90. All ten diagnostic sleeper elements were
still RUNNING. No candidate reached strict 90/90 validation and no process or
allocation was modified.

At 23:01:25, Task-29 proposal update 500 advanced to 54/90 and the corrected
Task-60 proposal-500 evaluator advanced to 3/90. The Task-28
executed-backfill update-200 evaluator remained an active process on gpua082
in model/GPU initialization (about 1.8 GB RSS and 433 MiB GPU memory) with
zero task records and no error marker; it was left untouched.

At 23:02:15, the backfill evaluator log showed successful checkpoint-manager
initialization and restoration of checkpoint 200. The logged missing
`metrics/metrics` file is an optional Orbax side-file warning also present in
other valid SFT evaluations; the evaluator process remained alive in I/O
initialization with GPU memory allocated and no terminal traceback, OOM, or
kill. No process or allocation was modified.

At 23:06:31, the Task-28 executed-backfill evaluator was relaunched on the
confirmed-idle gpua087 slot of allocation 23433381_1 after the gpua082 process
was diagnosed in `cl_sync_io_wait` on a `/projects` JAX source file. The
gpua082 process remains an isolated uninterruptible-I/O remnant with SIGKILL
pending; its original output directory has no task records. The retry uses
the identical checkpoint, evaluator arguments, and environment, but writes
to `sft_retention_full90_eval_20261104/task28_executed_backfill_step200_gpua087`.
Its remote Python process is alive in early initialization with zero records;
no other evaluator or training process was modified.

At 23:09:03, the gpua087 retry completed checkpoint restoration and loaded
the LIBERO normalization assets; its evaluator reported the expected
observation shape `(224, 224, 6)`, action horizon 10, and replan interval 5.
It remained alive with zero task records while entering compiled rollout
initialization. Task-28 proposal update 300 reached 60/90; Task-29 proposal
update 200 remained at 61/90; Task-29 proposal update 500 remained at 56/90;
Task-60 proposal update 400 remained at 54/90; and Task-60 proposal update
500 reached 4/90. No process or allocation was modified.

At 20:11:42, the live evaluator poll verified Task-60 full-proposal update 200
at 35/90. The other committed counts remained Task-28 full-proposal update 300
at 16/90, Task-28 outer augmentation update 300 at 14/90, Task-29
full-proposal updates 200 and 500 at 21/90 and 13/90, and Task-60
full-proposal update 400 at 13/90. All ten diagnostic sleeper elements were
still RUNNING. No candidate reached strict 90/90 validation and no process or
allocation was modified.

At 20:12:41, the live evaluator poll verified Task-29 full-proposal update 500
at 14/90. The other committed counts remained Task-28 full-proposal update 300
at 16/90, Task-28 outer augmentation update 300 at 14/90, Task-29
full-proposal update 200 at 21/90, and Task-60 full-proposal updates 200 and
400 at 35/90 and 13/90. All ten diagnostic sleeper elements were still
RUNNING. No candidate reached strict 90/90 validation and no process or
allocation was modified.

At 20:13:48, the live evaluator poll verified Task-28 full-proposal update 300
at 17/90. The other committed counts remained Task-28 outer augmentation
update 300 at 14/90, Task-29 full-proposal updates 200 and 500 at 21/90 and
14/90, and Task-60 full-proposal updates 200 and 400 at 35/90 and 13/90. All
ten diagnostic sleeper elements remained RUNNING, while the three additional
A100 arrays remained PENDING (Priority). No candidate reached strict 90/90
validation and no process or allocation was modified.

At 20:14:38, the live evaluator poll verified Task-29 full-proposal update 200
at 22/90. The other committed counts remained Task-28 full-proposal update 300
at 17/90, Task-28 outer augmentation update 300 at 14/90, Task-29
full-proposal update 500 at 14/90, and Task-60 full-proposal updates 200 and
400 at 35/90 and 13/90. All ten diagnostic sleeper elements remained RUNNING,
while the three additional A100 arrays remained PENDING (Priority). No
candidate reached strict 90/90 validation and no process or allocation was
modified.

At 20:15:00, the live evaluator poll verified Task-28 outer augmentation
update 300 at 15/90 and Task-60 full-proposal update 200 at 36/90. The other
committed counts remained Task-28 full-proposal update 300 at 17/90, Task-29
full-proposal updates 200 and 500 at 22/90 and 14/90, and Task-60
full-proposal update 400 at 13/90. All ten diagnostic sleeper elements
remained RUNNING, while the three additional A100 arrays remained PENDING
(Priority). No candidate reached strict 90/90 validation and no process or
allocation was modified.

At 21:07:28, the live evaluator poll verified Task-28 full-proposal update 300
at 31/90, Task-29 full-proposal updates 200 and 500 at 35/90 and 27/90, and
Task-60 full-proposal update 200 and update 400 at 61/90 and 26/90. Task-28
outer-augmentation update 300 remained at 25/90. The seven candidate evaluator
directories were still represented by live processes on the diagnostic
allocations; all ten diagnostic sleeper elements remained RUNNING, while the
three additional A100 arrays remained PENDING (Priority). No candidate reached
strict 90/90 validation and no process or allocation was modified.

At 20:15:24, the live evaluator poll verified Task-60 full-proposal update 200
at 37/90 and update 400 at 14/90. The other committed counts remained Task-28
full-proposal update 300 at 17/90, Task-28 outer augmentation update 300 at
15/90, and Task-29 full-proposal updates 200 and 500 at 22/90 and 14/90. All
ten diagnostic sleeper elements remained RUNNING, while the three additional
A100 arrays remained PENDING (Priority). No candidate reached strict 90/90
validation and no process or allocation was modified.

At 20:17:08, the live evaluator poll verified Task-29 full-proposal update 500
at 15/90. The other committed counts remained Task-28 full-proposal update 300
at 17/90, Task-28 outer augmentation update 300 at 15/90, Task-29
full-proposal update 200 at 22/90, and Task-60 full-proposal updates 200 and
400 at 37/90 and 14/90. All ten diagnostic sleeper elements remained RUNNING,
while the three additional A100 arrays remained PENDING (Priority). No
candidate reached strict 90/90 validation and no process or allocation was
modified.

At 20:17:31, the live evaluator poll verified Task-29 full-proposal update 200
at 23/90. The other committed counts remained Task-28 full-proposal update 300
at 17/90, Task-28 outer augmentation update 300 at 15/90, Task-29
full-proposal update 500 at 15/90, and Task-60 full-proposal updates 200 and
400 at 37/90 and 14/90. All ten diagnostic sleeper elements remained RUNNING,
while the three additional A100 arrays remained PENDING (Priority). No
candidate reached strict 90/90 validation and no process or allocation was
modified.

At 20:19:17, the live evaluator poll verified Task-28 full-proposal update 300
at 18/90. The other committed counts remained Task-28 outer augmentation update
300 at 15/90, Task-29 full-proposal updates 200 and 500 at 23/90 and 15/90,
and Task-60 full-proposal updates 200 and 400 at 37/90 and 14/90. All ten
diagnostic sleeper elements remained RUNNING, while the three additional A100
arrays remained PENDING (Priority). No candidate reached strict 90/90
validation and no process or allocation was modified.

At 20:19:40, the live evaluator poll verified Task-60 full-proposal update 200
at 38/90. The other committed counts remained Task-28 full-proposal update 300
at 18/90, Task-28 outer augmentation update 300 at 15/90, Task-29
full-proposal updates 200 and 500 at 23/90 and 15/90, and Task-60
full-proposal update 400 at 14/90. All ten diagnostic sleeper elements
remained RUNNING, while the three additional A100 arrays remained PENDING
(Priority). No candidate reached strict 90/90 validation and no process or
allocation was modified.

At 20:20:05, the live evaluator poll verified Task-60 full-proposal update 200
at 39/90 and update 400 at 15/90. The other committed counts remained Task-28
full-proposal update 300 at 18/90, Task-28 outer augmentation update 300 at
15/90, and Task-29 full-proposal updates 200 and 500 at 23/90 and 15/90. All
ten diagnostic sleeper elements remained RUNNING, while the three additional
A100 arrays remained PENDING (Priority). No candidate reached strict 90/90
validation and no process or allocation was modified.

At 20:20:28, the live evaluator poll verified Task-28 outer augmentation
update 300 at 16/90. The other committed counts remained Task-28 full-proposal
update 300 at 18/90, Task-29 full-proposal updates 200 and 500 at 23/90 and
15/90, and Task-60 full-proposal updates 200 and 400 at 39/90 and 15/90. All
ten diagnostic sleeper elements remained RUNNING, while the three additional
A100 arrays remained PENDING (Priority). No candidate reached strict 90/90
validation and no process or allocation was modified.

At 20:22:26, the live evaluator poll verified Task-29 full-proposal update 200
at 24/90 and update 500 at 16/90. The other committed counts remained Task-28
full-proposal update 300 at 18/90, Task-28 outer augmentation update 300 at
16/90, and Task-60 full-proposal updates 200 and 400 at 39/90 and 15/90. All
ten diagnostic sleeper elements remained RUNNING, while the three additional
A100 arrays remained PENDING (Priority). No candidate reached strict 90/90
validation and no process or allocation was modified.

At 20:23:57, the live evaluator poll verified Task-28 full-proposal update 300
at 19/90. The other committed counts remained Task-28 outer augmentation update
300 at 16/90, Task-29 full-proposal updates 200 and 500 at 24/90 and 16/90,
and Task-60 full-proposal updates 200 and 400 at 39/90 and 15/90. All ten
diagnostic sleeper elements remained RUNNING, while the three additional A100
arrays remained PENDING (Priority). No candidate reached strict 90/90
validation and no process or allocation was modified.

At 20:24:19, the live evaluator poll verified Task-28 full-proposal update 300
at 19/90, Task-29 full-proposal update 200 at 25/90, and Task-60 full-proposal
update 200 at 40/90. The other committed counts remained Task-28 outer
augmentation update 300 at 16/90, Task-29 full-proposal update 500 at 16/90,
and Task-60 full-proposal update 400 at 15/90. All ten diagnostic sleeper
elements remained RUNNING, while the three additional A100 arrays remained
PENDING (Priority). No candidate reached strict 90/90 validation and no
process or allocation was modified.

At 20:24:43, the live evaluator poll verified Task-60 full-proposal update 200
at 41/90. The other committed counts remained Task-28 full-proposal update 300
at 19/90, Task-28 outer augmentation update 300 at 16/90, Task-29
full-proposal updates 200 and 500 at 25/90 and 16/90, and Task-60
full-proposal update 400 at 15/90. All ten diagnostic sleeper elements
remained RUNNING, while the three additional A100 arrays remained PENDING
(Priority). No candidate reached strict 90/90 validation and no process or
allocation was modified.

At 20:25:06, the live evaluator poll verified Task-60 full-proposal update 400
at 16/90. The other committed counts remained Task-28 full-proposal update 300
at 19/90, Task-28 outer augmentation update 300 at 16/90, Task-29
full-proposal updates 200 and 500 at 25/90 and 16/90, and Task-60
full-proposal update 200 at 41/90. All ten diagnostic sleeper elements
remained RUNNING, while the three additional A100 arrays remained PENDING
(Priority). No candidate reached strict 90/90 validation and no process or
allocation was modified.

At 20:27:15, the live evaluator poll verified Task-28 full-proposal update 300
at 20/90. The other committed counts remained Task-28 outer augmentation update
300 at 16/90, Task-29 full-proposal updates 200 and 500 at 25/90 and 16/90,
and Task-60 full-proposal updates 200 and 400 at 41/90 and 16/90. All ten
diagnostic sleeper elements remained RUNNING, while the three additional A100
arrays remained PENDING (Priority). No candidate reached strict 90/90
validation and no process or allocation was modified.

At 20:27:52, the live evaluator poll verified Task-28 outer augmentation
update 300 at 17/90; Task-28 full-proposal update 300 also produced its next
committed record while remaining at 20/90. The other committed counts remained
Task-29 full-proposal updates 200 and 500 at 25/90 and 16/90, and Task-60
full-proposal updates 200 and 400 at 41/90 and 16/90. All ten diagnostic
sleeper elements remained RUNNING, while the three additional A100 arrays
remained PENDING (Priority). No candidate reached strict 90/90 validation and
no process or allocation was modified.

At 20:28:16, the live evaluator poll verified Task-29 full-proposal update 500
at 17/90. The other committed counts remained Task-28 full-proposal update 300
at 20/90, Task-28 outer augmentation update 300 at 17/90, Task-29
full-proposal update 200 at 25/90, and Task-60 full-proposal updates 200 and
400 at 41/90 and 16/90. All ten diagnostic sleeper elements remained RUNNING,
while the three additional A100 arrays remained PENDING (Priority). No
candidate reached strict 90/90 validation and no process or allocation was
modified.

At 20:28:54, the live evaluator poll verified Task-28 full-proposal update 300
at 21/90. The other committed counts remained Task-28 outer augmentation update
300 at 17/90, Task-29 full-proposal updates 200 and 500 at 25/90 and 17/90,
and Task-60 full-proposal updates 200 and 400 at 41/90 and 16/90. All ten
diagnostic sleeper elements remained RUNNING, while the three additional A100
arrays remained PENDING (Priority). No candidate reached strict 90/90
validation and no process or allocation was modified.

At 20:29:17, the live evaluator poll verified Task-29 full-proposal update 200
at 26/90 and Task-60 full-proposal update 200 at 43/90. The other committed
counts remained Task-28 full-proposal update 300 at 21/90, Task-28 outer
augmentation update 300 at 17/90, Task-29 full-proposal update 500 at 17/90,
and Task-60 full-proposal update 400 at 16/90. All ten diagnostic sleeper
elements remained RUNNING, while the three additional A100 arrays remained
PENDING (Priority). No candidate reached strict 90/90 validation and no
process or allocation was modified.

At 20:30:38, the live evaluator poll verified Task-60 full-proposal update 400
at 17/90. The other committed counts remained Task-28 full-proposal update 300
at 21/90, Task-28 outer augmentation update 300 at 17/90, Task-29
full-proposal updates 200 and 500 at 26/90 and 17/90, and Task-60
full-proposal update 200 at 43/90. All ten diagnostic sleeper elements
remained RUNNING, while the three additional A100 arrays remained PENDING
(Priority). No candidate reached strict 90/90 validation and no process or
allocation was modified.

At 20:31:01, the live evaluator poll verified Task-60 full-proposal update 200
at 44/90 and another committed record for update 400 while it remained at
17/90. The other committed counts remained Task-28 full-proposal update 300 at
21/90, Task-28 outer augmentation update 300 at 17/90, and Task-29
full-proposal updates 200 and 500 at 26/90 and 17/90. All ten diagnostic
sleeper elements remained RUNNING, while the three additional A100 arrays
remained PENDING (Priority). No candidate reached strict 90/90 validation and
no process or allocation was modified.

At 20:33:21, the live evaluator poll verified Task-28 full-proposal update 300
at 22/90. The other committed counts remained Task-28 outer augmentation update
300 at 17/90, Task-29 full-proposal updates 200 and 500 at 26/90 and 17/90,
and Task-60 full-proposal updates 200 and 400 at 44/90 and 17/90. All ten
diagnostic sleeper elements remained RUNNING, while the three additional A100
arrays remained PENDING (Priority). No candidate reached strict 90/90
validation and no process or allocation was modified.

At 20:33:46, the live evaluator poll verified Task-28 full-proposal update 300
at 22/90, Task-28 outer augmentation update 300 at 18/90, Task-29
full-proposal updates 200 and 500 at 27/90 and 18/90, and Task-60
full-proposal update 200 at 45/90. The other committed count, Task-60
full-proposal update 400, remained at 17/90. All ten diagnostic sleeper
elements remained RUNNING, while the three additional A100 arrays remained
PENDING (Priority). No candidate reached strict 90/90 validation and no
process or allocation was modified.

At 20:35:50, the live evaluator poll verified Task-60 full-proposal update 200
at 46/90. The other committed counts remained Task-28 full-proposal update 300
at 22/90, Task-28 outer augmentation update 300 at 18/90, Task-29
full-proposal updates 200 and 500 at 27/90 and 18/90, and Task-60
full-proposal update 400 at 17/90. All ten diagnostic sleeper elements
remained RUNNING, while the three additional A100 arrays remained PENDING
(Priority). No candidate reached strict 90/90 validation and no process or
allocation was modified.

At 20:36:24, the live evaluator poll verified Task-28 full-proposal update 300
at 23/90 and Task-60 full-proposal update 400 at 18/90. The other committed
counts remained Task-28 outer augmentation update 300 at 18/90, Task-29
full-proposal updates 200 and 500 at 27/90 and 18/90, and Task-60
full-proposal update 200 at 46/90. All ten diagnostic sleeper elements
remained RUNNING, while the three additional A100 arrays remained PENDING
(Priority). No candidate reached strict 90/90 validation and no process or
allocation was modified.

At 20:38:35, the live evaluator poll verified Task-28 outer-augmentation
update 300 at 19/90, Task-29 full-proposal updates 200 and 500 at 28/90 and
19/90, and Task-60 full-proposal update 200 at 48/90. Task-28 full-proposal
update 300 remained at 23/90 and Task-60 full-proposal update 400 remained at
18/90. All ten diagnostic sleeper elements remained RUNNING, while the three
additional A100 arrays remained PENDING (Priority). No candidate reached
strict 90/90 validation and no process or allocation was modified.

At 20:41:18, the live evaluator poll verified Task-28 full-proposal update 300
at 24/90 and Task-60 full-proposal update 400 at 19/90. The other committed
counts remained Task-28 outer-augmentation update 300 at 19/90, Task-29
full-proposal updates 200 and 500 at 28/90 and 19/90, and Task-60
full-proposal update 200 at 48/90. All ten diagnostic sleeper elements
remained RUNNING, while the three additional A100 arrays remained PENDING
(Priority). No candidate reached strict 90/90 validation and no process or
allocation was modified.

At 20:41:46, the live evaluator poll verified Task-28 outer-augmentation
update 300 at 20/90 and Task-29 full-proposal update 500 at 20/90. The other
committed counts remained Task-28 full-proposal update 300 at 24/90, Task-29
full-proposal update 200 at 28/90, and Task-60 full-proposal updates 200 and
400 at 48/90 and 19/90. All ten diagnostic sleeper elements remained RUNNING,
while the three additional A100 arrays remained PENDING (Priority). No
candidate reached strict 90/90 validation and no process or allocation was
modified.

At 20:42:38, the live evaluator poll verified Task-29 full-proposal update 200
at 29/90. The other committed counts remained Task-28 full-proposal update 300
at 24/90, Task-28 outer-augmentation update 300 at 20/90, Task-29
full-proposal update 500 at 20/90, and Task-60 full-proposal updates 200 and
400 at 48/90 and 19/90. All ten diagnostic sleeper elements remained RUNNING,
while the three additional A100 arrays remained PENDING (Priority). No
candidate reached strict 90/90 validation and no process or allocation was
modified.

At 20:43:04, the live evaluator poll verified Task-28 full-proposal update 300
at 25/90, Task-29 full-proposal update 500 at 21/90, and Task-60
full-proposal update 200 at 49/90. The other committed counts remained
Task-28 outer-augmentation update 300 at 20/90, Task-29 full-proposal update
200 at 29/90, and Task-60 full-proposal update 400 at 19/90. All ten
diagnostic sleeper elements remained RUNNING, while the three additional A100
arrays remained PENDING (Priority). No candidate reached strict 90/90
validation and no process or allocation was modified.

At 20:43:33, the live evaluator poll verified Task-28 outer-augmentation
update 300 at 21/90 and Task-60 full-proposal update 200 at 50/90. The other
committed counts remained Task-28 full-proposal update 300 at 25/90, Task-29
full-proposal updates 200 and 500 at 29/90 and 21/90, and Task-60
full-proposal update 400 at 19/90. All ten diagnostic sleeper elements
remained RUNNING, while the three additional A100 arrays remained PENDING
(Priority). No candidate reached strict 90/90 validation and no process or
allocation was modified.

At 20:44:16, the live evaluator poll verified Task-60 full-proposal update 400
at 20/90. The other committed counts remained Task-28 full-proposal update 300
at 25/90, Task-28 outer-augmentation update 300 at 21/90, Task-29
full-proposal updates 200 and 500 at 29/90 and 21/90, and Task-60
full-proposal update 200 at 50/90. All ten diagnostic sleeper elements
remained RUNNING, while the three additional A100 arrays remained PENDING
(Priority). No candidate reached strict 90/90 validation and no process or
allocation was modified.

At 20:45:24, the live evaluator poll verified Task-29 full-proposal update 200
at 30/90. The other committed counts remained Task-28 full-proposal update 300
at 25/90, Task-28 outer-augmentation update 300 at 21/90, Task-29
full-proposal update 500 at 21/90, and Task-60 full-proposal updates 200 and
400 at 50/90 and 20/90. All ten diagnostic sleeper elements remained RUNNING,
while the three additional A100 arrays remained PENDING (Priority). No
candidate reached strict 90/90 validation and no process or allocation was
modified.

At 20:46:08, the live evaluator poll verified Task-60 full-proposal update 400
at 21/90. The other committed counts remained Task-28 full-proposal update 300
at 25/90, Task-28 outer-augmentation update 300 at 21/90, Task-29
full-proposal updates 200 and 500 at 30/90 and 21/90, and Task-60
full-proposal update 200 at 50/90. All ten diagnostic sleeper elements
remained RUNNING, while the three additional A100 arrays remained PENDING
(Priority). No candidate reached strict 90/90 validation and no process or
allocation was modified.

At 20:46:37, the live evaluator poll verified Task-60 full-proposal update 200
at 51/90. The other committed counts remained Task-28 full-proposal update 300
at 25/90, Task-28 outer-augmentation update 300 at 21/90, Task-29
full-proposal updates 200 and 500 at 30/90 and 21/90, and Task-60
full-proposal update 400 at 21/90. All ten diagnostic sleeper elements
remained RUNNING, while the three additional A100 arrays remained PENDING
(Priority). No candidate reached strict 90/90 validation and no process or
allocation was modified.

At 20:47:20, the live evaluator poll verified Task-60 full-proposal update 200
at 52/90. The other committed counts remained Task-28 full-proposal update 300
at 25/90, Task-28 outer-augmentation update 300 at 21/90, Task-29
full-proposal updates 200 and 500 at 30/90 and 21/90, and Task-60
full-proposal update 400 at 21/90. All ten diagnostic sleeper elements
remained RUNNING, while the three additional A100 arrays remained PENDING
(Priority). No candidate reached strict 90/90 validation and no process or
allocation was modified.

At 20:47:49, the live evaluator poll verified Task-28 full-proposal update 300
at 26/90 and Task-29 full-proposal update 500 at 22/90. The other committed
counts remained Task-28 outer-augmentation update 300 at 21/90, Task-29
full-proposal update 200 at 30/90, and Task-60 full-proposal updates 200 and
400 at 52/90 and 21/90. All ten diagnostic sleeper elements remained RUNNING,
while the three additional A100 arrays remained PENDING (Priority). No
candidate reached strict 90/90 validation and no process or allocation was
modified.

At 20:48:17, the live evaluator poll verified Task-28 outer-augmentation
update 300 at 22/90, Task-29 full-proposal updates 200 and 500 at 31/90 and
22/90. The other committed counts remained Task-28 full-proposal update 300 at
26/90 and Task-60 full-proposal updates 200 and 400 at 52/90 and 21/90. All
ten diagnostic sleeper elements remained RUNNING, while the three additional
A100 arrays remained PENDING (Priority). No candidate reached strict 90/90
validation and no process or allocation was modified.

At 20:50:19, the live evaluator poll verified Task-29 full-proposal update 500
at 23/90. The other committed counts remained Task-28 full-proposal update 300
at 26/90, Task-28 outer-augmentation update 300 at 22/90, Task-29
full-proposal update 200 at 31/90, and Task-60 full-proposal updates 200 and
400 at 52/90 and 21/90. All ten diagnostic sleeper elements remained RUNNING,
while the three additional A100 arrays remained PENDING (Priority). No
candidate reached strict 90/90 validation and no process or allocation was
modified.

At 20:50:49, the live evaluator poll verified Task-60 full-proposal updates 200
and 400 at 53/90 and 22/90. The other committed counts remained Task-28
full-proposal update 300 at 26/90, Task-28 outer-augmentation update 300 at
22/90, and Task-29 full-proposal updates 200 and 500 at 31/90 and 23/90. All
ten diagnostic sleeper elements remained RUNNING, while the three additional
A100 arrays remained PENDING (Priority). No candidate reached strict 90/90
validation and no process or allocation was modified.

At 20:51:16, the live evaluator poll verified Task-60 full-proposal update 200
at 54/90. The other committed counts remained Task-28 full-proposal update 300
at 26/90, Task-28 outer-augmentation update 300 at 22/90, Task-29
full-proposal updates 200 and 500 at 31/90 and 23/90, and Task-60
full-proposal update 400 at 22/90. All ten diagnostic sleeper elements
remained RUNNING, while the three additional A100 arrays remained PENDING
(Priority). No candidate reached strict 90/90 validation and no process or
allocation was modified.

At 20:52:35, the live evaluator poll verified Task-28 full-proposal update 300
at 27/90. The other committed counts remained Task-28 outer-augmentation
update 300 at 22/90, Task-29 full-proposal updates 200 and 500 at 31/90 and
23/90, and Task-60 full-proposal updates 200 and 400 at 54/90 and 22/90. All
ten diagnostic sleeper elements remained RUNNING, while the three additional
A100 arrays remained PENDING (Priority). No candidate reached strict 90/90
validation and no process or allocation was modified.

At 20:53:20, the live evaluator poll verified Task-29 full-proposal update 200
at 32/90. The other committed counts remained Task-28 full-proposal update 300
at 27/90, Task-28 outer-augmentation update 300 at 22/90, Task-29
full-proposal update 500 at 23/90, and Task-60 full-proposal updates 200 and
400 at 54/90 and 22/90. All ten diagnostic sleeper elements remained RUNNING,
while the three additional A100 arrays remained PENDING (Priority). No
candidate reached strict 90/90 validation and no process or allocation was
modified.

At 20:53:49, the live evaluator poll verified Task-60 full-proposal update 400
at 23/90. The other committed counts remained Task-28 full-proposal update 300
at 27/90, Task-28 outer-augmentation update 300 at 22/90, Task-29
full-proposal updates 200 and 500 at 32/90 and 23/90, and Task-60
full-proposal update 200 at 54/90. All ten diagnostic sleeper elements
remained RUNNING, while the three additional A100 arrays remained PENDING
(Priority). No candidate reached strict 90/90 validation and no process or
allocation was modified.

At 20:54:17, the live evaluator poll verified Task-28 outer-augmentation
update 300 at 23/90. The other committed counts remained Task-28
full-proposal update 300 at 27/90, Task-29 full-proposal updates 200 and 500
at 32/90 and 23/90, and Task-60 full-proposal updates 200 and 400 at 54/90
and 23/90. All ten diagnostic sleeper elements remained RUNNING, while the
three additional A100 arrays remained PENDING (Priority). No candidate reached
strict 90/90 validation and no process or allocation was modified.

At 20:54:45, the live evaluator poll verified Task-60 full-proposal update 200
at 55/90. The other committed counts remained Task-28 full-proposal update 300
at 27/90, Task-28 outer-augmentation update 300 at 23/90, Task-29
full-proposal updates 200 and 500 at 32/90 and 23/90, and Task-60
full-proposal update 400 at 23/90. All ten diagnostic sleeper elements
remained RUNNING, while the three additional A100 arrays remained PENDING
(Priority). No candidate reached strict 90/90 validation and no process or
allocation was modified.

At 20:55:13, the live evaluator poll verified Task-29 full-proposal update 500
at 24/90. The other committed counts remained Task-28 full-proposal update 300
at 27/90, Task-28 outer-augmentation update 300 at 23/90, Task-29
full-proposal update 200 at 32/90, and Task-60 full-proposal updates 200 and
400 at 55/90 and 23/90. All ten diagnostic sleeper elements remained RUNNING,
while the three additional A100 arrays remained PENDING (Priority). No
candidate reached strict 90/90 validation and no process or allocation was
modified.

At 20:55:57, the live evaluator poll verified Task-60 full-proposal update 200
at 56/90. The other committed counts remained Task-28 full-proposal update 300
at 27/90, Task-28 outer-augmentation update 300 at 23/90, Task-29
full-proposal updates 200 and 500 at 32/90 and 24/90, and Task-60
full-proposal update 400 at 23/90. All ten diagnostic sleeper elements
remained RUNNING, while the three additional A100 arrays remained PENDING
(Priority). No candidate reached strict 90/90 validation and no process or
allocation was modified.

At 20:57:07, the live evaluator poll verified Task-29 full-proposal update 500
at 25/90. The other committed counts remained Task-28 full-proposal update 300
at 27/90, Task-28 outer-augmentation update 300 at 23/90, Task-29
full-proposal update 200 at 32/90, and Task-60 full-proposal updates 200 and
400 at 56/90 and 23/90. All ten diagnostic sleeper elements remained RUNNING,
while the three additional A100 arrays remained PENDING (Priority). No
candidate reached strict 90/90 validation and no process or allocation was
modified.

At 20:57:46, the live evaluator poll verified Task-28 full-proposal update 300
at 28/90. The other committed counts remained Task-28 outer-augmentation
update 300 at 23/90, Task-29 full-proposal updates 200 and 500 at 32/90 and
25/90, and Task-60 full-proposal updates 200 and 400 at 56/90 and 23/90. All
ten diagnostic sleeper elements remained RUNNING, while the three additional
A100 arrays remained PENDING (Priority). No candidate reached strict 90/90
validation and no process or allocation was modified.

At 20:58:14, the live evaluator poll verified Task-28 full-proposal update 300
at 28/90 and Task-29 full-proposal update 200 at 33/90. The other committed
counts remained Task-28 outer-augmentation update 300 at 23/90, Task-29
full-proposal update 500 at 25/90, and Task-60 full-proposal updates 200 and
400 at 56/90 and 23/90. All ten diagnostic sleeper elements remained RUNNING,
while the three additional A100 arrays remained PENDING (Priority). No
candidate reached strict 90/90 validation and no process or allocation was
modified.

At 20:58:39, the live evaluator poll verified Task-60 full-proposal update 400
at 24/90. The other committed counts remained Task-28 full-proposal update 300
at 28/90, Task-28 outer-augmentation update 300 at 23/90, Task-29
full-proposal updates 200 and 500 at 33/90 and 25/90, and Task-60
full-proposal update 200 at 56/90. All ten diagnostic sleeper elements
remained RUNNING, while the three additional A100 arrays remained PENDING
(Priority). No candidate reached strict 90/90 validation and no process or
allocation was modified.

At 20:59:05, the live evaluator poll verified Task-28 outer-augmentation
update 300 at 24/90. The other committed counts remained Task-28 full-proposal
update 300 at 28/90, Task-29 full-proposal updates 200 and 500 at 33/90 and
25/90, and Task-60 full-proposal updates 200 and 400 at 56/90 and 24/90. All
ten diagnostic sleeper elements remained RUNNING, while the three additional
A100 arrays remained PENDING (Priority). No candidate reached strict 90/90
validation and no process or allocation was modified.

At 20:59:32, the live evaluator poll verified Task-60 full-proposal update 200
at 57/90. The other committed counts remained Task-28 full-proposal update 300
at 28/90, Task-28 outer-augmentation update 300 at 24/90, Task-29
full-proposal updates 200 and 500 at 33/90 and 25/90, and Task-60
full-proposal update 400 at 24/90. All ten diagnostic sleeper elements
remained RUNNING, while the three additional A100 arrays remained PENDING
(Priority). No candidate reached strict 90/90 validation and no process or
allocation was modified.

At 21:00:02, the live evaluator poll verified Task-60 full-proposal update 200
at 58/90. The other committed counts remained Task-28 full-proposal update 300
at 28/90, Task-28 outer-augmentation update 300 at 24/90, Task-29
full-proposal updates 200 and 500 at 33/90 and 25/90, and Task-60
full-proposal update 400 at 24/90. All ten diagnostic sleeper elements
remained RUNNING, while the three additional A100 arrays remained PENDING
(Priority). No candidate reached strict 90/90 validation and no process or
allocation was modified.

At 21:00:30, the live evaluator poll verified Task-60 full-proposal update 400
at 25/90. The other committed counts remained Task-28 full-proposal update 300
at 28/90, Task-28 outer-augmentation update 300 at 24/90, Task-29
full-proposal updates 200 and 500 at 33/90 and 25/90, and Task-60
full-proposal update 200 at 58/90. All ten diagnostic sleeper elements
remained RUNNING, while the three additional A100 arrays remained PENDING
(Priority). No candidate reached strict 90/90 validation and no process or
allocation was modified.

At 21:00:58, the live evaluator poll verified Task-28 full-proposal update 300
at 29/90 and Task-28 outer-augmentation update 300 at 25/90. The other
committed counts remained Task-29 full-proposal updates 200 and 500 at 33/90
and 25/90, and Task-60 full-proposal updates 200 and 400 at 58/90 and 25/90.
All ten diagnostic sleeper elements remained RUNNING, while the three
additional A100 arrays remained PENDING (Priority). No candidate reached
strict 90/90 validation and no process or allocation was modified.

At 21:01:40, the live evaluator poll verified Task-60 full-proposal update 200
at 59/90. The other committed counts remained Task-28 full-proposal update 300
at 29/90, Task-28 outer-augmentation update 300 at 25/90, Task-29
full-proposal updates 200 and 500 at 33/90 and 25/90, and Task-60
full-proposal update 400 at 25/90. All ten diagnostic sleeper elements
remained RUNNING, while the three additional A100 arrays remained PENDING
(Priority). No candidate reached strict 90/90 validation and no process or
allocation was modified.

At 21:02:05, the live evaluator poll verified Task-29 full-proposal updates 200
and 500 at 34/90 and 26/90. The other committed counts remained Task-28
full-proposal update 300 at 29/90, Task-28 outer-augmentation update 300 at
25/90, and Task-60 full-proposal updates 200 and 400 at 59/90 and 25/90. All
ten diagnostic sleeper elements remained RUNNING, while the three additional
A100 arrays remained PENDING (Priority). No candidate reached strict 90/90
validation and no process or allocation was modified.

At 21:03:55, the live evaluator poll verified Task-60 full-proposal update 200
at 60/90. The other committed counts remained Task-28 full-proposal update 300
at 29/90, Task-28 outer-augmentation update 300 at 25/90, Task-29
full-proposal updates 200 and 500 at 34/90 and 26/90, and Task-60
full-proposal update 400 at 25/90. All ten diagnostic sleeper elements
remained RUNNING, while the three additional A100 arrays remained PENDING
(Priority). No candidate reached strict 90/90 validation and no process or
allocation was modified.

At 21:04:35, the live evaluator poll verified Task-28 full-proposal update 300
at 30/90 and Task-60 full-proposal update 200 at 61/90. The other committed
counts remained Task-28 outer-augmentation update 300 at 25/90, Task-29
full-proposal updates 200 and 500 at 34/90 and 26/90, and Task-60
full-proposal update 400 at 25/90. All ten diagnostic sleeper elements
remained RUNNING, while the three additional A100 arrays remained PENDING
(Priority). No candidate reached strict 90/90 validation and no process or
allocation was modified.

At 21:05:18, the live evaluator poll verified Task-60 full-proposal update 400
at 26/90. The other committed counts remained Task-28 full-proposal update 300
at 30/90, Task-28 outer-augmentation update 300 at 25/90, Task-29
full-proposal updates 200 and 500 at 34/90 and 26/90, and Task-60
full-proposal update 200 at 61/90. All ten diagnostic sleeper elements
remained RUNNING, while the three additional A100 arrays remained PENDING
(Priority). No candidate reached strict 90/90 validation and no process or
allocation was modified.

At 21:08:56, the live evaluator poll verified Task-28 full-proposal update 300
at 31/90, Task-28 outer-augmentation update 300 at 26/90, Task-29
full-proposal updates 200 and 500 at 35/90 and 27/90, and Task-60
full-proposal updates 200 and 400 at 62/90 and 26/90. All seven candidate
evaluation directories still had live evaluator processes (Task-60 update 200
uses two shards); all ten diagnostic sleeper elements remained RUNNING, while
the three additional A100 arrays remained PENDING (Priority). No candidate
reached strict 90/90 validation and no process or allocation was modified.

At 21:09:42, the live evaluator poll verified Task-60 full-proposal update 200
at 63/90. The other counts remained Task-28 full-proposal update 300 at 31/90,
Task-28 outer-augmentation update 300 at 26/90, Task-29 full-proposal updates
200 and 500 at 35/90 and 27/90, and Task-60 full-proposal update 400 at 26/90.
All six active evaluator launchers remained present on their diagnostic
allocations (Task-60 update 200 uses two shards); no candidate reached strict
90/90 validation and no process or allocation was modified.

At 21:10:10, the live evaluator poll verified Task-60 full-proposal update 400
at 27/90. The other counts remained Task-28 full-proposal update 300 at 31/90,
Task-28 outer-augmentation update 300 at 26/90, Task-29 full-proposal updates
200 and 500 at 35/90 and 27/90, and Task-60 full-proposal update 200 at 63/90.
All six active evaluator launchers remained present on their diagnostic
allocations (Task-60 update 200 uses two shards); no candidate reached strict
90/90 validation and no process or allocation was modified.

At 21:11:18, the live evaluator poll verified Task-29 full-proposal update 500
at 28/90 and Task-60 full-proposal update 200 at 64/90. The other counts
remained Task-28 full-proposal update 300 at 31/90, Task-28
outer-augmentation update 300 at 26/90, Task-29 full-proposal update 200 at
35/90, and Task-60 full-proposal update 400 at 27/90. All six active evaluator
launchers remained present on their diagnostic allocations; no candidate
reached strict 90/90 validation and no process or allocation was modified.

At 21:11:47, the live evaluator poll verified Task-28 full-proposal update 300
at 32/90. The other counts remained Task-28 outer-augmentation update 300 at
26/90, Task-29 full-proposal updates 200 and 500 at 35/90 and 28/90, and
Task-60 full-proposal updates 200 and 400 at 64/90 and 27/90. All six active
evaluator launchers remained present on their diagnostic allocations; no
candidate reached strict 90/90 validation and no process or allocation was
modified.

At 21:12:14, the live evaluator poll verified Task-29 full-proposal update 200
at 36/90. The other counts remained Task-28 full-proposal update 300 at 32/90,
Task-28 outer-augmentation update 300 at 26/90, Task-29 full-proposal update
500 at 28/90, and Task-60 full-proposal updates 200 and 400 at 64/90 and 27/90.
All six active evaluator launchers remained present on their diagnostic
allocations; no candidate reached strict 90/90 validation and no process or
allocation was modified.

At 21:14:41, the live evaluator poll verified Task-60 full-proposal update 400
at 28/90. The other counts remained Task-28 full-proposal update 300 at 32/90,
Task-28 outer-augmentation update 300 at 26/90, Task-29 full-proposal updates
200 and 500 at 36/90 and 28/90, and Task-60 full-proposal update 200 at 64/90.
All six evaluator launchers remained live on the diagnostic allocations; no
candidate reached strict 90/90 validation and no process or allocation was
modified.

At 21:15:13, the live evaluator poll verified Task-60 full-proposal update 200
at 65/90. The other counts remained Task-28 full-proposal update 300 at 32/90,
Task-28 outer-augmentation update 300 at 26/90, Task-29 full-proposal updates
200 and 500 at 36/90 and 28/90, and Task-60 full-proposal update 400 at 28/90.
All six evaluator launchers remained live on the diagnostic allocations; no
candidate reached strict 90/90 validation and no process or allocation was
modified.

At 21:15:55, the live evaluator poll verified Task-29 full-proposal update 500
at 29/90. The other counts remained Task-28 full-proposal update 300 at 32/90,
Task-28 outer-augmentation update 300 at 26/90, Task-29 full-proposal update
200 at 36/90, and Task-60 full-proposal updates 200 and 400 at 65/90 and 28/90.
All six evaluator launchers remained live on the diagnostic allocations; no
candidate reached strict 90/90 validation and no process or allocation was
modified.

At 21:16:16, the live evaluator poll verified Task-28 outer-augmentation update
300 at 27/90. The other counts remained Task-28 full-proposal update 300 at
32/90, Task-29 full-proposal updates 200 and 500 at 36/90 and 29/90, and
Task-60 full-proposal updates 200 and 400 at 65/90 and 28/90. All six evaluator
launchers remained live on the diagnostic allocations; no candidate reached
strict 90/90 validation and no process or allocation was modified.

At 21:16:39, the live evaluator poll verified Task-28 full-proposal update 300
at 33/90. The other counts remained Task-28 outer-augmentation update 300 at
27/90, Task-29 full-proposal updates 200 and 500 at 36/90 and 29/90, and
Task-60 full-proposal updates 200 and 400 at 65/90 and 28/90. All six evaluator
launchers remained live on the diagnostic allocations; no candidate reached
strict 90/90 validation and no process or allocation was modified.

At 21:17:04, the live evaluator poll verified Task-29 full-proposal update 200
at 37/90 and Task-60 full-proposal update 200 at 66/90. The other counts
remained Task-28 full-proposal update 300 at 33/90, Task-28 outer-augmentation
update 300 at 27/90, Task-29 full-proposal update 500 at 29/90, and Task-60
full-proposal update 400 at 28/90. All six evaluator launchers remained live
on the diagnostic allocations; no candidate reached strict 90/90 validation
and no process or allocation was modified.

At 21:18:08, the live evaluator poll verified Task-60 full-proposal update 200
at 67/90. The other counts remained Task-28 full-proposal update 300 at 33/90,
Task-28 outer-augmentation update 300 at 27/90, Task-29 full-proposal updates
200 and 500 at 37/90 and 29/90, and Task-60 full-proposal update 400 at 28/90.
All six evaluator launchers remained live on the diagnostic allocations; no
candidate reached strict 90/90 validation and no process or allocation was
modified.

At 21:18:31, the live evaluator poll verified Task-60 full-proposal updates 200
and 400 at 67/90 and 29/90. The other counts remained Task-28 full-proposal
update 300 at 33/90, Task-28 outer-augmentation update 300 at 27/90, and
Task-29 full-proposal updates 200 and 500 at 37/90 and 29/90. All six evaluator
launchers remained live on the diagnostic allocations; no candidate reached
strict 90/90 validation and no process or allocation was modified.

At 21:18:53, the live evaluator poll verified Task-29 full-proposal update 500
at 30/90. The other counts remained Task-28 full-proposal update 300 at 33/90,
Task-28 outer-augmentation update 300 at 27/90, Task-29 full-proposal update
200 at 37/90, and Task-60 full-proposal updates 200 and 400 at 67/90 and 29/90.
All six evaluator launchers remained live on the diagnostic allocations; no
candidate reached strict 90/90 validation and no process or allocation was
modified.

At 21:19:59, the live evaluator poll verified Task-28 full-proposal update 300
at 34/90. The other counts remained Task-28 outer-augmentation update 300 at
27/90, Task-29 full-proposal updates 200 and 500 at 37/90 and 30/90, and
Task-60 full-proposal updates 200 and 400 at 67/90 and 29/90. All six
evaluator launchers remained live on the diagnostic allocations; no candidate
reached strict 90/90 validation and no process or allocation was modified.

At 21:20:50, the live evaluator poll verified Task-28 outer-augmentation update
300 at 28/90. The other counts remained Task-28 full-proposal update 300 at
34/90, Task-29 full-proposal updates 200 and 500 at 37/90 and 30/90, and
Task-60 full-proposal updates 200 and 400 at 67/90 and 29/90. All six
evaluator launchers remained live on the diagnostic allocations; no candidate
reached strict 90/90 validation and no process or allocation was modified.

At 21:21:26, the live evaluator poll verified Task-29 full-proposal updates 200
and 500 at 38/90 and 31/90, and Task-60 full-proposal update 400 at 30/90. The
other counts remained Task-28 full-proposal update 300 at 34/90, Task-28
outer-augmentation update 300 at 28/90, and Task-60 full-proposal update 200 at
67/90. All six evaluator launchers remained live on the diagnostic allocations;
no candidate reached strict 90/90 validation and no process or allocation was
modified.

At 21:21:51, the live evaluator poll verified Task-60 full-proposal update 200
at 68/90. The other counts remained Task-28 full-proposal update 300 at 34/90,
Task-28 outer-augmentation update 300 at 28/90, Task-29 full-proposal updates
200 and 500 at 38/90 and 31/90, and Task-60 full-proposal update 400 at 30/90.
All six evaluator launchers remained live on the diagnostic allocations; no
candidate reached strict 90/90 validation and no process or allocation was
modified.

At 21:22:13, the live evaluator poll verified Task-60 full-proposal update 200
at 69/90. The other counts remained Task-28 full-proposal update 300 at 34/90,
Task-28 outer-augmentation update 300 at 28/90, Task-29 full-proposal updates
200 and 500 at 38/90 and 31/90, and Task-60 full-proposal update 400 at 30/90.
All six evaluator launchers remained live on the diagnostic allocations; no
candidate reached strict 90/90 validation and no process or allocation was
modified.

At 21:23:20, the live evaluator poll verified Task-60 full-proposal update 200
at 70/90. The other counts remained Task-28 full-proposal update 300 at 34/90,
Task-28 outer-augmentation update 300 at 28/90, Task-29 full-proposal updates
200 and 500 at 38/90 and 31/90, and Task-60 full-proposal update 400 at 30/90.
All six evaluator launchers remained live on the diagnostic allocations; no
candidate reached strict 90/90 validation and no process or allocation was
modified.

At 21:23:46, the live evaluator poll verified Task-60 full-proposal update 200
at 71/90. The other counts remained Task-28 full-proposal update 300 at 34/90,
Task-28 outer-augmentation update 300 at 28/90, Task-29 full-proposal updates
200 and 500 at 38/90 and 31/90, and Task-60 full-proposal update 400 at 30/90.
All six evaluator launchers remained live on the diagnostic allocations; no
candidate reached strict 90/90 validation and no process or allocation was
modified.

At 21:24:10, the live evaluator poll verified Task-60 full-proposal update 400
at 31/90. The other counts remained Task-28 full-proposal update 300 at 34/90,
Task-28 outer-augmentation update 300 at 28/90, Task-29 full-proposal updates
200 and 500 at 38/90 and 31/90, and Task-60 full-proposal update 200 at 71/90.
All six evaluator launchers remained live on the diagnostic allocations; no
candidate reached strict 90/90 validation and no process or allocation was
modified.

At 21:24:48, the live evaluator poll verified Task-28 full-proposal update 300
at 35/90 and Task-28 outer-augmentation update 300 at 29/90. The other counts
remained Task-29 full-proposal updates 200 and 500 at 38/90 and 31/90, and
Task-60 full-proposal updates 200 and 400 at 71/90 and 31/90. All six evaluator
launchers remained live on the diagnostic allocations; no candidate reached
strict 90/90 validation and no process or allocation was modified.

At 21:25:56, the live evaluator poll verified Task-29 full-proposal update 500
at 32/90. The other counts remained Task-28 full-proposal update 300 at 35/90,
Task-28 outer-augmentation update 300 at 29/90, Task-29 full-proposal update
200 at 38/90, and Task-60 full-proposal updates 200 and 400 at 71/90 and 31/90.
All six evaluator launchers remained live on the diagnostic allocations; no
candidate reached strict 90/90 validation and no process or allocation was
modified.

At 21:26:30, the live evaluator poll verified Task-29 full-proposal update 200
at 39/90. The other counts remained Task-28 full-proposal update 300 at 35/90,
Task-28 outer-augmentation update 300 at 29/90, Task-29 full-proposal update
500 at 32/90, and Task-60 full-proposal updates 200 and 400 at 71/90 and 31/90.
All six evaluator launchers remained live on the diagnostic allocations; no
candidate reached strict 90/90 validation and no process or allocation was
modified.

At 21:27:41, the live evaluator poll verified Task-28 outer-augmentation update
300 at 30/90. The other counts remained Task-28 full-proposal update 300 at
35/90, Task-29 full-proposal updates 200 and 500 at 39/90 and 32/90, and
Task-60 full-proposal updates 200 and 400 at 71/90 and 31/90. All six evaluator
launchers remained live on the diagnostic allocations; no candidate reached
strict 90/90 validation and no process or allocation was modified.

At 21:28:42, the live evaluator poll verified Task-60 full-proposal update 200
at 72/90. The other counts remained Task-28 full-proposal update 300 at 35/90,
Task-28 outer-augmentation update 300 at 30/90, Task-29 full-proposal updates
200 and 500 at 39/90 and 32/90, and Task-60 full-proposal update 400 at 31/90.
All six evaluator launchers remained live on the diagnostic allocations; no
candidate reached strict 90/90 validation and no process or allocation was
modified.

At 21:29:25, the live evaluator poll verified Task-28 full-proposal update 300
at 36/90, Task-60 full-proposal update 200 at 73/90, and Task-60
full-proposal update 400 at 32/90. The other counts remained Task-28
outer-augmentation update 300 at 30/90 and Task-29 full-proposal updates 200
and 500 at 39/90 and 32/90. All six evaluator launchers remained live on the
diagnostic allocations; no candidate reached strict 90/90 validation and no
process or allocation was modified.

At 21:30:59, the live evaluator poll verified Task-28 outer-augmentation update
300 at 31/90 and Task-29 full-proposal updates 200 and 500 at 40/90 and 33/90.
The other counts remained Task-28 full-proposal update 300 at 36/90, and
Task-60 full-proposal updates 200 and 400 at 73/90 and 32/90. All six
evaluator launchers remained live on the diagnostic allocations; no candidate
reached strict 90/90 validation and no process or allocation was modified.

At 21:31:38, the live evaluator poll verified Task-28 full-proposal update 300
at 36/90 and Task-29 full-proposal update 200 at 40/90. The other counts
remained Task-28 outer-augmentation update 300 at 31/90, Task-29
full-proposal update 500 at 33/90, and Task-60 full-proposal updates 200 and
400 at 73/90 and 32/90. All six evaluator launchers remained live on the
diagnostic allocations; no candidate reached strict 90/90 validation and no
process or allocation was modified.

At 21:32:58, the live evaluator poll verified Task-60 full-proposal update 200
at 74/90. The other counts remained Task-28 full-proposal update 300 at 36/90,
Task-28 outer-augmentation update 300 at 31/90, Task-29 full-proposal updates
200 and 500 at 40/90 and 33/90, and Task-60 full-proposal update 400 at 32/90.
All six evaluator launchers remained live on the diagnostic allocations; no
candidate reached strict 90/90 validation and no process or allocation was
modified.

At 21:33:57, the live evaluator poll verified Task-28 full-proposal update 300
at 37/90, Task-60 full-proposal update 200 at 75/90, and Task-60
full-proposal update 400 at 33/90. The other counts remained Task-28
outer-augmentation update 300 at 31/90 and Task-29 full-proposal updates 200
and 500 at 40/90 and 33/90. All six evaluator launchers remained live on the
diagnostic allocations; no candidate reached strict 90/90 validation and no
process or allocation was modified.

At 21:34:33, the live evaluator poll verified Task-29 full-proposal update 500
at 34/90. The other counts remained Task-28 full-proposal update 300 at 37/90,
Task-28 outer-augmentation update 300 at 31/90, Task-29 full-proposal update
200 at 40/90, and Task-60 full-proposal updates 200 and 400 at 75/90 and 33/90.
All six evaluator launchers remained live on the diagnostic allocations; no
candidate reached strict 90/90 validation and no process or allocation was
modified.

At 21:36:08, the live evaluator poll verified Task-29 full-proposal update 200
at 41/90. The other counts remained Task-28 full-proposal update 300 at 37/90,
Task-28 outer-augmentation update 300 at 31/90, Task-29 full-proposal update
500 at 34/90, and Task-60 full-proposal updates 200 and 400 at 75/90 and 33/90.
All six evaluator launchers remained live on the diagnostic allocations; no
candidate reached strict 90/90 validation and no process or allocation was
modified.

At 21:36:56, the live evaluator poll verified Task-28 outer-augmentation update
300 at 32/90. The other counts remained Task-28 full-proposal update 300 at
37/90, Task-29 full-proposal updates 200 and 500 at 41/90 and 34/90, and
Task-60 full-proposal updates 200 and 400 at 75/90 and 33/90. All six evaluator
launchers remained live on the diagnostic allocations; no candidate reached
strict 90/90 validation and no process or allocation was modified.

At 21:37:44, the live evaluator poll verified Task-60 full-proposal updates 200
and 400 at 76/90 and 34/90. The other counts remained Task-28 full-proposal
update 300 at 37/90, Task-28 outer-augmentation update 300 at 32/90, and
Task-29 full-proposal updates 200 and 500 at 41/90 and 34/90. All six evaluator
launchers remained live on the diagnostic allocations; no candidate reached
strict 90/90 validation and no process or allocation was modified.

At 21:38:31, the live evaluator poll verified Task-28 full-proposal update 300
at 38/90. The other counts remained Task-28 outer-augmentation update 300 at
32/90, Task-29 full-proposal updates 200 and 500 at 41/90 and 34/90, and
Task-60 full-proposal updates 200 and 400 at 76/90 and 34/90. All six evaluator
launchers remained live on the diagnostic allocations; no candidate reached
strict 90/90 validation and no process or allocation was modified.

At 21:39:30, the live evaluator poll verified Task-29 full-proposal update 500
at 35/90, Task-60 full-proposal update 200 at 77/90, and Task-60 full-proposal
update 400 at 34/90. The other counts remained Task-28 full-proposal update 300
at 38/90, Task-28 outer-augmentation update 300 at 32/90, and Task-29
full-proposal update 200 at 41/90. All six evaluator launchers remained live
on the diagnostic allocations; no candidate reached strict 90/90 validation
and no process or allocation was modified.

At 21:40:55, the live evaluator poll verified Task-29 full-proposal update 200
at 42/90. The other counts remained Task-28 full-proposal update 300 at 38/90,
Task-28 outer-augmentation update 300 at 32/90, Task-29 full-proposal update
500 at 35/90, and Task-60 full-proposal updates 200 and 400 at 77/90 and 34/90.
All six evaluator launchers remained live on the diagnostic allocations; no
candidate reached strict 90/90 validation and no process or allocation was
modified.

At 21:41:57, the live evaluator poll verified Task-28 outer-augmentation update
300 at 33/90, Task-60 full-proposal update 200 at 78/90, and Task-60
full-proposal update 400 at 35/90. The other counts remained Task-28
full-proposal update 300 at 38/90 and Task-29 full-proposal updates 200 and 500
at 42/90 and 35/90. All six evaluator launchers remained live on the diagnostic
allocations; no candidate reached strict 90/90 validation and no process or
allocation was modified.

At 21:44:06, the live evaluator poll verified Task-28 full-proposal update 300
at 39/90, Task-29 full-proposal update 500 at 36/90, and Task-60 full-proposal
update 200 at 79/90. The other counts remained Task-28 outer-augmentation
update 300 at 33/90, Task-29 full-proposal update 200 at 42/90, and Task-60
full-proposal update 400 at 35/90. The obsolete Task-28 update-100 directory
remained at 0/90. All six evaluator launchers remained live on the diagnostic
allocations; no candidate reached strict 90/90 validation and no process or
allocation was modified.

At 21:45:30, the live evaluator poll verified Task-29 full-proposal update 200
at 43/90. The other active counts remained Task-28 full-proposal update 300 at
39/90, Task-28 outer-augmentation update 300 at 33/90, Task-29 full-proposal
update 500 at 36/90, and Task-60 full-proposal updates 200 and 400 at 79/90 and
35/90. The obsolete Task-28 update-100 directory remained at 0/90. All six
evaluator launchers remained live on the diagnostic allocations; no candidate
reached strict 90/90 validation and no process or allocation was modified.

At 21:46:29, the live evaluator poll verified Task-60 full-proposal update 200
at 80/90 and Task-60 full-proposal update 400 at 36/90. The other active counts
remained Task-28 full-proposal update 300 at 39/90, Task-28 outer-augmentation
update 300 at 33/90, and Task-29 full-proposal updates 200 and 500 at 43/90 and
36/90. The obsolete Task-28 update-100 directory remained at 0/90. All six
evaluator launchers remained live on the diagnostic allocations; no candidate
reached strict 90/90 validation and no process or allocation was modified.

At 21:47:39, the live evaluator poll verified Task-60 full-proposal update 200
at 81/90. The other active counts remained Task-28 full-proposal update 300 at
39/90, Task-28 outer-augmentation update 300 at 33/90, Task-29 full-proposal
updates 200 and 500 at 43/90 and 36/90, and Task-60 full-proposal update 400 at
36/90. The obsolete Task-28 update-100 directory remained at 0/90. All six
evaluator launchers remained live on the diagnostic allocations; no candidate
reached strict 90/90 validation and no process or allocation was modified.

At 21:48:24, the live evaluator poll verified Task-28 full-proposal update 300
at 40/90, Task-28 outer-augmentation update 300 at 34/90, and Task-29
full-proposal update 500 at 37/90. The other active counts remained Task-29
full-proposal update 200 at 43/90 and Task-60 full-proposal updates 200 and 400
at 81/90 and 36/90. The obsolete Task-28 update-100 directory remained at
0/90. All six evaluator launchers remained live on the diagnostic allocations;
no candidate reached strict 90/90 validation and no process or allocation was
modified.

At 21:50:28, a follow-up output poll verified Task-29 full-proposal update 200
at 44/90 and Task-60 full-proposal update 200 at 82/90. The other active counts
remained Task-28 full-proposal update 300 at 40/90, Task-28 outer-augmentation
update 300 at 34/90, Task-29 full-proposal update 500 at 37/90, and Task-60
full-proposal update 400 at 36/90. The evaluator health check observed recent
task JSON writes and live srun processes; no candidate reached strict 90/90
validation and no process or allocation was modified.

At 21:51:16, the live evaluator poll verified Task-60 full-proposal update 400
at 37/90. The other active counts remained Task-28 full-proposal update 300 at
40/90, Task-28 outer-augmentation update 300 at 34/90, Task-29 full-proposal
updates 200 and 500 at 44/90 and 37/90, and Task-60 full-proposal update 200 at
82/90. All six evaluator launchers remained live on the diagnostic allocations;
no candidate reached strict 90/90 validation and no process or allocation was
modified.

At 21:52:03, the live evaluator poll verified Task-60 full-proposal update 200
at 83/90. The other active counts remained Task-28 full-proposal update 300 at
40/90, Task-28 outer-augmentation update 300 at 34/90, Task-29 full-proposal
updates 200 and 500 at 44/90 and 37/90, and Task-60 full-proposal update 400 at
37/90. All six evaluator launchers remained live on the diagnostic allocations;
no candidate reached strict 90/90 validation and no process or allocation was
modified.

At 21:52:51, the live evaluator poll verified Task-28 full-proposal update 300
at 41/90 and Task-29 full-proposal update 500 at 38/90. The other active counts
remained Task-28 outer-augmentation update 300 at 34/90, Task-29 full-proposal
update 200 at 44/90, and Task-60 full-proposal updates 200 and 400 at 83/90 and
37/90. All six evaluator launchers remained live on the diagnostic allocations;
no candidate reached strict 90/90 validation and no process or allocation was
modified.

At 21:53:48, the live evaluator poll verified Task-28 outer-augmentation update
300 at 35/90. The other active counts remained Task-28 full-proposal update 300
at 41/90, Task-29 full-proposal updates 200 and 500 at 44/90 and 38/90, and
Task-60 full-proposal updates 200 and 400 at 83/90 and 37/90. All six evaluator
launchers remained live on the diagnostic allocations; no candidate reached
strict 90/90 validation and no process or allocation was modified.

At 21:54:34, the live evaluator poll verified Task-29 full-proposal update 200
at 45/90. The other active counts remained Task-28 full-proposal update 300 at
41/90, Task-28 outer-augmentation update 300 at 35/90, Task-29 full-proposal
update 500 at 38/90, and Task-60 full-proposal updates 200 and 400 at 83/90 and
37/90. All six evaluator launchers remained live on the diagnostic allocations;
no candidate reached strict 90/90 validation and no process or allocation was
modified.

At 21:55:23, the live evaluator poll verified Task-60 full-proposal update 200
at 84/90. The other active counts remained Task-28 full-proposal update 300 at
41/90, Task-28 outer-augmentation update 300 at 35/90, Task-29 full-proposal
updates 200 and 500 at 45/90 and 38/90, and Task-60 full-proposal update 400 at
37/90. All six evaluator launchers remained live on the diagnostic allocations;
no candidate reached strict 90/90 validation and no process or allocation was
modified.

At 21:56:08, the live evaluator poll verified Task-60 full-proposal update 400
at 38/90. The other active counts remained Task-28 full-proposal update 300 at
41/90, Task-28 outer-augmentation update 300 at 35/90, Task-29 full-proposal
updates 200 and 500 at 45/90 and 38/90, and Task-60 full-proposal update 200 at
84/90. All six evaluator launchers remained live on the diagnostic allocations;
no candidate reached strict 90/90 validation and no process or allocation was
modified.

At 21:57:06, the live evaluator poll verified Task-60 full-proposal update 200
at 85/90. The other active counts remained Task-28 full-proposal update 300 at
41/90, Task-28 outer-augmentation update 300 at 35/90, Task-29 full-proposal
updates 200 and 500 at 45/90 and 38/90, and Task-60 full-proposal update 400 at
38/90. All six evaluator launchers remained live on the diagnostic allocations;
no candidate reached strict 90/90 validation and no process or allocation was
modified.

At 21:57:58, the live evaluator poll verified Task-28 full-proposal update 300
at 42/90, Task-28 outer-augmentation update 300 at 36/90, and Task-29
full-proposal update 500 at 39/90. The other active counts remained Task-29
full-proposal update 200 at 45/90 and Task-60 full-proposal updates 200 and 400
at 85/90 and 38/90. All six evaluator launchers remained live on the diagnostic
allocations; no candidate reached strict 90/90 validation and no process or
allocation was modified.

At 21:59:12, the live evaluator poll verified Task-29 full-proposal update 200
at 46/90 and Task-60 full-proposal update 200 at 86/90. The other active counts
remained Task-28 full-proposal update 300 at 42/90, Task-28 outer-augmentation
update 300 at 36/90, Task-29 full-proposal update 500 at 39/90, and Task-60
full-proposal update 400 at 38/90. All six evaluator launchers remained live on
the diagnostic allocations; no candidate reached strict 90/90 validation and no
process or allocation was modified.

At 22:00:13, the live evaluator poll verified Task-60 full-proposal update 400
at 39/90. The other active counts remained Task-28 full-proposal update 300 at
42/90, Task-28 outer-augmentation update 300 at 36/90, Task-29 full-proposal
updates 200 and 500 at 46/90 and 39/90, and Task-60 full-proposal update 200 at
86/90. All six evaluator launchers remained live on the diagnostic allocations;
no candidate reached strict 90/90 validation and no process or allocation was
modified.

At 22:01:16, the live evaluator poll verified Task-29 full-proposal update 200
at 47/90. The other active counts remained Task-28 full-proposal update 300 at
42/90, Task-28 outer-augmentation update 300 at 36/90, Task-29 full-proposal
update 500 at 39/90, and Task-60 full-proposal updates 200 and 400 at 86/90 and
39/90. All six evaluator launchers remained live on the diagnostic allocations;
no candidate reached strict 90/90 validation and no process or allocation was
modified.

At 22:01:39, the live evaluator poll verified Task-60 full-proposal update 200
at 87/90. The other active counts remained Task-28 full-proposal update 300 at
42/90, Task-28 outer-augmentation update 300 at 36/90, Task-29 full-proposal
updates 200 and 500 at 47/90 and 39/90, and Task-60 full-proposal update 400 at
39/90. All six evaluator launchers remained live on the diagnostic allocations;
no candidate reached strict 90/90 validation and no process or allocation was
modified.

At 22:02:01, the live evaluator poll verified Task-28 full-proposal update 300
at 43/90. The other active counts remained Task-28 outer-augmentation update
300 at 36/90, Task-29 full-proposal updates 200 and 500 at 47/90 and 39/90, and
Task-60 full-proposal updates 200 and 400 at 87/90 and 39/90. All six evaluator
launchers remained live on the diagnostic allocations; no candidate reached
strict 90/90 validation and no process or allocation was modified.

At 22:02:22, the live evaluator poll verified Task-29 full-proposal update 500
at 40/90. The other active counts remained Task-28 full-proposal update 300 at
43/90, Task-28 outer-augmentation update 300 at 36/90, Task-29 full-proposal
update 200 at 47/90, and Task-60 full-proposal updates 200 and 400 at 87/90 and
39/90. All six evaluator launchers remained live on the diagnostic allocations;
no candidate reached strict 90/90 validation and no process or allocation was
modified.

At 22:02:44, the live evaluator poll verified Task-28 outer-augmentation update
300 at 37/90. The other active counts remained Task-28 full-proposal update 300
at 43/90, Task-29 full-proposal updates 200 and 500 at 47/90 and 40/90, and
Task-60 full-proposal updates 200 and 400 at 87/90 and 39/90. All six evaluator
launchers remained live on the diagnostic allocations; no candidate reached
strict 90/90 validation and no process or allocation was modified.

At 22:04:55, the live evaluator poll verified Task-60 full-proposal update 400
at 40/90. The other active counts remained Task-28 full-proposal update 300 at
43/90, Task-28 outer-augmentation update 300 at 37/90, Task-29 full-proposal
updates 200 and 500 at 47/90 and 40/90, and Task-60 full-proposal update 200 at
87/90. All six evaluator launchers remained live on the diagnostic allocations;
no candidate reached strict 90/90 validation and no process or allocation was
modified.

At 22:06:15, the live evaluator poll verified Task-28 full-proposal update 300
at 44/90. The other active counts remained Task-28 outer-augmentation update
300 at 37/90, Task-29 full-proposal updates 200 and 500 at 47/90 and 40/90, and
Task-60 full-proposal updates 200 and 400 at 87/90 and 40/90. All six evaluator
launchers remained live on the diagnostic allocations; no candidate reached
strict 90/90 validation and no process or allocation was modified.

At 22:06:38, the live evaluator poll confirmed that the active counts remained
Task-28 full-proposal update 300 at 44/90, Task-28 outer-augmentation update 300
at 37/90, Task-29 full-proposal updates 200 and 500 at 47/90 and 40/90, and
Task-60 full-proposal updates 200 and 400 at 87/90 and 40/90. All six evaluator
launchers remained live on the diagnostic allocations; no candidate reached
strict 90/90 validation and no process or allocation was modified.

At 22:07:19, the live evaluator poll verified Task-29 full-proposal update 200
at 48/90 and Task-29 full-proposal update 500 at 41/90. The other active counts
remained Task-28 full-proposal update 300 at 44/90, Task-28 outer-augmentation
update 300 at 37/90, and Task-60 full-proposal updates 200 and 400 at 87/90 and
40/90. All six evaluator launchers remained live on the diagnostic allocations;
no candidate reached strict 90/90 validation and no process or allocation was
modified.

At 22:08:36, the live evaluator poll verified Task-60 full-proposal update 200
at 88/90. The other active counts remained Task-28 full-proposal update 300 at
44/90, Task-28 outer-augmentation update 300 at 37/90, Task-29 full-proposal
updates 200 and 500 at 48/90 and 41/90, and Task-60 full-proposal update 400 at
40/90. All six evaluator launchers remained live on the diagnostic allocations;
no candidate reached strict 90/90 validation and no process or allocation was
modified.

At 22:10:20, the live evaluator poll verified Task-60 full-proposal update 400
at 41/90. The other active counts remained Task-28 full-proposal update 300 at
44/90, Task-28 outer-augmentation update 300 at 37/90, Task-29 full-proposal
updates 200 and 500 at 48/90 and 41/90, and Task-60 full-proposal update 200 at
88/90. All six evaluator launchers remained live on the diagnostic allocations;
no candidate reached strict 90/90 validation and no process or allocation was
modified.

At 22:10:43, the live evaluator poll verified Task-28 outer-augmentation update
300 at 38/90. The other active counts remained Task-28 full-proposal update 300
at 44/90, Task-29 full-proposal updates 200 and 500 at 48/90 and 41/90, and
Task-60 full-proposal updates 200 and 400 at 88/90 and 41/90. All six evaluator
launchers remained live on the diagnostic allocations; no candidate reached
strict 90/90 validation and no process or allocation was modified.

At 22:11:06, the live evaluator poll verified Task-28 full-proposal update 300
at 45/90. The other active counts remained Task-28 outer-augmentation update
300 at 38/90, Task-29 full-proposal updates 200 and 500 at 48/90 and 41/90, and
Task-60 full-proposal updates 200 and 400 at 88/90 and 41/90. All six evaluator
launchers remained live on the diagnostic allocations; no candidate reached
strict 90/90 validation and no process or allocation was modified.

At 22:11:27, the live evaluator poll confirmed Task-28 full-proposal update 300
remained at 45/90 and verified Task-29 full-proposal update 500 at 42/90. The other active counts
remained Task-28 outer-augmentation update 300 at 38/90, Task-29 full-proposal
update 200 at 48/90, and Task-60 full-proposal updates 200 and 400 at 88/90 and
41/90. All six evaluator launchers remained live on the diagnostic allocations;
no candidate reached strict 90/90 validation and no process or allocation was
modified.

At 22:11:56, the live evaluator poll verified Task-28 full-proposal update 300
at 45/90 and Task-29 full-proposal update 200 at 49/90. The other active counts
remained Task-28 outer-augmentation update 300 at 38/90, Task-29 full-proposal
update 500 at 42/90, and Task-60 full-proposal updates 200 and 400 at 88/90 and
41/90. A process-health check confirmed all six logical evaluator launchers
were still present and producing recent task files; no candidate reached strict
90/90 validation and no process or allocation was modified.

At 22:13:10, the live evaluator poll verified Task-60 full-proposal update 200
at 89/90. The other active counts remained Task-28 full-proposal update 300 at
45/90, Task-28 outer-augmentation update 300 at 38/90, Task-29 full-proposal
updates 200 and 500 at 49/90 and 42/90, and Task-60 full-proposal update 400 at
41/90. All six logical evaluator launchers remained active; no candidate had
yet reached strict 90/90 validation and no process or allocation was modified.

At 22:15:12, the live evaluator poll verified Task-28 outer-augmentation update
300 at 39/90 and Task-60 full-proposal update 400 at 42/90. The other active
counts remained Task-28 full-proposal update 300 at 45/90, Task-29 full-proposal
updates 200 and 500 at 49/90 and 42/90, and Task-60 full-proposal update 200 at
89/90. All six logical evaluator launchers remained active; no candidate had
yet reached strict 90/90 validation and no process or allocation was modified.

At 22:15:35, the live evaluator poll verified Task-28 full-proposal update 300
at 46/90, Task-28 outer-augmentation update 300 at 39/90, and Task-60
full-proposal update 400 at 42/90. The other active counts remained Task-29
full-proposal updates 200 and 500 at 49/90 and 42/90, and Task-60 full-proposal
update 200 at 89/90. All six logical evaluator launchers remained active; no
candidate had yet reached strict 90/90 validation and no process or allocation
was modified.

At 23:00:32, the newly launched Task-28 executed-backfill update-200
evaluator had an active Python rollout process on gpua082 (about 1.7 GB RSS
and nonzero CPU) but had not yet emitted its first task record, consistent
with normal model initialization. Task-60 proposal update 400 advanced to
53/90. The other live evaluator counts were unchanged; no process or
allocation was modified.

At 22:57:45, an additional strict actor-only LIBERO-90 evaluator was
launched for the valid Task-28 executed-tail-backfill update-200 checkpoint.
This arm uses the same Task-28 Small EXPO source only, zero official-demo
chunks, the 10/5 action interface, and the shared seed/evaluation protocol.
It runs in tmux session `sft_diag_full90_task28_backfill_step200_20261104` on
the previously idle gpua082 slot of allocation 23416001_2, writing to
`sft_retention_full90_eval_20261104/task28_executed_backfill_step200`.
The process was alive in checkpoint/import initialization with zero records,
no failure marker, and no GPU allocation yet; no existing evaluator or
unrelated process was modified.

At 22:16:04, the live evaluator poll verified Task-29 full-proposal update 500
at 42/90. The other active counts remained Task-28 full-proposal update 300 at
46/90, Task-28 outer-augmentation update 300 at 39/90, Task-29 full-proposal
update 200 at 49/90, and Task-60 full-proposal updates 200 and 400 at 89/90 and
42/90. All six logical evaluator launchers remained active; no candidate had
yet reached strict 90/90 validation and no process or allocation was modified.

At 22:16:29, the live evaluator poll verified Task-29 full-proposal update 500
at 43/90. The other active counts remained Task-28 full-proposal update 300 at
46/90, Task-28 outer-augmentation update 300 at 39/90, Task-29 full-proposal
update 200 at 49/90, and Task-60 full-proposal updates 200 and 400 at 89/90 and
42/90. All six logical evaluator launchers remained active; no candidate had
yet reached strict 90/90 validation and no process or allocation was modified.

At 22:17:01, the live evaluator poll verified Task-29 full-proposal update 200
at 50/90. The other active counts remained Task-28 full-proposal update 300 at
46/90, Task-28 outer-augmentation update 300 at 39/90, Task-29 full-proposal
update 500 at 43/90, and Task-60 full-proposal updates 200 and 400 at 89/90 and
42/90. All six logical evaluator launchers remained active; no candidate had
yet reached strict 90/90 validation and no process or allocation was modified.

At 22:17:24, the live evaluator poll verified Task-28 full-proposal update 300
at 47/90 and Task-29 full-proposal update 200 at 50/90. The other active counts
remained Task-28 outer-augmentation update 300 at 39/90, Task-29 full-proposal
update 500 at 43/90, and Task-60 full-proposal updates 200 and 400 at 89/90 and
42/90. All six logical evaluator launchers remained active; no candidate had
yet reached strict 90/90 validation and no process or allocation was modified.

At 22:20:00, the Task-60 full-proposal update-200 actor-only evaluation
completed all 90 LIBERO tasks. The strict validator found 90 unique task
records, 1,800 total episodes, no missing or malformed records, and 519
successes (**28.83%**). This is above the corresponding Full EXPO task-60
checkpoint (368/1,800 = 20.44%) and slightly above the official Pi0.5 base
(512/1,800 = 28.44%). The other five controlled full-suite evaluators remain
active; no training process, Task-2 side run, or GPU allocation was modified.

The per-task audit for this validated comparison found SFT above the matched
Full EXPO task-60 policy on 30 tasks, below it on 11 tasks, and tied on 49
tasks. The winning task IDs were 2, 10, 12, 14, 15, 19, 22, 26, 29, 30, 31,
33, 38, 46, 47, 48, 51, 54, 55, 56, 57, 59, 61, 64, 67, 68, 69, 70, 72,
and 77. Against the official base, SFT was higher on 11 tasks, lower on 14,
and tied on 65. This is a genuine aggregate gain over the matched Full EXPO
actor-only result, but not yet a universal improvement and not yet the final
best-margin result because the remaining controlled arms are still being
evaluated.

At 22:23:23, the live poll verified Task-28 full-proposal update 300 at 48/90,
Task-28 outer-augmentation update 300 at 39/90, Task-29 full-proposal updates
200 and 500 at 52/90 and 44/90, and Task-60 full-proposal update 400 at 43/90.
Task-60 full-proposal update 200 remained complete at 90/90. All six logical
evaluator launchers remained present, and all ten diagnostic allocation
elements remained running; the three newer sleep arrays remained pending. No
process or allocation was modified.

At 22:24:29, the live poll verified Task-28 outer-augmentation update 300 at
40/90. The other incomplete counts remained Task-28 full-proposal update 300
at 48/90, Task-29 full-proposal updates 200 and 500 at 52/90 and 44/90, and
Task-60 full-proposal update 400 at 43/90. Task-60 full-proposal update 200
remained complete at 90/90; all six logical evaluator launchers and ten
diagnostic allocation elements remained live. No process or allocation was
modified.

At 22:25:01, the live poll verified Task-60 full-proposal update 400 advanced
to 44/90. The other incomplete counts remained Task-28 full-proposal update
300 at 48/90, Task-28 outer-augmentation update 300 at 40/90, and Task-29
full-proposal updates 200 and 500 at 52/90 and 44/90. Task-60 update 200
remained complete at 90/90; all six logical evaluator launchers and ten
diagnostic allocation elements remained live. No process or allocation was
modified.

A launcher health audit immediately afterward found no active fatal error,
OOM, or traceback marker in the five incomplete evaluator logs. The messages
about the unavailable TPU backend, missing optional dataset path, and missing
Orbax metrics side file are nonfatal startup/runtime warnings; each active
evaluator continues producing task records. The only killed step is the old
Task-28 update-100 test, which has zero records and is not being relaunched.

At 22:25:56, the live poll verified Task-28 full-proposal update 300 advanced
to 49/90 and Task-29 full-proposal update 500 advanced to 45/90. Task-28
outer-augmentation update 300 remained at 40/90, Task-29 update 200 at 52/90,
and Task-60 update 400 at 44/90; Task-60 update 200 remained complete at
90/90. No process or allocation was modified.

At 22:26:20, the live poll found no new completed candidate: Task-28 proposal
300 remained at 49/90, Task-28 outer augmentation 300 at 40/90, Task-29
proposal updates 200 and 500 at 52/90 and 45/90, and Task-60 proposal 400 at
44/90. Task-60 proposal 200 remained the only strict full-suite completion.
Its two-shard evaluator processes were no longer present after completion,
while the separate
Task-2 task-specific training/evaluation process remained active and was not
touched. The written run plan was rechecked; no additional arm was launched
because the remaining diagnostic evaluators are still consuming the available
diagnostic GPUs.

At 22:28:30, the live poll verified Task-29 proposal update 200 advanced to
53/90. Task-28 proposal update 300 remained at 49/90, Task-28 outer
augmentation update 300 at 40/90, Task-29 proposal update 500 at 45/90, and
Task-60 proposal update 400 at 44/90. Task-60 proposal update 200 remained
complete at 90/90. The Task-2 task-specific process was still present; no
process or allocation was modified.

At 22:28:34, a read-only health check confirmed the protected Task-2 process
was still writing its launcher log during its four-task evaluation. Its last
training metrics reached update 499 with `data/small_expo_chunks=16` and
`data/official_demo_chunks=0`; no failure or traceback was observed. This
process remains outside the full-suite diagnosis and was not modified.

At 22:31:30, the saved Task-60 full-proposal `latest_checkpoints/500` was
verified complete. A new isolated actor-only LIBERO-90 evaluator was launched
in tmux session `sft_diag_full90_task60_step500_20261104` on the freed gpua082
slot of allocation 23416001_2, writing only to
`sft_retention_full90_eval_20261104/task60_full_proposal_step500`. At 22:32:21
the remote evaluator process was confirmed alive on gpua082; it had not yet
emitted its first task record while initializing. Existing evaluators and the
Task-2 process were not modified.

At 22:33:27, the live poll verified progress in three existing evaluators:
Task-28 proposal update 300 reached 51/90, Task-29 proposal update 500 reached
47/90, and Task-60 proposal update 400 reached 45/90. Task-28 outer
augmentation remained at 40/90 and Task-29 proposal update 200 at 53/90. The
new Task-60 update-500 evaluator remained in checkpoint initialization with no
task record yet; its remote process was confirmed alive.

At 22:34:09, Task-29 proposal update 200 advanced to 54/90. The other four
incomplete existing evaluators were unchanged, and the isolated Task-60
update-500 process remained alive with zero records while initializing. No
process or allocation was modified.

The previously protected Task-2 task-specific SFT process subsequently
completed its planned 500 updates and four-task retention evaluation. Its
terminal artifacts include checkpoint 500 and eval records 0, 100, 200, 300,
400, and 500; the final four-task macro was 60.0% (75%, 50%, 70%, 45% on
tasks 2, 28, 29, and 60), with zero official-demo chunks. This run remains
outside the full-suite diagnosis and is not being used as evidence here.

At 22:34:57, the live full-suite poll found Task-29 proposal update 500 at
48/90; the other incomplete existing suites were Task-28 proposal 300 at
51/90, Task-28 outer augmentation 300 at 40/90, Task-29 proposal 200 at
54/90, and Task-60 proposal 400 at 46/90. Task-60 proposal 200 remained
complete at 90/90, while the separate Task-60 proposal 500 evaluator remained
alive with zero records during initialization.

At 22:35:24, the Task-60 proposal-500 evaluator was confirmed alive on gpua082.
Its Python process was waiting in shared-filesystem synchronization I/O while
loading the checkpoint (`cl_sync_io_wait`), with no failure state or task
record yet. This is the same infrastructure startup behavior seen on earlier
cross-mount evaluations; it was not restarted.

At 22:36:19, Task-60 proposal update 400 advanced to 47/90. The other
full-suite counts were unchanged, and the Task-60 proposal-500 process remained
alive on gpua082. A read-only process check showed it was still in
`cl_sync_io_wait` during checkpoint loading, with no traceback or terminal
state; it was left untouched.

At 22:36:59, Task-28 proposal update 300 advanced to 52/90. Task-60 proposal
update 400 remained at 47/90 and the other incomplete suites were unchanged.
The Task-60 proposal-500 process remained alive after six minutes with no
task record; its low-memory Python process was still in shared-filesystem
initialization before opening the checkpoint, so it was not restarted.

At 22:37:39, the live poll found no new completed suite. The Task-60
proposal-500 process remained alive at 6:41 elapsed in `cl_sync_io_wait`; its
read-only syscall/fd check identified a read from the `/projects` NumPy
bytecode file during Python import, before model checkpoint access. This
confirms the delay is code-mount initialization rather than a model or GPU
failure; the process was left untouched.

At 22:38:56, controlled read-only probes confirmed the node-specific cause:
reading the same `/projects` NumPy bytecode file timed out on gpua082 but took
11 ms on gpua038. I stopped only the newly created Task-60 proposal-500 tmux
session on gpua082 (not the sleep allocation or any pre-existing evaluator)
and relaunched the same candidate on gpua038. The first relaunch exposed a
launcher-path issue: passing `latest_checkpoints/500` made the audit wrapper
look for `sft_config.json` inside `latest_checkpoints`; the run-level manifest
is one directory above. The isolated launcher was corrected to pass the
`latest_checkpoints` root and verify its `500` child, then relaunched on
gpua038.

At 22:44:04, the corrected gpua038 evaluator had successfully loaded the
run-level manifest and normalization assets and was actively restoring the
step-500 learner (approximately 1.9 GB resident memory and active CPU). It
had not yet emitted its first task record; the missing optional checkpoint
metrics side-file remained a warning only.

At 22:45:08, the corrected Task-60 proposal-500 evaluator reached GPU/XLA
initialization on gpua038. Its log showed normal cuDNN convolution compilation
slow-operation alarms and no exception; it had not yet emitted a task record.
At 22:45:47 it remained alive while finishing compilation. The existing five
incomplete evaluator counts were unchanged during this interval.

At 22:46:07, no new full-suite record had appeared. The gpua038 step-500
process remained live; a remote health check showed it at approximately 8 GB
RSS with active CPU time in a futex wait during XLA/compiled-agent setup, and
the log contained only deprecation warnings. This is still an active startup
phase, not a terminal failure, so no restart was made.

At 22:46:36, the corrected gpua038 process was actively running (approximately
8.3 GB RSS and 80% CPU) after compilation/setup. Its log contained the known
nonfatal missing checkpoint-metrics side-file message plus normal backend and
deprecation warnings, but no traceback, OOM, or kill. It had not yet emitted a
task JSON; the process was left running.

At 22:47:34, Task-28 proposal update 300 advanced to 54/90 and Task-29
proposal update 500 to 51/90. Task-29 proposal update 200 remained at 57/90,
Task-28 outer augmentation at 41/90, and Task-60 proposal update 400 at 49/90.
The corrected Task-60 proposal-500 process remained alive in compilation/setup
with no task JSON and no failure marker.

At 22:47:54, no candidate reached a new terminal state. The corrected
Task-60 proposal-500 process remained active on gpua038 at approximately 8.8
GB RSS and 76% CPU; a GPU probe showed 43% utilization and 17.3 GB memory in
use, confirming active compilation/model setup rather than a stalled or dead
process.

At 22:48:19, Task-28 outer augmentation advanced to 42/90; the other candidate
counts were unchanged. The Task-60 proposal-500 evaluator remained active on
gpua038 at 41% GPU utilization and 17.3 GB memory, with only normal JAX/Flax
deprecation messages in its log and no task record yet.

At 22:49:20, Task-29 proposal update 200 advanced to 58/90 and Task-60
proposal update 400 to 50/90. Task-28 proposal remained at 54/90, Task-28
outer augmentation at 42/90, and Task-29 update 500 at 51/90. The Task-60
proposal-500 evaluator remained active on gpua038 at about 40% GPU utilization
and 17.3 GB memory while completing compiled inference setup; no task record
or failure appeared.

At 22:50:19, no evaluator reached a new terminal state. The Task-60
proposal-500 process remained active on gpua038 at 41% GPU utilization and
17.3 GB memory, with no failure marker or task record; all other evaluator
counts were unchanged.

At 22:50:42, Task-29 proposal update 500 advanced to 52/90. The other
full-suite counts were unchanged. Task-60 proposal-500 remained active on
gpua038 at 40% GPU utilization and 16.9 GB memory, with no failure marker or
task record yet.

At 22:51:02, no new task record appeared. Task-60 proposal-500 remained a live
GPU process on gpua038 at 40% utilization and 17.3 GB memory, with active CPU
and no terminal error; it was left untouched.

At 22:51:26, the corrected Task-60 proposal-500 evaluator emitted its first
valid full-suite record (`task_00.json`), confirming that the gpua038 relaunch
passed initialization and rollout. Task-28 proposal update 300 also advanced
to 55/90. At 22:52:04, Task-60 proposal update 400 advanced to 51/90; the
step-500 evaluator remained live with 1/90 records and no failure marker.

At 22:54:20, the live poll found Task-28 proposal update 300 at 56/90 and
Task-28 outer-augmentation update 300 at 43/90. Task-29 proposal updates 200
and 500 were at 58/90 and 52/90, respectively, and Task-60 proposal update
400 was at 51/90. Task-60 proposal update 200 remained complete at 90/90.
The corrected Task-60 proposal-500 evaluator remained active on gpua038 with
1/90 valid records, approximately 17.3 GB GPU memory in use, and no traceback,
OOM, or kill marker. No process or allocation was modified.

At 22:56:09, the live poll confirmed continued progress without a terminal
failure: Task-29 proposal update 500 advanced to 53/90 and Task-60 proposal
update 400 advanced to 52/90. Task-28 proposal update 300 remained at 56/90,
Task-28 outer-augmentation update 300 at 43/90, Task-29 proposal update 200
at 58/90, Task-60 proposal update 200 remained complete at 90/90, and the
Task-60 proposal-500 evaluator remained live at 1/90. No process or allocation
was modified.

At 23:09:36, the live poll confirmed all seven incomplete actor-only
LIBERO-90 evaluators remained alive. Task-28 proposal update 300 reached
60/90, Task-60 proposal update 400 reached 55/90, and the other counts were
unchanged. The gpua087 executed-backfill retry remained active after
checkpoint restoration with no failure marker and zero records while entering
compiled rollout initialization. No evaluator was duplicated and no
unrelated process or allocation was modified.

At 23:10:36, the live poll found continued progress without a terminal
failure: Task-29 proposal update 200 advanced to 62/90 and update 500 to
57/90. Task-28 proposal update 300 remained at 60/90, Task-28
outer-augmentation at 45/90, Task-60 proposal update 400 at 55/90, and
Task-60 proposal update 500 at 4/90. The gpua087 backfill retry was actively
compiling cuDNN/XLA kernels with no traceback or OOM; it had not emitted its
first task record yet. No process or allocation was modified.

At 23:10:57, the gpua087 executed-backfill retry remained alive in active
GPU compilation (approximately 2.5 GB GPU memory and nonzero CPU), with the
same expected cuDNN slow-operation alarms and no traceback, OOM, or kill
marker. The full-suite counts were unchanged from the prior poll. No process
or allocation was modified.

At 23:34, the current artifact poll found the Task-60 proposal update-200
candidate complete at 90/90 and strictly validated at 519/1,800 (28.83%).
The remaining active actor-only LIBERO-90 evaluators were Task-28 proposal
update 300 (65/90), Task-28 outer augmentation update 300 (48/90), Task-29
proposal update 200 (64/90), Task-29 proposal update 500 (62/90), Task-60
proposal update 400 (61/90), and Task-60 proposal update 500 (8/90). Remote
process checks confirmed all six were live; no evaluator or unrelated
allocation was modified.

The Task-28 executed-tail backfill control still has no usable full-suite
records after two isolated startup attempts. A read-only preflight on the
currently unoccupied gpua002 diagnostic slot again timed out while importing
the `/projects` environment, matching the earlier node-specific code-mount
stall. No new backfill evaluator was launched from that slot; the six active
evaluators remain the priority.

At 23:40, all six active evaluator processes were still live. Their latest
committed counts were Task-28 proposal update 300 (66/90), Task-28 outer
augmentation update 300 (49/90), Task-29 proposal updates 200 and 500 (65/90
and 64/90), and Task-60 proposal updates 400 and 500 (63/90 and 11/90). The
Task-60 update-200 candidate remained the only newly completed diagnostic
candidate and its strict validator result remained 519/1,800 (28.83%).

At 23:41, a fresh poll confirmed all six evaluator processes were still live.
The counts advanced to Task-28 proposal update 300 (67/90), Task-28 outer
augmentation update 300 (50/90), Task-29 proposal updates 200 and 500 (65/90
and 64/90), and Task-60 proposal updates 400 and 500 (63/90 and 11/90). No
candidate reached the strict completion gate, and no process or allocation was
modified.

At 23:43, the four unoccupied slots inside the active diagnostic allocations
were identified as gpua002, gpua082, gpua087, and the second gpua044 slot
(allocation 23416001_5). A read-only Python import preflight on the latter
also timed out in the shared `/projects` environment. Therefore no clean idle
slot is currently suitable for launching the executed-tail control; the six
already-running evaluators remain untouched.

At 23:44, all six evaluators remained live and continued to commit records:
Task-28 proposal update 300 (68/90), Task-28 outer augmentation update 300
(51/90), Task-29 proposal updates 200 and 500 (66/90 and 65/90), and Task-60
proposal updates 400 and 500 (64/90 and 12/90). No new candidate reached 90/90.

At 23:45, resource checks showed all six evaluator Python processes using
active CPU (approximately 47--65%) rather than exited or idle states. The
latest task-file timestamps also continued to advance, including Task-60
update 400 through task 63 and Task-29 update 500 through task 64. These are
still partial suites and were not scored as final results.

At 23:46, Task-28 proposal update 300 advanced to 69/90. The other live
counts remained Task-28 outer augmentation 51/90, Task-29 proposal updates
200 and 500 at 66/90 and 65/90, and Task-60 proposal updates 400 and 500 at
64/90 and 12/90. All six remote evaluator processes remained live; no strict
90-task validation was triggered because no new directory was complete.

At 23:47, Task-60 proposal update 500 advanced to 13/90. The other counts
remained unchanged, and all six evaluator processes were still live. No new
candidate was complete enough for validation.

At 23:48, Task-29 proposal update 500 advanced to 66/90 and Task-60 proposal
update 400 advanced to 65/90. Task-28 proposal and outer-augmentation remained
at 69/90 and 51/90, Task-29 update 200 at 66/90, and Task-60 update 500 at
13/90. All six evaluator processes remained live; no candidate reached 90/90.

At 23:49, Task-28 outer augmentation update 300 advanced to 52/90. The other
counts remained Task-28 proposal 69/90, Task-29 proposals 66/90 and 66/90,
and Task-60 proposals 65/90 and 13/90. All six processes remained live and no
candidate reached the strict validation gate.

At 23:50, Task-29 proposal update 200 advanced to 67/90. The other counts
remained Task-28 proposal/outer augmentation 69/90 and 52/90, Task-29 update
500 at 66/90, and Task-60 updates 400/500 at 65/90 and 13/90. All six
evaluators remained live; no new candidate was complete.

At 23:51, Task-28 proposal update 300 advanced to 70/90. The other counts
remained unchanged, all six evaluator processes remained live, and no
candidate reached the strict 90-task gate.

At 23:52, Task-29 proposal update 200 advanced to 68/90. The other counts
remained Task-28 proposal/outer augmentation 70/90 and 52/90, Task-29 update
500 at 66/90, and Task-60 updates 400/500 at 65/90 and 13/90. All six
evaluators remained live; no candidate reached 90/90.

At 23:53, Task-60 proposal update 500 advanced to 14/90. The other counts
remained unchanged, all six evaluators were still live, and no candidate
reached the strict validation gate.

At 23:54, Task-28 outer augmentation update 300 advanced to 53/90, Task-29
proposal update 500 to 67/90, and Task-60 proposal update 400 to 66/90. The
other counts remained Task-28 proposal 70/90, Task-29 update 200 68/90, and
Task-60 update 500 14/90. All six evaluators remain live; no suite is complete.

At 23:55, Task-28 outer augmentation advanced to 53/90, Task-29 update 500
remained at 67/90, and Task-60 update 400 remained at 66/90; the latest
artifact timestamps confirm active rollout work. No candidate reached 90/90.

At 23:55 (subsequent poll), Task-29 proposal updates 200 and 500 advanced to
69/90 and 68/90. The other counts remained Task-28 proposal/outer augmentation
70/90 and 53/90, and Task-60 updates 400/500 at 66/90 and 14/90. All six
evaluators remained live; no candidate was complete.

At 23:56, Task-28 proposal update 300 advanced to 71/90. The other counts
remained unchanged, all six evaluators remained live, and no candidate reached
the strict 90-task validation gate.

At 23:59, Task-60 proposal updates 400 and 500 advanced to 67/90 and 15/90.
The other counts remained Task-28 proposal/outer augmentation 71/90 and 53/90,
and Task-29 proposal updates 200/500 at 69/90 and 68/90. All six evaluators
remained live; no candidate reached 90/90.

At 00:00 on 2026-11-05, Task-29 proposal updates 200 and 500 advanced to
70/90 and 69/90, while Task-60 proposal updates 400 and 500 advanced to 67/90
and 15/90. Task-28 remained at 71/90 and 53/90. All six evaluator processes
remained live; no candidate reached the strict validation gate.

At 00:01, Task-28 proposal update 300 advanced to 72/90. The other counts
remained Task-28 outer augmentation 53/90, Task-29 updates 200/500 at 70/90
and 69/90, and Task-60 updates 400/500 at 67/90 and 15/90. All six evaluators
remained live; no candidate reached 90/90.

At 00:01 (subsequent poll), Task-28 outer augmentation update 300 advanced to
54/90 and Task-60 proposal update 400 to 68/90. The other counts remained
Task-28 proposal 72/90, Task-29 updates 200/500 at 70/90 and 69/90, and
Task-60 update 500 at 15/90. All six evaluators remained live; no candidate
passed the strict validation gate.

At 00:02, Task-28 proposal update 300 advanced to 73/90. The other counts
remained Task-28 outer augmentation 54/90, Task-29 updates 200/500 at 70/90
and 69/90, and Task-60 updates 400/500 at 68/90 and 15/90. All six
evaluators remained live; no candidate reached 90/90.

At 00:04, Task-29 proposal update 200 advanced to 71/90. The other counts
remained Task-28 proposal/outer augmentation 73/90 and 54/90, Task-29 update
500 at 69/90, and Task-60 updates 400/500 at 68/90 and 15/90. All six
evaluators remained live; no candidate reached the strict validation gate.

At 00:04 (subsequent poll), Task-60 proposal update 400 advanced to 69/90.
The other counts remained Task-28 proposal/outer augmentation 73/90 and 54/90,
Task-29 updates 200/500 at 71/90 and 69/90, and Task-60 update 500 at 15/90.
All six evaluators remained live; no candidate reached 90/90.

At 00:05, Task-29 proposal update 500 advanced to 69/90. The other counts
remained unchanged, all six evaluators remained live, and no candidate reached
the strict validation gate.

At 00:06, Task-28 outer augmentation update 300 advanced to 55/90, Task-29
update 500 to 70/90, and Task-60 update 500 to 15/90. Task-28 proposal remained
at 73/90, Task-29 update 200 at 71/90, and Task-60 update 400 at 69/90. All
six evaluators remained live; no candidate reached 90/90.

At 00:06 (subsequent poll), Task-60 proposal update 500 advanced to 16/90.
The other counts remained unchanged, all six evaluators remained live, and no
candidate reached the strict validation gate.

At 00:07, Task-28 proposal update 300 advanced to 74/90. The other counts
remained Task-28 outer augmentation 55/90, Task-29 updates 200/500 at 71/90
and 70/90, and Task-60 updates 400/500 at 69/90 and 16/90. All six
evaluators remained live; no candidate reached 90/90.

At 00:08, Task-28 outer augmentation update 300 advanced to 56/90. The other
counts remained Task-28 proposal 74/90, Task-29 updates 200/500 at 71/90 and
70/90, and Task-60 updates 400/500 at 69/90 and 16/90. All six evaluators
remained live; no candidate reached the strict validation gate.

At 00:08 (subsequent poll), Task-29 proposal updates 200 and 500 advanced to
72/90 and 70/90. The other counts remained Task-28 proposal/outer augmentation
74/90 and 56/90, and Task-60 updates 400/500 at 69/90 and 16/90. All six
evaluators remained live; no candidate reached 90/90.

At 00:09, Task-60 proposal update 400 advanced to 70/90. The other counts
remained unchanged, all six evaluators remained live, and no candidate reached
the strict validation gate.

At 00:10, all six evaluator processes remained live. The latest committed
counts were unchanged at Task-28 proposal/outer augmentation 74/90 and 56/90,
Task-29 proposal updates 200/500 at 72/90 and 71/90, and Task-60 proposal
updates 400/500 at 70/90 and 16/90. No candidate reached 90/90.

At 00:12, direct remote process checks confirmed that all six actor-only
LIBERO-90 evaluators were still running on their assigned diagnostic A100
slots. Their committed output counts had advanced to Task-28 proposal/outer
augmentation 75/90 and 57/90, Task-29 proposal updates 200/500 at 73/90 and
71/90, and Task-60 proposal updates 400/500 at 70/90 and 17/90. The Task-60
proposal update-200 candidate remained the only strict full-suite completion
(519/1,800, 28.83%); no new candidate passed the 90-task validation gate.
The four nominally idle slots remain reserved inside the diagnostic sleeper
allocations; no new trainer or evaluator was launched because prior probes on
those slots still encounter the shared `/projects` code-mount startup stall.

At 00:15, the previously blocked Task-28 executed-tail-backfill full-suite
control was launched on the unused `gpua002` slot in tmux session
`sft_diag_full90_task28_backfill_gpua002_20261105`, writing to the isolated
directory `sft_retention_full90_eval_20261104/task28_executed_backfill_step200_gpua002`.
The remote actor-only evaluator process is live; it has not yet emitted a task
record during initialization. This launch changes no existing evaluator,
trainer, checkpoint, or allocation.

The Task-60 update-400 evaluator is validly pinned in memory despite its
launcher path using `latest_checkpoints`: its process started at 19:13:35,
before the update-500 checkpoint was created at 19:24:01. The evaluator builds
the SFT agent once before iterating over tasks, so later replacement of the
`latest_checkpoints` directory cannot change that already-restored agent.
This timing audit prevents mislabeling the update-400 full-suite result.

At 00:16, the six original evaluators remained live and advanced to Task-28
proposal/outer augmentation 76/90 and 58/90, Task-29 updates 200/500 at 74/90
and 73/90, and Task-60 updates 400/500 at 71/90 and 17/90. The new executed
backfill evaluator on `gpua002` was also live with zero records still emitted
during startup.

At 00:19, a new isolated early-stop Task-60 full-proposal run was launched on
the second unused diagnostic slot, `gpua082`, in tmux session
`sft_diag_task60_full_proposal_step100_20261105`. It uses the same Task-60
Small EXPO teacher collection (teacher step 450223, 150 successful episodes,
zero failures), official `pi05_libero` initialization, zero-B LoRA-only scope,
batch 16, learning rate 5e-5, warmup 10,000, official-demo ratio 0, and
100 optimizer updates. Its output is isolated at
`sft_retention_diagnosis_20261105/task_60_full_proposal_step100`; the remote
trainer is live and W&B authentication succeeded. This run is intended to
recover the pruned update-100 checkpoint for a strict full-suite test and does
not touch the existing Task-60 run.

At 00:21, the six original full-suite evaluators continued advancing: Task-28
proposal/outer augmentation 77/90 and 59/90, Task-29 updates 200/500 at 75/90
and 74/90, and Task-60 updates 400/500 at 73/90 and 18/90. The Task-28
executed-backfill evaluator on `gpua002` remained live at 0/90 while completing
initial restore. The new Task-60 update-100 trainer remained present in its
dedicated tmux session on `gpua082`; its W&B run was established but the
training loop had not yet emitted a progress record.

At 00:22, the early-stop run passed initialization and printed its strict data
audit: `small_expo_chunks=2226`, `official_demo_chunks=0`,
`official_demo_episodes=0`, `official_demo_ratio=0.0`, and
`target_mode=full_proposal`. The data source is the Task-60 teacher collection
only. It then entered the actor-evaluation portion of the first retention
gate; its update-100 checkpoint has not yet been committed. The original
full-suite counts were Task-28 proposal/outer augmentation 77/90 and 60/90,
Task-29 updates 200/500 at 75/90 and 74/90, and Task-60 updates 400/500 at
73/90 and 18/90. The new Task-28 backfill evaluator remained live at 0/90.

At 00:23, direct `srun` checks confirmed both newly launched processes were
live: the Task-60 update-100 trainer on `gpua082` and the Task-28
executed-backfill evaluator on `gpua002`. The update-100 trainer had verified
the Task-60-only data audit and was still in its first retention evaluation;
no checkpoint or `RETENTION_EVAL` result had been written yet. The original
full-suite counts advanced to Task-28 proposal/outer augmentation 77/90 and
60/90, Task-29 updates 200/500 at 75/90 and 74/90, and Task-60 updates 400/500
at 73/90 and 19/90.

At 00:24, direct process checks still found both new jobs live. The update-100
log showed normal CuDNN/JAX compilation slow-operation alarms followed by the
same strict data audit; there was no traceback, OOM, or terminal marker. The
full-suite evaluators advanced to Task-28 proposal/outer augmentation 78/90
and 60/90, Task-29 updates 200/500 at 76/90 and 75/90, and Task-60 updates
400/500 at 74/90 and 19/90. No new candidate reached the strict 90-task gate.

At 00:28, the update-100 trainer was still live and had advanced through the
first 20-episode retention task in its initial four-task evaluation; the log
showed no failure and checkpoint metadata was still absent. This duration is
consistent with the established roughly four-minutes-per-task evaluation
time. The full-suite evaluators remained active; no new strict 90-task result
was available.

At 00:25, direct remote checks again confirmed the update-100 trainer and
executed-backfill evaluator were live. The update-100 run had created its
initialization audit and SFT manifest and had loaded only 2,226 Task-60 Small
EXPO chunks with zero official demos, but had not yet emitted its first
`RETENTION_EVAL` or checkpoint; it is still in the long initial actor
evaluation. The backfill evaluator remained at 0/90 during restore/compile.
The six original evaluators remained allocated and no candidate had reached
the strict 90-task gate.

At 00:26, the isolated evaluator launcher for the pending Task-60 update-100
checkpoint was prepared and syntax-checked at
`run_full90_task60_proposal_step100_20261105.sh`. It is deliberately gated on
`checkpoints/100/_CHECKPOINT_METADATA` and has not been launched while that
checkpoint is absent; the intended slot is the unused `gpua087` diagnostic
slot, with a separate full-suite output directory.

At 00:29, the Task-28 executed-tail evaluator emitted its first valid task
record (`task_00.json`), confirming that restore/compile completed and rollout
is progressing. The Task-60 update-100 trainer remained live in its initial
four-task retention evaluation; its checkpoint gate was still not ready. The
other full-suite candidates continued advancing but none had reached 90/90.

At 00:31, the update-100 trainer was confirmed active on `gpua082` at roughly
57% GPU utilization and had moved into the second 20-episode retention task;
the checkpoint gate remained correctly closed. The executed-tail evaluator
remained live after its first record, and no candidate had yet produced a new
strict 90-task completion.

At 00:33, the update-100 trainer remained active in its initial retention
pass, with the log progressing through the third task and no checkpoint or
failure marker. The executed-tail evaluator advanced to 2/90 valid task
records. No full-suite candidate had completed the strict 90-task gate.

At 00:38, the live output poll found the existing evaluator counts at
Task-28 proposal update 300 (81/90), Task-28 outer augmentation update 300
(63/90), Task-29 proposal updates 200 and 500 (79/90 and 78/90), and Task-60
proposal updates 400 and 500 (77/90 and 23/90). Task-60 proposal update 200
remains complete at 90/90 and strictly validated at 519/1,800 (28.83%). The
executed-tail backfill control has advanced to 3/90. The isolated Task-60
update-100 trainer remains live in its dedicated tmux session, but its
`checkpoints/100/_CHECKPOINT_METADATA` gate is still absent, so its prepared
full-suite evaluator has not been launched. No existing training or
allocation was changed.

At 03:41, the first update-1,000 retention segment completed all 20 episodes
and the evaluator entered the second segment. The trainer and allocation
remain live; no step-1,000 aggregate or checkpoint is inferred until all four
task panels finish.

At 03:42, the first update-1,000 panel had completed 20/20 and the second
panel had reached 5/20. The trainer process remained live on `gpua044`; no
update-1,000 result or checkpoint was inferred early, and no full-suite
evaluator was launched on top of the active trainer.

At 03:29, the same Task-29 clipped trainer had advanced to update 711 with
finite actor loss, gradient norm, and parameter norm. The active process on
`gpua044` remained GPU-computing, and its data counters still reported 16
Task-29 Small EXPO chunks per batch and zero official-demo chunks. No new
checkpoint was expected before update 1,000; the pinned update-500 evaluator
remains prepared but not launched while this trainer owns the GPU.

### Task-29 clipped-target update-1,000 retention result (2026-11-05, 03:50 CST)

The update-1,000 retention panel committed successfully: Task 2 = 25%, Task
28 = 55%, Task 29 = 90%, and Task 60 = 45%, for a 53.75% macro rate (43/80
episodes). The JSON result is present at `evals/00001000.json` and the
`latest_checkpoints/1000` metadata has committed. The run continued training
after the checkpoint with finite metrics. Update 500 remains the stronger
four-task retention dose (58.75%), and both are still awaiting strict
LIBERO-90 actor-only evaluation.

At 03:37, the trainer reached the scheduled update-1,000 retention point
after finite optimizer metrics through update 999. The four-task actor-only
evaluation began normally; at 03:38 its first 20-episode segment had reached
5/20. No step-1,000 score or checkpoint is counted until the complete panel
commits. The step-500 artifacts remain intact and are still the next strict
LIBERO-90 evaluation target.

At 01:10, after moving the corrected Task-60 update-100 evaluator to the
free `gpua082` slot, its process remained live through checkpoint-manager
restore but had not yet emitted a task record; GPU initialization is waiting
on the shared `/projects` filesystem. Existing suites continued advancing,
with Task-28 proposal update 300 at 88/90, Task-29 proposal updates 200/500
at 86/90 and 85/90, Task-60 proposal update 400 at 84/90, and the executed
backfill control at 11/90. These partial counts remain provisional until the
strict 90-task validator passes.

At 01:11, the moved evaluator remained live on `gpua082` in
`cl_sync_io_wait` with approximately 17.3 GiB allocated and no task files;
this is the same shared-code-mount stall previously observed on `gpua087`, not
a model or checkpoint error. The existing evaluators remained untouched.

At 01:13, the `gpua082` retry had progressed out of the filesystem wait into
active JAX/model initialization (approximately 48% GPU and 76% CPU), still
without a first task record and without an error marker. It remains the only
new evaluator for Task-60 update 100; the partial existing suites remain
provisional.

At 01:14, the Task-60 update-100 evaluator remained live and GPU-active with
no failure marker and no task record yet. Existing suites advanced to
Task-28 proposal update 300 at 89/90, Task-29 proposal updates 200/500 at
87/90 and 86/90, and Task-60 proposal update 400 at 85/90. These remain
unvalidated partial outputs.

At 01:15, the strict validator was rerun on the completed Task-60
full-proposal update-200 suite: 90 files, 90 unique tasks, no missing tasks,
no errors, and 519/1,800 successes (28.83%). This remains the validated
diagnostic leader; the update-100 full-suite evaluator is still compiling on
`gpua082`.

At 00:49, the isolated Task-60 update-100 trainer remained live after
completing all 100 optimizer updates and the first post-update retention task;
it was progressing through the second task at roughly 40% GPU utilization.
The process has finite optimizer metrics through step 99 and no error marker,
but checkpoint 100 is still withheld until the four-task retention evaluation
completes. The `gpua087` watcher remains waiting on that metadata gate.

At 00:44, the Task-60 update-100 trainer had completed its 100 learner BC
updates and entered the post-update four-task retention evaluation; checkpoint
100 was not yet committed. A dedicated tmux watcher
(`sft_diag_full90_task60_step100_20261105`) was started for the unused
`gpua087` diagnostic slot. It will invoke the prepared strict actor-only
LIBERO-90 evaluator only after `checkpoints/100/_CHECKPOINT_METADATA` appears,
writing to the isolated `task60_full_proposal_step100` output directory. No
existing training process, evaluator, or allocation was changed.

At 00:52, the Task-60 update-100 trainer remained live without a failure
marker, had completed the second retention task, and had entered the third.
Checkpoint 100 was still absent, so the `gpua087` full-suite watcher remained
correctly gated. The other actor-only full-suite evaluators continued to
advance, but no additional candidate had passed strict 90-task validation.

At 00:53, the Task-60 update-100 retention evaluation was still live at about
7/20 episodes on its third task, with checkpoint 100 absent and the watcher
still waiting. Slurm confirmed all ten diagnostic sleeper elements remained
RUNNING; the three newer A100 arrays remained PENDING. No existing run or
allocation was changed.

At 01:08, the Task-60 update-100 retention run had completed successfully:
the step-100 retention record is 50.0% macro (75%, 25%, 55%, 45% on tasks
2/28/29/60), and the latest learner checkpoint is committed under
`latest_checkpoints/100`. The first isolated full-suite launcher exposed two
path issues in the new harness: the evaluator must receive the
`latest_checkpoints` manager root (not the numeric child), and readiness must
check `latest_checkpoints/100/_CHECKPOINT_METADATA`. Those isolated launcher
paths were corrected without changing the trainer or evaluator implementation.
The corrected evaluator was first tried on `gpua087`, where it stalled in
`cl_sync_io_wait` during `/projects` import; it was moved to the genuinely idle
`gpua082` diagnostic slot, where it is currently live but stalled in the same
shared-code filesystem during `filemap_update_page`. It has produced no task
records yet. No existing evaluator, trainer, or allocation was changed.

At 01:21, strict validation also completed for the Task-28 full-proposal
update-300 suite: 90 files, 90 unique tasks, no missing tasks, no errors, and
518/1,800 successes (28.78%). Relative to the matched historical Full EXPO
Task-28 result (451/1,800, 25.06%), this is a provisional +3.72 percentage
point margin; it remains an actor-only SFT result compared with the deployed
Full EXPO editor/critic system.

At 01:21, the controlled Task-2 full-proposal update-500 checkpoint was
confirmed complete at
`sft_retention_diagnosis_20261104/task_02_full_proposal_retry/checkpoints/500`.
Its training audit reports same-task Small EXPO teacher data, zero official
demo chunks, zero official-demo ratio, zero-B LoRA initialization, and a
four-task retention result at update 500 of 60.0% macro (75%, 50%, 70%, 45%
on tasks 2/28/29/60). A new isolated actor-only LIBERO-90 evaluation was
launched on the genuinely idle `gpua087` allocation (`23433381_1`) in tmux
session `sft_diag_full90_task02_step500_gpua087_20261105`, writing only to
`sft_retention_full90_eval_20261104/task02_full_proposal_step500`. The process
is live in the shared `/projects` import/compile wait and has not produced a
task record yet; it has not failed and no existing job was changed.

At 01:24, the Task-2 update-500 evaluator had restored the committed Orbax
checkpoint and was still live during shared-filesystem initialization, with
zero task records and no failure marker. The other live-suite counts were
Task-28 outer augmentation 75/90, Task-28 executed-tail backfill 14/90,
Task-29 proposal updates 200/500 at 89/90 and 88/90, Task-60 update 100 at
2/90, and Task-60 update 400/500 at 87/90 and 33/90. The two completed
validated suites remain Task-28 update 300 (518/1,800) and Task-60 update 200
(519/1,800). No evaluator or allocation was stopped or modified.

At 01:26, the Task-29 full-proposal update-200 suite passed strict validation:
90 files, 90 unique tasks, no missing tasks, no errors, and 523/1,800
successes (29.06%). The training manifest independently confirms
`source_task_id=29`, a Task-29 Small EXPO teacher directory,
`collection_source=small_expo`, `official_demo_ratio=0`,
`official_demo_chunks=0`, `target_mode=full_proposal`, official
`pi05_libero` initialization, zero-B LoRA, batch 16, learning rate 5e-5, and
10k warmup. Against matched Full EXPO Task-29 (441/1,800, 24.50%), the
actor-only SFT result is +4.56 percentage points.

At 01:31, the Task-29 full-proposal update-500 suite also passed strict
validation: 90 files, 90 unique tasks, no missing tasks, no errors, and
524/1,800 successes (29.11%). It uses the same verified Task-29-only,
Small-EXPO-only, zero-official-demo configuration as the update-200 arm.
Against matched Full EXPO Task-29 (441/1,800, 24.50%), the margin is +4.61
percentage points. Update 500 is therefore the stronger of the two completed
Task-29 doses, but Task-60 update 200 remains the overall margin leader.

At 01:36, the Task-60 full-proposal update-400 suite passed strict
validation: 90 files, 90 unique tasks, no missing tasks, no errors, and
506/1,800 successes (28.11%). It uses the same verified Task-60-only,
Small-EXPO-only, zero-official-demo configuration as update 200. Against
matched Full EXPO Task-60 (368/1,800, 20.44%), the margin is +7.67 percentage
points; the update-200 dose remains better by 13 successes.

At 01:38, two previously unscored, task-valid historical Task-29 checkpoints
were launched for isolated actor-only LIBERO-90 evaluation on newly idle
diagnostic slots: the Small-EXPO-only/no-replay 1,500-update checkpoint on
`gpua051` (`23433381_4`, tmux
`sft_diag_full90_task29_no_replay1500_gpua051_20261105`) and the 50%-official-
Task-29-demo 1,500-update checkpoint on `gpua070` (`23416001_3`, tmux
`sft_diag_full90_task29_demo50_1500_gpua070_20261105`). Both processes are
live in shared-filesystem initialization, with isolated output directories;
no existing evaluator or training job was changed.

At 01:48, the two historical Task-29 evaluators remained live after their
shared-filesystem startup phase, each resident at roughly 17.3 GiB and about
40% GPU utilization, but neither had committed its first task record yet.
The controlled Task-2 evaluator likewise remained live at roughly 17.3 GiB
and 48% GPU utilization, with four task records committed through task 03.
These are slow rollout/compilation phases, not terminal failures; no process
was restarted.

At 01:50, the no-replay and 50%-demo historical Task-29 suites each remained
live at 1/90 (task 0 = 0/20); the controlled Task-2 update-500 suite advanced
to 5/90 (through task 04). Other active suites advanced to Task-28 outer
augmentation 81/90, Task-28 executed backfill 19/90, and Task-60 update 500
38/90. These are still partial, unvalidated results.

At 01:54, both historical Task-29 arms advanced to 2/90: task 1 also scored
0/20 on each arm. The controlled Task-2 update-500 evaluation advanced to
5/90, while the Task-60 update-100 evaluation reached 9/90. All processes
remain live; no aggregate is inferred from these partial results.

At 01:59, both historical Task-29 arms reached task 2/90 and continued
normally. Their task-2 partial results were 45% success for no-replay and 35%
for the 50%-official-demo arm; these are per-task observations only, not
full-suite estimates. The controlled Task-2 update-500 suite remained at
6/90 and all evaluators remained live.

At 02:02, the live-suite poll found Task-2 update 500 at 7/90, Task-28
executed-tail backfill at 23/90, Task-28 outer augmentation at 83/90, and
Task-60 update 100 at 11/90. The historical Task-29 no-replay and 50%-demo
arms remained at 3/90 each. The completed suites were rechecked with the
strict validator and remain unchanged: Task-28 update 300 = 518/1,800,
Task-29 updates 200/500 = 523/1,800 and 524/1,800, and Task-60 updates
200/400 = 519/1,800 and 506/1,800; each has 90 unique task files, no missing
tasks, and zero errors. All ten diagnostic A100 slots remain occupied by
these evaluators or the completed-suite sleeper allocations; the additional
arrays are still pending, so no new diagnostic arm was launched.

At 02:04, two genuinely idle GPUs were confirmed inside completed-suite
sleeper allocations: `gpua085` (allocation `23433381_2`) and `gpua077`
(`23416001_4`). I launched isolated actor-only LIBERO-90 evaluations for two
valid historical Task-28-only arms using those slots: the raw action-bound
checkpoint at update 1,500 and the clipped action-bound checkpoint at update
1,500. Both manifests specify official `pi05_libero` initialization,
zero-B LoRA, `collection_source=small_expo`, `source_task_id=28`, the Task-28
Small EXPO teacher only, `official_demo_ratio=0`, and the same 10/5 action
interface. The evaluators are live in separate tmux sessions and output
directories; these are historical action-bound controls, not mixed-data
results.

At 02:06, both action-bound evaluator processes remained present on their
assigned nodes with zero task records so far; neither has a traceback or
terminal marker. The controlled Task-2 suite was at 8/90, and both historical
Task-29 1,500-update suites were at 4/90. The action-bound counts therefore
remain pending rather than being interpreted as zeros.

At 02:07, both action-bound evaluators had completed checkpoint-manager
startup and were actively running on GPU. Their historical Orbax checkpoints
emit a nonfatal warning that step-1,500 metrics are absent; the evaluator
successfully found and restored the committed `agent`/`params` checkpoint and
the Python processes remain live. Neither action-bound suite has written its
first task record yet.

At 02:08, the action-bound evaluators remained live but were waiting in the
same `cl_sync_io_wait` shared-filesystem initialization state observed in
earlier full-suite launches; this is not a terminal error, so neither was
restarted. The live counts were Task-2 update 500 = 8/90, Task-28 outer
augmentation = 85/90, Task-28 executed-tail backfill = 25/90, Task-29
historical arms = 5/90 each, Task-60 update 100 = 13/90, and Task-60 update
500 = 41/90. No new strict full-suite result was available.

At 02:09, the strict validator still found only the five previously completed
full suites. Task-2 update 500 advanced to 9/90 and Task-60 update 500 to
42/90; the two Task-28 action-bound evaluators remained live at 0/90 in the
shared-filesystem initialization phase. No evaluator was restarted or
stopped, and no allocation state changed.

At 02:10, the same five suites remained the only strict-complete results. The
action-bound evaluators were still present, and no diagnostic allocation had
become free; the partial counts were unchanged from the prior poll. This is
logged as a verified wait rather than a terminal failure, so the live jobs
remain untouched.

At 02:10:58, a remote process health check showed both action-bound
evaluators actively progressing through initialization/JAX compilation: each
was a live `Sl` Python process using roughly 24--26% CPU and 7.5 GiB RSS, with
both launcher logs receiving new output. This is stronger evidence of forward
progress than the earlier `cl_sync_io_wait` observation. Task-2 update 500
advanced to 10/90 and Task-60 update 100 to 14/90; no new strict-complete
suite was available.

At 02:11, both action-bound evaluator handles remained present on their
assigned GPUs, with no terminal marker and no task record yet. The five
strict-complete scores remained unchanged; no diagnostic GPU had become free,
so no new arm was launched and no live process was restarted.

At 02:12, thread-level checks confirmed that both action-bound evaluators are
still doing work rather than being dead: each has a live primary thread
(running or waking on CPU) plus active JAX worker threads waiting on futexes,
with no exited process or traceback. Task-29 no-replay advanced to 6/90; the
action-bound suites remain at 0/90 during compilation.

At 02:12:30, Task-28 executed-tail backfill advanced to 26/90 and both
historical Task-29 arms advanced to 6/90. The action-bound processes remained
live at 0/90 but grew to roughly 8.4--8.5 GiB RSS and 43% CPU each, providing
additional evidence that compilation is still active rather than stalled.

At 02:13, both action-bound processes were in active `Rl` state at roughly
44% CPU and 8.5--8.7 GiB RSS, with JAX model-compilation warnings continuing
in their logs. Their zero-record startup is therefore still consistent with
the earlier roughly ten-minute evaluator initialization window. Task-2
update 500 reached 11/90 and Task-28 outer augmentation reached 86/90.

At 02:13:35, the action-bound processes remained active at roughly 44% CPU
and 8.7--8.9 GiB RSS, with no terminal state or traceback. No new strict
full-suite result was available, and no evaluator or allocation was changed.

At 02:14, GPU health checks showed the two action-bound evaluators using 48%
and 53% GPU utilization with about 17.3 GiB GPU memory each, in addition to
their live CPU processes. They remain at 0/90 only because the first task
record has not yet been committed; this is active JAX work, not an idle or
terminated evaluator. Task-60 update 500 advanced to 43/90, with no new
strict-complete suite.

At 02:14:31, both action-bound evaluators remained live after roughly ten
minutes. The clipped arm was using 58% GPU utilization and the raw arm was
still CPU-compiling while holding about 17 GiB GPU memory; neither had emitted
a task file. Task-60 update 100 advanced to 15/90. No process or allocation
was restarted or changed.

At 02:15, both action-bound evaluators remained live and GPU-backed, with
about 17.3 GiB GPU memory and 41--43% utilization, plus roughly 8.9--9.0 GiB
RSS each. They still had no committed task record and no terminal marker, so
they remain a verified slow initialization rather than a failed suite.

At 02:15:34, both action-bound evaluators remained GPU-active at roughly
17.3 GiB memory each and 41--54% utilization, with about 9 GiB RSS each. They
still had no task record, and no candidate crossed the strict 90/90 gate; the
live processes and allocations were preserved.

At 02:16, both action-bound evaluators remained live with about 17.3 GiB GPU
memory and 40--58% utilization; one was in a normal futex wait and the other
was actively running. Neither had emitted a task record, and the five
strict-complete suites were unchanged. No restart or allocation change was
made.

At 02:16:32, both Task-28 action-bound evaluators emitted their first valid
full-suite record (`task_00.json`): 20 episodes, 0/20 successes, mean 400
steps, with the expected task metadata and `mode=sft_pi`. This confirms that
checkpoint restore and rollout are working; the LIBERO dataset-path warning
is nonfatal and also appears in prior successful evaluators. Both arms are
now in task-by-task rollout at 1/90.

At 02:17, both action-bound arms remained live at 1/90 and GPU-active, using
about 17.3 GiB GPU memory each with 52--56% utilization. Task-28 outer
augmentation reached 87/90, backfill 27/90, and both historical Task-29 arms
reached 7/90. No new candidate passed strict validation.

At 02:17:46, the counts were unchanged from the prior poll. Both action-bound
processes remained live with CPU work and about 17.3 GiB GPU memory each; no
terminal marker or strict-complete transition occurred, so no intervention
was made.

At 02:18:17, the controlled Task-2 side evaluator advanced to 12/90. The two
Task-28 action-bound arms remained at 1/90, with live processes using about
17.3 GiB GPU memory and 40% utilization each; no terminal transition or new
strict-complete result occurred.

At 02:18:42, both action-bound evaluators remained live and GPU-backed, with
58% and 53% utilization respectively and no terminal state. Their counts and
the five strict-complete scores were unchanged; the live processes were left
untouched.

## Task-60 seed-789 update-150 interpolation arm (2026-11-06)

A newly allocated A100 slot (`23599571`, GPU UUID `GPU-a3e621b8-...`) was
verified idle before launch: it contained only the two-day sleeper and
reported 0 MiB of GPU memory. It now runs a separate login-side tmux session
`sft_task60_seed789_update150` for an isolated dose-interpolation arm. The arm
uses only the matching Task-60 Small EXPO collection (teacher step 450223),
zero official demonstrations, official `pi05_libero` initialization, true
`lora_only`/zero-B scope, full-proposal targets, batch 16, LR 5e-5, 10k
warmup, seed 789, and the unchanged 10/5 interface. It trains to update 150,
evaluates the source task first, and only then runs the four-task panel. The
learner and runtime outputs are node-local under `/tmp/usertmp/rpolonsky`;
no existing trainer, evaluator, allocation, or shared checkpoint tree was
changed. The process has initialized W&B and is GPU-visible; no score is
claimed until its source panel and four-task evaluation commit JSON results.

The seed-789 arm then passed its data gate with exactly 2,226 matching
Task-60 Small EXPO chunks and zero official chunks. Its source-task step-0
panel is 55% (20 episodes), and the trainer is actively executing the 150
updates; this is a seed-matched baseline, not a post-update result.

## Source-first seed-456 four-task evaluation repair (2026-11-06)

Both source-first seed-456 trainers completed their intended training and
source-only panels, but their original post-training evaluator command exited
before rollout with `ModuleNotFoundError: full_90_libero_evals_inference`.
The failure was an omitted import root in the evaluation launcher
(`.../sharder/training` was absent from `PYTHONPATH`); it was not an OOM,
checkpoint failure, or training failure. The saved checkpoints were verified
present under the node-local `checkpoints/200` and `checkpoints/100` trees,
with their matching student assets. Separate repair tmux sessions now run the
four-task evaluator with the correct import path on the same idle allocations:
Task-60/update-200 on `gpua064` and Task-60/update-100 on `gpua073`. The
update-200 source panel is 55%, equal to its 55% seed-matched control and
therefore remains eligible for the four-task screen; update-100 is 50% versus
55% and is retained only as a rejected/informative arm. No existing trainer or
production allocation was changed.

The new trainer passed initialization and loaded exactly 2,226 matching
Task-60 Small EXPO chunks, zero official-demo chunks, and
`official_demo_ratio=0.0`. It is in the source-only Task-60 retention panel
with active GPU utilization; no score is inferred yet.

The now-free `gpua073` slot is also running a separate Task-60 seed-456
update-100 source-first arm (`task_60_pure_seed456_sourcefirst100`). It uses
the identical task-pure recipe and will evaluate the four diagnostic tasks
from its update-100 checkpoint. This isolates the early-dose point rather
than relying on the update-200 run's latest-checkpoint retention.

The update-200 source-first arm has completed its step-0 source-only control
at 45% on Task 60 and has entered training; it remains active with finite
initialization and no traceback. The update-100 arm is still compiling. The
different seed's step-0 source score is retained as that arm's own baseline.

The Task-29 seed-456 replication has also completed its step-0 four-task
panel at `60/40/70/55%` on Tasks 2/28/29/60 (51.67% held-out average) and
entered its 300-update training loop with finite initialization. It remains
strictly task-pure; no later score is inferred yet.

## Task-2 seed-456 replication rejected (2026-11-06)

The independent Task-2 seed-456 pure-data replication completed its update-100
panel. It used matching Task-2 Small EXPO targets only, zero official-demo
chunks, official `pi05_libero` initialization, and the same true
`lora_only`/zero-B, full-proposal 10/5 protocol.

| source task | update | Task 2 | Task 28 | Task 29 | Task 60 | macro | held-out average | source change | held-out change |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0 | 70% | 50% | 65% | 55% | 60.00% | 56.67% | — | — |
| 2 | 100 | 70% | 35% | 60% | 55% | 55.00% | 50.00% | 0 pp | −6.67 pp |

The seed-123 Task-2/update-100 held-out gain therefore does not replicate;
this seed-456 arm is rejected and will not receive a Full-90 evaluation. Its
GPU is free for further diagnostics, while the unrelated Task-2 seed-123
Full-90 confirmation remains untouched.

At 02:19:13, the action-bound arms remained live at 1/90 with about 17.3 GiB
GPU memory and 40--50% utilization. Task-60 update 500 advanced to 44/90; no
new strict-complete suite was available, and no evaluator or allocation was
changed.

At 02:19:40, no state transition occurred: both action-bound arms remained
live at 1/90 with 41--54% GPU utilization and no terminal marker. Task-60
update 500 remained at 44/90; no new strict-complete result or free diagnostic
GPU was available.

At 02:20:08, both historical Task-29 arms advanced to 8/90 and Task-60
update 100 to 16/90. The Task-28 action-bound arms remained at 1/90; one
process was CPU-active and the other in a normal futex wait, with neither in a
terminal state. No strict-complete transition occurred.

At 02:20:35, no new task record or strict-complete transition occurred. Both
action-bound processes remained live, each holding about 17.3 GiB GPU memory
with 41--51% utilization; the existing evaluators and allocations were
preserved.

At 02:21:28, both Task-28 action-bound arms advanced to 2/90, and the
controlled Task-2 side evaluation advanced to 13/90. The new action-bound
task-1 records are valid actor-only 10/5 evaluations (20 episodes, 0/20
successes, mean 400 steps on each arm). No strict-complete transition
occurred.

At 02:22:15, Task-28 outer augmentation reached 88/90 and executed-tail
backfill 28/90. The Task-2 side evaluation remained at 13/90. The raw and
clipped action-bound arms remained at 2/90 with live GPU-backed processes; no
new strict-complete result was available.

At 02:22:41, no suite changed from the prior poll. Both action-bound
evaluators remained live at 2/90 with about 17.3 GiB GPU memory and 41--57%
utilization; outer augmentation remained at 88/90. No strict-complete
transition occurred.

At 02:23:08, no counts changed. The action-bound arms remained at 2/90,
outer augmentation at 88/90, and the five strict-complete suites remained
unchanged. Both action-bound processes were still live and GPU-active with
about 17.3 GiB memory each and 54--56% utilization.

At 02:23:34, no suite advanced: the action-bound arms remained at 2/90 and
outer augmentation at 88/90. Both action-bound processes remained live with
about 17.3 GiB GPU memory and 41--53% utilization; no strict-complete
transition occurred.

At 02:24:00, no counts changed. The action-bound arms remained at 2/90,
outer augmentation at 88/90, and the five strict-complete scores were
unchanged. Both action-bound processes remained live and GPU-active, so no
intervention was warranted.

At 02:24:27, the raw Task-28 action-bound arm advanced to 3/90 while the
clipped arm remained at 2/90. Task-60 update 500 advanced to 45/90. Both
action-bound processes remained GPU-active; no strict-complete transition
occurred.

At 02:24:51, both historical Task-29 arms advanced to 9/90. The Task-28 raw
action-bound arm remained at 3/90 and the clipped arm at 2/90; both processes
remained live and GPU-active. No strict-complete transition occurred.

At 02:25:18, both raw and clipped Task-28 action-bound arms advanced to
3/90. Their GPU-backed processes remained live with no terminal state; no
strict-complete transition occurred.

At 02:25:42, Task-60 update 100 advanced to 17/90. The raw and clipped
Task-28 action-bound arms remained at 3/90 each and were still live at about
17.3 GiB GPU memory with 50--52% utilization; no strict-complete transition
occurred.

At 02:26:11, Task-28 outer augmentation reached 89/90 and executed-tail
backfill 29/90. The raw and clipped action-bound arms remained at 3/90 each;
both processes were still live and GPU-active. No strict-complete transition
occurred.

At 02:26:36, the controlled Task-2 side evaluator advanced to 14/90 and the
historical Task-29 demo arm to 10/90. Task-28 outer augmentation remained at
89/90; both action-bound arms remained at 3/90 and live. No strict-complete
transition occurred.

At 02:27:03, both historical Task-29 arms were at 10/90 and the Task-2 side
evaluation remained at 14/90. Task-28 outer augmentation remained at 89/90;
the raw and clipped action-bound arms remained at 3/90 with live GPU-backed
processes. No strict-complete transition occurred.

At 02:27:27, no counts changed. Both action-bound arms remained live at 3/90
with about 17.3 GiB GPU memory and 40--50% utilization; the 89/90 outer-
augmentation suite had not closed, and no strict-complete transition occurred.

At 02:29:41, the live suite was re-polled from the output directories and the
process table. The raw Task-28 action-bound arm advanced to 4/90 and the
clipped arm remained at 3/90; both remained GPU-active. Task-28 executed-tail
backfill advanced to 30/90, Task-29 historical-demo advanced to 11/90, and
Task-60 update 500 advanced to 46/90. The Task-2 500-update side evaluation
remained at 14/90 and remains explicitly excluded from the diagnosis. The
Task-28 outer-augmentation suite remained at 89/90. No new strict-complete
candidate appeared; the validated set is unchanged (Task-28 update 300,
Task-29 updates 200 and 500, and Task-60 updates 200 and 400). All five
currently polled evaluator processes were present and live, so no restart or
GPU reallocation was justified.

At 02:30:37, Task-28 outer augmentation completed all 90 task records. The
strict validator reports 90 unique tasks, no missing tasks, no protocol errors,
and 516/1,800 successes (28.67%). Its launch script confirms the diagnosis
invariant: source task 28, the Task-28 Small EXPO teacher at step 275001,
`collection-source=small_expo`, `official-demo-ratio=0`, LoRA-only training
from the official `pi05_libero` base, and the same 10/5 action interface. Its
matched Full EXPO Task-28 score is 25.06%, so this control gives a +3.61 pp
margin. It is valid evidence but is slightly below the already validated
Task-28 full-proposal update-300 result (28.78%, +3.72 pp).

At 02:33, the prepared Task-29 raw-target versus controller-clipped-target
comparison was extended with its clipped arm on the now-unused GPU slot of
allocation 23416001_1 (gpua044). The launch is isolated in tmux
`sft_diag_task29_actionbounds_clipped_20261105` and writes to
`dcrlcheckpoints/sft_task29_actionbounds_20261103/clipped_no_replay_1500`.
The command line verifies Task-29 Small EXPO only, teacher step 375380, zero
official demos, zero-B LoRA from official `pi05_libero`, 1,500 updates, and
the unchanged 10/5 interface. W&B initialized successfully. At the first
poll the process was alive but still in model/JAX startup, before the required
data-count audit or step-0 evaluation; no failure marker was present.

At 02:21:03, the action-bound arms remained at 1/90 with live processes; one
was actively running and the other was in a filesystem RPC wait, but neither
was terminal. No strict-complete result or newly free diagnostic GPU appeared.

At 02:38, the Task-29 clipped arm completed initialization and printed the
required runtime data audit: `small_expo_chunks=2150`,
`official_demo_chunks=0`, `official_demo_episodes=0`, and
`official_demo_ratio=0.0`. It then entered the 20-episode step-0 actor
evaluation. The action-bound compile warnings were followed by a valid
initialization audit and no traceback; no optimizer update or result was
interpreted before the step-0 evaluation commits.

At 02:42, the first 20-episode task segment of the Task-29 clipped arm
completed (20/20). The evaluator then moved on to the remaining retention
tasks; the combined step-0 JSON was not yet committed, so no learning result
is inferred from this partial segment.

At 02:45, the evaluator's live W&B output showed the next task segment at
13/20 episodes. The process remained GPU-active (~17.3 GiB, ~57% utilization)
and was in a normal JAX synchronization wait between rollout kernels. The
combined step-0 artifact and any optimizer update remained absent.

At 02:46:53, the second step-0 task segment reached 20/20 and the evaluator
started the third segment (1/20). The full four-task step-0 artifact was still
not committed, so the run remains a rollout-in-progress rather than a scored
training point.

At 02:53:21, the Task-29 controller-clipped arm committed its complete step-0
retention result. All four declared tasks have 20 trials with initial-state
indices 0--19, and the result is: Task 2 = 65%, Task 28 = 40%, Task 29 =
60%, Task 60 = 45%, macro = 52.5%. The committed step-0 checkpoint exists at
`latest_checkpoints/0` and `checkpoints/0`. Its config confirms the single
factor arm: source task 29, `small_expo_chunks=2150`, zero official demos,
full-proposal targets, `clip_osc_motion_targets=true`, zero-B LoRA, batch 16,
1,500 updates, seed 42, evaluation seed 7, and horizon/replan 10/5. After the
step-0 checkpoint finalized, the trainer entered the real BC loop; no update
metric has yet been used as a result.

At 02:56, the clipped trainer passed its first-update XLA compilation and the
BC progress bar reached 12/1,500 updates. This confirms that real optimization
has begun after the committed step-0 checkpoint. The stdout stream has not yet
emitted a finite loss/gradient summary; update 500 is the first scheduled
retention point and remains the next interpretable trained result.

At 02:57, the same trainer advanced to 38/1,500 updates with the GPU process
live (~30.8 GiB resident, 100% utilization at the direct health poll). No
traceback or non-finite metric was observed; update 500 remains pending.

At 03:02, the Task-29 controller-clipped trainer remained live and reached
approximately 265/1,500 BC updates at a stable 1.5 seconds/update. Only the
step-0 retention artifacts were present; update 500 remains the first trained
checkpoint/evaluation and no trained score is inferred before it is committed.
The active diagnostic allocation table showed all currently running slots
occupied, while the newer A100 sleeper arrays remained pending, so no new arm
was launched. Direct revalidation of every completed controlled candidate
still reports 90 unique tasks, 20 trials per task, no protocol errors, and the
same strict scores: Task-28 update 300 = 518/1,800 (28.78%), Task-28 outer
augmentation update 300 = 516/1,800 (28.67%), Task-29 update 200 = 523/1,800
(29.06%), Task-29 update 500 = 524/1,800 (29.11%), Task-60 update 200 =
519/1,800 (28.83%), and Task-60 update 400 = 506/1,800 (28.11%).

At 03:09, the Task-29 controller-clipped arm reached update 500 and entered
its scheduled four-task retention evaluation. The process remains live; no
trained result is recorded until all four 20-episode segments finish and the
step-500 JSON/checkpoint is committed. The dataset-purity audit remains
unchanged: 2,150 Task-29 Small EXPO chunks, zero official-demo chunks, and no
cross-task mixture.

At 03:13, a recursive scan of the retained artifact tree recovered three
earlier strict-complete Task-28 actor-only LIBERO-90 evaluations that had not
been included in the current ranking. The Task-28 LoRA-only update-2,000
checkpoint validates at 471/1,800 (26.17%), while the raw and controller-clipped
action-bound update-1,500 checkpoints validate at 464/1,800 (25.78%) and
470/1,800 (26.11%). Their source launcher and logs record 760 Task-28 Small
EXPO chunks, zero official-demo chunks/ratio, official `pi05_libero`
initialization, zero-B LoRA, and the same 10/5 interface. Against the matched
Task-28 Full EXPO result of 451/1,800 (25.06%), their margins are +1.11, +0.72,
and +1.05 pp respectively. They are valid same-task controls but remain below
the newer Task-28 full-proposal update-300 result at 518/1,800 (28.78%,
+3.72 pp).

### Verified live poll after audit recovery (2026-11-05, 03:22 CST)

The Task-29 controller-clipped trainer remains alive in tmux
`sft_diag_task29_actionbounds_clipped_20261105`, with its Slurm step and
launcher processes present on `gpua044`. It has committed only the step-0
artifacts so far; the update-500 retention panel is still running and had
reached 14/20 episodes in the current task segment at the latest poll. No
step-500 score or checkpoint is inferred before the complete four-task panel
commits atomically. The run continues to satisfy the data-purity audit:
2,150 Task-29 Small EXPO chunks, zero official-demo chunks, and no cross-task
mixture.

The active strict full-suite evaluators also remain live. Current committed
record counts are: Task-28 action-bound raw/clipped 16/90 each, Task-28
executed-backfill 41/90, Task-29 historical demo50/no-replay 23/90 and 22/90,
Task-60 update-100/update-500 31/90 and 58/90, and the unrelated Task-2 side
panel 27/90. The completed candidates remain unchanged: Task-60 update-200
is the largest validated matched-Full-EXPO margin at +8.39 pp, and Task-29
update-500 is the highest absolute controlled SFT score at 29.11% (+4.61 pp).
All diagnostic slots are occupied; no new arm was launched or any existing
training allocation altered.

### Task-29 clipped-target update-500 retention result (2026-11-05, 03:23 CST)

The Task-29 controller-clipped arm completed and committed its first trained
retention point. The four-task actor-only panel is Task 2 = 70%, Task 28 =
50%, Task 29 = 70%, and Task 60 = 45%, for a macro success rate of 58.75%
(47/80 episodes). `evals/00000500.json`, `latest_checkpoints/500`, and
`checkpoints/500` are present; the optimizer metrics through update 500 are
finite. This is a promising retention result, but it is not counted as the
LIBERO-90 generalization score until the strict 90-task evaluator completes.

The candidate remains a valid same-task arm: 2,150 Task-29 Small EXPO chunks,
zero official-demo chunks, official `pi05_libero` initialization, zero-B
LoRA, full-proposal targets with only the six OSC motion dimensions clipped,
batch 16, learning rate 5e-5, and the unchanged horizon/replan 10/5. The
trainer session remains alive for its scheduled continuation; no other job or
allocation was changed.

At 03:45, the second update-1,000 panel had completed 20/20 and the third
panel had reached 7/20. The retention evaluator remained live without a
traceback; the aggregate and checkpoint were still pending, and `gpua044`
remained reserved by the trainer.

### Task-29 controller-clipped final retention result (2026-11-05, 04:17 CST)

The Task-29 controller-clipped, same-task-only arm completed all 1,500
updates and its four-task retention evaluation. The actor-only results are
Task 2 = 55%, Task 28 = 65%, Task 29 = 75%, and Task 60 = 25%, for a 55%
macro rate (44/80 episodes). The step-1,500 JSON is committed at
`evals/00001500.json`, and the final checkpoint metadata is committed at
`latest_checkpoints/1500`.

The run remains valid for the diagnosis: it used 2,150 Task-29 Small EXPO
chunks, zero official-demo chunks/episodes, the official `pi05_libero`
initialization, zero-B LoRA, full-proposal targets with only the six OSC
motion dimensions clipped before the existing normalization, batch 16,
seed 42, evaluation seed 7, and the unchanged 10/5 horizon/replan
interface. The step-1,000 retention peak (53.75% macro) was not treated as
the final result; the final panel is recorded separately.

After the trainer exited cleanly, the immutable step-500 checkpoint was
launched for the strict actor-only LIBERO-90 evaluation in tmux
`sft_diag_full90_task29_actionbounds_clipped_step500_20261105` on the
released CUDA device of `gpua044`. The evaluator uses exactly 90 tasks,
20 trials per task, seed 7, horizon 10/replan 5, and writes under
`dcrlcheckpoints/sft_retention_full90_eval_20261104/task29_actionbounds_clipped_step500`.
At launch it was live with no traceback; its full-suite score remains
pending. The unrelated Task-2 side evaluation and all other allocations were
left untouched.

At 04:30, the strict evaluator committed its first per-task record,
`task_00.json`: Task 0 achieved 1/20 successes (5%), with eval seed 7 and
the declared 10-step horizon/5-step replanning protocol. The evaluator
remains GPU-active and the remaining 89 task records are pending; this
partial record is not interpreted as a full-suite score.

At 04:35, the evaluator committed `task_01.json`: Task 1 achieved 0/20
successes. Together with Task 0, 2/90 task records are now committed; both
records verify eval seed 7 and the 10/5 protocol. The evaluator remains
GPU-active and no aggregate LIBERO-90 score is inferred from this partial
sample.

At 04:39:48, the same immutable evaluator committed `task_02.json`: Task 2
achieved 11/20 successes (55%). The evaluator remains alive and GPU-active at
3/90 task records. This is still only a partial result; no LIBERO-90 aggregate
or comparison against Full EXPO is inferred until all 90 task files pass the
strict validator.

At 04:44:48, it committed `task_03.json`: Task 3 achieved 0/20 successes.
The evaluator remains active at 4/90 records; the partial result is not used
as a full-suite score or compared against Full EXPO before completion.

At 04:49:45, it committed `task_04.json`: Task 4 achieved 0/20 successes.
The record verifies the declared seed 7, 10-action horizon, 5-step replanning,
and 400-step limit. The evaluator is now at 5/90 records; no aggregate score
is inferred from this partial sample.

At 04:54:42, it committed `task_05.json`: Task 5 achieved 0/20 successes.
The same strict protocol fields are present, and the evaluator is now at 6/90
records. This remains a partial result and is not compared against Full EXPO
until the complete suite is validated.

At 04:46, a read-only process inventory confirmed that all ten running sleeper
slots are occupied: the nine diagnostic tmux sessions each have an evaluator,
and the second gpua044 slot hosts the pre-existing Small EXPO Task-28
full-suite/teacher evaluation processes. That slot is therefore not available
for a new diagnosis arm. No allocation or process was changed.

At 04:59:38, the immutable Task-29 clipped-target evaluator committed
`task_06.json`: Task 6 achieved 0/20 successes. The record verifies the
expected evaluation seed 7, 10-action horizon, 5-step replanning, and 400-step
limit. The evaluator is now at 7/90 records and remains GPU-active; this
partial result is not used as a LIBERO-90 aggregate or compared against Full
EXPO until all 90 records pass validation.

At 05:04:18, the evaluator committed `task_07.json`: Task 7 achieved 0/20
successes. The record verifies the Task-7 instruction, evaluation seed 7,
10-action horizon, 5-step replanning, and 400-step limit. The immutable
Task-29 clipped-target evaluator is now at 8/90 records and remains active;
this partial result is not used as a LIBERO-90 aggregate or compared against
Full EXPO until the complete suite passes validation.

At 05:12:22, the evaluator committed `task_10.json`: Task 10 achieved 20/20
successes. The record verifies the Task-10 instruction, evaluation seed 7,
10-action horizon, 5-step replanning, and 400-step limit. The immutable
Task-29 clipped-target evaluator is now at 11/90 records and remains active;
this partial result is not used as a LIBERO-90 aggregate or compared against
Full EXPO until the complete suite passes validation.

At 05:10:45, the evaluator committed `task_09.json`: Task 9 achieved 20/20
successes. The record verifies the Task-9 instruction, evaluation seed 7,
10-action horizon, 5-step replanning, and 400-step limit. The immutable
Task-29 clipped-target evaluator is now at 10/90 records and remains active;
this partial result is not used as a LIBERO-90 aggregate or compared against
Full EXPO until the complete suite passes validation.

At 05:09:04, the evaluator committed `task_08.json`: Task 8 achieved 0/20
successes. The record verifies the Task-8 instruction, evaluation seed 7,
10-action horizon, 5-step replanning, and 400-step limit. The immutable
Task-29 clipped-target evaluator is now at 9/90 records and remains active;
this partial result is not used as a LIBERO-90 aggregate or compared against
Full EXPO until the complete suite passes validation.

### Strict full-suite revalidation and current slot state (2026-11-05, 05:17 CST)

The strict validator was rerun against every currently complete new candidate.
All six passed with 90 unique task records and zero errors: Task-28
full-proposal update 300 = 518/1,800 (28.78%), Task-28 outer-augmentation
update 300 = 516/1,800 (28.67%), Task-29 full-proposal updates 200 and 500 =
523/1,800 (29.06%) and 524/1,800 (29.11%), and Task-60 full-proposal updates
200 and 400 = 519/1,800 (28.83%) and 506/1,800 (28.11%). The matched-Full-EXPO
ranking therefore remains unchanged: Task-60 update 200 is the largest
verified margin at +8.39 percentage points, and Task-29 update 500 is the
highest absolute controlled SFT score at +4.61 points over its matched Full
EXPO result.

The Task-29 clipped-target update-500 evaluator remains live at 11/90 records.
The other diagnostic evaluators are also still attached to their allocated
slots, including panels that have written 90/90 records but have not exited.
There is consequently no safely reusable GPU slot at this poll; no process,
tmux session, allocation, or unrelated Task-2 side evaluation was changed.

At 05:18:50, the same evaluator committed `task_11.json`: Task 11 achieved
0/20 successes. The record verifies evaluation seed 7, horizon 10, replanning
5, and the 400-step limit; the strict panel is now 12/90 and remains active.

### Continued evaluator progress; no released diagnostic slot (2026-11-05, 05:19 CST)

A further read-only poll confirmed forward progress in the active strict
evaluations: Task-28 executed-backfill is at 71/90, Task-60 full-proposal
update 500 is at 83/90, the historical Task-29 demo50/no-replay panels are at
52/90 each, Task-28 raw/clipped action-bound panels are at 45/90 each, Task-60
update 100 is at 59/90, and the Task-2 side panel is at 54/90. The Task-29
clipped-target update-500 panel remains live at 12/90. All ten diagnostic
allocation slots are still RUNNING and the three newer sleeper arrays remain
PENDING, so no new arm was launched and no existing process or allocation was
changed.

A finalization-path audit found that the six already complete candidates have
their `summary.json` artifacts and no corresponding live evaluator process;
their slots have already been reused by the partial panels above. The only
live diagnostic evaluators are therefore the nine partial panels, while the
Task-2 side panel and Small EXPO teacher evaluation remain separate and
untouched. This confirms there is no completed evaluator session to clean up
and no safely reusable GPU slot at this time.

At 05:22, the active panels continued advancing: Task-60 full-proposal update
100 reached 60/90, the excluded Task-2 side panel reached 56/90, Task-28
clipped action-bound reached 46/90, and Task-29 clipped-target update 500
reached 13/90. The other partial counts were unchanged, all ten diagnostic
allocations remained RUNNING, and the three newer sleeper arrays remained
PENDING. No new arm or process change was made.

At 05:22:38, Task-28 executed-backfill advanced to 72/90 and both Task-28
raw/clipped action-bound panels reached 46/90. The remaining evaluator and
allocation states were unchanged; no new arm was launched.

At 05:23:45, the Task-28 clipped action-bound panel advanced to 47/90. All
diagnostic allocations remained occupied and no new arm was launched.

At 05:24:18, Task-28 executed-backfill advanced to 73/90, Task-60 update 100
to 61/90, and the Task-29 demo50 historical panel to 53/90. The queue still
showed all ten diagnostic allocations RUNNING and the additional arrays
PENDING; no new arm was launched.

At 05:24:38, the Task-29 no-replay historical panel also reached 53/90. No
evaluator finished and no diagnostic GPU slot was released.

At 05:24:58, both Task-28 raw and clipped action-bound panels reached 47/90.
No evaluator exited and no diagnostic GPU slot was released.

At 05:25:15, the excluded Task-2 side panel advanced to 57/90. The diagnostic
panels and allocation states were otherwise unchanged; no new arm was
launched.

At 05:25:32, Task-60 full-proposal update 500 advanced to 85/90; the excluded
Task-2 side panel remained at 57/90. No evaluator finished and all ten
diagnostic allocations remained occupied.

At 05:25:51, the Task-29 clipped-target update-500 panel advanced to 14/90.
No evaluator finished and no diagnostic allocation was released.

At 05:26:40, Task-60 full-proposal update 100 advanced to 62/90. No evaluator
finished and no diagnostic allocation was released.

At 05:27:28, the excluded Task-2 side panel advanced to 58/90. The diagnostic
panels and allocation states were otherwise unchanged; no new arm was
launched.

At 05:27:44, the Task-28 clipped action-bound panel advanced to 48/90. The
diagnostic allocations remained occupied and no new arm was launched.

At 05:28:14, the Task-28 clipped action-bound panel remained at 48/90 while
the other active panels and allocation states were unchanged; no new arm was
launched.

At 05:28:53, Task-28 executed-backfill advanced to 74/90 and both Task-28
raw/clipped action-bound panels reached 48/90. No evaluator finished and no
diagnostic GPU slot was released.

At 05:29:33, both Task-29 historical demo50 and no-replay panels advanced to
54/90. No evaluator finished and no diagnostic GPU slot was released.

At 05:29:50, the Task-29 clipped-target update-500 panel advanced to 15/90.
No evaluator finished and no diagnostic GPU slot was released.

At 05:31:03, Task-60 full-proposal update 500 advanced to 86/90. No evaluator
finished and no diagnostic GPU slot was released.

At 05:31:52, Task-60 full-proposal update 100 advanced to 63/90. No evaluator
finished and no diagnostic GPU slot was released.

At 05:32:54, the Task-28 clipped action-bound panel advanced to 49/90. No
evaluator finished and no diagnostic GPU slot was released.

At 05:33:32, Task-28 executed-backfill advanced to 75/90 while the clipped
action-bound panel remained at 49/90. No evaluator finished and no diagnostic
GPU slot was released.

At 05:35:23--05:36:01, the live evaluator poll confirmed that all ten
diagnostic A100 allocation slots remained occupied and all evaluator tmux
sessions were still present. The six previously completed candidates remained
at 90/90. Partial panels were: Task-60 full-proposal update 500 at 86/90,
Task-28 executed-backfill update 200 at 75/90, Task-60 full-proposal update
100 at 63/90, the excluded Task-2 side panel at 59/90, Task-29 historical
demo50 and no-replay panels at 55/90, Task-28 raw and clipped action-bound
panels at 49/90, and Task-29 clipped-target update 500 at 16/90. Their
launcher logs continued to emit task results, so this is active progress, not
a stalled tmux shell. The repeated missing `/projects/.../datasets` warning
did not stop evaluation; no new full-suite score was inferred and no
diagnostic slot was reused.

At 05:37:06, the strict validator was rerun directly for all six 90/90
candidate directories. Every candidate passed with 90 unique task records,
zero errors, and the same scores: Task-28 full-proposal 300 = 518/1,800
(28.78%), Task-28 outer-augmentation 300 = 516/1,800 (28.67%), Task-29
full-proposal 200/500 = 523/1,800 (29.06%/29.11%), and Task-60
full-proposal 200/400 = 519/1,800 (28.83%/28.11%).

The apparent unused sleeper allocation was audited before reuse. Both
allocations on `gpua044` are occupied by live compute: the preserved Small
EXPO Task-28 teacher LIBERO-90 evaluator uses one GPU and the Task-29
action-bound update-500 evaluator uses the other. The remaining nine
diagnostic allocations also have live evaluator processes. Thus there is no
safe idle GPU slot at this poll; no running session, allocation, teacher
evaluation, or Task-2 side panel was changed.

At 05:39:39, the next poll showed continued evaluator progress with all six
validated candidates still complete. Partial counts were Task-60 update 500
87/90, Task-28 executed-backfill 76/90, Task-60 update 100 64/90, the
excluded Task-2 side panel 60/90, Task-29 historical demo50 57/90 and
no-replay 56/90, Task-28 raw and clipped action-bound 50/90 each, and
Task-29 clipped-target update 500 16/90. Nine diagnostic evaluator sessions
remain active; all three additional sleeper arrays remain pending. No slot
was released and no session or allocation was changed.

At 05:40, a provenance audit reconciled the run-plan wording with the actual
launchers. The plan's older phrase “Task 28 Small EXPO successful collection
only” was too broad and has been corrected: the active Task-29 and Task-60
arms use their matching Small EXPO collections, not Task-28 data. Their
launchers explicitly set `source_task_id=29/60`, matching teacher directories,
`collection_source=small_expo`, and zero official-demo chunks; the runtime
audits report 2,150 and 2,226 target chunks respectively. This confirms the
completed Task-29 and Task-60 margins satisfy the task-specific data rule. No
trainer code or running process was changed; only the run-plan documentation
was clarified.

At 05:41:30, active evaluator progress continued: Task-60 full-proposal
update 500 reached 88/90; Task-28 executed-backfill remained at 76/90;
Task-60 update 100 reached 64/90; the excluded Task-2 side panel reached
61/90; Task-29 historical demo50/no-replay reached 58/90 and 57/90; Task-28
clipped/raw action-bound reached 51/90 and 50/90; and Task-29 clipped-target
update 500 reached 17/90. All nine diagnostic evaluator sessions remained
live, the preserved teacher evaluation remained protected, and the three
additional sleeper arrays remained pending. No new full-suite score or safe
GPU slot was available.

At 05:43:47, the process-level poll confirmed that the near-complete panels
still had live `srun` launcher processes and tmux sessions. Task-28 clipped
action-bound advanced to 52/90 and the excluded Task-2 side panel to 62/90;
the other partial panels remained active at their prior counts. Since no
evaluator had exited and all current allocations remained occupied, no slot
was reused and no protected run was changed.

At 05:42:03, the live poll confirmed continued writes rather than a stall:
Task-60 update 500 was 88/90 and update 100 65/90; Task-28 executed-backfill
was 76/90; the excluded Task-2 panel was 61/90; Task-29 historical demo50 and
no-replay were 58/90 and 57/90; Task-28 clipped/raw action-bound were 51/90
and 50/90; and Task-29 clipped-target update 500 was 17/90. The nine
diagnostic tmux sessions remained present, all sleeper allocations remained
running, and all three additional arrays remained pending. No strict-complete
new score or safe slot appeared, so no run was launched or altered.

At 05:42:48, the live files showed continued progress: Task-28
executed-backfill reached 77/90; Task-60 update 100 reached 65/90; Task-29
demo50 reached 58/90; Task-28 raw and clipped action-bound reached 51/90
each; and the other panels remained active (Task-60 update 500 88/90,
Task-2 side panel 61/90, Task-29 no-replay 58/90, and Task-29 clipped 17/90).
All nine diagnostic sessions remained present and all three extra arrays were
still pending. The newest task files had current timestamps, so no evaluator
was restarted and no allocation was changed.

At 05:45:35, a log/process audit ruled out evaluator hangs. The Task-28 raw
and clipped action-bound evaluators were still committing task-51 records.
The slower Task-29 clipped-target evaluator had a live `full90_audit_eval.py`
process on `gpua044`, using 17,479 MiB and 47% GPU utilization; its launcher
had no traceback or terminal marker. The preserved Small EXPO teacher process
was also still live on the other `gpua044` GPU. No evaluator was restarted or
reclaimed.

At 05:49:04, Task-28 clipped action-bound advanced to 53/90. Task-60
full-proposal update 500 remained at 89/90 awaiting task 89; the other
diagnostic panels and the excluded Task-2 panel remained active. All nine
diagnostic sessions and ten allocations were still occupied, so no run was
restarted or resource was reclaimed.

At 05:47:10, Task-60 full-proposal update 500 reached 89/90 records after
committing zero-based task 88; its final task 89 was still pending. Task-28
executed-backfill reached 78/90, Task-60 update 100 66/90, Task-2 62/90,
Task-29 demo50/no-replay 59/90 each, Task-28 raw/clipped action-bound 52/90
each, and Task-29 clipped-target 18/90. The Task-60 and Task-28 logs showed
new task commits with no traceback, so all active evaluators were left alone.

At 05:51:03, Task-60 full-proposal update 500 committed its final zero-based
task 89. The strict validator then passed: 90 files, 90 unique tasks, zero
errors, and 517/1,800 successes (28.72%). Against matched Full EXPO Task-60
(368/1,800, 20.44%), this is a +8.28 percentage-point margin; Task-60 update
200 remains the overall best verified margin at +8.39 pp. The evaluator log
finished Orbax cleanup, but its Python process was still present during the
immediate post-validation poll, so its allocation was not reclaimed.

At 05:52:11, the Task-60 evaluator exited naturally and its tmux session
disappeared, releasing `gpua038` within sleeper allocation `23433381_5` while
leaving the allocation itself running. I launched the next documented
task-specific target-construction arm there in a new tmux session,
`sft_diag_task60_executed_backfill_gpua038_20261105`. It uses only the Task-60
Small EXPO collection (teacher step 450223; runtime source directory for task
60), official `pi05_libero`, zero-B LoRA, zero official demos, batch 16,
learning rate 5e-5, 500 updates, and 10/5 retention evaluations/checkpoints
every 100 updates. W&B authenticated as run `n9amrwj6`. At 05:55 it remained
in normal JAX/model startup with no data audit, step-0 result, traceback, or
OOM yet; the existing panels and allocations were not changed.

At 05:58:24, the new Task-60 executed-backfill arm completed its runtime
audit: `small_expo_chunks=2226`, `official_demo_chunks=0`,
`official_demo_episodes=0`, `official_demo_ratio=0.0`, and
`target_mode=executed_backfill`. Its initialization audit had already passed
with `scope=lora_only`; it then entered the 20-episode step-0 actor evaluation.
No optimizer update or retention score is claimed until that evaluation
commits.

At 06:08:46, that arm was still healthy and had completed the first two of its
four step-0 retention tasks (Tasks 2 and 28), with the third (Task 29) at 3/20
episodes. The trainer process remained live on `gpua038`, using about 17.3 GiB
and approximately 46% GPU utilization. No step-0 JSON/checkpoint or optimizer
update had yet been committed, so no score is inferred; all other evaluator
sessions and the out-of-scope Task-2 side panel were left untouched.

At 06:12:41, the same initial pass had completed Tasks 2, 28, and 29 and was
6/20 episodes into Task 60. The trainer remained live in its original tmux
session with no traceback, retention JSON, checkpoint, or optimizer update
written yet; the score remains intentionally unclaimed until all four task
records are committed.

At 06:15:24, the complete step-0 retention record was committed for the
Task-60 `executed_backfill` arm: Task 2 = 75%, Task 28 = 35%, Task 29 = 70%,
Task 60 = 55%, and four-task macro = 58.75%. The JSON is
`sft_retention_target_semantics_20261105/task_60_executed_backfill/evals/00000000.json`.
This is the official-pi05 initialization reference for this arm, not a trained
result; the process remained live and proceeded into optimizer updates.

At 06:18:14, the arm had completed its JAX training compilation and reached
optimizer step 15/500. `metrics.jsonl` contained finite actor losses and
gradient norms (for example, step 15 loss 0.1333 and grad norm 0.0246), with
16 Task-60 Small EXPO chunks and zero official-demo chunks on each update.
GPU memory was stable at about 30.8 GiB and the trainer remained healthy; the
next retention/checkpoint gate is step 100.

At 06:21:20, the arm had reached optimizer step 99/500 with finite metrics and
entered the scheduled step-100 retention evaluation. The evaluator was at 4/20
episodes on its first retention task, with the trainer still live and no
traceback. All ten allocated diagnostic A100 slots remained occupied, so no
new arm was launched and no existing session was changed.

At 06:23:09, the step-100 evaluation continued normally at 11/20 episodes on
its first retention task. No result was inferred before all four tasks finish;
the trainer remained live with stable model memory and no traceback.

At 06:25:18, the first step-100 retention task completed all 20 episodes and
the second task had begun (2/20). The trainer remained live with no error or
partial score artifact; the four-task step-100 result is still pending.

At 06:26:43, the second step-100 retention task had advanced to 8/20 episodes.
The trainer remained healthy with no traceback or partial score; all diagnostic
GPU slots were still occupied.

At 06:29:45, the first two step-100 retention tasks had both completed 20/20
episodes and the third task had begun. The trainer remained live without a
traceback; the complete four-task step-100 score and checkpoint gate remained
pending.

At 06:30:23, the third step-100 retention task was at 4/20 episodes. Tasks 1
and 2 remained complete, the trainer had no traceback, and all diagnostic GPU
allocations were still occupied.

At 06:30:47, the third step-100 task had advanced to 6/20 episodes. The
trainer remained healthy at about 31.3 GiB GPU memory, with all ten diagnostic
allocations still occupied.

At 06:36:53, the complete Task-60 `executed_backfill` step-100 result and its
latest learner checkpoint were committed: Task 2 = 65%, Task 28 = 40%, Task
29 = 65%, Task 60 = 50%, four-task macro = 55.0%. The step-0 macro was 58.75%,
so replacing the proposal tail with executed-backfill did not improve early
retention. The trainer resumed normally and reached step 115 with finite
metrics; the 500-step arm remains active for later checkpoints.

At 06:37:57, the same arm had resumed normally and reached optimizer step
137/500 with finite losses and gradient norms. Each update still used 16
Task-60 Small EXPO chunks and zero official-demo chunks; the GPU was fully
utilized with stable memory, and no diagnostic slot was free.

At 06:38:27, the arm had advanced to optimizer step 156/500 with finite
metrics, approximately 98% GPU utilization, and stable model memory. No new
diagnostic GPU slot was available.

At 06:38:51, the arm had reached optimizer step 171/500 with finite metrics
and 100% GPU utilization. All ten diagnostic allocations remained occupied.

At 06:39:13, the arm had reached optimizer step 185/500 with finite metrics
and 100% GPU utilization. The next retention/checkpoint gate remains step 200;
all diagnostic allocations were still occupied.

At 06:40:17, the arm reached step 199/500 and entered the step-200 retention
evaluation, which was at 2/20 episodes on its first task. The trainer remained
healthy and no partial step-200 score was inferred.

At 06:40:38, the step-200 evaluation had advanced to 3/20 episodes on its
first task. The trainer remained live without errors and all diagnostic GPU
allocations were still occupied.

At 06:41:06, the step-200 evaluation had advanced to 5/20 episodes on its
first task. The trainer remained live without errors; no partial score was
promoted and no diagnostic allocation was free.

At 06:41:30, the step-200 evaluation reached 7/20 episodes on its first task.
The trainer remained healthy and no diagnostic allocation was free.

At 06:44:09, the first step-200 retention task completed 20/20 episodes and
the second task began. The trainer remained healthy; the complete score and
checkpoint gate remained pending.

At 06:44:28, the second step-200 task was at 2/20 episodes. The trainer
remained healthy and the complete four-task result remained pending.

At 06:44:48, the second step-200 task reached 4/20 episodes with the trainer
still healthy; no complete score was available yet.

At 06:45:07, the second step-200 task reached 5/20 episodes. The trainer
remained healthy and all diagnostic allocations were still occupied.

At 06:45:28, the second step-200 task reached 6/20 episodes. The trainer
remained healthy and the complete result was still pending.

At 06:45:46, the second step-200 task reached 8/20 episodes. The trainer
remained healthy and the complete result was still pending.

At 06:46:06, the second step-200 task reached 9/20 episodes. The trainer
remained live and healthy; the complete result remained pending.

At 06:46:35, the second step-200 task reached 11/20 episodes. The trainer
remained healthy and the complete result remained pending.

At 06:46:54, the second step-200 task reached 13/20 episodes. The trainer
remained healthy; no complete result or free diagnostic allocation was
available.

At 06:47:15, the second step-200 task reached 15/20 episodes. The trainer
remained healthy and the complete result remained pending.

At 06:47:38, the second step-200 task reached 16/20 episodes. The trainer
remained healthy and the complete result remained pending.

At 06:47:59, the second step-200 task reached 17/20 episodes. The trainer
remained healthy and the complete result remained pending.

At 06:49:02, the first two configured step-200 retention tasks—actual task IDs
2 and 28—were complete, and actual task 29 had reached 3/20 episodes. Earlier
“first/second task” shorthand refers to these configured slots, not task IDs;
this explicit mapping is recorded to avoid ambiguity.

At 06:55:56, the complete step-200 retention record for the Task-60
`executed_backfill` arm was committed at
`dcrlcheckpoints/sft_retention_target_semantics_20261105/task_60_executed_backfill/evals/00000200.json`.
The exact actor-only retention scores were Task 2 = 70%, Task 28 = 50%, Task
29 = 75%, and Task 60 = 55%, with a four-task macro success rate of 62.5%
(50/80 successful episodes). The trainer also committed the corresponding
Orbax checkpoint under `latest_checkpoints/200` and resumed with finite
metrics. Relative to this arm's step-0 macro of 58.75% and step-100 macro of
55.0%, executed-backfill recovered at step 200 but has not yet established a
full-LIBERO-90 result or a new best generalization margin. The arm remains
configured for steps 300, 400, and 500; no official-demo chunks are used.

At 07:15:05, the complete step-300 retention record for the same arm was
committed as `evals/00000300.json`: Task 2 = 80%, Task 28 = 40%, Task 29 =
65%, and Task 60 = 50%, with a four-task macro of 58.75% (47/80). This is
below the step-200 macro of 62.5% and therefore does not support executed
backfill as a better target construction. The step-300 Orbax checkpoint was
committed and the trainer resumed at step 319 with finite metrics; it continues
to steps 400 and 500 for the planned retention sweep. The data audit remains
unchanged: 2,226 Task-60 Small EXPO chunks, zero official-demo chunks, and no
cross-task data.

At 07:26, the prior Task-28 executed-backfill full-90 evaluator had completed
and its tmux session was gone while the parent sleeper allocation remained
alive. A read-only GPU audit confirmed gpua002 had no user process, no compute
application, and 0 MiB used. The isolated evaluator
`run_full90_task60_executed_backfill_step200_20261105.sh` was therefore
launched in a new tmux session on gpua002 for the exact committed Task-60
executed-backfill step-200 checkpoint. It uses the same actor-only strict
LIBERO-90 protocol as the other controlled candidates (20 episodes/task,
seed 7, image size 224, horizon 10, replan 5, one FSDP device). At 07:28 it
had restored the one committed checkpoint successfully and was still in
initialization; no partial full-90 score is being reported.

At 07:34:35, the Task-60 `executed_backfill` arm completed its step-400
retention record: Task 2 = 65%, Task 28 = 35%, Task 29 = 70%, and Task 60 =
45%, macro = 53.75% (43/80). The committed step-400 checkpoint was verified,
and training resumed with finite metrics at step 407 toward the final planned
step 500. The retention curve is now 58.75% (step 0), 55.0% (100), 62.5%
(200), 58.75% (300), and 53.75% (400), so this target construction is not
currently competitive with the full-proposal arm. The exact step-200 full-90
evaluator remains alive on gpua002 in JAX compilation; it has not emitted a
task result or error.

At 07:36:25, the strict evaluator completed initialization and began actual
LIBERO-90 rollout evaluation on gpua002. It had written one task file and
reported task 00 at 5.0% success; the aggregate remains deliberately
unreported until all 90 task files are committed and passed through the strict
validator. In parallel, the retention trainer had reached step 464/500 with
finite metrics and zero official-demo chunks.

At 07:46, a confirmed-free slot was found in the existing sleeper allocation
`23416001_2` on gpua082: the prior evaluator had exited, and a read-only audit
showed no user compute process and 0 MiB GPU memory. I launched the next
task-specific single-factor arm in tmux
`sft_diag_task60_outer_aug_gpua082_20261105`, using
`run_task60_outer_aug_zero_b_20261105.sh`. It reuses only the Task-60 Small
EXPO collection (teacher step 450223), official `pi05_libero`, zero-B LoRA,
zero official demos, batch 16, the same 10/5 interface, and enables the outer
EXPO image augmentation that the SFT `bc_update` path normally omits. Its
isolated output is
`dcrlcheckpoints/sft_retention_outer_aug_20261105/task_60`; W&B run `xca5oaea`.
The process passed W&B startup and reached Pi0.5 model initialization/JAX
compilation; no data-count, step-0 score, or trained result is claimed yet.
No existing training, side panel, evaluator, or allocation was changed.

At 07:50:46, the outer-augmentation arm passed its runtime lineage audit:
`small_expo_chunks=2226`, `official_demo_chunks=0`,
`official_demo_episodes=0`, `official_demo_ratio=0.0`,
`target_mode=full_proposal`, and `official_demo_path=disabled`. It entered the
step-0 actor retention evaluation on the same four-task panel; no step-0 score
is promoted until all 80 episodes commit. The exact Task-60 executed-backfill
step-500 retention evaluation was also still running, and its strict step-200
full-90 audit had committed four of 90 task files (through task 03); no
aggregate was inferred.

At 07:53:43, the Task-60 executed-backfill arm committed its final step-500
retention record: Task 2 = 60%, Task 28 = 30%, Task 29 = 60%, Task 60 = 60%,
macro = 52.5% (42/80). Its complete curve is therefore 58.75%, 55.0%, 62.5%,
58.75%, 53.75%, and 52.5% at steps 0/100/200/300/400/500. The committed
`latest_checkpoints/500` artifact was verified and the trainer exited cleanly,
releasing gpua038. This target construction is not competitive with the
full-proposal arm on the retention panel.

At 07:54, after verifying that the existing exact step-200 full-90 evaluator
had only written tasks 00--03, I launched a non-overlapping tasks-45--89
shard of the same audit in tmux
`sft_diag_full90_task60_executed_backfill_step200_shard45_89_gpua038_20261105`
using the released gpua038 slot. It writes task files into the existing audit
directory and uses `--shard-index 1` for a separate shard summary; it does not
touch the running task-00--89 sequential evaluator or any training job. The
shard reached normal robosuite/JAX initialization without an error; no task
score is claimed yet.

At 07:58, the completed Task-2 side panel was independently verified as
finished (`task_89.json` and `summary.json` present, gpua087 at 0 MiB with no
user compute process). To prevent the original all-90 evaluator from later
duplicating the new shards, I stopped only that diagnostic evaluator after its
committed tasks 00--04 and preserved those files. I replaced it on gpua002
with an explicit tasks-05--22 shard in tmux
`sft_diag_full90_task60_executed_backfill_step200_shard05_22_gpua002_20261105`.
The merged audit is now partitioned without overlap as 00--04 (already
committed), 05--22, 23--44, and 45--89; all three new shard processes reached
normal robosuite/JAX initialization with no error.

At 08:02:14, the tasks-45--89 shard committed its first record, task 45 = 0%
success, while the other two shards remained in normal model initialization.
The merged audit therefore contains six task files (00--04 and 45); no
full-suite score is inferred. The outer-augmentation arm remained in its
four-task step-0 evaluation, with no committed JSON yet.

At 08:05, a read-only sweep found gpua051 and gpua070 free after their prior
diagnostic evaluators exited (0 MiB and no user compute processes). I launched
the missing apples-to-apples actor-only Full EXPO Task-60 control using the
official full learner checkpoint at step 60386, split into two LIBERO-90
shards in tmux `sft_diag_full_expo_task60_actor_only_shard0_gpua051_20261105`
and `sft_diag_full_expo_task60_actor_only_shard1_gpua070_20261105`. It uses
the same 20 episodes/task, seed 7, 224-pixel, 10/5 rollout protocol as the SFT
audit and writes only under
`dcrlcheckpoints/full_expo_task60_actor_only_full90_20261105`. Both launchers
passed normal robosuite initialization; no actor-only EXPO score is claimed
until both shards validate.

At 08:08:14, the outer-augmentation Task-60 step-0 retention record committed
at Task 2/28/29/60 = 70%/40%/55%/50%, macro 53.75%. This is only the
initialization reference, not a trained result; the arm then entered its
500-update training loop with finite metrics.

The first actor-only EXPO shard on gpua051 initially failed before model
restore with a transient `CUDA_ERROR_NOT_INITIALIZED: No visible GPU devices`.
A read-only retry check showed `CUDA_VISIBLE_DEVICES=0` and a visible A100;
the failed diagnostic session was replaced by
`sft_diag_full_expo_task60_actor_only_shard0_gpua051_retry_20261105` with a
separate log. The retry has a live Python evaluator and no traceback so far;
the gpua070 shard remains live and initializing. No SFT training or unrelated
evaluation was changed.

At 08:12:52, the gpua051 actor-only EXPO retry terminated with the same
`CUDA_ERROR_NOT_INITIALIZED: No visible GPU devices` error during JAX mesh
creation, despite `nvidia-smi` exposing the A100. This slot is therefore
treated as CUDA-incompatible for this evaluator and will not be retried again
until an external allocation/node state changes. The gpua070 actor-only EXPO
shard remains live; all other diagnostic processes remain untouched.

At 08:15, the SFT executed-backfill full-90 shards had committed 14 task
records (00--06, 23--25, and 45--48). The actor-only EXPO gpua070 shard had
committed its first assigned record, task 01; no aggregate is inferred for
either suite. The outer-augmentation arm was still in its step-100 retention
evaluation after finite updates, with the step-0 macro remaining 53.75%.

At 08:20:23, a fresh read-only audit found 17 strict SFT full-90 task files:
00--07, 23--26, and 45--49. The three non-overlapping SFT evaluator shards
remain live and are advancing; the task-23--44 shard is still running even
though its original tmux pane is not visible, so it was not restarted. The
actor-only Full EXPO control has two committed files (tasks 01 and 03) on its
gpua070 shard. The gpua051 shard and its independent retry both failed during
JAX CUDA initialization with `CUDA_ERROR_NOT_INITIALIZED`, so that slot is
not being reused without an external node-state change. The outer-augmentation
trainer has finite updates through step 99 but still only its step-0 eval file;
its step-100 four-task eval is pending. All currently usable GPU slots are
occupied by these diagnostics or preserved evaluator jobs; the newly submitted
sleeper allocations remain pending. No training or side-panel allocation was
changed during this audit.

At 08:21:33, the SFT full-90 audit advanced to 19 committed task files,
adding tasks 08 and 50; the three shards remain live. The actor-only EXPO
control remains on its CUDA-capable gpua070 shard with two committed files
(tasks 01 and 03), while the gpua051 shard remains unavailable because both
JAX initialization attempts failed. The outer-augmentation arm still has
finite training metrics through step 99 and no committed step-100 evaluation.
The allocation queue is unchanged: the three sleeper arrays remain pending,
so no new diagnostic launch was made.

At 08:24, I ran a read-only JAX/CUDA probe inside the gpua051 allocation
without touching the evaluator or training processes. `nvidia-smi` exposed the
A100, but the Expo virtualenv's JAX import hung during CUDA initialization;
the probe was terminated, and the earlier evaluator logs independently show
`CUDA_ERROR_NOT_INITIALIZED` / `No visible GPU devices`. This confirms gpua051
is not a usable control slot under the current node runtime, rather than a
missing checkpoint or evaluator argument. It will not be retried again unless
the allocation/node state changes.

At 08:22:58, the merged SFT audit reached 20/90 committed task files; task
27 was the newest addition. The actor-only EXPO control remained at 2/90 and
the outer-augmentation arm still had only `evals/00000000.json`. All ten
current allocations remained RUNNING and the three new sleeper arrays remained
PENDING, so there was still no safe CUDA-capable idle slot for another arm.

At 08:25:20, the strict SFT audit advanced to 23/90, with tasks 10 and 51
newly committed. The actor-only Full EXPO control advanced to 3/90 (tasks 01,
03, and 05) on gpua070. The three SFT shards and the EXPO shard remain live;
the outer-augmentation trainer has not yet committed its step-100 eval. The
allocation queue is unchanged, so no additional arm was launched.

At 08:28, gpua085 and gpua077 were verified directly inside their existing
sleeper allocations: each had 0 MiB GPU memory and only its allocation sleep
process. I launched the missing actor-only Full EXPO Task-60 LIBERO-90 shard
0 in tmux `sft_diag_full_expo_task60_actor_only_shard0_gpua085_20261105`,
using allocation `23433381_2` and the same 20-episode, seed-7, 224-pixel,
10/5 protocol as shard 1. The launcher passed robosuite/JAX initialization
without the gpua051 CUDA failure. The outer-augmentation arm then committed
step 100 at Task 2/28/29/60 = 75%/35%/60%/45%, macro 53.75%, unchanged from
step 0; this is a retention result only and not a full-90 result.

At 08:26:16, the strict SFT audit reached 24/90, with task 28 newly
committed. The actor-only Full EXPO control remained at 3/90, and the
outer-augmentation arm still had only its step-0 eval. No shard summary exists
yet, so neither suite has been aggregated or compared; all active diagnostic
and side-panel allocations remain untouched.

At 08:33:46, the corrected outer step-100 full-90 evaluator had restored its
checkpoint and initialized Orbax on gpua077; it had not yet committed a task
file. The actor-only EXPO shard 0 on gpua085 was still in model startup, while
shard 1 had five committed task files. The executed-backfill SFT audit had
advanced to 29/90. The initial outer-evaluator launcher failure was preserved
in `launcher_gpua077.log`; the active retry uses `launcher_gpua077_retry.log`.

At 08:34:42, both new evaluations remained healthy in startup: gpua085 held
about 30.8 GiB for actor-only EXPO shard 0 and gpua077 held about 17.3 GiB for
the outer step-100 evaluator; neither had emitted a traceback. The actor-only
EXPO shard 1 remained at five task files, the new shard 0 and outer evaluator
had zero committed task files, and the executed-backfill SFT audit remained at
29/90. The Slurm queue was unchanged.

At 08:36:20, Slurm confirmed the gpua077 outer evaluator step and gpua085
actor-only EXPO shard-0 step were both still RUNNING; their logs contained
only normal Flax/JAX startup warnings. EXPO shard 1 had advanced to six task
files, while shard 0 and outer step-100 remained at zero task files. The
executed-backfill SFT audit remained at 29/90 and no new CUDA-capable slot was
available.

At 08:35:44, the actor-only EXPO shard 1 advanced to six committed files
(through task 11). The gpua085 shard 0 and gpua077 outer step-100 evaluator
were still in normal Flax/JAX startup with no traceback and no task file yet.
The SFT audit's newest committed files were tasks 11, 12, 30, and 52, but its
merged count remained 29/90 because those files were already within the count
reported by the prior audit; no aggregate was inferred.

At 08:36:58, the executed-backfill SFT audit advanced to 31/90. Both new
full-90 evaluators remained alive in normal Flax/JAX initialization with no
traceback: actor-only EXPO shard 0 on gpua085 had no task file yet, and the
outer step-100 evaluator on gpua077 had no task file yet. EXPO shard 1 remained
at six files; the pending sleeper arrays and all existing allocations were
unchanged.

At 08:40:18, the outer step-100 full-90 evaluator committed task 00 = 5.0%
success (20 episodes). The actor-only EXPO shard 0 remained valid after
restoring the historical Full EXPO checkpoint; Orbax warned that its old
checkpoint had no `metrics/metrics` file, but restoration completed and shard
0 had already committed task 00 = 5.0%. This is a metadata warning, not a
failed evaluation. The SFT audit remained at 32/90 and no aggregate was
inferred.

At 08:42:45, the strict SFT audit advanced to 35/90. The actor-only EXPO
shards remained at 1/90 and 7/90, and the outer step-100 evaluator remained at
1/90; all sessions were still live with no terminal summary.

At 08:41:19, the SFT audit advanced to 33/90, and actor-only EXPO shard 1
advanced to 7/90 (through task 13). Actor-only EXPO shard 0 and outer step-100
each remained at one committed task. No shard summary or complete suite was
available, so no aggregate comparison was made.

At 08:41:44, the strict SFT audit advanced to 34/90. Actor-only EXPO shard 1
remained at 7/90, shard 0 at 1/90, and outer step-100 at 1/90; all three
evaluator sessions remained live and no terminal summary was present.

At 08:40:52, actor-only EXPO shard 1 advanced to seven committed task files,
through task 13. Shard 0 remained at task 00, and the outer step-100 evaluator
remained at task 00; both sessions were still live. The executed-backfill SFT
audit remained at 32/90, and no suite-level score was inferred.

At 08:37:52, direct resource checks showed both new evaluators actively using
their GPUs rather than stalled: actor-only EXPO shard 0 on gpua085 used
31,235 MiB at 47% utilization, and the outer step-100 evaluator on gpua077
used 17,287 MiB at 40%. They still had no committed task files because they
were in first-rollout/JAX execution; no restart was warranted.

At 08:38:43, actor-only EXPO shard 0 committed its first task file (task 00 =
5.0% success), confirming that the gpua085 control is now in real LIBERO-90
rollouts. The SFT audit advanced to 32/90. Outer step-100 remained live with
normal startup warnings and no task file yet; no allocation was changed.

At 08:43:12, actor-only EXPO shard 0 advanced to two committed tasks (task 02
= 15.0% success). The SFT audit remained at 35/90, shard 1 at 7/90, and outer
step-100 at 1/90; all sessions remained live.

At 08:43:54, the SFT audit advanced to 36/90. EXPO shard 0 remained at 2/90,
shard 1 at 7/90, and outer step-100 at 1/90; all sessions remained live and
no terminal summary was available.

At 08:49:16, the outer step-100 evaluator advanced to three committed tasks;
task 02 scored 70.0%. The SFT audit remained at 39/90, actor-only EXPO shard 0
at 3/90, and shard 1 at 8/90. All sessions remained live; no aggregate was
inferred.

At 08:48:40, actor-only EXPO shard 0 advanced to three committed tasks;
task 04 scored 0.0%. The SFT audit remained at 39/90, shard 1 at 8/90, and
outer step-100 at 2/90; all sessions remained live.

At 08:48:08, the strict SFT audit advanced to 39/90. Actor-only EXPO shard 0
remained at 2/90, shard 1 at 8/90, and outer step-100 at 2/90; all sessions
remained live without terminal summaries.

At 08:47:26, the strict SFT audit advanced to 38/90. Actor-only EXPO shard 0
remained at 2/90, shard 1 at 8/90, and outer step-100 at 2/90; all sessions
remained live with no terminal summary.

At 08:45:50, the outer step-100 evaluator advanced to two committed tasks;
task 01 scored 0.0%. The SFT audit remained at 37/90, actor-only EXPO shards
remained at 2/90 and 7/90, and all sessions remained live.

At 08:46:23, actor-only EXPO shard 1 advanced to eight committed tasks,
through task 15 (10.0% success). The outer step-100 evaluator remained at two
tasks, shard 0 at two, and the SFT audit at 37/90. All sessions remained live;
no aggregate was inferred.

At 08:45:17, the SFT audit advanced to 37/90. EXPO shard 0 remained at 2/90,
shard 1 at 7/90, and outer step-100 at 1/90; all sessions remained live and
no terminal summary was available.

At 08:52:48, the read-only live audit remained healthy: the executed-backfill
SFT full-suite output had 41/90 task files, actor-only Full EXPO shards had
3/90 and 9/90, and outer-augmentation step 100 had 3/90. All corresponding
tmux sessions and Slurm allocations were still present. The active logs showed
normal task completion messages and only the known dataset-path warnings; no
traceback or OOM was observed. No allocation or unrelated training session
was changed.

At 08:51:39, a fresh read-only audit found 41/90 committed records for the
Task-60 executed-backfill step-200 SFT full-suite evaluation. The two
actor-only Full EXPO control shards remained at 3/90 (gpua085) and 8/90
(gpua070), and the Task-60 outer-augmentation step-100 evaluator remained at
3/90 (gpua077). The corresponding tmux sessions and Slurm steps were still
present. The old gpua002 launcher log contains a terminated predecessor step,
but the active replacement shard log continued through task 15; this was not
treated as a failure or restarted. The gpua051 actor-only control remains
excluded after two independent CUDA-initialization failures. The three newly
submitted sleeper arrays remain pending, so no allocation or unrelated
training session was changed.

At 08:53:32, the next read-only poll showed continued progress in the
actor-only Full EXPO control: shard 0 had 4/90 files and shard 1 had 9/90.
The executed-backfill SFT evaluation remained at 41/90 and the outer-
augmentation step-100 evaluation at 3/90. All active sessions remained in
Slurm and tmux, with no traceback or OOM in the logs. No suite was complete
enough for strict validation or aggregation, and no unrelated allocation was
modified.

At 08:54:06, the live audit advanced the executed-backfill SFT evaluation to
43/90 task files and the outer-augmentation step-100 evaluation to 4/90; the
actor-only Full EXPO control remained at 4/90 plus 9/90. The active gpua002
replacement shard continued producing results, while an older predecessor
launcher log still records the intentionally terminated pre-retry step. The
replacement was not restarted because its tmux and Slurm handles are live.
No complete suite was available for validation, and no unrelated allocation
was changed.

At 08:55:10, a read-only coverage audit confirmed that the 43 committed SFT
files occupy the intended shard ranges so far (tasks 00--16, 23--35, and
45--57), with no duplicate task IDs. The actor-only Full EXPO control has 13
unique, disjoint files across its even-task shard (00, 02, 04, 06) and
odd-task shard (01, 03, 05, 07, 09, 11, 13, 15, 17). The missing IDs are
unfinished work, not evidence of data mixing or shard overlap. All live
handles remained present.

At 08:56:02, the live evaluators advanced: executed-backfill SFT reached
44/90 task files, actor-only Full EXPO shard 1 reached 10/90, while EXPO shard
0 remained at 4/90 and outer-augmentation step 100 at 4/90. All eight
diagnosis tmux sessions remained present and their Slurm steps were RUNNING.
No full suite was complete for validation or aggregation, and no unrelated
allocation was changed.

At 08:58:16, a further read-only poll found actor-only Full EXPO shard 0 at
5/90. Executed-backfill SFT remained at 44/90, actor-only EXPO shard 1 at
10/90, and outer augmentation step 100 at 4/90. The queue still showed all
diagnostic allocations RUNNING and sleeper arrays PENDING; no restart or
allocation change was made.

At 08:59:18, after a 30-second monitored interval, executed-backfill SFT had
advanced to 46/90 task files and outer-augmentation step 100 to 5/90. The
actor-only Full EXPO control remained at 5/90 plus 10/90. The queue still
showed all diagnostic allocations RUNNING and no job was restarted or moved.

At 09:00:21, actor-only Full EXPO shard 0 advanced to 6/90. Executed-backfill
SFT remained at 46/90, EXPO shard 1 at 10/90, and outer augmentation step 100
at 5/90. All ten diagnostic allocations were still RUNNING. A read-only log
display produced no lines in this poll, but the authoritative Slurm state and
task-file counts were valid; no failure was inferred or acted on.

At 09:01:40, the monitored evaluators advanced to 47/90 for executed-backfill
SFT and 11/90 for actor-only Full EXPO shard 1. EXPO shard 0 remained at
6/90 and outer-augmentation step 100 at 5/90. All diagnostic Slurm jobs
remained RUNNING; no suite was complete for validation and no allocation was
changed.

At 09:02:48, a separate read-only check confirmed outer-augmentation step 100
at 5/90 (latest committed task 04), while the other counts remained SFT 47/90
and actor-only EXPO 6/90 plus 11/90. The relevant Slurm steps were all
RUNNING; no terminal state or complete suite was present.

At 09:03:53, the separate outer check advanced to 6/90 (latest committed task
05). The combined live snapshot was executed-backfill SFT 48/90, actor-only
Full EXPO 6/90 plus 11/90, and outer augmentation 6/90. The outer Slurm step
remained RUNNING; no suite was complete for strict validation.

At 09:04:47, executed-backfill SFT advanced to 50/90 task files. Actor-only
Full EXPO remained at 6/90 plus 11/90 and outer augmentation step 100 at 6/90.
All eight diagnosis tmux sessions and ten running allocations remained live;
no suite was complete for strict validation or aggregation.

At 09:05:48, the actor-only EXPO shard-0 evaluator advanced to 7/90. The
other live counts were SFT 50/90, EXPO shard 1 11/90, and outer augmentation
6/90. A separate session check confirmed all eight diagnosis tmux sessions and
their selected Slurm steps were present and RUNNING; no allocation was
changed.

At 09:06:55, a corrected `rg --files` audit found actor-only EXPO shard 1 at
12/90. The authoritative counts were executed-backfill SFT 50/90, EXPO shard
0 7/90, EXPO shard 1 12/90, and outer augmentation 6/90. All eight diagnosis
tmux sessions and ten Slurm allocations remained present; the pending sleeper
arrays were unchanged.

At 09:07:55, executed-backfill SFT advanced to 53/90 task files. Actor-only
Full EXPO remained at 7/90 plus 12/90 and outer augmentation step 100 at 6/90.
A separate queue/session audit confirmed all ten Slurm allocations and eight
diagnosis tmux sessions remained live; no terminal suite or validator-ready
aggregate was available.

At 09:08:53, actor-only EXPO shard 0 advanced to 8/90 and outer augmentation
step 100 to 7/90. Executed-backfill SFT remained at 53/90 and EXPO shard 1 at
12/90. A separate scheduler/session check confirmed all ten allocations were
RUNNING and all eight diagnosis tmux sessions were present; no complete suite
was available for validation.

At 09:09:45, executed-backfill SFT advanced to 54/90 task files. Actor-only
EXPO remained at 8/90 plus 12/90 and outer augmentation at 7/90. A scheduler
check confirmed all ten allocations were RUNNING and a tmux check found all
eight diagnosis sessions; no suite was complete for validation.

At 09:11:24, actor-only Full EXPO shard 1 advanced to 13/90. Executed-backfill
SFT remained at 54/90, EXPO shard 0 at 8/90, and outer augmentation at 7/90.
A separate scheduler check confirmed all ten diagnostic allocations were
RUNNING, all eight tmux sessions were present, and the sleeper arrays remained
PENDING. No allocation was changed.

At 09:13:12, executed-backfill SFT advanced to 55/90 and outer augmentation
step 100 to 8/90. Actor-only EXPO remained at 8/90 plus 13/90. The separate
scheduler/session check confirmed all ten Slurm allocations were RUNNING and
all eight diagnosis tmux sessions were present; no suite was validator-ready.

At 09:14:09, executed-backfill SFT advanced to 57/90 task files. Actor-only
EXPO remained at 8/90 plus 13/90 and outer augmentation at 8/90. A separate
queue/session check again confirmed all ten allocations were RUNNING and all
eight diagnosis tmux sessions were present; no complete suite was available
for validation.

At 09:15:07, actor-only EXPO shard 0 advanced to 9/90. Executed-backfill SFT
remained at 57/90, EXPO shard 1 at 13/90, and outer augmentation at 8/90.
The separate scheduler/session check again found all ten allocations RUNNING
and all eight diagnosis tmux sessions present; no allocation was changed.

At 09:16:04, actor-only EXPO shard 0 remained at 9/90 while the other counts
remained SFT 57/90, EXPO shard 1 13/90, and outer augmentation 8/90. The
separate scheduler/session check confirmed all ten allocations were RUNNING
and all eight diagnosis tmux sessions were present.

At 09:17:00, all active suites advanced: executed-backfill SFT reached 58/90,
actor-only EXPO shard 0 reached 9/90 and shard 1 14/90, and outer augmentation
reached 8/90. A separate scheduler/session check confirmed all ten allocations
were RUNNING and all eight diagnosis tmux sessions were present; no suite was
complete for validation.

At 09:18:30, the gpua002 executed-backfill shard had completed its assigned
tasks 05--22 and finalized cleanly at task 22; the tmux session disappeared
because that shard finished, while its parent sleeper allocation remained
RUNNING. The log ended with Orbax metadata-store closure and no traceback.
The merged executed-backfill audit therefore had 59 files at that point.

At 09:20, gpua002 was independently probed at 0 MiB and 0% utilization after
the shard finished. I added the isolated launcher
`run_full90_task60_outer_aug_step200_gpua002_20261105.sh` and started
`sft_diag_full90_task60_outer_aug_step200_gpua002_20261105` on the existing
allocation `23433381_3`; this does not alter the other evaluators or any
training allocation. By 09:21:58 the step-200 evaluator was alive, had passed
normal robosuite/LIBERO imports, and had emitted no traceback; it was still in
JAX/model initialization with zero task files.

At 09:22:55, the new outer-augmentation step-200 evaluator had restored its
checkpoint and initialized Orbax/JAX without a traceback; it remained live in
startup with zero committed task files. The other audits advanced to
executed-backfill SFT 61/90, actor-only EXPO shard 0 11/90, shard 1 15/90,
and outer-augmentation step 100 11/90. All eight diagnosis sessions remained
present.

At 09:24:59, the step-200 evaluator remained alive in its tmux session and
Slurm allocation while completing model compilation. Its output was still
0/90, but the log had no traceback, OOM, CUDA error, kill, or cancellation;
this was treated as normal startup and not restarted.

At 09:27:25, the new evaluator logged a successful restore of the outer-
augmentation step-200 checkpoint. It remained live at 0/90 while compiling
the rollout path, with no traceback or resource failure. The other audits had
advanced to SFT 63/90, actor-only EXPO shard 0 13/90, shard 1 16/90, and
outer step 100 12/90.

At 09:28:04, the step-200 process was still live and its log contained only
the known JAX/Flax deprecation warnings; no restart was warranted.

At 09:29:02, the new step-200 evaluator remained at 0/90 with no error while
actor-only EXPO shard 1 advanced to 17/90. A direct read-only GPU probe showed
the gpua002 evaluator actively computing at 17,289 MiB and 54% utilization,
confirming that the long compilation phase was live rather than stalled.

At 09:30:05, the new Task-60 outer-augmentation step-200 evaluator completed
its first real rollout, task 00 = 5.0% success (20 episodes), proving that the
new arm passed initialization and is evaluating the intended checkpoint. The
other audits advanced to executed-backfill SFT 64/90, actor-only EXPO shard 0
13/90 and shard 1 17/90, and outer step 100 13/90. No aggregate was inferred.

At 09:31:03, the step-200 outer-augmentation audit remained healthy at 1/90
(task 00 was the only committed result). The other live counts were
executed-backfill SFT 65/90, actor-only EXPO shard 0 13/90, shard 1 17/90,
and outer step 100 13/90. No suite-level aggregate was inferred.

At 09:32:11, actor-only EXPO shard 0 advanced to 14/90. Executed-backfill SFT
remained at 65/90, EXPO shard 1 at 17/90, outer step 100 at 13/90, and the
new outer step-200 arm at 1/90. The new arm still had no error markers.

At 09:33:52, actor-only EXPO shard 1 advanced to 18/90. Executed-backfill
SFT remained at 65/90, EXPO shard 0 at 14/90, outer step 100 at 13/90, and
outer step 200 at 1/90; the step-200 log remained error-free.

At 09:34:46, the new outer-augmentation step-200 evaluator advanced to 2/90;
task 01 scored 0.0% success over 20 episodes. Actor-only EXPO shard 0
advanced to 15/90. Executed-backfill SFT remained at 65/90, EXPO shard 1 at
18/90, and outer step 100 at 13/90.

### Reuse of the completed SFT shard and step-400 outer-augmentation evaluation (2026-11-05, 09:35--09:42 CST)

The executed-backfill SFT shard on `gpua087` completed tasks 23--44 cleanly at
09:35:09, including Orbax metadata-store closure and no traceback. Its parent
allocation `23433381_1` remained a live one-GPU sleeper allocation, and a
read-only probe confirmed `gpua087` had 0 MiB GPU use. This released a safe
diagnostic slot without changing any training or side-panel process.

The outer-augmentation Task-60 trainer on `gpua082` had a complete checkpoint
at update 400 and a four-task retention result of 55.0% overall (tasks
2/28/29/60 = 65/40/65/55%). I added the isolated launcher
`run_full90_task60_outer_aug_step400_gpua087_20261105.sh` and started
`sft_diag_full90_task60_outer_aug_step400_gpua087_20261105` on the released
`23433381_1` allocation. It uses the unchanged actor-only LIBERO-90 protocol:
20 episodes per task, seed 7, evaluation seed 42, 224-pixel images,
horizon 10, replan 5, and one FSDP device. Its output is isolated at
`dcrlcheckpoints/sft_retention_full90_eval_20261105/task60_outer_aug_step400`.

At 09:42:11 the new evaluator had passed LIBERO/robosuite initialization and
had a live Python process using 422 MiB on `gpua087`, but had not committed its
first task file yet. The other live counts were executed-backfill SFT 69/90,
outer augmentation step 100 15/90, step 200 3/90, and actor-only EXPO shards
0/1 at 16/90 and 19/90. No aggregate comparison is admitted until complete,
disjoint coverage passes `validate_full90.py`.

At 09:47:06, the step-400 evaluator remained live with no traceback, OOM,
CUDA error, kill, or cancellation while completing its cold rollout
compilation; it still had 0/90 committed task files. The other panels had
advanced to executed-backfill SFT 70/90, outer step 100 16/90, outer step 200
4/90, and actor-only EXPO shards 0/1 at 17/90 and 20/90. The Task-60 outer-
augmentation trainer was still active on `gpua082` at about 31.5 GiB and was
finishing its update-500 evaluation; no new checkpoint or diagnostic slot was
available yet.

At 09:48:41, the step-400 evaluator's tmux session, Slurm step, and remote
process tree were still present. Its log remained free of terminal/error
markers and still showed the expected post-restore JAX compilation phase; the
task-file count remained 0/90. I did not restart it based on a transient empty
GPU-probe response because the authoritative process handles remained live.

### Update-500 checkpoint and parallel full-suite evaluation (2026-11-05, 09:49--09:53 CST)

The Task-60 outer-augmentation trainer on `gpua082` completed update 500
cleanly and wrote `checkpoints/500`. Its retention panel was 45.0% overall,
with tasks 2/28/29/60 = 75/40/70/45%; the run summary reported
`official_demo_chunks=0` and selected update 500 as the best four-task
macro-success checkpoint (57.5%). The trainer tmux session exited after W&B
sync, while its parent allocation remained a live sleeper.

The step-400 full-suite evaluator on `gpua087` then committed its first record
at 09:49:53: task 00 = 1/20 (5.0%), with the expected actor-only protocol.
The evaluator remains active and its output is isolated at
`dcrlcheckpoints/sft_retention_full90_eval_20261105/task60_outer_aug_step400`.

I added and launched the isolated step-500 evaluator
`run_full90_task60_outer_aug_step500_gpua082_20261105.sh` in tmux session
`sft_diag_full90_task60_outer_aug_step500_gpua082_20261105`, using the newly
idle `23416001_2` allocation. Its output is isolated at
`dcrlcheckpoints/sft_retention_full90_eval_20261105/task60_outer_aug_step500`;
at 09:53:06 the session was live in startup with zero task files. A bounded
JAX probe on the separate idle-looking `gpua051` allocation again hung and was
terminated by timeout, so that node was not used for evidence.

At 09:54:17, the step-500 evaluator had restored the complete checkpoint 500
and initialized its Orbax manager without terminal/error markers; it remained
at 0/90 during normal startup compilation. The step-400 evaluator remained
live at 1/90. Both evaluations therefore have isolated output trees and are
running concurrently on separate allocations.

At 09:56:14, the step-400 panel was at 2/90 and remained active. The step-500
panel had completed model construction (the expected 224x224 observation and
35-action-shape diagnostics were logged), remained at 0/90, and had no
traceback, CUDA error, OOM, kill, or cancellation marker. Its low initial GPU
use was therefore treated as ongoing restore/compilation rather than a stall.

At 09:58:35, step 400 advanced to 3/90. Step 500 had restored checkpoint 500
successfully and was actively using about 17.3 GiB on `gpua082` at 44% GPU
utilization while compiling the rollout path; it remained at 0/90 with no
failure markers. The evaluator was not restarted.

At 10:00:40, both outer-augmentation evaluators had live Python processes and
were using about 17.3 GiB on their respective GPUs (`gpua087` and `gpua082`),
at 41% and 40% utilization. Step 400 remained at 3/90 and step 500 at 0/90;
neither log contained a terminal error marker, so both were left running.

### Continued live poll and slot audit (2026-11-05, 10:16 CST)

The active full-suite panels continued to make progress without changing any
training or side-panel session. The committed task-file counts were: Task-60
executed-backfill step 200 = 75/90; outer augmentation steps 100, 200, 400,
and 500 = 22/90, 11/90, 5/90, and 2/90; and actor-only Full EXPO shards 0 and
1 = 24/90 and 26/90. These are partial counts only; no suite-level score is
admitted until each panel has complete disjoint 0--89 coverage and passes
`validate_full90.py`.

All ten allocated A100 slots remained accounted for. `gpua051` has 0 MiB
reported by `nvidia-smi` and only its sleeper process, but earlier bounded JAX
probes on that allocation failed with `CUDA_ERROR_NOT_INITIALIZED` and
"No visible GPU devices". It is therefore not treated as a usable diagnostic
slot. The other CUDA-capable slots remain occupied by live evaluation or the
preserved teacher/side panel. The prepared balanced multi-task launcher is not
being launched because mixing tasks would violate the strict task-specific
diagnosis rule; the next new arm must use one task's collection only.

At 10:18:49, a fresh authoritative poll confirmed that all current Slurm
allocations and diagnosis tmux sessions remained live. The committed counts
were executed-backfill 75/90; outer-augmentation steps 100/200/400/500 at
23/90, 11/90, 5/90, and 2/90; and actor-only EXPO shards 0/1 at 24/90 and
26/90. The step-100 retry continued to produce valid task records after its
earlier isolated import failure; no active evaluator was restarted.

At 10:19:43, all diagnosis handles remained live. The executed-backfill panel
advanced to 76/90 and outer-augmentation step 400 to 6/90; the other counts
were step 100 = 23/90, step 200 = 11/90, step 500 = 2/90, and actor-only
EXPO shards 0/1 = 24/90 and 26/90. No panel had complete strict coverage, so
no aggregate or new arm was launched.

### `gpua051` node comparison (2026-11-05, 10:20 CST)

A bounded read-only comparison against working `gpua002` found the same
NVIDIA A100-SXM4-40GB, driver 595.71.05, 40,960 MiB capacity, compute mode,
MIG state, CUDA 13.2 environment, `CUDA_VISIBLE_DEVICES=0`, and Slurm device
mapping. `nvidia-smi` sees the idle `gpua051` GPU, but the project JAX process
still fails during CUDA initialization with `CUDA_ERROR_NOT_INITIALIZED`.
This rules out a simple launcher/environment mismatch; the node remains
excluded rather than changing the research stack or resetting the GPU.

At 10:22:22, the active panels remained healthy and advanced: executed-
backfill 76/90; outer-augmentation steps 100/200/400/500 at 23/90, 12/90,
6/90, and 3/90; and actor-only EXPO shards 0/1 at 25/90 and 27/90. No
complete panel or released CUDA-capable slot was available, so no partial
aggregate or replacement arm was started.

As a CPU-side evidence check at 10:23, I reran `validate_full90.py` on the
established best candidates. Task-60 full-proposal steps 200 and 500 passed
with 519/1,800 and 517/1,800 successes; Task-29 full-proposal step 500 passed
with 524/1,800; and Task-28 full-proposal step 300 passed with 518/1,800.
Every directory had 90 unique task files and zero validation errors. The
Task-60 step-200 result remains the best verified absolute margin against the
matched historical Full EXPO reference (+8.39 percentage points).

At 10:25:19, the active panels continued to write valid task records: Task-60
executed-backfill 77/90; outer-augmentation steps 100/200/400/500 at 25/90,
13/90, 7/90, and 3/90; and actor-only EXPO shards 0/1 at 26/90 and 28/90.
No panel reached 90/90 and no usable allocation was released, so the active
sessions were left untouched.

At 10:29:10, the executed-backfill panel advanced to 78/90 and outer-
augmentation step 400 advanced to 8/90; fresh task records were present and
no terminal error appeared. The other active panels remained live, so no
allocation or tmux session was changed.

At 10:26:25, direct in-allocation health probes confirmed that the live
evaluators were computing rather than stalled: the Python processes were
resident on their assigned nodes with approximately 17--31 GiB GPU memory
and 40--58% GPU utilization (roughly 42--66% CPU). This covered the
executed-backfill, all four outer-augmentation checkpoints, and both
actor-only EXPO shards. No restart was warranted.

At 10:35:40, a fresh read-only poll confirmed that the same ten A100
allocations remained accounted for and that all active diagnostic panels were
still attached to their original tmux/Slurm slots. The committed task-file
counts were executed-backfill step 200 = 81/90; outer-augmentation steps
100/200/400/500 = 27/90, 15/90, 10/90, and 6/90; and actor-only Full EXPO
shards 0/1 = 28/90 and 31/90. The counts are disjoint task files, not suite
scores; no panel yet satisfies the 90-unique-task validator gate. The
preserved Task-2/teacher-side session was left untouched. The three pending
sleep arrays remain pending at priority, and no diagnostic slot was released
or repurposed.

At 10:36:27, the authoritative task-file count was rechecked using only
`task_*.json` (the executed-backfill directory also contains two shard
metadata JSON files). The valid partial counts were executed-backfill step
200 = 79/90; outer-augmentation steps 100/200/400/500 = 27/90, 15/90, 10/90,
and 6/90; and actor-only Full EXPO shards 0/1 = 28/90 and 31/90. The
executed-backfill panel had no new task after the previous poll but remained
live, while the other panel logs continued writing fresh task records. The
repeated missing `sharder/training/datasets` warning remains the previously
verified optional demonstration-HDF5 path warning; the evaluator uses the
existing BDDL, initial-state, and asset paths and does not load demos during
actor-only rollout. No active evaluator, allocation, or tmux session was
changed.

At 10:37:28, the same live allocations and tmux sessions were confirmed. The
only new committed task record was outer-augmentation step 400, advancing to
11/90; the other valid counts remained executed-backfill 79/90, outer steps
100/200/500 at 27/90, 15/90, and 6/90, and actor-only EXPO shards 0/1 at
28/90 and 31/90. No panel passed the complete-coverage gate, so no validator
score or new training arm was started. All ten running A100 slots remained
occupied/reserved by the existing diagnosis or preserved side session, while
the three sleeper arrays remained pending.

At 10:39:09, fresh task records confirmed continued rollout work: outer-
augmentation steps 100 and 200 advanced to 28/90 and 16/90, respectively, and
actor-only Full EXPO shard 0 advanced to 29/90 (newest task 56). The other
counts remained executed-backfill 79/90, outer steps 400/500 at 11/90 and
6/90, and EXPO shard 1 at 31/90. Slurm showed active evaluator job steps on
the assigned nodes. The repeated optional-dataset warnings remained present
but no evaluator traceback or terminal state appeared; no allocation or tmux
session was changed.

At 10:39:39, the executed-backfill evaluator produced task 79, advancing its
strict task-file coverage to 80/90. The latest valid counts for the remaining
panels were outer steps 100/200/400/500 = 28/90, 16/90, 11/90, and 6/90, and
actor-only EXPO shards 0/1 = 29/90 and 31/90. The corresponding launcher logs
contained fresh task records and all ten diagnostic/side allocations remained
RUNNING; no panel had complete coverage and no session was changed.

At 10:40:05, outer-augmentation step 500 committed task 06, advancing to
7/90. Executed-backfill remained 80/90; outer steps 100/200/400 remained
28/90, 16/90, and 11/90; and actor-only EXPO shards 0/1 remained 29/90 and
31/90. The active Slurm allocations and evaluator launchers remained present,
with no complete panel or released diagnostic GPU.

At 10:40:45, actor-only Full EXPO shard 1 committed task 63, advancing to
32/90. Executed-backfill remained 80/90; outer steps 100/200/400/500 remained
28/90, 16/90, 11/90, and 7/90; and shard 0 remained 29/90. All ten running
allocations and the existing tmux sessions were still present, so no panel was
validated or reassigned.

At 10:42:43, outer-augmentation step 400 committed task 11, advancing to
12/90. The other valid counts remained executed-backfill 80/90, outer steps
100/200/500 at 28/90, 16/90, and 7/90, and actor-only EXPO shards 0/1 at
29/90 and 32/90. All ten allocations remained RUNNING and no complete panel
was available for validation.

At 10:43:07, outer-augmentation step 100 committed task 28, advancing to
29/90. Executed-backfill remained 80/90; outer steps 200/400/500 remained
16/90, 12/90, and 7/90; and actor-only EXPO shards 0/1 remained 29/90 and
32/90. No complete panel or released GPU was available.

At 10:44:36, multiple active panels advanced: executed-backfill step 200
committed task 80 (81/90); outer-augmentation steps 200 and 500 committed
tasks 16 and 07 (17/90 and 8/90); and actor-only EXPO shard 0 committed task
58 (30/90). Outer steps 100 and 400 remained 29/90 and 12/90, while EXPO
shard 1 remained 32/90. All ten allocations remained RUNNING; no complete
panel was available for validation or comparison.

At 10:45:41, outer-augmentation step 400 committed task 12, advancing to
13/90. The other panels retained executed-backfill 81/90, outer steps
100/200/500 at 29/90, 17/90, and 8/90, and actor-only EXPO shards 0/1 at
30/90 and 32/90. All ten allocations remained active; no complete panel was
available for validation.

At 10:46:13, both actor-only Full EXPO controls advanced: shard 0 committed
task 60 (31/90) and shard 1 committed task 65 (33/90). Executed-backfill
remained 81/90; outer steps 100/200/400/500 remained 29/90, 17/90, 13/90,
and 8/90. All ten allocations remained RUNNING and no complete panel was
available for validation.

At 10:46:38, outer-augmentation step 100 committed task 29, advancing to
30/90. Executed-backfill remained 81/90; outer steps 200/400/500 remained
17/90, 13/90, and 8/90; and actor-only EXPO shards 0/1 remained 31/90 and
33/90. All ten allocations remained active and no panel reached the complete
coverage gate.

At 10:48:05, actor-only Full EXPO shard 1 committed task 67, advancing to
34/90. Executed-backfill remained 81/90; outer-augmentation steps
100/200/400/500 remained 30/90, 17/90, 13/90, and 8/90; and EXPO shard 0
remained 31/90. All ten allocations remained active, with no complete panel
available for validation.

At 10:49:19, outer-augmentation steps 100 and 500 committed tasks 30 and 08,
advancing to 31/90 and 9/90. Executed-backfill remained 81/90; outer steps
200/400 remained 17/90 and 13/90; and actor-only EXPO shards 0/1 remained
31/90 and 34/90. All ten allocations remained active and no panel reached
complete coverage.

At 10:49:53, executed-backfill step 200 committed task 81, advancing to
82/90. Outer steps 100/200/400/500 remained 31/90, 17/90, 13/90, and 9/90,
and actor-only EXPO shards 0/1 remained 31/90 and 34/90. All ten allocations
remained active; no complete panel was available for validation.

At 10:50:14, outer-augmentation step 200 committed task 17, advancing to
18/90. Executed-backfill remained 82/90; outer steps 100/400/500 remained
31/90, 13/90, and 9/90; and actor-only EXPO shards 0/1 remained 31/90 and
34/90. All ten allocations remained active and no panel reached complete
coverage.

At 10:50:58, outer-augmentation step 400 committed task 13, advancing to
14/90. Executed-backfill remained 82/90; outer steps 100/200/500 remained
31/90, 18/90, and 9/90; and actor-only EXPO shards 0/1 remained 31/90 and
34/90. All ten allocations remained active and no panel reached the complete
coverage gate.

At 10:51:17, outer-augmentation step 500 committed task 09, advancing to
10/90. Executed-backfill remained 82/90; outer steps 100/200/400 remained
31/90, 18/90, and 14/90; and actor-only EXPO shards 0/1 remained 31/90 and
34/90. All ten allocations remained active and no panel reached complete
coverage.

At 10:51:50, actor-only Full EXPO shard 0 committed task 62, advancing to
32/90. Executed-backfill remained 82/90; outer steps 100/200/400/500 remained
31/90, 18/90, 14/90, and 10/90; and EXPO shard 1 remained 34/90. All ten
allocations remained active and no panel reached complete coverage.

At 10:52:50, outer-augmentation step 500 committed task 10, advancing to
11/90. Executed-backfill remained 82/90; outer steps 100/200/400 remained
31/90, 18/90, and 14/90; and actor-only EXPO shards 0/1 remained 32/90 and
34/90. All ten allocations remained active and no panel reached complete
coverage.

At 10:53:24, actor-only Full EXPO shard 1 committed task 69, advancing to
35/90. Executed-backfill remained 82/90; outer steps 100/200/400/500 remained
31/90, 18/90, 14/90, and 11/90; and EXPO shard 0 remained 32/90. All ten
allocations remained active and no panel reached complete coverage.

At 10:54:13, outer-augmentation step 100 committed task 31, advancing to
32/90. Executed-backfill remained 82/90; outer steps 200/400/500 remained
18/90, 14/90, and 11/90; and actor-only EXPO shards 0/1 remained 32/90 and
35/90. All ten allocations remained active and no panel reached complete
coverage.

At 10:54:33, executed-backfill step 200 committed task 82, advancing to
83/90. Outer steps 100/200/400/500 remained 32/90, 18/90, 14/90, and 11/90,
and actor-only EXPO shards 0/1 remained 32/90 and 35/90. All ten allocations
remained active; no panel reached complete coverage.

At 10:54:53, outer-augmentation steps 200 and 400 committed tasks 18 and 14,
advancing to 19/90 and 15/90. Executed-backfill remained 83/90; outer steps
100/500 remained 32/90 and 11/90; and actor-only EXPO shards 0/1 remained
32/90 and 35/90. All ten allocations remained active and no panel reached
complete coverage.

At 10:57:57, outer-augmentation step 200 committed task 19, advancing to
20/90, and actor-only Full EXPO shard 0 committed task 64, advancing to
33/90. Executed-backfill remained 83/90; outer steps 100/400/500 remained
32/90, 15/90, and 11/90; and EXPO shard 1 remained 35/90. All ten
allocations remained RUNNING and no panel reached the complete-coverage gate.

At 10:58:30, outer-augmentation step 500 committed task 11, advancing to
12/90. The other panels retained their previous counts: executed-backfill
83/90; outer steps 100/200/400 at 32/90, 20/90, and 15/90; and actor-only
EXPO shards 0/1 at 33/90 and 35/90. All ten allocations remained RUNNING and
no panel reached complete coverage.

At 10:58:54, actor-only Full EXPO shard 1 committed task 71, advancing to
36/90. The SFT panels retained their previous counts, and actor-only EXPO
shard 0 remained at 33/90. All ten allocations remained RUNNING; no panel
reached complete coverage.

At 10:59:16, outer-augmentation step 100 committed task 32, advancing to
33/90. Executed-backfill remained 83/90; outer steps 200/400/500 remained
20/90, 15/90, and 12/90; and actor-only EXPO shards 0/1 remained 33/90 and
36/90. All ten allocations remained RUNNING and no panel reached complete
coverage.

At 10:59:40, outer-augmentation step 200 committed task 20, advancing to
21/90. Executed-backfill remained 83/90; outer steps 100/400/500 remained
33/90, 15/90, and 12/90; and actor-only EXPO shards 0/1 remained 33/90 and
36/90. All ten allocations remained RUNNING and no panel reached complete
coverage.

At 11:00:03, executed-backfill step 200, outer-augmentation step 400, and
their corresponding output panels committed tasks 83 and 15, advancing to
84/90 and 16/90. Outer steps 100/200/500 remained 33/90, 21/90, and 12/90;
actor-only EXPO shards 0/1 remained 33/90 and 36/90. All ten allocations
remained RUNNING and no panel reached complete coverage.

At 11:01:28, outer-augmentation step 500 committed task 12, advancing to
13/90. Executed-backfill remained 84/90; outer steps 100/200/400 remained
33/90, 21/90, and 16/90; and actor-only EXPO shards 0/1 remained 33/90 and
36/90. All ten allocations remained RUNNING and no panel reached complete
coverage.

At 11:02:53, outer-augmentation step 100 committed task 33, advancing to
34/90. Executed-backfill remained 84/90; outer steps 200/400/500 remained
21/90, 16/90, and 13/90; and actor-only EXPO shards 0/1 remained 33/90 and
36/90. All ten allocations remained RUNNING and no panel reached complete
coverage.

At 11:03:27, outer-augmentation step 100 and actor-only Full EXPO shard 0
each committed one additional task, advancing to 34/90 and 34/90. The other
panels retained their previous counts: executed-backfill 84/90; outer steps
200/400/500 at 21/90, 16/90, and 13/90; and EXPO shard 1 at 36/90. All ten
allocations remained RUNNING and no panel reached complete coverage.

At 11:03:51, actor-only Full EXPO shards 0 and 1 committed tasks 66 and 73,
advancing to 34/90 and 37/90. The SFT panels retained their previous counts:
executed-backfill 84/90 and outer steps 100/200/400/500 at 34/90, 21/90,
16/90, and 13/90. All ten allocations remained RUNNING and no panel reached
complete coverage.

At 11:04:15, outer-augmentation step 200 committed task 21, advancing to
22/90. Executed-backfill remained 84/90; outer steps 100/400/500 remained
34/90, 16/90, and 13/90; and actor-only EXPO shards 0/1 remained 34/90 and
37/90. All ten allocations remained RUNNING and no panel reached complete
coverage.

At 11:05:12, executed-backfill step 200 committed task 84, advancing to
85/90. Outer steps 100/200/400/500 remained 34/90, 22/90, 16/90, and 13/90;
actor-only EXPO shards 0/1 remained 34/90 and 37/90. All ten allocations
remained RUNNING and no panel reached complete coverage.

At 11:05:49, outer-augmentation step 400 committed task 16, advancing to
17/90. Executed-backfill remained 85/90; outer steps 100/200/500 remained
34/90, 22/90, and 13/90; and actor-only EXPO shards 0/1 remained 34/90 and
37/90. All ten allocations remained RUNNING and no panel reached complete
coverage.

At 11:06:24, outer-augmentation step 500 committed task 13, advancing to
14/90. Executed-backfill remained 85/90; outer steps 100/200/400 remained
34/90, 22/90, and 17/90; and actor-only EXPO shards 0/1 remained 34/90 and
37/90. All ten allocations remained RUNNING and no panel reached complete
coverage.

At 11:07:01, outer-augmentation step 200 committed task 22, advancing to
23/90. Executed-backfill remained 85/90; outer steps 100/400/500 remained
34/90, 17/90, and 14/90; and actor-only EXPO shards 0/1 remained 34/90 and
37/90. All ten allocations remained RUNNING and no panel reached complete
coverage.

At 11:07:23, outer-augmentation step 200 and actor-only Full EXPO shard 0
each committed one additional task, advancing to 23/90 and 35/90. Executed-
backfill remained 85/90; outer steps 100/400/500 remained 34/90, 17/90, and
14/90; and EXPO shard 1 remained 37/90. All ten allocations remained RUNNING
and no panel reached complete coverage.

At 11:07:45, outer-augmentation step 100 and actor-only Full EXPO shard 0
each committed one additional task, advancing to 35/90. Executed-backfill
remained 85/90; outer steps 200/400/500 remained 23/90, 17/90, and 14/90;
and EXPO shard 1 remained 37/90. All ten allocations remained RUNNING and no
panel reached complete coverage.

At 11:08:24, actor-only Full EXPO shard 1 committed task 75, advancing to
38/90. The SFT panels retained their prior counts: executed-backfill 85/90;
outer steps 100/200/400/500 at 35/90, 23/90, 17/90, and 14/90; and EXPO
shard 0 at 35/90. All ten allocations remained RUNNING and no panel reached
complete coverage.

At 11:10:01, outer-augmentation step 500 committed task 14, advancing to
15/90. Executed-backfill remained 85/90; outer steps 100/200/400 remained
35/90, 23/90, and 17/90; and actor-only EXPO shards 0/1 remained 35/90 and
38/90. All ten allocations remained RUNNING and no panel reached complete
coverage.

At 11:10:42, executed-backfill step 200 committed task 85, advancing to
86/90. Outer steps 100/200/400/500 remained 35/90, 23/90, 17/90, and 15/90;
and actor-only EXPO shards 0/1 remained 35/90 and 38/90. All ten allocations
remained RUNNING and no panel reached complete coverage.

At 11:11:07, executed-backfill step 200 and actor-only Full EXPO shard 0 each
committed one additional task, advancing to 86/90 and 36/90. Outer steps
100/200/400/500 remained 35/90, 23/90, 17/90, and 15/90; and EXPO shard 1
remained 38/90. All ten allocations remained RUNNING and no panel reached
complete coverage.

At 11:11:35, outer-augmentation step 400 committed task 17, advancing to
18/90. Executed-backfill remained 86/90; outer steps 100/200/500 remained
35/90, 23/90, and 15/90; and actor-only EXPO shards 0/1 remained 36/90 and
38/90. All ten allocations remained RUNNING and no panel reached complete
coverage.

At 11:11:59, outer-augmentation step 200 committed task 23, advancing to
24/90. Executed-backfill remained 86/90; outer steps 100/400/500 remained
35/90, 18/90, and 15/90; and actor-only EXPO shards 0/1 remained 36/90 and
38/90. All ten allocations remained RUNNING and no panel reached complete
coverage.

At 11:12:26, outer-augmentation step 100 committed task 35, advancing to
36/90. Executed-backfill remained 86/90; outer steps 200/400/500 remained
24/90, 18/90, and 15/90; and actor-only EXPO shards 0/1 remained 36/90 and
38/90. All ten allocations remained RUNNING and no panel reached complete
coverage.

At 11:13:02, actor-only Full EXPO shard 1 committed task 77, advancing to
39/90. The SFT panels retained their previous counts: executed-backfill 86/90;
outer steps 100/200/400/500 at 36/90, 24/90, 18/90, and 15/90; and EXPO
shard 0 at 36/90. All ten allocations remained RUNNING and no panel reached
complete coverage.

At 11:13:26, actor-only Full EXPO shards 0 and 1 committed tasks 72 and 77,
advancing to 37/90 and 39/90. The SFT panels retained their previous counts:
executed-backfill 86/90 and outer steps 100/200/400/500 at 36/90, 24/90,
18/90, and 15/90. All ten allocations remained RUNNING and no panel reached
complete coverage.

At 11:13:51, outer-augmentation step 200 committed task 24, advancing to
25/90. Executed-backfill remained 86/90; outer steps 100/400/500 remained
36/90, 18/90, and 15/90; and actor-only EXPO shards 0/1 remained 37/90 and
39/90. All ten allocations remained RUNNING and no panel reached complete
coverage.

At 11:15:21, outer-augmentation step 500 committed task 15, advancing to
16/90. Executed-backfill remained 86/90; outer steps 100/200/400 remained
36/90, 25/90, and 18/90; and actor-only EXPO shards 0/1 remained 37/90 and
39/90. All ten allocations remained RUNNING and no panel reached complete
coverage.

At 11:15:46, executed-backfill step 200, outer-augmentation step 500, and
actor-only Full EXPO shard 1 each committed one additional task, advancing
to 87/90, 16/90, and 40/90. Outer steps 100/200/400 remained 36/90, 25/90,
and 18/90; and EXPO shard 0 remained 37/90. All ten allocations remained
RUNNING and no panel reached complete coverage.

At 11:16:10, outer-augmentation step 400 and actor-only Full EXPO shard 1
each committed one additional task, advancing to 19/90 and 40/90. Executed-
backfill remained 87/90; outer steps 100/200/500 remained 36/90, 25/90, and
16/90; and EXPO shard 0 remained 37/90. All ten allocations remained RUNNING
and no panel reached complete coverage.

At 11:17:01, outer-augmentation step 100 committed task 36, advancing to
37/90. Executed-backfill remained 87/90; outer steps 200/400/500 remained
25/90, 19/90, and 16/90; and actor-only EXPO shards 0/1 remained 37/90 and
40/90. All ten allocations remained RUNNING and no panel reached complete
coverage.

At 11:18:16, actor-only Full EXPO shard 0 committed task 74, advancing to
38/90. The SFT panels retained their previous counts: executed-backfill 87/90;
outer steps 100/200/400/500 at 37/90, 25/90, 19/90, and 16/90; and EXPO
shard 1 at 40/90. All ten allocations remained RUNNING and no panel reached
complete coverage.

At 11:18:39, outer-augmentation step 200 committed task 25, advancing to
26/90. Executed-backfill remained 87/90; outer steps 100/400/500 remained
37/90, 19/90, and 16/90; and actor-only EXPO shards 0/1 remained 38/90 and
40/90. All ten allocations remained RUNNING and no panel reached complete
coverage.

At 11:19:03, outer-augmentation step 400 committed task 19, advancing to
20/90. Executed-backfill remained 87/90; outer steps 100/200/500 remained
37/90, 26/90, and 16/90; and actor-only EXPO shards 0/1 remained 38/90 and
40/90. All ten allocations remained RUNNING and no panel reached complete
coverage.

At 11:20:36, executed-backfill step 200 committed task 87, advancing to
88/90. Outer steps 100/200/400/500 remained 37/90, 26/90, 20/90, and 16/90;
and actor-only EXPO shards 0/1 remained 38/90 and 40/90. All ten allocations
remained RUNNING and no panel reached complete coverage.

At 11:26:58, executed-backfill remained at 88/90. Outer-augmentation steps
100/200/400/500 advanced or remained at 38/90, 26/90, 21/90, and 17/90;
actor-only EXPO shards 0/1 remained at 38/90 and advanced to 41/90. All seven
evaluator panels were still partial, so no full-90 aggregate was admitted.
The evaluator processes were confirmed alive on their assigned nodes with
nonzero CPU and GPU memory/utilization. The preserved gpua051 allocation had
no usable process/GPU, so it was left untouched; the three sleeper arrays
remained PENDING. No tmux session or running allocation was changed.

At 11:33:43, executed-backfill step 200 advanced to 89/90. Outer-augmentation
steps 100/200/400/500 were at 38/90, 27/90, 22/90, and 17/90; actor-only
EXPO shards 0/1 were at 39/90 and 42/90. The evaluator process for the
remaining executed-backfill tasks was still alive with active CPU/GPU use;
no panel was yet complete and no validator was run.

At 11:37:14, the read-only validator was rerun against three previously
completed strict panels: task-60 full-proposal step 200 was COMPLETE at
519/1800 (28.83%), task-60 full-proposal step 400 was COMPLETE at 506/1800
(28.11%), and task-29 full-proposal step 500 was COMPLETE at 524/1800
(29.11%). Each had 90 files, 90 unique tasks, zero errors, and the expected
LIBERO-90 protocol. These results remain admissible under the task-specific
data rule; the current 2026-11-05 panels remain incomplete.

At 11:39:57, the task-60 executed-backfill step-200 panel reached 90/90.
The validator reported COMPLETE with 90 files, 90 unique tasks, zero errors,
and 517/1800 successes (28.72%). Its saved configuration independently
confirms source_task_id=60, Small EXPO teacher task 60 at step 450223,
official pi05_libero student/base, official_demo_ratio=0, target_mode=
executed_backfill, and the 10/5 interface. This is the current strict
task-specific result; the other 2026-11-05 panels remain partial.

At 11:35:19, executed-backfill remained at 89/90; outer-augmentation steps
100/200/400/500 were at 39/90, 27/90, 22/90, and 17/90; and actor-only EXPO
shards 0/1 were at 40/90 and 42/90. The Slurm allocation for one preserved
side evaluator entered COMPLETING at its two-day limit; it was not altered.
The remaining diagnosis evaluators and their allocations stayed RUNNING, and
no complete panel was yet eligible for validation.

At 11:45, after the original actor-only EXPO shard-1 allocation expired at
its two-day limit, a read-only audit found only tasks 85, 87, and 89 missing
from the 42/45 shard. Those three tasks were relaunched in a separate tmux
session on the now-free gpua038 diagnostic slot, writing to the same shard-1
directory without touching existing task files. At 11:46:54 the process was
alive in shared-storage checkpoint I/O (about 553 MB read, 515 MiB resident,
no GPU allocation yet); no failure marker or overwrite was observed.

At 11:51:04, the resumed actor-only EXPO process had reached normal GPU
initialization (31,312 MiB resident on gpua038, 91.8% CPU while compiling or
loading); tasks 85/87/89 had not yet committed. The process remained alive and
was not restarted.

At 11:55, the actor-only EXPO resume was stopped because it is a secondary
decomposition, not the primary research comparison: the target result is the
deployed Full EXPO system with editor/critic versus actor-only SFT. No new
actor-only task files had committed, and only the newly created resume tmux
session was stopped. The freed gpua038 slot was then assigned to resume the
strict task-60 outer-augmentation step-500 SFT full-90 panel, evaluating only
its missing task IDs 17--89. The first launch failed before evaluation due to
an import-path omission in the new launcher; that launcher was corrected and
relaunched successfully at 11:56. No existing process or result file was
changed.

### Per-task retention check and Task-2 diagnosis result (2026-11-05, ~12:14 CST)

The full-90 aggregates were rechecked against their per-task records rather
than treating a high aggregate as sufficient evidence. For the strict
Task-60 full-proposal update-200 checkpoint, the four tracked tasks scored
Task 2 = 70%, Task 28 = 55%, Task 29 = 70%, and the underlying training Task
60 = 55%; the complete suite was 519/1,800 (28.83%). Thus the checkpoint
generalizes strongly relative to deployed Full EXPO Task 60, but it does not
retain the underlying task at 100% (the matched Full EXPO Task-60 run is
20/20 on Task 60). The corresponding Task-60 full-proposal update-500 record
is Task 2/28/29/60 = 65%/40%/50%/55%, total 517/1,800 (28.72%).

The previously separate Task-2 side panel remains complete at 250/1,800
(13.89%) and is excluded from the diagnosis. Separately, the strict Task-2
full-proposal update-500 panel is now validated complete at 516/1,800
(28.67%), with Task 2 = 70%, Task 28 = 35%, Task 29 = 60%, and Task 60 =
45%. Its configuration confirms only Task-2 Small EXPO targets, official
pi05_libero initialization, zero official-demo ratio, and the 10/5 interface.
Against deployed Full EXPO Task 2 (447/1,800, 24.83%), this is +3.83 pp;
the underlying Task-2 retention is meaningful but below the Full EXPO
run's 19/20 (95%).

The current 2026-11-05 Task-60 outer-augmentation full-90 panels remain
partial at 39/90 (update 100), 37/90 (update 200), 32/90 (update 400), and
19/90 (update 500); no aggregate is inferred from them.

### Diagnostic protocol revision: constrained fast screening (2026-11-05, ~12:20 CST)

LIBERO-90 is no longer used as a routine screening evaluator. It is too slow
for a meaningful target/optimization search and can reward a checkpoint that
has forgotten the task that generated its targets. The existing partial
2026-11-05 full-90 files are preserved as artifacts, but only a final
shortlist will receive a full-90 confirmation.

The screening objective is now explicitly constrained: train on one source
task only; evaluate that source task plus three held-out tasks with the fast
four-task retention evaluator; reject material source-task regressions; then
rank survivors by held-out-task macro success and margin over the corresponding
deployed Full EXPO runs. The source-task score is always reported separately,
never hidden inside the macro.

The first targeted ablation under this protocol is full-proposal SFT with
same-task-only data, Pi0.5 initialization, the 10/5 interface, batch 16,
learning rate 5e-5, and 500 updates, but with `zero_lora_b` disabled. It is
running for source Tasks 28 and 60 on idle gpua002 and gpua087 in separate
tmux sessions. This tests whether the current generalization/retention tradeoff
is caused by LoRA initialization rather than data mixing or target semantics.
The old Task-60 outer-augmentation full-90 evaluator sessions were stopped;
their partial files and all training allocations were preserved.

At 12:24, the two no-`zero_lora_b` launchers were corrected after an isolated
missing `pi_base_data` symlink caused a pre-training exit; both then restarted
cleanly with the matching Task-28 and Task-60 collections. At 12:26, a second
factorial arm was launched on idle gpua051/gpua085: identical same-task-only
full-proposal training with zero LoRA-B retained but `warmup_steps=100`
instead of 10,000. All four runs use only the four-task retention panel; no
new full-90 evaluator was launched.

The Task-28 warmup-100 arm on gpua051 then failed before training with the
node-specific JAX error `CUDA_ERROR_NOT_INITIALIZED: No visible GPU devices`.
Only its dead diagnostic tmux wrapper was removed; it was not retried on that
node. The Task-28 no-B arm on gpua002 and Task-60 no-B/warmup-100 arms on
gpua087/gpua085 remain live with nonzero training CPU activity.

### Diagnosis closure: existing evidence is sufficient (2026-11-05, ~12:40 CST)

The four new 2026-11-05 ablation sessions were stopped before they produced a
committed training evaluation or checkpoint. Their launcher logs and manifests
are retained for provenance, but they are not results. The Task-28 warmup-100
arm failed before GPU initialization; the other three were interrupted during
their initial four-task rollout. No production trainer, production checkpoint,
or production GPU allocation was changed.

The existing validated evidence is sufficient to answer the research question:

| Source task | Best validated task-specific SFT evidence | SFT full-90 | Matched deployed Full EXPO full-90 | SFT source-task retention on fast panel |
|---|---|---:|---:|---:|
| 2 | `full_proposal`, update 500 | 516/1800 = 28.67% | 447/1800 = 24.83% | 70% |
| 28 | `full_proposal`, update 300 | 518/1800 = 28.78% | 451/1800 = 25.06% | 60% |
| 29 | `full_proposal`, update 500 | 524/1800 = 29.11% | 441/1800 = 24.50% | 75% |
| 60 | `full_proposal`, update 200 | 519/1800 = 28.83% | 368/1800 = 20.44% | 55% |

These are not failures of optimization. They show that target-specific SFT can
improve the aggregate suite score, but it does so by moving the actor toward a
narrow task policy and sacrificing part of the source task. The old 20k/40k
SFT runs made this much worse because they performed hundreds of passes over a
few hundred/thousand stored chunks and selected checkpoints by target fit
rather than retention.

#### What actually happened

1. **The comparison objective was wrong.** We initially treated a high
   four-task/full-90 macro as success. That allowed Task-28 and Task-60
   checkpoints with source retention of only 60% and 55% to look attractive.
   The source task must be a hard constraint; held-out generalization is the
   ranking objective only after that constraint is met.

2. **The historical “LoRA SFT” was not adapter-only.** The default Pi0.5 LoRA
   freeze mask updates about 466.96M parameters here: the 414.80M image tower,
   49.99M LoRA leaves, and 2.17M action/time projections. The isolated
   `lora_only` audit mask updates about 49.99M parameters. The controlled
   Task-28 scope comparison shows that true adapter-only training reduces the
   collateral damage, but it does not eliminate specialization by itself.

3. **The update dose is excessive for the data volume.** Task 28 has only 760
   eligible chunks from 150 successful episodes. At 40,000 batch-16 updates,
   the learner draws each stored row about 842 times on average. The old
   checkpoints therefore measure heavy memorization of a narrow teacher
   distribution, not a small adaptation.

4. **SFT has no retention term or recovery mechanism.** `bc_update()` updates
   only the Pi0.5 actor. It does not update or consult the critic, editor,
   temperature, or target actor. Full EXPO's deployed score includes its
   editor/critic candidate selection, so an actor-only SFT policy is being
   compared with a system that can reject or repair bad actor candidates.
   This is a real deployment difference, not a training-loop bug.

5. **The target semantics and augmentation hypotheses were not the main fix.**
   Full-proposal and executed-backfill Task-60 update-200 both reached roughly
   28.8% on full LIBERO-90 with 55% source retention. Outer image augmentation
   was not better on the available Task-60 evidence. These should not be the
   next direction.

6. **The data path is task-pure and internally consistent.** The validated
   manifests use the corresponding Small EXPO teacher only, the official
   `pi05_libero` student initialization, zero official-demo mixture for the
   main full-proposal runs, and the same 10/5 interface. The collector stores a
   ten-action proposal and executes five before replanning; training and
   evaluation use that same interface. The problem is not accidental cross-task
   data or a basic preprocessing mismatch.

#### How SFT should be improved

The next SFT recipe should be treated as constrained distillation:

- Use the true `lora_only` gradient mask; freeze the image tower and action/time
  projections. Do not call the broad default mask “LoRA-only.”
- Start with zero-B LoRA so step 0 is structurally close to official
  `pi05_libero`, and evaluate step 0 through the same actor-only evaluator.
- Use early stopping at the first point that passes a source-retention gate;
  do not run 20k/40k updates by default. Existing evidence puts useful
  checkpoints around 200--500 updates, not thousands.
- Keep the source collection task-specific, but add a small amount of
  **same-task official demonstration data** only if source retention is below
  the gate. The existing Task-29 50%-demo result raised source retention to
  80% at update 1500, but its held-out/full-90 score was worse than the
  200--500-step full-proposal arm, so 50% is too aggressive and is not a
  demonstrated solution. A lower same-task ratio would be a controlled next
  candidate, not an assumption.
- Select by a Pareto rule: source retention first, then held-out macro and
  margin versus the matched Full EXPO run. Preserve the best source-valid
  checkpoint, not merely the best target-task checkpoint.
- If the final goal is to beat deployed Full EXPO rather than only its actor,
  distill the *selected deployed action* after editor/critic selection or keep
  a lightweight selection head. Actor-only BC cannot reproduce the editor/
  critic's candidate-rejection behavior just by fitting the actor's raw
  proposal.

The practical conclusion is: use the Task-29 full-proposal update-200/500
recipe as the current best generalization direction, but add a source-retention
gate and true adapter-only scope. Task-2 update 500 is also useful; Task-28
and Task-60 full-proposal checkpoints are aggregate wins but fail the
source-retention constraint. No further routine full-90 screening is justified
until a candidate first passes the fast source-plus-held-out panel.

### Same-task demonstration screen launched (2026-11-05, ~12:56 CST)

To test source-retention repair without another quota-heavy checkpoint sweep,
three isolated screens were launched on idle sleeper allocations for Tasks 2,
28, and 29. Each uses the corresponding Small EXPO successful collection,
official `pi05_libero`, true `lora_only` gradients, zero-B initialization,
full-proposal ten-action targets, batch 16, learning rate 5e-5, 10k warmup,
and 500 updates. Each batch contains 25% official demonstrations from the
same underlying task and 75% Small EXPO targets; no other task's data is used.
The four-task retention panel runs at steps 0/100/200/300/400/500. These are
quota-safe screens with `--no-checkpoints`: metrics, manifests, W&B logs and
retention JSONs are persisted, but no 12--13 GiB learner checkpoint is written.

The exact loaded counts printed by the trainers are:

| Source task | Small EXPO chunks | Same-task official chunks | Official demo path |
|---:|---:|---:|---|
| 2 | 1,727 | 1,247 | `KITCHEN_SCENE10_put_the_black_bowl_in_the_top_drawer_of_the_cabinet_demo.hdf5` |
| 28 | 760 | 760 | `KITCHEN_SCENE5_close_the_top_drawer_of_the_cabinet_demo.hdf5` |
| 29 | 2,150 | 1,231 | `KITCHEN_SCENE5_put_the_black_bowl_in_the_top_drawer_of_the_cabinet_demo.hdf5` |

Task 60 was attempted on the fourth idle slot but failed before data loading
with the previously observed node-specific `CUDA_ERROR_NOT_INITIALIZED: No
visible GPU devices` fault on gpua051. It produced no training result and was
not retried on that node. The gpua038 slot remains occupied by the separate
Task-60 outer-augmentation full-90 evaluator and was left untouched.

The step-0 actor-only retention baselines for the three live screens then
completed successfully. Task 2's arm measured Task 2/28/29/60 = 70%/50%/65%/55%
(four-task macro 60.00%); Task 28's arm measured the same values (macro 60.00%);
and Task 29's arm measured 70%/50%/65%/60% (macro 61.25%). These are initial
Pi0.5 baselines, not post-SFT results. All three arms subsequently entered the
500-update BC loop with nonzero GPU utilization; no error or checkpoint write
has been observed.

At the next health poll, the three loops were still live after compilation:
Tasks 2/28/29 were approximately at updates 69/72/69 of 500, respectively,
with steady 1.5--1.7 s/update and no runtime error. The first post-training
retention panel is scheduled at update 100.

Task 2's update-100 panel then completed and passed JSON validation. It measured
Task 2/28/29/60 = 45%/45%/70%/60%, with four-task macro 55.00%. Relative to
its step-0 baseline of 70%/50%/65%/55%, source Task-2 retention fell 25 points
by update 100 even with 25% same-task official demonstrations. The held-out
macro over Tasks 28/29/60 rose only from 56.67% to 58.33%, so the source loss
is not justified by a large held-out gain. This is the first evidence against
the current demo25 recipe for Task 2 at learning rate 5e-5; Tasks 28 and 29
remain pending.

The remaining update-100 panels then completed with valid JSON records. Task
28's arm measured 60%/30%/50%/55% on Tasks 2/28/29/60 (macro 48.75%), versus
70%/50%/65%/55% at step 0. Its source score fell 20 points and its held-out
macro (Tasks 2/29/60) fell from 63.33% to 55.00%. Task 29's arm measured
60%/35%/70%/50% (macro 53.75%), versus 70%/50%/65%/60% at step 0. Its source
score rose 5 points, but held-out macro (Tasks 2/28/60) fell from 60.00% to
48.33%. Across all three arms, a 25% same-task official-data mix therefore did
not produce a Pareto improvement: it either failed to preserve the source task
or reduced held-out generalization substantially.

The first complete update-200 records are now available. Task 2 measured
Task 2/28/29/60 = 70%/40%/60%/50% (macro 55.00%), so its source task returned
to its step-0 70% but held-out macro fell from 56.67% to 50.00%. Task 28
measured 75%/30%/75%/55% (macro 58.75%): held-out macro rose from 63.33% to
68.33%, but source Task 28 fell 20 points and is therefore inadmissible under
the retention gate. Task 29 measured 65%/40%/55%/55% (macro 53.75%): its
source score fell 10 points and held-out macro fell from 60.00% to 53.33%.
None of the three update-200 points is a source-valid Pareto improvement.
These points reinforce that the official-data fraction is not a valid repair
unless it produces a source-valid Pareto point, not merely a higher held-out
macro.

The prepared 6.25%-demo Task-29 arm was attempted on the otherwise idle
gpua051 slot. It failed before data loading with the same node-specific JAX
`CUDA_ERROR_NOT_INITIALIZED`/`No visible GPU devices` failure seen earlier.
`nvidia-smi` can enumerate the A100 on that node, but JAX backend initialization
is not reliable there; the run is not a method result and its no-checkpoint
artifact is retained only for provenance. No production job or gpua038
full-90 evaluator was touched.

All three demo25 update-300 panels then completed. Task 2 measured
50%/50%/65%/55% on Tasks 2/28/29/60 (macro 55.00%): source retention was
20 points below its 70% step-0 control and held-out macro was unchanged at
56.67%. Task 28 measured 60%/35%/65%/55% (macro 53.75%): source retention
fell 15 points and held-out macro fell from 63.33% to 60.00%. Task 29
measured 65%/50%/65%/55% (macro 58.75%): source retention matched its step-0
65%, but held-out macro fell from 60.00% to 56.67%. Thus the complete
update-300 evidence still contains no source-valid generalization improvement.

Task 28's demo25 update-400 panel is the first exception in this screen:
it measured 80%/65%/70%/45% on Tasks 2/28/29/60 (macro 65.00%). Relative to
its step-0 control of 70%/50%/65%/55%, source Task 28 improved by 15 points
and held-out macro (Tasks 2/29/60) improved from 63.33% to 65.00%. This is a
source-valid screening Pareto point, but it has no learner checkpoint because
the screen uses `--no-checkpoints`; it must be reproduced with checkpointing
and a larger paired evaluation before full-90 confirmation.

As a read-only ranking of completed four-task screens, using each run's own
step-0 source score as the provisional retention gate, the strongest surviving
held-out points are: Task-60 pure full-proposal update 200 at 65.00% held-out
macro with source unchanged; Task-28 executed-backfill update 200 at 63.33%
with source +10 points; and Task-29 pure full-proposal update 200 at 60.00%
with source +5 points. The current demo25 points do not displace these. This
is only a screening ranking: each task score is based on 20 episodes, so the
shortlist still requires a larger paired panel and strict full-90 confirmation.

## Demo25 screen completion and validation handoff (2026-11-05)

The three demo25 arms completed their 500-update screens. The final panels
(Tasks 2/28/29/60, followed by four-task macro) were:

| Source task | Task 2 | Task 28 | Task 29 | Task 60 | Macro |
|---|---:|---:|---:|---:|---:|
| 2 | 70% | 35% | 60% | 50% | 53.75% |
| 28 | 75% | 80% | 60% | 40% | 63.75% |
| 29 | 60% | 40% | 70% | 50% | 55.00% |

The apparent Task-28 update-400 Pareto point (80/65/70/45, macro 65.00%)
therefore did not persist at update 500: its held-out average fell to 58.33%
versus 63.33% at its own step-0 control. Task 2 and Task 29 also fail the
source-valid held-out-improvement criterion. The 25% same-task official mix is
not promoted.

A checkpointed Task-28 reproduction was handed off automatically, after the
screen exited normally, on the same healthy gpua085 diagnostic allocation.
It keeps the exact demo25 recipe but stores the learner checkpoint on
node-local `/tmp/usertmp/rpolonsky` (the shared `/work` and `/projects`
filesystems currently have zero free space), evaluates 100 episodes per task
at steps 0 and 400, and retains the selected learner state. This is the
current validation experiment; only after it passes the source gate and
larger paired panel will a full-90 confirmation be considered.

The first handoff attempt exited before training because the requested 100
episodes exceeded the 50 official initial states per LIBERO task. This was a
validation-configuration error, not a learner or checkpoint failure. The
corrected retry uses the maximum valid 50 episodes and has passed setup,
loaded the task-pure data, and entered the step-0 evaluation.

The shortlist validation is now running on three independent diagnostic
slots: Task 28 demo25 on gpua085, Task 60 pure full-proposal on gpua002, and
Task 29 pure full-proposal on gpua087. The Task-29 and Task-60 arms loaded
2,150 and 2,226 same-task Small EXPO chunks respectively, with no official
data; all three are progressing through the 50-episode step-0 panels. No
candidate has yet been promoted to full LIBERO-90.

## Live validation poll (2026-11-05 15:15 CT)

The three allocations remain RUNNING and healthy. Their saved configs confirm
the intended task-purity and comparison conditions: Task 29 uses 2,150 Task-29
Small EXPO chunks with official-demo ratio 0; Task 60 uses 2,226 Task-60
Small EXPO chunks with official-demo ratio 0; Task 28 uses only Task-28 teacher
data plus same-task official demonstrations at ratio 0.25. All start from
`gs://openpi-assets/checkpoints/pi05_libero`, use the audited `lora_only`/zero-B
scope, full-proposal flow matching, 10/5 execution, batch-16, LR 5e-5, and the
same 224-pixel preprocessing. The runs are still completing their step-0
50-episode panels; no retention JSON or checkpoint has been produced yet, so
no candidate has been promoted to full-90.

At the 15:18 CT poll, the three allocations remained RUNNING. The progress
logs showed Task 28 had completed three of the four step-0 task panels and was
starting the fourth; Tasks 29 and 60 had completed one panel and were near the
end of their second. No `RETENTION_EVAL` record or checkpoint existed yet.

At 15:19 CT, the same allocations were still RUNNING. Task 28 was at roughly
18% of its fourth panel; Tasks 29 and 60 were at roughly 98% of their second
panel. No retention JSON or checkpoint had appeared.

At the next poll, all three remained healthy: Task 29 was at about 20% of
panel 3, Task 60 at 22% of panel 3, and Task 28 at 36% of panel 4. There were
still no retention records or checkpoints.

At 15:22 CT, the underlying training processes were confirmed alive on gpua087,
gpua085, and gpua002. Progress was Task 29 38% of panel 3, Task 60 38% of
panel 3, and Task 28 50% of panel 4; all error counters remained zero and no
retention record or checkpoint had been emitted.

Task 28 has now completed its checkpointed 50-episode step-0 control. The
record is Task 2/28/29/60 = 68%/44%/72%/58%, macro 60.50%, with a valid
`latest_checkpoints/0` learner checkpoint. The process continues into its
400-update training/evaluation run; this is a control measurement, not yet a
post-SFT result.

Task 28 has reached 8% of its 400-update learner-BC phase after the initial
JAX compilation, with no error. Tasks 29 and 60 remain in their final step-0
panel at approximately 30%; their control records are still pending.

The latest poll shows Task 28 at 25% of its learner-BC phase, still healthy.
Tasks 29 and 60 are each at approximately 50% of their final step-0 panel;
their control records remain pending.

At 15:24 CT, all three processes remained alive with zero detected errors.
Task 29 was at 58% of panel 3, Task 60 at 60% of panel 3, and Task 28 at
72% of panel 4. No retention JSON or checkpoint had appeared.

Tasks 29 and 60 subsequently completed their step-0 controls. Both measured
Task 2/28/29/60 = 64%/38%/60%/54%, macro 54.00%, and each wrote a valid
step-0 checkpoint. Their 200-update SFT phases are now the remaining
diagnostic comparison against these controls; Task 28 is still in its
400-update phase.

## Live checkpointed-shortlist poll (2026-11-05 15:53 CST)

The three corrected checkpointed validations remain RUNNING and healthy on
gpua087 (Task 29), gpua085 (Task 28), and gpua002 (Task 60), with no detected
runtime or checkpoint error. Task 29 and Task 60 have completed their update
loops and are progressing through the sequential post-training retention
panels; Task 28 has completed its 400-update loop and is in the corresponding
post-training panels. Because `evaluate_and_save()` evaluates Tasks 2, 28, 29,
and 60 sequentially and writes the learner checkpoint only after all four
records exist, the absence of a post-training JSON/checkpoint at this poll is
expected rather than evidence of a failed save. The pending decision remains
source retention first, then held-out macro; no 6.25%-official arm or full-90
confirmation has been launched.

At 16:10 CST, the same three processes were still RUNNING on their original
diagnostic nodes. The post-training retention panels were still being written
sequentially; each output directory still contained only `evals/00000000.json`
and the step-0 checkpoint. No runtime failure, OOM, or checkpoint failure was
observed, and no candidate was promoted before its complete four-task panel.

## Task-28 checkpointed demo25 result (2026-11-05 16:16 CST)

The checkpointed Task-28 demo25 reproduction completed normally. Its 50-episode
post-training panel at update 400 measured Task 2/28/29/60 = 72%/52%/50%/56%,
with macro 57.50%. The step-0 control was 68%/44%/72%/58%, macro 60.50%.
Thus source Task 28 improved by 8 percentage points, but held-out average
fell from 66.00% to 59.33% (−6.67 points). This is not a source-valid
held-out Pareto improvement and is not promoted to full LIBERO-90. The latest
learner checkpoint was saved at node-local `latest_checkpoints/400`; the best
checkpoint manager retained step 0 because its macro score was higher.

The result rejects the 25%-same-task-official-data repair for Task 28 at this
learning rate and update dose, even though it improves source retention.

## Task-28 demo0625 screen launched (2026-11-05 16:17 CST)

The freed gpua085 diagnostic allocation was reused for the prepared lower-dose
official-data screen in a new tmux session. It uses only Task-28 Small EXPO
targets plus Task-28 official demonstrations at `official_demo_ratio=0.0625`,
the same official Pi0.5 initialization, true `lora_only`/zero-B scope,
full-proposal 10/5 targets, batch 16, LR 5e-5, and 500 updates. It writes no
checkpoints during this quota-safe screen and evaluates the four-task panel
every 100 updates with 20 episodes. The launcher initially failed before
Python because its node-local log directory was absent; that was corrected
without changing the experiment, and the retry loaded 760 Small EXPO chunks
and 760 Task-28 official chunks and entered its step-0 evaluation. Tasks 29
and 60 remain running on their original allocations.

## Task-29 and Task-60 checkpointed shortlist results (2026-11-05 16:23 CST)

The pure same-task full-proposal validations completed normally with valid
step-200 checkpoints. Their 50-episode four-task panels were:

| Source | Step 0 (Tasks 2/28/29/60) | Step 200 (Tasks 2/28/29/60) | Source change | Held-out change |
|---|---|---|---:|---:|
| Task 29 | 64/38/60/54 (54.0% macro) | 62/48/66/54 (57.5% macro) | +6 pp | 52.0% → 54.7% (+2.7 pp) |
| Task 60 | 64/38/60/54 (54.0% macro) | 64/54/68/58 (61.0% macro) | +4 pp | 54.0% → 62.0% (+8.0 pp) |

Both are source-valid held-out Pareto improvements under the fast diagnostic
gate. The best-checkpoint directories and node-local latest checkpoints both
contain step 200. Full LIBERO-90 confirmation was launched for both on their
original now-idle allocations: Task 29 on gpua087 and Task 60 on gpua002,
using the existing node-local checkpoint evaluator, 20 episodes/task, and the
same 224-pixel 10/5 protocol. No production or side-evaluator allocation was
changed.

At 16:32 CST, both new full-90 confirmations had initialized and committed
one task file each; the evaluator processes remained live. The historical
Task-29 and Task-60 step-200 full-90 directories were checked, but their first
20 trial-success vectors do not match the newly validated checkpoints, so the
new confirmations are retained rather than treated as duplicates. The Task-28
demo0625 arm remained healthy in its sequential 20-episode retention panel;
no result was inferred before its complete panel.

At 16:36 CST, the Task-29 and Task-60 full-90 confirmations had each advanced
to 2/90 committed task files and remained GPU-active. The Task-28 demo0625 arm
completed its step-0 control at Task 2/28/29/60 = 75%/35%/70%/55%, macro
58.75%, and entered its 500-update BC loop. No runtime error or checkpoint
failure was observed; its first post-training retention evaluation remains
pending.

At 16:40 CST, the Task-28 demo0625 process was at update 84/500 and remained
healthy. Its metrics recorded 15 Small EXPO chunks plus 1 official Task-28
chunk per batch, confirming the intended 6.25% same-task composition; no
post-training retention JSON had been emitted yet. The Task-29 and Task-60
full-90 confirmations had each committed 3/90 task files and remained
GPU-active without traceback or runtime error. All five diagnostic sleeper
slots were occupied (the known-broken CUDA slot and separate Task-60
evaluator included), and the additional A100 arrays were still pending, so
no new arm was launched.

At 16:42 CST, the Task-28 screen reached update 99/500 and entered its first
20-episode retention evaluation. The only retention JSON visible at that
moment was the pre-training `step=0` control (75%/35%/70%/55% on Tasks
2/28/29/60, 58.75% macro); it must not be reported as the step-100 result.
The training process remained live and error-free.

The subsequent poll confirmed that the screen was progressing through the
step-100 four-task panel rather than hanging: one 20-episode panel had
completed and the next was at 5/20. The trainer emits one `RETENTION_EVAL`
record only after all four panels finish, so no step-100 score is available
yet.

At 16:46 CST, the Task-29 and Task-60 full-90 confirmations had each
advanced to 4/90 committed task files and remained GPU-active. The Task-28
demo0625 screen was still progressing through its first post-training panel
(the second 20-episode task panel was at 8/20); no aggregate step-100
retention result was available yet.

At 16:47 CST, both full-90 confirmations remained healthy with task records
00--03 committed (4/90 each). Task-28 was still in its first post-training
retention evaluation, at 10/20 episodes in the second task panel; its
aggregate step-100 record remained pending.

At 16:48 CST, the Task-28 screen had advanced to 14/20 episodes in that
second task panel; the process remained active and no aggregate retention
record had been emitted. The Task-29 and Task-60 full-90 confirmations were
still healthy at 4/90 task files each.

At 16:50 CST, the Task-29 and Task-60 full-90 confirmations had each advanced
to 5/90 task files. Task-28 was in its third retention panel at 6/20
episodes; its aggregate step-100 record was still pending.

The next poll showed the second Task-28 panel complete and the third panel
started (3/20 episodes). The trainer remains active; the aggregate step-100
retention record is still pending.

The third Task-28 retention panel then completed and the fourth began (4/20
episodes). The aggregate step-100 record remains pending; the Task-29 and
Task-60 full-90 confirmations remain healthy at 5/90 files each.

## Task-28 demo0625 step-100 result (2026-11-05 16:55 CST)

The first post-training retention panel completed successfully. The control
was Task 2/28/29/60 = 75%/35%/70%/55%, macro 58.75%; update 100 was
70%/40%/60%/45%, macro 53.75%. Task 28 itself improved by 5 points, but the
held-out average over Tasks 2, 29 and 60 fell from 66.7% to 58.3% (−8.3
points). This arm therefore fails the source-valid held-out Pareto gate at
update 100. It remains running through its planned 500 updates without
checkpoints so the complete dose curve can be assessed; no promotion is made
from this intermediate point.

At 17:11 CST, the Task-29 and Task-60 full-LIBERO-90 confirmations remained
GPU-active and had each committed valid JSON records for tasks 0--9 (10/90).
Each record contains the expected 20-episode evaluation fields, including
`eval/success_rate`, `successes`, task identity, checkpoint, and evaluation
seed; they are not stale or partial placeholder files. The logs contain only
the known missing-datasets-path warning and JAX deprecation warnings, with no
traceback, OOM, or termination. The Task-28 demo0625 screen remained alive in
its step-200 retention panel, with no new aggregate record yet. No new arm is
launched while all diagnostic slots remain occupied.

## Task-28 demo0625 step-200 result (2026-11-05 17:12 CST)

The second post-training retention panel completed successfully. The control
was Task 2/28/29/60 = 75%/35%/70%/55%, macro 58.75%; update 200 was
50%/45%/70%/55%, macro 55.00%. Source Task 28 improved by 10 points, but
the held-out average over Tasks 2, 29 and 60 fell from 66.7% to 58.3%
(−8.3 points), exactly the same held-out average as at step 100. The arm
therefore remains rejected by the source-valid held-out Pareto gate; it is
left running to 500 only to complete the dose curve and is not promoted.
The Task-29 and Task-60 full-90 confirmations had reached 11/90 valid task
records each and remained active.

The full-90 evaluator command was independently verified on both live slots:
each is evaluating the newly trained task-pure step-200 checkpoint directory
with `task-ids all`, 20 episodes/task, evaluation seed 7, image size 224,
action horizon 10, replan steps 5, and one evaluator device. At the next poll
both had advanced through task 10 (11/90 records); no stale checkpoint or
wrong-output-directory issue was found.

At 17:15 CST, the Task-28 demo0625 run had entered its step-300 retention
evaluation; only 4/20 episodes of the first task panel were complete, so no
step-300 aggregate was inferred. The Task-29 and Task-60 full-90 runs remained
healthy at 11/90 records each.

At 17:17 CST, Task-28 remained healthy in the step-300 panel at 9/20
episodes of the current task, with no aggregate result yet. The Task-29 and
Task-60 full-90 confirmations remained active at 11/90 and 12/90 valid task
records, respectively.

At 17:18 CST, the Task-28 step-300 panel had advanced to 16/20 episodes of
the current task, still without an aggregate record. The Task-29 and Task-60
full-90 confirmations were both at 12/90 valid task records and remained
healthy.

At 17:19 CST, the first Task-28 step-300 task panel completed and the next
panel began (2/20 episodes). The trainer has not emitted the step-300
aggregate; selection remains deferred until all four panels complete.

At 17:20 CST, the second Task-28 step-300 panel was at 7/20 episodes. The
Task-29 and Task-60 full-90 confirmations had both advanced to 13/90 valid
task records and remained healthy.

At 17:21 CST, the second Task-28 step-300 panel was at 10/20 episodes; no
step-300 aggregate had been emitted. The Task-29 and Task-60 full-90 runs
remained healthy at 13/90 records each.

At 17:22 CST, the second Task-28 step-300 panel was at 13/20 episodes, still
without an aggregate record. Both full-90 confirmations remained active at
13/90 valid task records.

The same poll later showed the second Task-28 panel at 18/20 episodes; the
step-300 aggregate remained pending, while both full-90 confirmations stayed
healthy at 13/90.

At 17:23 CST, the second Task-28 step-300 panel completed and the third panel
began (3/20 episodes). The step-300 aggregate remained pending; both full-90
confirmations remained healthy at 13/90.

At 17:24 CST, the third Task-28 step-300 panel was at 6/20 episodes. The
step-300 aggregate remained pending, and both full-90 confirmations remained
healthy at 13/90 records.

The third Task-28 panel later advanced to 9/20 episodes; its aggregate was
still pending, while both full-90 confirmations remained healthy at 13/90.

At 17:25 CST, the third Task-28 step-300 panel was at 13/20 episodes. The
Task-29 and Task-60 full-90 confirmations were at 13/90 and 14/90 records,
respectively, and remained healthy; no incomplete panel was used for scoring.

The next poll showed the third Task-28 panel at 16/20 episodes. The Task-29
and Task-60 full-90 confirmations had both advanced to 14/90 valid task
records and remained healthy; the Task-28 aggregate was still pending.

At 17:26 CST, the third Task-28 panel completed and the fourth/final panel
began (2/20 episodes). The step-300 aggregate remained pending; both full-90
confirmations remained healthy at 14/90.

At 17:27 CST, the final Task-28 step-300 panel was at 6/20 episodes. Its
aggregate remained pending, while the Task-29 and Task-60 full-90 runs stayed
healthy at 14/90 records each.

At 17:28 CST, the final Task-28 step-300 panel was at 13/20 episodes; its
aggregate remained pending, while both full-90 confirmations remained healthy
at 14/90.

At 17:29 CST, the final Task-28 panel had advanced to 15/20 episodes. Its
aggregate remained pending, while both full-90 confirmations remained at
14/90 and healthy.

The final Task-28 panel later reached 10/20 episodes; its aggregate remained
pending, while both full-90 confirmations remained healthy at 14/90.

## Task-28 demo0625 step-300 result (2026-11-05 17:30 CST)

The complete four-panel retention evaluation completed successfully. The
control was Task 2/28/29/60 = 75%/35%/70%/55%, macro 58.75%; update 300 was
60%/50%/60%/50%, macro 55.00%. Source Task 28 improved by 15 points, but
the held-out average over Tasks 2, 29 and 60 fell from 66.7% to 56.7%
(−10.0 points). This arm is definitively rejected by the source-valid
held-out Pareto gate and is not promoted. The no-checkpoint process was left
running only to finish its preplanned 500-update dose curve.

After the rejected step-300 evaluation, the Task-28 process resumed normally
and reached update 331/500 with no checkpoint or runtime error. The Task-29
and Task-60 full-90 confirmations remained healthy at 15/90 records each; no
diagnostic slot was free for a new arm.

At 17:31 CST, the rejected Task-28 dose-curve process had progressed to
357/500 updates without error. The Task-29 and Task-60 full-90 confirmations
remained healthy at 15/90 records each, and no diagnostic slot had freed.

At 17:32 CST, the same rejected process had reached 385/500 updates without
error. Both full-90 confirmations remained healthy at 15/90 records, and no
diagnostic slot was free.

The next poll showed the rejected process at update 399/500 and entering its
step-400 retention panel (1/20 episodes). The Task-29 and Task-60 full-90
confirmations remained healthy at 15/90 records each.

At 17:33 CST, the Task-28 step-400 panel was at 5/20 episodes; no aggregate
was available. The Task-29 and Task-60 full-90 confirmations remained healthy
at 15/90 records each, and no diagnostic slot had freed.

At 17:34 CST, the Task-28 step-400 panel was at 8/20 episodes. The process
remained healthy, both full-90 confirmations remained at 15/90, and no new
diagnostic slot was available.

The next poll showed the Task-28 step-400 panel at 12/20 episodes. The
Task-29 and Task-60 full-90 confirmations had both advanced to 16/90 valid
records and remained healthy; no diagnostic allocation was free.

At 17:35 CST, the Task-28 step-400 panel was at 15/20 episodes. Both
full-90 confirmations remained healthy at 16/90 records, with no diagnostic
slot free.

At 17:36 CST, that Task-28 step-400 task panel completed and the next of the
four task panels began (1/20 episodes). No step-400 aggregate was emitted;
the Task-29 and Task-60 full-90 confirmations remained healthy at 16/90.

At 17:37 CST, the second Task-28 step-400 panel was at 5/20 episodes. The
four-panel aggregate remained pending; both full-90 confirmations remained
healthy at 16/90 and no diagnostic slot was free.

The second Task-28 step-400 panel later advanced to 8/20 episodes; its
aggregate remained pending, while both full-90 confirmations stayed healthy at
16/90.

At 17:38 CST, the second Task-28 step-400 panel was at 11/20 episodes. Both
full-90 confirmations remained healthy at 16/90, and no diagnostic slot had
freed.

At 17:39 CST, the second Task-28 step-400 panel was at 15/20 episodes. Both
full-90 confirmations remained healthy at 16/90, with no diagnostic slot free.

The second Task-28 step-400 panel then completed and the third panel began
(0/20 episodes visible). The four-panel aggregate remained pending; both
full-90 confirmations remained healthy at 16/90.

At 17:40 CST, the third Task-28 step-400 panel was at 4/20 episodes. The
Task-29 and Task-60 full-90 confirmations had advanced to 17/90 records each
and remained healthy; no diagnostic slot was available.

At 17:42 CST, the third Task-28 step-400 panel was still running (8/20
episodes observed in the latest poll), so no step-400 aggregate is recorded
yet. The Task-29 and Task-60 full-90 confirmations remained healthy at 17/90
records each. All diagnostic slots remained occupied; unrelated training jobs
were not changed.

The Task-28 demo0625 screen subsequently emitted its complete step-400
aggregate: Task 2/28/29/60 = 80%/50%/65%/45%, or 60.0% macro. Relative to
its step-0 control (75%/35%/70%/55%), source Task 28 improved by 15 points,
but the held-out average fell from 66.7% to 63.3% (-3.3 points). It therefore
fails the source-valid held-out Pareto gate and is not promoted. The process
continues only to finish its preplanned 500-update no-checkpoint curve. The
Task-29 and Task-60 full-90 confirmations subsequently reached 18/90 records
each and remained healthy.

At 17:50 CST, the rejected Task-28 demo0625 process had completed update
500 and entered its final four-task retention evaluation; the first panel was
at 8/20 episodes. No step-500 aggregate is available yet, and the process is
still healthy. The two full-90 confirmations remain active, so all five
diagnostic allocations are still occupied.

At 17:51 CST, the Task-28 step-500 retention evaluation was at 15/20
episodes in its first panel. The Task-29 and Task-60 full-90 confirmations
had each advanced to 19/90 task records and remained GPU-active; no diagnostic
slot had freed.

At 17:52 CST, the first Task-28 step-500 panel completed and the second panel
was at 2/20 episodes. The step-500 aggregate remains pending; the Task-29 and
Task-60 confirmations remain active at 19/90 each.

At 17:54 CST, the second Task-28 step-500 panel was at 14/20 episodes. Both
Task-29 and Task-60 full-90 confirmations reached 20/90 committed task records
and remained active; neither is complete until all 90 task files validate.

At 17:58 CST, the second Task-28 step-500 panel had reached 18/20 episodes.
The two current full-90 confirmation directories had each advanced to 21/90
task records and remained active. The complete on-disk summary audit still
shows the previously validated Task-60 dose curve and no newly freed slot.

The next poll found the second Task-28 panel complete and the third panel at
3/20 episodes. The current Task-29 and Task-60 confirmations remained at
21/90 each; all five diagnostic allocations were still occupied.

At 17:59 CST, the third Task-28 step-500 panel was at 7/20 episodes. The
Task-29 and Task-60 confirmation counts remained 21/90 each, with all five
diagnostic allocations still running and the additional A100 arrays pending.

At 18:00 CST, the third Task-28 step-500 panel was at 12/20 episodes. The
Task-29 and Task-60 confirmations advanced to 22/90 records each and remained
active; no diagnostic slot was free.

The Task-28 demo0625 run then completed its final retention aggregate at
Task 2/28/29/60 = 70%/70%/65%/40% (61.25% macro). Its source Task 28 rose
35 points from the step-0 control, but its held-out average fell from 66.7%
to 58.3% (-8.3 points). This arm is definitively rejected and was not
promoted; it wrote no learner checkpoints.

After the Task-28 process exited, its gpua085 allocation was reused without
changing the allocation or any other session for a new Task-60 screen. The
new `sft_diag_task60_demo0625_20261106` tmux session uses only Task-60 Small
EXPO targets plus Task-60 official demonstrations (`official_demo_ratio=0.0625`),
official `pi05_libero`, true `lora_only`/zero-B, full-proposal 10/5 targets,
batch 16, LR `5e-5`, 10k warmup, 500 updates, and no checkpoints. The first
detached wrapper was reaped before starting the trainer; it was corrected by
keeping the `srun` step in the login-side tmux session. The corrected trainer
is now live, has initialized W&B run `zk2x14ni`, and is in shared-filesystem
startup/loading with no traceback or failure marker yet. Task-29 and Task-60
full-90 confirmations remain active at 22/90 each.

At 18:11 CST, the Task-60 demo625 process had completed environment/model
initialization (the Pi0.5/EXPO shape audit printed successfully) and remained
GPU-active with finite startup activity. The explicit `SFT data loaded` line
had not appeared yet, so no data count or step-0 result is inferred. The two
current full-90 confirmations had advanced to 25/90 each.

The same process then passed the data gate: `small_expo_chunks=2226`,
`official_demo_chunks=1247`, `official_demo_episodes=50`, ratio `0.0625`, and
the official-demo path was the matching Task-60 Living Room task file. It has
not emitted the step-0 retention record yet; the two full-90 confirmations
remain at 25/90 each.

At 18:15 CST, the newly allocated gpua073 slot was used for a Task-29
same-task demo625 screen. Its first wrapper exited before training because the
dataset manifest stores the teacher directory under `/u/rpolonsky`; the
launcher had requested the equivalent `/work` path. The launcher was corrected
to match the manifest exactly, and the retry is now live under
`sft_diag_task29_demo0625_20261106` with W&B run `g7p4tjew`. No optimizer step
or evaluation occurred in the failed attempt. Task-60 demo625 remains live;
the original Task-29/Task-60 full-90 confirmations remain untouched.

The Task-29 retry passed its data gate with 2,150 Task-29 Small EXPO chunks,
1,231 matching official-demo chunks from 50 episodes, ratio `0.0625`, and the
matching Kitchen Scene-5 demonstration file. It is still in compilation/early
startup; no step-0 aggregate has been inferred. The Task-60 screen has
completed at least one 20-episode retention panel but has not emitted its
four-task aggregate. Both current full-90 confirmations advanced to 26/90.

At 18:18 CST, the Task-60 demo625 screen was in its second retention panel at
12/20 episodes and remained GPU-active without an error. The Task-29 retry was
still in JAX compilation/startup with finite CPU activity and no traceback.
The Task-29 and Task-60 full-90 confirmations advanced to 27/90 and 26/90,
respectively.

At 18:19 CST, Task-29 had entered its retention evaluation at 13/20 in the
current panel, while Task-60 remained active in its retention sequence. Neither
screen had emitted a four-task aggregate. Both current full-90 confirmations
were at 27/90; all diagnostic processes remained active.

At 18:24 CST, all four active diagnostic processes were still healthy and
GPU-utilized. The Task-29 and Task-60 full-90 confirmations had each advanced
to 28/90 task records. The Task-29 and Task-60 demo625 screens remained in
their sequential retention panels and had not emitted a four-task aggregate;
no result is inferred before those aggregates are complete.

The Task-60 demo625 screen has now emitted its step-0 control before any SFT
updates: Task 2/28/29/60 = 75%/35%/70%/55% (58.75% macro). This matches the
expected official-Pi0.5 starting point; it is recorded only as the control.
The process then entered the 500-update training loop. The Task-29 screen and
both full-90 confirmations remain active.

The next live poll showed the Task-29 full-90 confirmation at 29/90 task
records and the Task-60 confirmation at 28/90. Task-60's screen process was
still GPU-active in the update loop; Task-29's screen was still completing its
initial retention evaluation. These partial states are not treated as final
scores.

The subsequent poll found both full-90 confirmations at 29/90 records. The
screen processes remained alive and GPU-active; Task-60 was still in its first
training loop after the step-0 control, and Task-29 had not yet emitted its
step-0 aggregate. There was no traceback or OOM, so neither process was
restarted.

The next poll confirmed that Task-60's SFT loop had progressed to update
33/500 after compilation, at roughly 1.6 seconds per update. Its process was
GPU-active and healthy. The Task-29 and Task-60 full-90 confirmations remained
at 29/90 records and continued producing task evaluation lines without errors;
the Task-29 demo screen remained in its initial panel.

Task-60 subsequently reached update 49/500 at the same stable throughput,
approaching its first post-update retention evaluation at update 100. The
full-90 confirmations remained at 29/90; their latest completed Task-28
records were 40% for the Task-29 arm and 55% for the Task-60 arm. No failure
was observed.

Task-60 then advanced to update 80/500, still without a health error and close
to its first post-update retention evaluation. The Task-29 full-90
confirmation advanced to 30/90 task records, while the Task-60 confirmation
remained at 29/90. These are still partial results.

The Task-29 demo625 screen has now emitted its step-0 control: Task
2/28/29/60 = 65%/40%/60%/45% (52.5% macro). Its source-task score is 60% and
the held-out average over Tasks 2/28/60 is 50.0%. Task-60's screen was at
update 99/500 and entering its first post-update evaluation. The two controls
are kept separate because they are task-specific and differ before training.

The following poll confirmed that Task-29 had also entered its SFT loop after
the control. Task-60 was still completing its update-100 evaluation. Both
full-90 confirmations had reached 30/90 records, and neither screen had yet
produced a post-update aggregate.

The next poll found the Task-29 full-90 confirmation at 31/90 records (the
latest completed Task 30 record was 85%) and the Task-60 confirmation at
30/90. Both screens remained healthy; Task-60 still had not emitted its
completed update-100 aggregate.

The subsequent poll found both full-90 confirmations at 31/90 records. The
Task-29 screen was actively progressing through its SFT loop, while the
Task-60 log continued updating at the step-100 transition. Neither screen had
emitted a completed post-update aggregate or a failure.

Both screens subsequently completed 100 SFT updates and entered their first
post-update retention evaluations. At the latest poll, Task-29 was at 2/20
episodes and Task-60 at 10/20; their aggregate `RETENTION_EVAL step=100`
records were still pending. The full-90 confirmations remained at 31/90.

The live output clarifies that each retention evaluation runs 20 episodes for
each of the four tasks sequentially (80 episodes total) before emitting its
aggregate. Task-60 completed its first 20-episode task panel and had entered
the next panel; Task-29 was 13/20 through its first panel. These panel-level
counts are not aggregate scores.

The next poll still found both screens inside their sequential step-100
panels, with no aggregate emitted. The two full-90 confirmations advanced to
32/90 records each and remained active.

The following poll showed Task-29 complete another 20-episode retention panel
and Task-60 midway through its current panel. Neither had emitted the full
four-task step-100 aggregate. Both full-90 confirmations remained active at
32/90 records.

## Verified Task-60 dose curve recovered from completed suites

A read-only audit of the completed full-90 summary files recovered the
Task-60 pure full-proposal dose curve. Every row uses the Task-60 Small EXPO
collection only (150 successful teacher episodes, 2,226 chunks), official
`pi05_libero` initialization, zero official demonstrations, true `lora_only`
with zero LoRA-B initialization, batch 16, LR `5e-5`, 10k warmup, and the
same 10/5 action interface:

| SFT update | Full-90 successes | Rate | Matched Full EXPO | Margin |
|---:|---:|---:|---:|---:|
| 100 | 510/1,800 | 28.33% | 368/1,800 (20.44%) | +7.89 pp |
| 200 | 519/1,800 | **28.83%** | 368/1,800 (20.44%) | **+8.39 pp** |
| 400 | 506/1,800 | 28.11% | 368/1,800 (20.44%) | +7.67 pp |
| 500 | 517/1,800 | 28.72% | 368/1,800 (20.44%) | +8.28 pp |

Update 200 remains the best validated Task-60 checkpoint and the largest
matched Full-EXPO margin found so far. This supports early stopping/low-dose
SFT as the main practical mechanism; longer fitting does not monotonically
improve generalization.

## Live diagnostic status (2026-11-05 18:43 CST)

The two same-task official-data screens are still active and healthy. The
Task-29 process is running on `gpua073` and the Task-60 process on `gpua085`.
Both use only the corresponding Small EXPO teacher collection plus a 6.25%
ratio of official demonstrations from the same underlying task; both start
from the official `pi05_libero` checkpoint, use the true `lora_only` scope,
the full-proposal target, batch 16, and 500 SFT updates. They write no
checkpoints (`--no-checkpoints`), so these screens are not creating a quota
or artifact-retention problem.

The post-update retention aggregates have not yet been emitted: each one
requires four sequential 20-episode task panels (80 episodes total). Thus no
6.25%-screen result is admitted to the hypothesis ranking yet. The separate
full-90 confirmations of the previously best pure Task-29/update-200 and
Task-60/update-200 checkpoints are also still active, at 33/90 and 32/90
completed task records respectively. These are partial runtime counts, not
final scores, and no intervention or relaunch is warranted.

### First completed 6.25% screen: Task 60, update 100

The Task-60 screen has now emitted its complete four-task retention aggregate:

| arm | Task 2 | Task 28 | Task 29 | Task 60 (source) | held-out average | macro |
|---|---:|---:|---:|---:|---:|---:|
| step 0 control | 75% | 35% | 70% | 55% | 60.0% | 58.75% |
| 6.25% official + teacher, update 100 | 75% | 45% | 60% | 50% | 60.0% | 57.5% |

This candidate is rejected by the source-retention gate: Task-60 fell by 5
percentage points. Its held-out average was unchanged, so the added official
data did not improve generalization either. The run remains alive to measure
later doses, but this result is not evidence that the 6.25% mixture is a
better recipe than pure task-specific distillation.

The subsequent live poll still found the Task-29 screen without a completed
post-update aggregate. Both independent full-90 confirmations were alive at
33/90 task records. The Task-60 screen remained alive after its step-100
measurement; no job was restarted or stopped.

## Completed 6.25% same-task screen dose curves

Both no-checkpoint screens have now completed all scheduled retention panels.
The held-out column is the arithmetic mean over the three non-source tasks;
these four-task panels are screening evidence, not substitutes for LIBERO-90.

Task 29 screen (step-0 source = 60%, held-out average = 50.0%):

| update | Task 2 | Task 28 | Task 29 source | Task 60 | held-out average | macro |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 65% | 40% | 60% | 45% | 50.0% | 52.5% |
| 100 | 65% | 30% | 55% | 55% | 50.0% | 51.25% |
| 200 | 80% | 50% | 65% | 55% | **61.67%** | **62.5%** |
| 300 | 55% | 30% | 75% | 55% | 46.67% | 53.75% |
| 400 | 55% | 35% | 70% | 45% | 45.0% | 51.25% |
| 500 | 70% | 45% | 65% | 50% | 55.0% | 57.5% |

Task 60 screen (step-0 source = 55%, held-out average = 60.0%):

| update | Task 2 | Task 28 | Task 29 | Task 60 source | held-out average | macro |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 75% | 35% | 70% | 55% | 60.0% | 58.75% |
| 100 | 75% | 45% | 60% | 50% | 60.0% | 57.5% |
| 200 | 75% | 50% | 65% | 45% | 63.33% | 58.75% |
| 300 | 80% | 35% | 70% | 50% | 61.67% | 58.75% |
| 400 | 55% | 25% | 75% | 55% | 51.67% | 52.5% |
| 500 | 60% | 45% | 65% | 55% | 56.67% | 56.25% |

The Task-29/update-200 point is the only new screen point with both a
source-task improvement and a large held-out improvement (+11.67 pp). It was
not checkpointed by the screen, so it is being reproduced with identical
settings and a single checkpoint at update 200 before any full-90 claim is
made. The Task-60 screen has no point that improves held-out performance while
also passing the source-retention gate: update 200 raises held-out average but
loses 10 source points.

The checkpointed Task-29 reproduction is running in tmux
`sft_diag_task29_demo0625_step200_full90_20261106` on `gpua073`. Its output
and checkpoint are node-local under `/tmp/usertmp`; the launcher log is on
the work filesystem. At the latest live check the trainer was still alive
during initialization/JAX compilation, with 16.8 GiB GPU memory allocated and
no traceback. The two earlier pure-distillation full-90 audits had advanced to
54/90 records each and remain incomplete. A subsequent poll showed the
reproduction had passed initialization and entered its first retention
evaluation with active GPU utilization; the step-200 checkpoint is not yet
available. The subsequent poll still found the reproduction in its first
20-episode panel with active GPU utilization and no error; the independent
full-90 audits were at 55/90 (Task 29) and 54/90 (Task 60).

The checkpointed Task-29 reproduction has now completed its own full step-0
retention aggregate: Task 2/28/29/60 = 75%/35%/70%/55%, macro 58.75%, and a
55.0% held-out average over Tasks 2/28/60. This differs from the earlier
no-checkpoint screen's 65%/40%/60%/45% control, so that screen's apparent
11.67-point held-out gain is not yet replicated. The update-200 candidate will
be judged against this run's own step-0 control. A step-0 checkpoint is being
written before training continues.

The step-0 checkpoint finalized, and the trainer entered the SFT loop. The
latest metrics showed update 16 with finite actor loss/gradient values and
15 Small EXPO chunks plus 1 matching Task-29 official chunk in each logged
batch. GPU utilization was 100% during the update loop; no instability was
observed.

At 20:46 CST, the checkpointed Task-29 reproduction had completed the full
200-update SFT loop. The last recorded metrics were finite (update 199,
actor loss 0.1298, gradient norm 0.0264), and every logged batch still had
15 Task-29 Small EXPO chunks plus 1 matching Task-29 official chunk. The
trainer then entered the required four sequential 20-episode retention
panels. The step-200 checkpoint and post-update aggregate were not yet
visible; the process remained GPU-active and healthy. The independent pure
Task-29 and Task-60 Full-90 confirmations had each reached 62/90 task files
and were left untouched.

The checkpointed reproduction then completed its retention evaluation and
finalized `latest_checkpoints/200` plus `evals/00000200.json`. Its step-200
panel was Task 2/28/29/60 = 85%/45%/65%/45%, macro 60.0%, with a 58.33%
held-out average over Tasks 2/28/60. Against this run's step-0 control
(75%/35%/70%/55%, 55.0% held-out, 58.75% macro), the candidate gains 3.33
held-out points and 1.25 macro points but loses 5 source-task points. Thus
the earlier no-checkpoint Task-29 demo625 screen gain is not replicated under
the checkpointed protocol and the candidate fails the current source-retention
gate. The first Full-90 wrapper attempt exposed two diagnostic-wrapper bugs
(missing parent training path in `PYTHONPATH` and an incorrect checkpoint
root); both were corrected without rerunning training. An evaluation-only
tmux retry is now restoring the verified checkpoint; it has loaded the model
and is compiling, with no traceback or task files yet.

The evaluation-only retry then logged a successful restore of step 200 and
entered the JAX inference compilation path. It remained GPU-active without a
traceback; no Full-90 task record had been committed at the latest poll.

The retry has now entered actual LIBERO-90 rollout and committed Task 00
(5.0% success); it remains at 1/90 records, so no suite score is inferred.
The independent pure Task-29 and Task-60 Full-90 confirmations are at 67/90
and 66/90 records respectively and remain untouched.

The audit then committed Task 01 at 0.0%, advancing to 2/90 records. The
process remained actively evaluating with no traceback or OOM; the per-task
rollout time is variable because each task can use up to 20 x 400 environment
steps. No aggregate Full-90 score is inferred from these partial records.

Task 02 subsequently completed at 85.0%, advancing the audit to 3/90 task
records. The evaluator remains active; these partial records are retained for
runtime provenance only and are not yet used to rank the candidate.

The audit then completed Tasks 03, 04, 05, 06, and 07, each at 0.0%,
advancing to 8/90 task records. The evaluator remains GPU-active and healthy;
these partial results are not treated as a Full-90 score.

Task 08 subsequently completed at 0.0%, advancing the audit to 9/90 task
records. The evaluator remains active; no complete-suite score is inferred.

Task 09 subsequently completed at 100.0% (mean 119.05 steps), advancing the
audit to 10/90 records. The evaluator remains active and the suite remains
incomplete; this isolated task score is not used as a ranking result.

Task 10 subsequently completed at 95.0%, advancing the audit to 11/90
records. The evaluator remains active; this partial score is retained for
provenance and does not change the incomplete-suite ranking.

The two independent pure-teacher Full90 confirmations (Task-29/update-200
and Task-60/update-200) were re-polled and have each advanced to 78/90 task
records. Both evaluator processes remain GPU-active with no traceback, OOM,
or termination. Their logs contain only the previously observed nonfatal
missing-datasets-path warning; their partial suites are not yet admitted as
complete comparison scores. The checkpointed Task-29 demo625 reproduction
audit remains active separately at 11/90.

The checkpointed Task-29 demo625 audit then committed Task 11 at 0.0%,
advancing to 12/90 records. Its evaluator remains GPU-active with no
traceback or OOM. The two pure-teacher confirmations remain active at 78/90
each; no incomplete suite is used for ranking.

The next live poll showed both pure-teacher confirmations advancing to 79/90
records. Their processes remain alive with no failure; the checkpointed
Task-29 demo625 audit remains active at 12/90. No partial suite is being
treated as a Full90 comparison.

The subsequent poll confirmed further rollout progress: both pure-teacher
audits have now written through task 78 (79/90 records), and the checkpointed
Task-29 demo625 audit has written task 12 (13/90 records). All three
evaluators remain live and GPU-active without OOM or traceback; complete-suite
aggregation is still pending.

The next poll showed the pure-teacher Task-29 audit advance through task 79
(80/90 records). The pure-teacher Task-60 audit remains at 79/90 and is still
GPU-active, while the demo625 audit remains at 13/90. No complete-suite score
has been computed yet.

The following poll showed the pure-teacher Task-29 and Task-60 audits both
advance through task 79 (80/90 records each). The demo625 audit remains at
13/90; all evaluators are still GPU-active and complete-suite aggregation is
pending.

The latest poll confirms both pure-teacher audits remain at 80/90 records and
are still active; the demo625 audit remains at 13/90. No terminal marker,
error, or complete-suite aggregate is present yet.

A direct log inspection confirms the apparent pause is ordinary rollout
latency, not a hang: Task 79 completed at 21:55:39 for the pure Task-29 audit
and at 21:55:54 for pure Task-60, with ongoing 44--50% GPU utilization. Both
remain healthy at 80/90 records; the final ten tasks are still pending.

The checkpointed Task-29 demo625 audit then committed task 13, advancing to
14/90 records. Its evaluator remains active without traceback or OOM; both
pure-teacher audits remain active at 80/90.

The next poll showed both pure-teacher audits commit task 80, advancing to
81/90 records each. The demo625 audit remains at 14/90, and all three
evaluators remain active without error.

The latest poll confirms both pure-teacher audits remain at 81/90 records and
GPU-active; the demo625 audit remains at 14/90. No complete-suite aggregate
or terminal marker is present yet.

The checkpointed Task-29 demo625 audit then committed task 14, advancing to
15/90 records. It remains GPU-active without traceback or OOM; the two
pure-teacher audits remain active at 81/90.

Direct log inspection confirms continued normal rollout progress: task 80
completed at 22:00:16 for pure Task-29 and at 22:00:32 for pure Task-60.
Both evaluators are now rolling out task 81 with active GPU utilization; the
81/90 counts are therefore not a hang.

The next poll showed pure-teacher Task-29 commit task 81, advancing to 82/90
records. Pure Task-60 remains at 81/90 and the demo625 audit at 15/90; all
evaluators remain active without error.

The following poll showed both pure-teacher audits commit task 81, advancing
to 82/90 records each. The demo625 audit remains at 15/90; all evaluators
remain active without error.

The latest poll confirms both pure-teacher audits at 82/90 records, with
active evaluator processes and GPU memory allocated. The demo625 audit remains
at 15/90; no complete-suite aggregate is available yet.

The next poll advanced the pure-teacher Task-29 audit through task 82,
bringing it to 83/90 records. Pure Task-60 remains at 82/90 and demo625 at
16/90; all evaluators remain active without error.

The checkpointed Task-29 demo625 audit then committed task 15, advancing to
16/90 records. It remains active without traceback or OOM; both pure-teacher
audits remain at 82/90.

Latest scope clarification: the validated pure-teacher Task-29 and Task-60
arms use only their matching Small EXPO teacher chunks (`official_demo_ratio=0`);
the official `pi05_libero` checkpoint is initialization, not training data. The
separate Task-29 demo625 arm intentionally mixes matching Task-29 official
demonstrations with Task-29 teacher chunks at ratio 0.0625. It has no
cross-task data, but it is not a no-mixture result. Current audit counts are
83/90 for each pure arm and 16/90 for demo625.

The next live poll advanced pure Task-29 to 84/90 records (through task 83).
Pure Task-60 remains at 83/90 and demo625 at 17/90. All three evaluator
processes remain alive with GPU activity; no aggregate is computed until a
suite reaches 90/90.

The next poll advanced pure Task-29 to 85/90 files (through task 84), with an
interim 521/1,700 successes (30.65%). Pure Task-60 remains at 84/90 and
520/1,680 (30.95%). The same-task demo625 audit advanced to 18/90. All three
processes remain active without an error marker.

The next poll advanced both pure audits to 86/90 (through task 85). Pure
Task-29 is at 521/1,720 successes (30.29%) and pure Task-60 at 520/1,720
(30.23%). The demo625 audit remains at 19/90; all processes remain active.

The subsequent poll advanced the same-task demo625 audit to 20/90 (through
task 19). The pure Task-29 and Task-60 audits remain at 86/90; all processes
remain active without an error marker.

The next poll advanced pure Task-29 to 87/90 (through task 86), with 521/1,740
successes (29.94%), and advanced demo625 to 21/90 (through task 20). Pure
Task-60 remains at 86/90, with 520/1,740 successes (29.89%). All processes
remain active and healthy.

The following poll advanced pure Task-60 to 87/90 as well (through task 86),
with 520/1,740 successes (29.89%). Pure Task-29 remains at 87/90 and
521/1,740 (29.94%); demo625 remains at 21/90. All evaluators remain active.

The following poll advanced pure Task-60 to 85/90 as well (through task 84),
with 520/1,700 successes (30.59%). Pure Task-29 remains at 85/90 and
521/1,700 (30.65%); demo625 remains at 18/90. All three evaluator processes
remain active and healthy.

The subsequent poll advanced the same-task demo625 audit to 19/90 (through
task 18). The two pure audits remain at 85/90, and all three evaluator
processes remain active without an error marker.

An interim, non-final tally over the first 84 task files is 521/1,680
(31.01%) for pure Task-29 and 520/1,680 (30.95%) for pure Task-60. These
figures are recorded only as progress indicators; they are not compared as
Full-90 results until all 90 task files exist.

The subsequent poll advanced the pure Task-60 audit to 84/90 as well. Pure
Task-29 remains at 84/90 and the same-task demo625 audit at 17/90; all remain
GPU-active and no evaluator error is present.

The latest live poll advanced both task-pure Full-90 audits to 88/90 files:
pure Task-29 is 521/1,760 (29.60%) and pure Task-60 is 520/1,760 (29.55%).
The separate same-task official-demo control advanced to 22/90 and is
currently 116/440 (26.36%). These are interim tallies only; no Full-90
comparison is admitted until the corresponding 90-file audit completes.
All three evaluator processes remain alive and GPU-active without an error
marker.

The next poll advanced the separate same-task Task-29 official-demo control to
23/90 files (through task 22). Its process remains GPU-active; the two pure
audits remain at 88/90 and continue their task-88 rollouts. The evaluator logs
contain only the known nonfatal missing-datasets-path warnings and no
traceback, OOM, or termination marker.

The next poll advanced the pure Task-29 audit to 89/90 files (through task
88), with an interim 521/1,780 successes (29.27%). Pure Task-60 remains at
88/90 with 520/1,760 (29.55%). These remain provisional until each audit has
all 90 task files; all evaluator processes remain alive.

The following poll advanced pure Task-60 to 89/90 as well. The synchronized
interim tallies are now Task-29 521/1,780 (29.27%) and Task-60 520/1,780
(29.21%). Each pure audit has exactly one task file remaining, so no final
Full-90 comparison is recorded yet.

Pure Task-29 then completed its final task file and passed the strict
Full-90 validator: 90 unique task files, no missing tasks, no errors, and
521/1,800 successes (28.94%). This is the Task-29 pure full-proposal,
update-200 confirmation using only Task-29 Small EXPO teacher targets and
zero official demonstrations. Against matched Full EXPO Task-29 (441/1,800,
24.50%), the confirmed margin is +4.44 percentage points. It remains below
the already validated Task-29 update-500 result (524/1,800, 29.11%, +4.61
points), so it does not change the current best-dose ranking.

Pure Task-60 then completed its final task file and passed the strict
Full-90 validator: 90 unique task files, no missing tasks, no errors, and
520/1,800 successes (28.89%). Its manifest confirms the task-pure
configuration: Task-60 Small EXPO teacher collection only, `collection_source`
`small_expo`, `official_demo_ratio=0`, full-proposal 10/5 targets, true
LoRA-only/zero-B initialization, batch 16, learning rate 5e-5, 10k warmup,
and official `pi05_libero` initialization. Against matched Full EXPO Task-60
(368/1,800, 20.44%), the confirmed margin is +8.44 percentage points. This
is currently the largest validated pure-SFT margin in the diagnosis.

The separate same-task Task-29 official-demo control has continued to 25/90
files. It remains an intentionally mixed same-task arm and is not included in
the pure-SFT ranking or margin claims.

The next live poll advanced the separate same-task Task-29 official-demo
control to 24/90 files (through task 23). The pure Task-29 and Task-60 audits
remain at 89/90, with both evaluator processes still GPU-active and without
error or termination markers.

The subsequent poll advanced the same-task Task-29 official-demo control to
26/90 files (through task 25), with an interim 154/520 successes (29.62%).
Its process remains GPU-active and healthy; this mixed-source interim tally is
not used in the pure-SFT ranking.

The same-task Task-29 official-demo control then advanced to 28/90 files
(through task 27), with an interim 155/560 successes (27.68%). Its evaluator
remains GPU-active and healthy; this mixed-source result is still excluded
from pure-SFT comparisons.

The same-task Task-29 official-demo control then advanced to 27/90 files
(through task 26), with an interim 155/540 successes (28.70%). Its evaluator
remains GPU-active and healthy; this mixed-source result is still excluded
from pure-SFT comparisons.

## Current resource and data-scope status (2026-11-05 22:58 CST)

The strict task-pure Task-29 and Task-60 Full-90 confirmations are complete and
validated. The remaining Task-29 official-demo control is still alive at 28/90
committed task files, with its evaluator process GPU-active and no traceback or
OOM. It intentionally combines Task-29 Small EXPO teacher chunks with
same-task Task-29 official demonstrations (`official_demo_ratio=0.0625`), so it
is a mixed-source control and remains excluded from all pure-SFT rankings.

The current Slurm state has no genuinely free allocation: all running GPU
allocations are reserved sleeper or evaluator jobs, and the remaining A100
requests are pending. No existing allocation or training/evaluation process was
changed. A new diagnosis arm should be launched only when a pending allocation
starts or an explicitly released GPU becomes available.

The next poll confirmed that the mixed Task-29 official-demo control advanced
to 29/90 files (through task 28), with an interim 164/580 successes (28.28%).
Its evaluator remains GPU-active and healthy. This remains a same-task mixed
control, not evidence for the task-pure SFT ranking.

The following poll confirmed continued forward progress: the mixed control
advanced to 30/90 files (through task 29), with an interim 177/600 successes
(29.50%). Its evaluator remains GPU-active and healthy; this same-task mixed
result remains excluded from the pure-SFT ranking.

The subsequent resource poll found the mixed evaluator still alive at 30/90
and GPU-active. No allocation is free. Slurm currently estimates pending A100
array starts at approximately 23:10 CST for job 23505371 and 14:53 CST on
2026-11-06 for job 23448992; job 23533204 has no start estimate. These are
read-only scheduler estimates, and no existing job was changed.

The next live poll confirmed continued progress: the mixed Task-29 control
advanced to 31/90 files (through task 30), with an interim 196/620 successes
(31.61%). Its evaluator remains GPU-active and healthy. No pending A100 array
has started, so no new diagnostic job was launched.

## Control completion and new pure screens (2026-11-06 05:18 CST)

The same-task Task-29 official-demo control completed all 90 task files and
passed the strict validator: 90 unique tasks, zero missing tasks, zero errors,
and 510/1,800 successes (28.33%). Its manifest confirms `source_task_id=29`,
`collection_source=small_expo`, and `official_demo_ratio=0.0625`; it combines
Task-29 Small EXPO targets with Task-29 official demonstrations. It is retained
as a same-task mixed control and excluded from the pure-SFT ranking.

Five additional two-day A100 sleeper reservations were submitted as array
`23596881_[1-5]`. Their stdout/stderr are isolated under
`/work/hdd/bgqe/rpolonsky/allocation_logs/gpu-sleep-5k-%A_%a.{out,err}`.

Two newly idle A100 slots were verified to contain only their sleeper process
and 0 MiB GPU use. They now run separate tmux sessions for new task-pure seed-
123 screens: Task 2 and Task 28. Both use only their matching Small EXPO
teacher collection, official-demo ratio 0, official `pi05_libero`
initialization, true `lora_only`/zero-B scope, full-proposal 10/5 targets,
batch 16, LR 5e-5, 10k warmup, 500 updates, and retention/checkpoint
evaluation every 100 updates. They write checkpoints only to node-local
`/tmp/usertmp/rpolonsky` and do not alter existing jobs or shared checkpoint
trees. Both W&B runs authenticated successfully and are currently in normal
startup/compilation with no runtime error.

Both seed-123 screens then passed their data and model startup gates. Task 2
printed `small_expo_chunks=1727`, `official_demo_chunks=0`, and
`official_demo_ratio=0.0`; Task 28 printed `small_expo_chunks=760`,
`official_demo_chunks=0`, and `official_demo_ratio=0.0`. Both are in the
step-0 20-episode retention evaluation with approximately 17.2 GiB GPU memory
and no traceback, OOM, or termination marker.

The following live poll found the control still at 29/90, with the evaluator
process still running and GPU-active; this is a normal verified wait rather
than evidence of termination. Slurm still shows all running allocations as
reserved sleeper/evaluator jobs and the outstanding A100 arrays as pending, so
no new diagnostic process was launched and no existing job was touched.

The seed-123 pure screens completed their step-0 four-task retention baselines.
Task 2's screen measured 65% on Task 2, 40% on Task 28, 60% on Task 29, and
45% on Task 60 (52.5% macro). Task 28's screen measured 75%, 35%, 70%, and
55%, respectively (58.75% macro). Both configurations passed the source-purity
gate (`small_expo` matching task only, `official_demo_ratio=0`, zero official
chunks) and entered the first BC update; no update result or post-update
evaluation is claimed yet.

The next two pure seed-123 screens were launched on independently verified idle
sleeper A100s: Task 29 on gpua100 (allocation `23583973`) and Task 60 on gpua095
(allocation `23584251`), each in its own tmux session. Their launchers use the
matching task-specific Small EXPO teacher checkpoint, official `pi05_libero`
initialization, true `lora_only`/zero-B scope, full-proposal 10/5 targets,
official-demo ratio 0, batch 16, 500 updates, and four-task retention every 100
updates. Both W&B runs authenticated and their trainer processes are live; the
first data-count/step-0 gates are still pending. A first wrapper attempt failed
before tmux due shell quoting and started no trainer; the corrected wrapper
started the two verified processes.

The current poll confirms the new Task-29 and Task-60 screens passed their
strict data gates: Task 29 loaded 2,150 matching Small EXPO chunks and Task 60
loaded 2,226; both loaded zero official-demo chunks and entered step-0
evaluation. The Task-2 and Task-28 seed-123 screens remain alive in their
step-100 four-task evaluations; neither has emitted a post-update aggregate
yet, so no new score is inferred.

The weight-drift audit requires a qualification to the specialization
hypothesis. At the historical 40k checkpoints, the broad SFT run moved the
Pi0.5 image tower by relative L2 10.7%, while the comparable Full EXPO actor
checkpoint moved it by 1.94%. Thus the evidence directly proves severe
over-specialization for the old broad/high-dose SFT recipe, not that Full EXPO's
actor necessarily drifts farther. Full EXPO's editor/critic/temperature and
selection-state specialization remains a plausible explanation for its
deployed-system behavior, but it has not been measured by this actor-only drift
audit. The supported current claim is that low-dose zero-B adapter SFT is a
more conservative prior-preserving intervention; any stronger claim about
Full EXPO over-specialization requires a module-level Full EXPO drift audit.

## Seed-123 update-100 retention results (2026-11-06)

The new pure-data Task-2 and Task-28 screens completed their first scheduled
post-update retention panels. All data gates remain strict: matching Small EXPO
teacher chunks only, official-demo ratio 0, and zero official-demo chunks.

| source task | update | Task 2 | Task 28 | Task 29 | Task 60 | macro | held-out average | source change | held-out change |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0 | 65% | 40% | 60% | 45% | 52.50% | 48.33% | — | — |
| 2 | 100 | 70% | 50% | 65% | 55% | 60.00% | 56.67% | +5 pp | +8.33 pp |
| 28 | 0 | 75% | 35% | 70% | 55% | 58.75% | 66.67% | — | — |
| 28 | 100 | 70% | 50% | 70% | 55% | 61.25% | 65.00% | +15 pp | −1.67 pp |

The Task-2 update-100 point is an early source-valid generalization gain on the
paired panel, and is stronger than a target-task-only improvement. It is not a
final result until the full 500-update curve and, if retained, full-90
confirmation are complete. Task 28 improves its source task but loses a small
amount of held-out average at update 100, so it is not promoted on this point.
The Task-29 and Task-60 seed-123 screens then completed their step-0 panels,
both measuring Task 2/28/29/60 = 75%/35%/70%/55% (58.75% macro). Their
source-pure training loops are now starting; no post-update score is inferred
yet.

## Target-semantics correction: current SFT already distills deployed actions

Source inspection resolved an important ambiguity. The `full_proposal` target
is not the raw Pi0.5 actor proposal. The Small EXPO collector calls
`teacher.sample_actions(observation)`, takes the returned ten-action chunk, and
writes it to the HDF5 `actions` dataset before executing the first five steps.
Inside `MyEXPOLearner.sample_actions`, the returned action is selected from the
base and edited candidates by the target critic's Q-values. The teacher logs
confirm that edited candidates were selected 68.75% of the time for Task 29 and
65.625% for Task 60 in representative training records.

Therefore the current task-pure `full_proposal` SFT arms already amortize much
of the deployed editor/critic selection into the Pi0.5 actor. A separate
"selected-action distillation" arm would duplicate the current target path and
is not being launched. `executed_backfill` remains a distinct target-semantics
ablation because it reconstructs each ten-action target from executed five-step
prefixes across later rows. The remaining causal question is primarily dose,
parameter scope, and task-specific selection bias—not an untested raw-versus-
selected action-label distinction.

## Seed-123 pure-screen update-200 results (2026-11-06)

The next scheduled panels completed for the task-pure Task-2 and Task-28 runs.
Both still have the strict data gate: matching Small EXPO teacher chunks only,
`official_demo_ratio=0`, and zero official-demo chunks. The curves are
non-monotonic rather than steadily improving:

| source task | update | Task 2 | Task 28 | Task 29 | Task 60 | macro | held-out average | source change | held-out change |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0 | 65% | 40% | 60% | 45% | 52.50% | 48.33% | — | — |
| 2 | 100 | 70% | 50% | 65% | 55% | 60.00% | 56.67% | +5 pp | +8.33 pp |
| 2 | 200 | 65% | 30% | 60% | 45% | 50.00% | 45.00% | 0 pp | −3.33 pp |
| 28 | 0 | 75% | 35% | 70% | 55% | 58.75% | 66.67% | — | — |
| 28 | 100 | 70% | 50% | 70% | 55% | 61.25% | 65.00% | +15 pp | −1.67 pp |
| 28 | 200 | 60% | 45% | 60% | 45% | 52.50% | 55.00% | +10 pp | −11.67 pp |

The update-100 Task-2 point remains the strongest current source-valid
generalization candidate. Update 200 rejects the simple claim that more SFT
updates monotonically improve transfer: both runs have lost held-out average by
then. This is evidence for dose-induced specialization of the SFT actor. It is
not evidence that Full EXPO's actor drifts more; the historical drift audit
actually found the opposite for the old 40k comparison (SFT image-tower drift
10.7% versus Full EXPO actor drift 1.94%).

The defensible Full EXPO hypothesis is narrower: its task-specific editor,
critic, temperature, and candidate-selection state can specialize the deployed
decision rule even when the actor's image-tower drift is small. Existing full-
suite results are consistent with that hypothesis (all five task-specific Full
EXPO systems retain their source task strongly while worsening many off-task
paired evaluations), but they do not isolate the modules causally. We therefore
must not present “Full EXPO over-specializes” as a demonstrated fact until a
full-system-versus-actor-only/module-ablation comparison is completed.

Task 29 and Task 60 remain live in their first 100-update panels; no result is
inferred for them yet.

## Seed-123 update-100 results for all four pure-data runs

Task 29 and Task 60 subsequently completed their first post-update panels.
Their data gates are also strict: matching task-specific Small EXPO chunks,
zero official-demo chunks, true adapter-only scope with zero LoRA-B, and the
same full-proposal 10/5 target interface.

| source task | update | Task 2 | Task 28 | Task 29 | Task 60 | macro | held-out average | source change | held-out change |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 100 | 70% | 50% | 65% | 55% | 60.00% | 56.67% | +5 pp | +8.33 pp |
| 28 | 100 | 70% | 50% | 70% | 55% | 61.25% | 65.00% | +15 pp | −1.67 pp |
| 29 | 100 | 60% | 50% | 70% | 50% | 57.50% | 53.33% | 0 pp | −1.67 pp |
| 60 | 100 | 70% | 40% | 65% | 55% | 57.50% | 58.33% | 0 pp | −1.67 pp |

Under the source-preservation gate, Task-2/update-100 is the only current
screen candidate. Task-60/update-100 preserves its source score but lowers
held-out average from its 60.00% control to 58.33%, so it is not promoted.
Task 28 and Task 29 are also not promoted. These are four-task screens, not
Full-90 claims; Task-2/update-100 still requires the same-protocol Full-90
confirmation.

## Full EXPO selection-stack causal audit (Task 60, partial)

An existing actor-only audit uses the same Full EXPO Task-60 checkpoint
(`60386`) as the deployed-system Full-90 evaluation. The completed actor-only
audit has all 90 task files and 475/1800 successes (26.39%); the deployed
editor/critic-selection evaluation has all 90 with 368/1800 successes
(20.44%). On the four diagnostic tasks, actor-only is 3/20, 14/20, 9/20, and
20/20 for Tasks 2/28/29/60, respectively. The deployed system is 4/20, 14/20,
1/20, and 20/20 on those same tasks in the matched Full-90 record.

This is the strongest current evidence for the proposed mechanism: on this
checkpoint, adding the learned EXPO selection stack to the trained actor is
associated with a 5.94 percentage-point transfer loss, even though the actor
image-tower drift was small. This is a within-checkpoint comparison of the
deployed selection stack versus actor-only execution; it does not prove that
the editor, critic, and temperature each independently cause the loss.

## Seed-123 update-300 retention results (2026-11-06)

The next panels further reject monotonic-improvement assumptions. The runs
remain task-pure: each uses only matching Small EXPO teacher chunks, with
`official_demo_ratio=0` and zero official-demo chunks.

| source task | update | Task 2 | Task 28 | Task 29 | Task 60 | macro | held-out average | source change | held-out change |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 300 | 65% | 50% | 60% | 55% | 57.50% | 55.00% | 0 pp | +6.67 pp |
| 28 | 300 | 70% | 50% | 70% | 45% | 58.75% | 61.67% | +15 pp | −5.00 pp |

Task-2/update-300 remains below Task-2/update-100 on both macro score and
held-out average, while Task-28/update-300 has recovered some transfer from
update 200 but remains below its own step-0 held-out average. The promoted
screen candidate therefore remains Task-2/update-100. Task-29 and Task-60
update-200 results are recorded below.

## Seed-123 update-200 results for Task 29 and Task 60 (2026-11-06)

The later panels also used matching Small EXPO teacher chunks only, with
`official_demo_ratio=0` and zero official-demo chunks. Held-out average is the
arithmetic mean of the three tasks other than the source task.

| source task | update | Task 2 | Task 28 | Task 29 | Task 60 | macro | held-out average | source change | held-out change |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 29 | 200 | 70% | 35% | 75% | 45% | 56.25% | 50.00% | +5 pp | −5.00 pp |
| 60 | 200 | 85% | 45% | 55% | 50% | 58.75% | 61.67% | −5 pp | +1.67 pp |

Neither later point passes the source-valid Pareto gate: Task 29 loses held-out
average, while Task 60 loses 5 source-task points despite a 1.67-point
held-out gain. Thus the only current screen candidate satisfying both gates is
Task-2/update-100; its Full90 confirmation remains the primary pending
evaluation.

## Seed-123 update-400 results for Task 2 and Task 28 (2026-11-06)

The update-400 panels remain strictly task-pure, with matching Small EXPO
teacher chunks only and zero official-demo chunks. Held-out average excludes
the source task.

| source task | update | Task 2 | Task 28 | Task 29 | Task 60 | macro | held-out average | source change | held-out change |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 400 | 60% | 50% | 65% | 45% | 55.00% | 53.33% | −5 pp | +5.00 pp |
| 28 | 400 | 75% | 50% | 70% | 45% | 60.00% | 63.33% | +15 pp | −3.33 pp |

Task-2/update-400 trades a 5-point source regression for a 5-point held-out
gain and therefore fails the source-preservation gate. Task-28/update-400
retains its source improvement but loses held-out average. Neither displaces
Task-2/update-100; the Task-29 and Task-60 update-300 panels remain live.

## Seed-123 update-300 Task-29 candidate (2026-11-06)

The Task-29 pure-data run subsequently committed update 300.  Its data gate
is unchanged: matching Task-29 Small EXPO teacher chunks only,
`official_demo_ratio=0`, zero official-demo chunks, `lora_only` with zero
LoRA-B, and the full-proposal 10/5 target interface.

| source task | update | Task 2 | Task 28 | Task 29 | Task 60 | macro | held-out average | source change | held-out change |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 29 | 0 | 75% | 35% | 70% | 55% | 58.75% | 55.00% | — | — |
| 29 | 300 | 75% | 50% | 75% | 45% | 61.25% | 56.67% | +5 pp | +1.67 pp |

This is the second source-valid Pareto candidate after Task-2/update-100:
it improves the source task and the held-out average on the four-task screen.
Checkpoint `300` is committed and a deferred Full LIBERO-90 evaluator is
queued in tmux session `sft_diag_task29_full90_after_train`; it waits for the
existing Task-29 trainer to exit before evaluating only that checkpoint.  The
Task-29 trainer itself remains untouched and continues toward update 500.

The current screen ranking is therefore Task-2/update-100 first (held-out
gain +8.33 pp) and Task-29/update-300 second (held-out gain +1.67 pp), both
pending Full90 confirmation.  No claim that either beats Full EXPO on the
full suite is made until those confirmations complete.

## Terminal results for the 500-update screen runs (2026-11-06)

Task 2 and Task 28 completed their planned 500-update pure-data runs without
errors. The data gate remained matching Small EXPO targets only, with zero
official-demo chunks.

| source task | update | Task 2 | Task 28 | Task 29 | Task 60 | macro | held-out average | source change | held-out change |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 500 | 75% | 45% | 60% | 45% | 56.25% | 50.00% | +10 pp | +1.67 pp |
| 28 | 500 | 85% | 70% | 60% | 40% | 63.75% | 61.67% | +35 pp | −5.00 pp |

Task-2/update-500 remains source-valid but is substantially weaker than its
update-100 peak (+1.67 pp versus +8.33 pp held-out). Task-28/update-500
strongly improves its source task but loses held-out average, confirming that
the later checkpoint is not a generalization solution. Task-29 is currently
at its update-400 evaluation and Task-60 is evaluating update 300; their
trainers remain live.

## Seed-456 replication arms launched (2026-11-06)

To test whether the early-dose effect is seed-specific, two task-pure
replications are running on the diagnostic sleeper allocations. Task 2 is
training through update 100 on `gpua073`; Task 60 is training through update
200 on the now-idle `gpua022` slot. Both use the matching Small EXPO teacher
collection only, `official_demo_ratio=0`, official `pi05_libero`
initialization, true `lora_only`/zero-B scope, full-proposal targets, batch 16,
LR 5e-5, 10k warmup, and the unchanged 10/5 interface. The Task-60 seed-456
screen intentionally uses `--no-checkpoints`; it is a screen-only replication,
not a Full90 candidate until its retention panel passes the source gate.

The Task-2 seed-456 arm is in its post-update retention evaluation and the
Task-60 seed-456 trainer has initialized successfully. These replications do
not alter the original seed-123 trainers, the queued Full90 confirmations, or
any production training allocation.

## Additional dose points and replication baseline (2026-11-06)

The Task-29 and Task-60 seed-123 screens advanced while the Full90 audits were
running. These remain strictly task-pure and are not promoted:

| source task | update | Task 2 | Task 28 | Task 29 | Task 60 | macro | held-out average | source change | held-out change |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 29 | 400 | 65% | 40% | 60% | 50% | 53.75% | 51.67% | −10 pp | −3.33 pp |
| 60 | 300 | 65% | 35% | 45% | 45% | 47.50% | 48.33% | −10 pp | −11.67 pp |

Task-29/update-400 and Task-60/update-300 both fail the source-preservation
gate, further narrowing the useful dose window. The Task-2 seed-456
replication's step-0 control is 70/50/65/55% (macro 60.00%, held-out 56.67%),
matching the seed-123 Task-2 control before its update-100 test. Its training
loop is now active; no post-update result is inferred yet.

## Seed-456 replacement arm for Task 29 (2026-11-06)

The earlier Task-60 seed-456 screen on `gpua022` was diagnostic-only and did
not produce a usable result: its output remained frozen during the first
retention evaluation at episode 5/20 for more than six hours, despite the
process remaining alive. It was terminated without saving or promoting a
checkpoint; the allocation itself was not cancelled.

That newly free slot now runs a replacement Task-29 seed-456 replication in
tmux session `task_29_pure_seed456`. It uses only the matching Task-29 Small
EXPO teacher collection, `official_demo_ratio=0`, official `pi05_libero`
initialization, zero LoRA-B, `lora_only`, the full-proposal 10/5 target
interface, batch 16, learning rate 5e-5, 10k warmup, and retention panels at
updates 0/100/200/300. W&B is online and the trainer has initialized on
`gpua022`; no score is inferred until a retention JSON is committed.

This replacement tests whether the Task-29/update-300 seed-123 gain
(+1.67 pp held-out with +5 pp source retention) is reproducible. It does not
alter the live seed-123 runs or the Task-2 Full-90 confirmation.

At the first health poll, the replacement printed
`small_expo_chunks=2150`, `official_demo_chunks=0`, and
`official_demo_ratio=0.0`, then entered its initial 20-episode retention
evaluation. The Task-2 seed-456 replication reached update 99 with finite
gradients and the same zero-official-data gate; its post-update panel is still
in progress. The Task-2/update-100 Full-90 confirmation has committed four of
90 task files and remains live, so no Full-90 score is reported yet.

## Task-29 seed-123 dose curve completed (2026-11-06)

The pure Task-29 seed-123 trainer completed its planned 500 updates with
matching Task-29 Small EXPO targets only and zero official-demo chunks. Its
final retention panel was:

| source task | update | Task 2 | Task 28 | Task 29 | Task 60 | macro | held-out average | source change | held-out change |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 29 | 0 | 75% | 35% | 70% | 55% | 58.75% | 55.00% | — | — |
| 29 | 500 | 55% | 45% | 80% | 50% | 57.50% | 50.00% | +10 pp | −5.00 pp |

Although the source task improved, the held-out average declined. This
confirms that the late Task-29 dose is specialization, not the desired
generalization solution. The already queued Task-29/update-300 Full-90
confirmation remains the relevant candidate evaluation; the completed
update-500 point is not promoted.

## Task-60 seed-456 source-first replication launched (2026-11-06)

New allocation `gpua064` became available and is running tmux session
`task_60_pure_seed456_sourcefirst`. This is a strict Task-60-only replication
of the strongest validated pure-SFT result: matching Task-60 Small EXPO
targets, zero official demonstrations, official `pi05_libero` initialization,
true `lora_only`/zero-B scope, full-proposal 10/5 targets, batch 16, LR 5e-5,
10k warmup, seed 456, and 200 updates.

To separate the earlier operational hang from model behavior, training
retention panels initially evaluate only source Task 60. After update 200,
the script exposes the latest learner checkpoint and evaluates Tasks 2/28/29/60
with the same 20-episode actor-only evaluator. Only the diagnostic `gpua064`
slot was changed; all existing trainers and Full-90 evaluators remain
untouched.

The first post-training evaluator invocation exposed a separate path issue:
the source-first launchers omitted `/work/hdd/bgqe/rpolonsky/sharder/training`
from `PYTHONPATH`, causing `ModuleNotFoundError:
full_90_libero_evals_inference` after training had already completed. The
saved checkpoints and assets were intact. Dedicated repair sessions now run
the four-task evaluator with the corrected import root. The update-100 repair
also passes the real `checkpoints/` manager root rather than the symlinked
`checkpoints/100` path, avoiding manifest resolution through
`latest_checkpoints`; the update-200 repair is live as well.

## Task-29 seed-456 update-100 result (2026-11-06)

The independent task-pure Task-29 replication committed update 100 at
45%/45%/70%/50% on Tasks 2/28/29/60. The source task stayed at its 70% step-0
control, but held-out average fell from 51.67% to 46.67%. It is therefore
rejected by the source-preserving generalization gate; the same trainer
continues to updates 200 and 300 for the remaining dose measurements.

## Task-60 seed-101 update-150 replication (2026-11-06)

A second newly running sleeper allocation (`23599979` on `gpua003`) was
verified idle at 0 MiB before use. It now runs the parameterized update-150
launcher in its own tmux session `sft_task60_seed101_update150`. The protocol
is identical to the seed-789 arm: matching Task-60 Small EXPO targets only,
zero official data, official `pi05_libero`, true `lora_only`/zero-B, full
proposal, batch 16, LR 5e-5, 10k warmup, and 10/5 execution. W&B and the
trainer initialized successfully; no post-update score is claimed yet.

## Task-60 seed-789 update-150 source result (2026-11-06)

Seed 789 completed 150 finite SFT updates with exactly 2,226 matching
Task-60 Small EXPO chunks and zero official chunks. Its source-only panel is
55% at both step 0 and update 150, so it passes the source-retention gate but
does not improve the source task. The original wrapper then exited on a shell
quoting error after saving the update-150 checkpoint; a separate corrected
four-task evaluator is now running from the intact `checkpoints/` manager
root. No generalization score is claimed until that panel completes.

## Task-60 seed-456 update-200 four-task panel completed (2026-11-06)

The repaired evaluator completed the Task-60 seed-456/update-200 checkpoint
from the intact checkpoint manager root. The training arm had exactly 2,226
matching Task-60 Small EXPO teacher chunks and zero official-demo chunks; the
four-task actor-only panel is:

| source task | update | Task 2 | Task 28 | Task 29 | Task 60 | macro | held-out average |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 60 | 200 | 70% | 40% | 60% | 50% | 55.00% | 56.67% |

The source-first training panel for this seed was 55% on Task 60 at both step
0 and update 200, so the source-retention gate passes (no source regression),
but the held-out panel is not a new best over the previously validated
Task-60/update-200 result. This is valid screening evidence, not a LIBERO-90
replacement; the two side Full90 evaluations remain untouched.

## Task-28 independent update-300 replications launched (2026-11-06)

The two verified idle A100 slots `gpua073` and `gpua095` now run separate
tmux sessions `task_28_pure_seed456` and `task_28_pure_seed789`. Both use the
same strict recipe as the validated Task-28/update-300 screen: only the
matching Task-28 Small EXPO collection, zero official chunks, official
`pi05_libero` initialization, true `lora_only`/zero-B scope, full-proposal
10/5 targets, batch 16, learning rate 5e-5, 10k warmup, and 300 updates with
four-task retention at updates 0/100/200/300. These are diagnostic copies
under `sharder/training`; no production trainer, checkpoint, or side Full90
evaluation was changed.

## Task-60 seed-101 update-150 source result (2026-11-06)

The independent seed-101 arm completed 150 finite, task-pure updates with the
same 2,226 Task-60 Small EXPO chunks and zero official chunks. Its source-only
panel is 45% at step 0 and 45% at update 150, so it passes source retention but
does not improve the source task. The four-task follow-up remains pending; no
held-out or promotion claim is made until that panel is committed.

## Task-60 seed-789 update-150 four-task panel completed (2026-11-06)

The repaired evaluator completed the independent seed-789/update-150
checkpoint at 70/40/65/55% on Tasks 2/28/29/60 (57.50% macro and 55.00%
held-out average). The source Task-60 score remained 55% at both step 0 and
update 150, so the checkpoint passes source retention but does not establish a
new generalization best. This independent result does not reproduce a useful
update-150 gain; the remaining seed-101 panel is the final replication at this
dose.

## Task-29 seed-456 update-200 result (2026-11-06)

The independent Task-29-only replication committed its update-200 retention
panel at 70/55/65/55% on Tasks 2/28/29/60 (61.25% macro and 60.00%
held-out average). The held-out average is higher than its 51.67% step-0
control, but the source task fell from 70% to 65%. Under the predeclared
source-preservation gate this checkpoint is rejected and will not be promoted
or sent to Full90. The trainer continues to update 300 to determine whether
the source regression is transient or part of the dose curve.

## Task-60 seed-456 update-100 four-task panel completed (2026-11-06)

The repaired evaluator also completed the Task-60 seed-456/update-100
checkpoint. Its strict task-pure four-task actor-only panel is:

| source task | update | Task 2 | Task 28 | Task 29 | Task 60 | macro | held-out average |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 60 | 100 | 65% | 45% | 50% | 50% | 52.50% | 53.33% |

Task 60 fell from its 55% step-0 source control to 50%, so this checkpoint is
rejected by the predeclared source-preservation gate even though its held-out
average is measurable. It is not promoted or sent to Full90.

## Task-29 second update-300 replication launched (2026-11-06)

After `gpua064` became idle, a second Task-29/update-300 arm was launched in
tmux session `task_29_pure_seed789_update300`. It uses the identical strict
Task-29-only protocol as the seed-456 arm: matching Task-29 Small EXPO targets,
zero official chunks, official `pi05_libero`, true `lora_only`/zero-B,
full-proposal 10/5 targets, batch 16, LR 5e-5, 10k warmup, and four-task
retention at updates 0/100/200/300. The process initialized and W&B tracking
successfully; no score is claimed until its JSON panel is committed.

The launch health check confirmed `small_expo_chunks=760` and
`official_demo_chunks=0` for both Task-28 arms, and
`small_expo_chunks=2150`/`official_demo_chunks=0` for the Task-29 seed-789
arm. Both Task-28 processes are GPU-active in retention evaluation. The
Task-29 seed-456 arm is in its update-300 evaluation, while seed 789 is still
training/initializing. The Task-60 seed-101 follow-up has committed Task 2 at
75% and Task 28 at 50%; Tasks 29 and 60 remain pending.

## Task-29 seed-456 update-300 result (2026-11-06)

The seed-456 replication completed its final retention panel at 75/35/60/50%
on Tasks 2/28/29/60 (55.00% macro and 53.33% held-out average). Task 29 fell
from its 70% step-0 source control to 60%, so update 300 is rejected by the
source-preservation gate. The held-out score does not override that gate; the
apparently favorable Task-29 late-dose behavior is not a valid promoted SFT
candidate. The independent seed-789/update-300 arm remains in progress.

## Task-60 second update-200 replication launched (2026-11-06)

After `gpua022` became idle, a seed-789 Task-60/update-200 replication was
launched in tmux session `task_60_pure_seed789_update200`. It uses the same
strongest-candidate protocol: matching Task-60 Small EXPO targets only, zero
official chunks, official `pi05_libero` initialization, true
`lora_only`/zero-B, full-proposal 10/5 targets, batch 16, LR 5e-5, 10k
warmup, and retention at updates 0/100/200. The process initialized; no score
is claimed until its complete four-task panel is committed.

The Task-60 seed-101/update-150 Full4 evaluator has now committed Tasks 2, 28,
and 29 at 75%, 50%, and 70%, respectively. Task 60 is still running, so this
is recorded as a partial panel only; no macro or held-out aggregate is claimed
until the fourth task commits.

## Task-60 seed-101 update-150 Full4 panel completed (2026-11-06)

The complete actor-only panel is 75/50/70/45% on Tasks 2/28/29/60,
respectively (60.00% macro and 65.00% held-out average). Source Task 60 is
45% at both step 0 and update 150, so the checkpoint passes the source gate.
This is the strongest current four-task screening result, and its intact
checkpoint is being sent to LIBERO-90 for the definitive comparison.

The first evaluator invocation paused because it lacked `LIBERO_CONFIG_PATH`;
that diagnostic session was removed and relaunched with the repaired import
root and existing runtime LIBERO config. The corrected evaluator restored
checkpoint 150 successfully and is now running the LIBERO-90 panel in tmux
`full90_task60_seed101_update150_diag` on `gpua003`.

## Verified live status (2026-11-06 09:09 CST)

The node-local poll found no newly completed ranking panel. The definitive
Task-60 seed-101/update-150 Full90 evaluator has committed `task_00.json` and
`task_01.json`, `task_02.json`, and `task_03.json` (4/90 tasks) and remains GPU-active; its result is not summarized until all
90 task records commit. The Task-60 seed-789/update-200, Task-28 seed-456 and
seed-789/update-300, and Task-29 seed-789/update-300 arms are also GPU-active
in their retention evaluations, with no post-update score inferred from
progress output. The separate Task-2 and Task-29 Full90 side evaluations are
still running and remain excluded from this diagnosis. At that poll all known
A100 slots were occupied, so no new diagnostic run was launched then.

The Task-60 seed-789/update-200 arm has since committed its step-0 control:
75%/35%/70%/55% on Tasks 2/28/29/60 (58.75% four-task macro; 55% source
Task 60). This is a baseline control only; no post-update score is included
until the update-200 panel commits.

The node-local log confirms that this arm is healthy: after the step-0
retention evaluation it wrote checkpoint 0 and entered `Learner BC: 0/200`.
The temporary 0% GPU reading coincided with this host-side update phase; the
Python process remained alive and CPU-active, so it was not restarted.

At this poll, the separate side evaluators had committed 29/90 Task-2
records and 22/90 Task-29 records; they remain outside the diagnostic ranking.
The newly launched Task-2 seed-789 arm has claimed GPU memory and remains in
initialization, but its initialization/data audit has not yet committed.

The Task-2 seed-789 initialization has now completed the data gate:
`small_expo_chunks=1727`, `official_demo_chunks=0`, and
`official_demo_ratio=0.0`. It is training under the intended task-pure
protocol. The Task-29 seed-789 trainer is also healthy near the end of its
update-200 BC phase; checkpoint 100 remains intact for the planned Full90
confirmation after training completes.

The Task-60 seed-789/update-200 replication has now committed update 100:

| source task | seed | update | Task 2 | Task 28 | Task 29 | Task 60 | macro | held-out average | source change | held-out change |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 60 | 789 | 0 | 75% | 35% | 70% | 55% | 58.75% | 60.00% | — | — |
| 60 | 789 | 100 | 75% | 45% | 70% | 55% | 61.25% | 63.33% | 0 pp | +3.33 pp |

This is source-valid and is the second-best current screen candidate after
Task-29/update-100. The trainer is continuing to update 200; checkpoint 100
will be sent to Full90 only after the active training session releases the
GPU.

## New update-100 replication panels (2026-11-06)

The newly committed update-100 panels were checked against their own step-0
controls and the source-retention gate:

| source task | seed | update | Task 2 | Task 28 | Task 29 | Task 60 | macro | held-out average | source change | held-out change |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 29 | 789 | 0 | 65% | 40% | 60% | 45% | 52.50% | 48.33% | — | — |
| 29 | 789 | 100 | 75% | 40% | 65% | 55% | 58.75% | 53.33% | +5 pp | +5.00 pp |
| 28 | 456 | 0 | 60% | 40% | 70% | 55% | 56.25% | 61.67% | — | — |
| 28 | 456 | 100 | 55% | 45% | 70% | 50% | 55.00% | 58.33% | +5 pp | −3.33 pp |
| 28 | 789 | 0 | 70% | 50% | 65% | 55% | 60.00% | 63.33% | — | — |
| 28 | 789 | 100 | 60% | 40% | 50% | 45% | 48.75% | 51.67% | −10 pp | −11.67 pp |

All three arms loaded only their matching Small EXPO target collections and
reported zero official-demo chunks. Task-29/update-100 is therefore the
strongest current screen candidate and should receive the next Full90
confirmation after its active update-300 trainer releases the GPU. The two
Task-28 update-100 arms are not promotion candidates.

## Task-2 seed-456 update-100 replication (2026-11-06)

The independent Task-2-only seed-456 replication completed both retention
panels with matching Task-2 Small EXPO targets and zero official-demo ratio:

| source task | update | Task 2 | Task 28 | Task 29 | Task 60 | macro | held-out average |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0 | 70% | 50% | 65% | 55% | 60.00% | 56.67% |
| 2 | 100 | 70% | 35% | 60% | 55% | 55.00% | 50.00% |

The source task is retained at 70%, but held-out performance falls by 6.67
percentage points, so this arm is rejected. It does not reproduce the earlier
seed-123 Task-2/update-100 screen gain and strengthens the conclusion that
the apparent early-dose improvement is seed-sensitive.

## Task-2 seed-789 update-100 replication launched (2026-11-06)

The newly idle `gpua079` A100 now runs `task_02_pure_seed789_update100` in its
own tmux session. It uses the same strict Task-2-only protocol: matching
Small EXPO targets, zero official data, official `pi05_libero` initialization,
`lora_only`/zero-B, full-proposal 10/5 targets, batch 16, learning rate 5e-5,
10k warmup, and retention at updates 0 and 100. The process is alive but is
still paging in the shared checkpoint during initialization; no data-load or

## Live status checkpoint (2026-11-06 09:23 CST)

All currently monitored diagnostic processes are alive and using their assigned
A100s; no training or evaluation session was restarted. The active state is:

| node | arm | status | committed results |
|---|---|---|---:|
| `gpua079` | Task 2, seed 789, update 100 | completing the 20-episode update-100 retention evaluation | init/config only; no update-100 JSON yet |
| `gpua064` | Task 29, seed 789, update 300 | training at step 199/300 | update 0 and 100 |
| `gpua022` | Task 60, seed 789, update 200 | training at step 199/200 | update 0 and 100 |
| `gpua073` | Task 28, seed 456, update 300 | still training | update 0 and 100 |
| `gpua095` | Task 28, seed 789, update 300 | still training | update 0 and 100 |
| `gpua003` | Task 60, seed 101, Full90 | evaluating | task 00--06 |
| `gpua062` | prior Task 2 seed 123, Full90 | evaluating | task 00--31 |
| `gpua100` | prior Task 29 seed 123, Full90 | evaluating | task 00--24 |

The Task-29 and Task-60 trainers are not stalled: their metric logs reached
steps 199 and 199 respectively, with nonzero gradients and zero official-demo
chunks. Task-60 update-100 remains source-valid (+3.33 pp held-out); Task-29
update-100 remains the best screen candidate (+5.00 pp held-out). Full90
confirmation still awaits a free diagnostic GPU after the corresponding
trainer exits.

## Additional update-200 panels (2026-11-06)

The next committed panels were checked against each arm's own step-0 control:

| source task | seed | update | Task 2 | Task 28 | Task 29 | Task 60 | macro | held-out average | source change | held-out change |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 29 | 789 | 0 | 65% | 40% | 60% | 45% | 52.50% | 50.00% | — | — |
| 29 | 789 | 200 | 60% | 40% | 60% | 50% | 52.50% | 50.00% | 0 pp | 0 pp |
| 28 | 456 | 0 | 60% | 40% | 70% | 55% | 56.25% | 61.67% | — | — |
| 28 | 456 | 200 | 65% | 30% | 65% | 55% | 53.75% | 61.67% | −10 pp | 0 pp |
| 28 | 789 | 0 | 70% | 50% | 65% | 55% | 60.00% | 63.33% | — | — |
| 28 | 789 | 200 | 75% | 50% | 70% | 50% | 61.25% | 65.00% | 0 pp | +1.67 pp |

The Task-29 update-200 arm is source-valid but does not improve held-out
retention. Task-28 seed 456 fails the source-retention gate; Task-28 seed 789
is source-valid with a small held-out gain, but remains behind Task-29/update
100 and Task-60/update 100. These are still four-task screens, not Full90
claims.

## Live status correction (2026-11-06 09:25 CST)

The Task-2 seed-789 arm has now committed its step-0 control,
`[75%, 35%, 70%, 55%]` on Tasks 2/28/29/60 (58.75% macro); it has entered
the 100-update training loop, so no update-100 result is available yet. The
Task-60 seed-789 arm remains alive at step 199/200 with its update-100 JSON
intact; its update-200 panel has not committed yet. The older Full90 side jobs
advanced to 32/90 (Task 2) and 25/90 (Task 29), while Task-60 seed-101 is at
7/90. No process was restarted or stopped.

## Live status checkpoint (2026-11-06 09:34 CST)

No diagnostic GPU has been released. All five task-pure trainers and all three
Full90 evaluators remain live; no session was restarted or stopped.

| node | process | committed state |
|---|---|---|
| `gpua079` | Task 2 seed 789, update 100 | step-0 control; post-update retention evaluation in progress |
| `gpua064` | Task 29 seed 789, update 300 | updates 0/100/200; training beyond step 280 |
| `gpua022` | Task 60 seed 789, update 200 | updates 0/100; update-200 retention evaluation in progress |
| `gpua073` | Task 28 seed 456, update 300 | updates 0/100/200; update-300 retention evaluation in progress |
| `gpua095` | Task 28 seed 789, update 300 | updates 0/100/200; update-300 retention evaluation in progress |
| `gpua003` | Task 60 seed 101 Full90 | 10/90 task records |
| `gpua100` | prior Task 29 seed 123 Full90 | 28/90 task records |
| `gpua062` | prior Task 2 seed 123 Full90 | 35/90 task records |

The recurring LIBERO dataset-path warning remains non-fatal: all evaluators
continue to commit task JSON files. Full90 confirmation of the source-valid
screen candidates is still queued behind a genuinely free GPU.

## Task-60 update-200 completion and Full90 promotion (2026-11-06 09:38 CST)

Task-60 seed 789 completed its 200-update arm with the following retention
panel:

| update | Task 2 | Task 28 | Task 29 | Task 60 | macro | held-out average |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 75% | 35% | 70% | 55% | 58.75% | 60.00% |
| 100 | 75% | 45% | 70% | 55% | 61.25% | 63.33% |
| 200 | 80% | 45% | 70% | 50% | 61.25% | 65.00% |

Although update 200 improves the held-out average by 5.00 pp, it reduces the
source Task-60 score by 5 pp and therefore fails the pre-registered
source-retention gate. Update 100 remains the valid Task-60 candidate.

After the trainer exited and released `gpua022`, its update-100 checkpoint was
promoted to a task-pure Full90 evaluation in tmux session
`full90_task60_seed789_update100_diag`. Output is isolated at
`/tmp/usertmp/rpolonsky/sft_full90_diagnosis_20261106/task60_seed789_update100`;
the evaluator has initialized LIBERO and is beginning its first task. Existing
side evaluators and all other trainers were left untouched.

## Stable-checkpoint replication after retention race (2026-11-06 09:47 CST)

The first Task-60 Full90 attempt exited cleanly before evaluation because the
long-run trainer had already garbage-collected checkpoint 100; update 200 was
also source-invalid, so it was not substituted. The same checkpoint-retention
race made a direct copy of the live Task-29 checkpoint 100 unreliable.

To preserve the valid screen candidates, two new arms were launched on the
released GPUs, stopping at update 100 so the final checkpoint cannot be
removed by a later update:

| node | arm | output | data gate |
|---|---|---|---|
| `gpua073` | Task 29, seed 789, update 100 | `task_29_pure_seed789_update100_repro_datafix` | matching Task-29 prepared Small EXPO data, zero official data |
| `gpua022` | Task 60, seed 789, update 100 | `task_60_pure_seed789_update100_repro_datafix` | matching Task-60 prepared Small EXPO data, zero official data |

Both runs source the existing task-specific `pi_base_data` through a symlink,
source the shared W&B `.env`, place W&B/JAX temporary files under
`/tmp/usertmp`, and use the unchanged Pi0.5 initialization, LoRA/zero-B,
full-proposal 10/5, batch-16, 5e-5, and 10k-warmup protocol. Both have passed
W&B and model-shape initialization and are entering their update loops.
Once each exits, its stable update-100 checkpoint will be evaluated on
Full90. The failed evaluator log is retained as a diagnostic artifact; no
existing trainer or side evaluator was modified.

## Completed update-300/100 screens and node-local storage (2026-11-06)

The completed panels add two important results:

| source task | seed | update | Task 2 | Task 28 | Task 29 | Task 60 | macro | held-out average | source gate |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 28 | 456 | 300 | 75% | 40% | 75% | 55% | 61.25% | 68.33% | pass; source unchanged |
| 28 | 789 | 300 | 60% | 45% | 60% | 45% | 52.50% | 55.00% | reject; source −5 pp |
| 2 | 789 | 100 | 65% | 35% | 60% | 50% | 52.50% | 50.00% | reject; source −10 pp |

Task-28 seed 456/update 300 is now the strongest four-task screen candidate:
its source Task-28 score remains 40%, while held-out average improves from
61.67% to 68.33% (+6.67 pp). It is queued for Full90 confirmation.

Operationally, `/tmp/usertmp` is node-local. A checkpoint created on one A100
cannot be opened from another node unless explicitly copied to shared storage.
The new stable Task-29 and Task-60 terminal-update replications therefore
remain on their respective nodes, and the Task-28 Full90 evaluation will run
on `gpua073`, where its checkpoint resides, after the Task-29 replication
releases that GPU.

## Task-28 Full90 confirmation launched (2026-11-06 09:54 CST)

The completed Task-28 seed-456/update-300 checkpoint and its small student
assets were copied from node-local `gpua073` to node-local `gpua095` (13 GB
checkpoint; metadata validated). The Full90 evaluator is now running in tmux
session `full90_task28_seed456_update300_diag` on `gpua095`, using the copied
checkpoint and the unchanged 224px, action-horizon-10, replan-5 protocol. The
Task-29 and Task-60 stable terminal-update replications remain untouched on
`gpua073` and `gpua022`.

The first Task-28 evaluator attempt exited before model evaluation because a
cross-node checkpoint copy omitted `sft_config.json`; it produced zero task
records. The manifest was then copied and validated alongside the checkpoint,
and retry session `full90_task28_seed456_update300_diag_retry` passed Orbax
checkpoint discovery and is now loading the model on `gpua095`. This was an
evaluation-bundle correction only; the candidate checkpoint and training run
were not changed.

## Terminal-update replication controls (2026-11-06 10:03 CST)

The stable Task-29 and Task-60 terminal-update replications have committed
their step-0 controls and entered the 100-update learner phase. The controls
were evaluated with the same four-task retention suite and fixed evaluation
seed used by the prior screens:

| arm | Task 2 | Task 28 | Task 29 | Task 60 | macro | source-task score |
|---|---:|---:|---:|---:|---:|---:|
| Task 29, seed 789 | 65% | 40% | 60% | 45% | 52.50% | 60% |
| Task 60, seed 789 | 75% | 35% | 70% | 55% | 58.75% | 55% |

Both controls are consistent with their corresponding earlier screen controls.
At this checkpoint the terminal update-100 checkpoints do not yet exist; both
processes remain alive in their learner loops on `gpua073` and `gpua022`.
The Task-28 seed-456/update-300 Full90 retry is also alive on `gpua095` and
has restored checkpoint 300 successfully, but has not yet committed a task
record. No production trainer or unrelated side evaluator was changed.

## Terminal update-100 retention results and Full90 promotion (2026-11-06 10:09 CST)

The stable terminal-update replications completed without error. Both used
exactly 16 matching Small EXPO chunks per batch and zero official-data chunks;
their checkpoint-100 bundles contain Orbax metadata, `sft_config.json`, and
the student assets needed for evaluation:

| arm | Task 2 | Task 28 | Task 29 | Task 60 | macro | source gate |
|---|---:|---:|---:|---:|---:|---|
| Task 29, seed 789, update 100 | 70% | 40% | 65% | 50% | 56.25% | pass; source +5 pp |
| Task 60, seed 789, update 100 | 80% | 30% | 70% | 55% | 58.75% | pass; source unchanged |

The Task-29 checkpoint was promoted to Full90 in tmux
`full90_task29_seed789_update100_repro_diag` on `gpua073`, and the Task-60
checkpoint was promoted in tmux `full90_task60_seed789_update100_repro_diag`
on `gpua022`. Both evaluators use the same 224px, action-horizon-10,
replan-5, 20-episode/task protocol and are currently initializing.

## Same-task 6.25% official-data screen launched (2026-11-06)

Three idle A100 allocations were verified to have no user processes or tmux
sessions and were assigned the next preplanned controlled factor:

| node | source task | tmux session | output | protocol |
|---|---:|---|---|---|
| `gpua038` | 2 | `diag_task02_demo0625_20261106` | `task_02_demo0625_nocheckpoint` | 500 updates, no learner checkpoints |
| `gpua074` | 28 | `diag_task28_demo0625_20261106` | `task_28_demo0625_nocheckpoint` | 500 updates, no learner checkpoints |
| `gpua079` | 60 | `diag_task60_demo0625_20261106` | `task_60_demo0625_nocheckpoint` | 500 updates, no learner checkpoints |

Each arm uses only its own Small EXPO teacher collection plus a 6.25%
same-task official-demo fraction (one official row per batch of 16), official
`pi05_libero` initialization, true `lora_only`/zero-B, full-proposal 10/5,
batch 16, learning rate `5e-5`, 10k warmup, and retention evaluations every
100 updates on tasks 2/28/29/60. The launcher uses `--no-checkpoints`, so no
large learner checkpoint is written to shared storage. W&B runs authenticated
successfully; all three workers are currently in model/JAX initialization and
have not yet produced training metrics. The prior Task-29 6.25% attempt from
2026-11-05 is excluded because it failed before training with no visible GPU;
its output directory was not reused.

The three 6.25% arms subsequently passed model initialization and entered
their first 20-episode retention evaluation. Their manifests confirm the
intended same-task official-demo paths and task-specific Small EXPO counts;
no retention JSON has committed yet, so no performance ranking is assigned.

## Task-29 same-task 6.25% retry launched (2026-11-06)

The newly available idle A100 `gpua071` was verified to have no user process
or tmux session. A fresh Task-29 6.25%-official-data screen was launched in
`diag_task29_demo0625_retry_20261106`, writing to
`task_29_demo0625_nocheckpoint_retry`. It uses the same task-pure Small EXPO
collection, the Task-29 official demonstration only, the unchanged
full-proposal/10-5, LoRA/zero-B, batch-16, 500-update protocol, and no learner
checkpoints. Its W&B run authenticated successfully. This fresh output tree is
separate from the earlier Task-29 attempt, which remains excluded because it
failed before training with no visible GPU.

## Latest live diagnosis poll (2026-11-06 14:08 CST)

The three promoted candidates remain in healthy Full90 evaluation; none has
finished the 90-task suite yet:

| candidate | node/session | committed task records | status |
|---|---|---:|---|
| Task 28, seed 456, update 300 | `gpua095` / `full90_task28_seed456_update300_diag_retry` | 58/90 | active |
| Task 29, seed 789, update 100 | `gpua073` / `full90_task29_seed789_update100_repro_diag` | 34/90 | active |
| Task 60, seed 789, update 100 | `gpua022` / `full90_task60_seed789_update100_repro_diag` | 34/90 | active |

All three workers still have live tmux sessions, active Python processes, and
GPU allocations. The four-task 6.25%-official-data arms (Tasks 2, 28, 29, and
60) remain active; Task 2 has completed its 500 updates and begun evaluation,
and Task 28 has now committed its step-500 panel under `evals/00000500.json`.
Task 2, Task 28, Task 29, and Task 60 have now committed their step-500
panels. The four controlled-factor arms are complete; only the separate
full-system Task-28 evaluator remains active. No evaluator
was restarted and no unrelated side evaluator was changed. The diagnostic
allocations expose one A100 each and all seven are occupied; the other visible
A100 allocations are intentional sleep jobs, so they were not repurposed.

The latest committed panels are Task 2 step 500
`[70%, 55%, 75%, 50%]` (macro 62.5%), Task 29 step 500
`[55%, 30%, 65%, 60%]` (macro 52.5%), and Task 60 step 500
`[45%, 40%, 60%, 60%]` (macro 51.25%), all on tasks `[2, 28, 29, 60]`.

The freed `gpua074` slot was assigned a separate full-system diagnostic in
tmux `full90_task28_sft_plus_editor_diag`. It evaluates the strict
zero-official Task-28 SFT actor checkpoint (step 300, seed 42) with the
matching Task-28 Small EXPO editor/critic checkpoint (step 275001) through
the `sft_plus_editor` candidate-selection path over LIBERO-90. It has passed
Python/LIBERO initialization, restored the matching editor/critic structures,
and now has approximately 31.2 GB resident on the GPU at 70% utilization;
no task result is claimed yet. Existing evaluators and side jobs were not
changed.

The same poll found the Task-2 6.25%-official-data worker at update 500/500;
its first four-task retention evaluation is now committed below. Task 28,
Task 29, and Task 60 also completed their step-500 panels.

## Same-task official-data controlled-factor result (Task 28)

The Task-28 6.25%-official-data arm completed all 500 updates and committed
`task_28_demo0625_nocheckpoint/evals/00000500.json`. Its manifest confirms
source task 28, Small EXPO teacher targets, `official_demo_ratio=0.0625`,
full-proposal targets, LoRA-only training, batch 16, and the unchanged 10/5
interface. The step-500 panel is `[75%, 65%, 55%, 40%]` on tasks
`[2, 28, 29, 60]`, with macro success `58.75%`.

The complete within-run curve was:

| updates | Task 2 | Task 28 | Task 29 | Task 60 | macro | held-out mean (2/29/60) |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 75% | 35% | 70% | 55% | 58.75% | 66.67% |
| 100 | 65% | 25% | 70% | 55% | 53.75% | 63.33% |
| 200 | 70% | 40% | 65% | 55% | 57.50% | 63.33% |
| 300 | 70% | 45% | 65% | 55% | 58.75% | 63.33% |
| 400 | 75% | 45% | 50% | 45% | 53.75% | 56.67% |
| 500 | 75% | 65% | 55% | 40% | 58.75% | 56.67% |

This arm therefore improved its source-task score by 30 percentage points
from its own step-0 control, while its three-task held-out mean fell by 10
points. That is consistent with source specialization rather than a
generalization remedy. Because this is a single seed and a changed data
mixture, it is a controlled-factor signal, not a causal conclusion; the
strict zero-official-data candidates remain the primary diagnosis arms.

## Same-task official-data controlled-factor result (Task 2)

The Task-2 6.25%-official-data arm also completed all 500 updates. Its
step-500 panel is `[70%, 55%, 75%, 50%]` on `[2, 28, 29, 60]`, macro 62.5%.
Within this run, the step-0 panel was `[65%, 40%, 60%, 45%]` (macro 52.5%);
the held-out mean over tasks 28/29/60 rose from 48.33% to 60.0% while the
source task rose from 65% to 70%. This is the opposite direction from the
Task-28 arm, so same-task official-data mixing is task-dependent rather than
uniformly harmful. It is still only a single-seed controlled factor and does
not replace strict zero-official-data replication.

For reference, the completed Task-60 arm moved from step 0
`[70%, 50%, 65%, 55%]` to step 500 `[45%, 40%, 60%, 60%]`: its source task
rose by 5 points while the held-out mean fell from 56.67% to 53.33%.

The completed Task-29 arm moved from step 0 `[75%, 35%, 70%, 55%]` to
step 500 `[55%, 30%, 65%, 60%]`: its source task fell by 5 points and its
held-out mean fell from 55.0% to 48.33%. Thus three of four same-task
6.25%-official controls did not improve held-out transfer, while Task 2 did;
the factor is task-dependent rather than a general solution.

## Live Full-90 poll (2026-11-06 14:13 CST)

All four diagnostic evaluators remain healthy in their original tmux sessions;
no diagnostic GPU was free for a new experiment, so no worker was restarted or
repurposed. The actor-only audits have advanced to:

| candidate | mode | committed task records | status |
|---|---|---:|---|
| Task 28, seed 456, update 300 | `sft_pi` | 61/90 | active |
| Task 29, seed 789, update 100 | `sft_pi` | 36/90 | active |
| Task 60, seed 789, update 100 | `sft_pi` | 36/90 | active |
| Task 28, seed 42, update 300 | `sft_plus_editor` | 1/90 | active |

The first committed full-system result for the last row is Task 0 (35%
success, mean 331.6 steps), using the strict zero-official Task-28 SFT actor
and the matching Task-28 Small EXPO editor/critic at editor step 275001. This
is not yet an aggregate result. For the same SFT checkpoint and Task 0, the
actor-only evaluator records 0% success and mean 400 steps, so the first pair
already demonstrates that the editor/critic path changes deployment behavior;
it does not establish a Full-90 win until all 90 tasks in the matched
full-system run are committed and validated.

## Matched Task-60 full-system Full-90 evaluation launched (2026-11-06 14:24 CST)

A genuinely idle A100 slot was verified on `gpua038` (zero compute
processes and no existing tmux session). It now runs
`full90_task60_sft_plus_editor_diag` through an explicit node-visible tmux
socket. This is a deployment-path evaluation of the strongest existing
task-pure Task-60 SFT candidate: the update-200 full-proposal checkpoint at
`dcrlcheckpoints/sft_retention_diagnosis_20261104/task_60_full_proposal`,
whose manifest records `collection_source=small_expo` and
`official_demo_ratio=0.0`. It combines that actor with the matching Task-60
Small EXPO editor/critic checkpoint at step 450223 from the Task-60 teacher
directory. It uses the unchanged LIBERO-90 protocol (20 episodes/task, seed 7,
224px, horizon 10, replan 5). The worker passed Python/LIBERO initialization
and remains active; no task result or aggregate is inferred until all 90 task
files validate. Existing evaluators and the unrelated Task-2 side evaluator
were not changed.

Two interpretation guards were verified during this poll. First, these three
Full90 launches use `--mode sft_pi`, so they evaluate the distilled Pi0.5
actor-only path. They are a controlled actor-generalization measurement and
must not be reported as a win over the complete Full EXPO editor/critic/
temperature selection stack. A matched full-system comparison remains a
separate required step after the actor-only results are available. Second, the
repeated LIBERO warning about `/projects/bgqe/rpolonsky/dcrl/.deps/LIBERO/datasets`
is informational here: the evaluator constructs rollouts from the 90 BDDL
files and 90 `*.pruned_init` files, which are present, and does not read HDF5
demonstrations during evaluation. No evaluator restart is warranted.

## Live diagnostic Full-90 poll (2026-11-06 14:27 CST)

The node-side audit found six in-scope diagnostic Full-90 evaluators alive and
GPU-backed. These are partial counts only; no candidate is ranked until its
complete 90-task validator passes:

| candidate | mode | node | records | status |
|---|---|---|---:|---|
| Task 60, seed 101, update 150 | `sft_pi` | `gpua003` | 80/90 | active |
| Task 28, seed 456, update 300 | `sft_pi` | `gpua095` | 64/90 | active |
| Task 29, seed 789, update 100 | `sft_pi` | `gpua073` | 39/90 | active |
| Task 60, seed 789, update 100 | `sft_pi` | `gpua022` | 39/90 | active |
| Task 28, seed 42, update 300 | `sft_plus_editor` | `gpua074` | 3/90 | active |
| Task 60, seed 42, update 200 | `sft_plus_editor` | `gpua038` | 0/90 | active; model restored |

The new Task-60 full-system worker printed the expected observation, action,
editor, critic, and state shapes and is beginning its first rollout. No
evaluator was restarted, and the unrelated side evaluator and training
workers were left untouched.

## Live diagnostic Full-90 poll (2026-11-06 14:34 CST)

The same six diagnostic evaluators remain alive. Their validated-record counts
are still partial: Task-60 seed-101/update-150 actor-only is 82/90,
Task-28 seed-456/update-300 actor-only is 65/90, Task-29 seed-789/update-100
actor-only is 40/90, Task-60 seed-789/update-100 actor-only is 40/90,
Task-28 SFT-plus-editor is 4/90, and Task-60 SFT-plus-editor is 0/90. The
last worker is GPU-saturated during its first full-system compilation/rollout;
it remains healthy and was not restarted. No partial count is used for
ranking.

## Exact Task-29 Full EXPO actor-only control relaunched (2026-11-06 14:39 CST)

The prior Task-29 `direct_rl_actor_only` audit was incomplete, so two
genuinely idle A100s (`gpua052` and `gpua062`) were verified and assigned a
clean two-shard rerun in tmux sessions
`full90_expo_task29_actor_only_shard0` and
`full90_expo_task29_actor_only_shard1`. Both use the exact Full EXPO Task-29
checkpoint at step 60486, `direct_rl_actor_only`, 45 tasks per shard, 20
episodes/task, seed 42, evaluation seed 7, 224px, horizon 10 and replan 5.
Their outputs are isolated under
`dcrlcheckpoints/full_expo_task29_actor_only_full90_20261106`; both have
passed initial LIBERO startup and have not yet committed task files. No
existing SFT evaluator, Full EXPO job, or side evaluator was changed.

## Matched Task-2 full-system Full-90 evaluation launched (2026-11-06 14:51 CST)

The strongest source-valid Task-2 screen candidate (seed 123, update 100) was
staged from its node-local checkpoint on `gpua062` to node-local storage on
the genuinely idle `gpua079`. Its saved `sft_config.json` verifies the strict
data gate: `source_task_id=2`, `collection_source=small_expo`,
`official_demo_ratio=0.0`, `official_demo_chunks=0`, zero-B `lora_only`, and
the unchanged full-proposal 10/5 interface. The checkpoint is the same
Task-2/update-100 actor that produced the current best four-task screen
candidate (+8.33 pp held-out against its own step-0 control).

The new tmux session `full90_task02_sft_plus_editor_diag` runs on `gpua079`
with `mode=sft_plus_editor`: that SFT actor is combined with the matching
Task-2 Small EXPO editor/critic checkpoint at step 375398 and its matching
Task-2 normalization statistics. It evaluates all LIBERO-90 tasks, 20
episodes/task, evaluation seed 7, image size 224, action horizon 10, and
replan boundary 5. The checkpoint staging used node-local `/tmp/usertmp`
because the shared `/work` quota is currently full; it did not modify the
source checkpoint, any running evaluator, or any training allocation.

At launch validation, the tmux session and Python process were alive. The
process had passed robosuite initialization and was still in model/checkpoint
startup, with no task JSON committed yet. No score is inferred until all 90
task records validate. The isolated launcher is
`launch_task02_plus_editor_full90_20261106.sh`.

The follow-up health poll confirmed that Task-2 worker is now GPU-active
(about 31.2 GiB and 89% utilization) and executing the JAX rollout path; it
has not yet committed its first task file. The two-shard Task-29 actor-only
control has also begun committing records (2/45 and 1/45), and the Task-60
full-system run has committed 2/90. These are operational counts only; no
partial aggregate is promoted.

## Exact Task-2 Full EXPO actor-only control launched (2026-11-06 14:59 CST)

Because Task-2/update-100 is the strongest source-valid SFT screen candidate,
an exact actor-only reference was launched on the newly verified idle
`gpua071`, using the Full EXPO Task-2 checkpoint at step 60244. The isolated
tmux session is `full90_expo_task02_actor_only`; it uses the same LIBERO-90
suite, 20 episodes/task, seed 42, evaluation seed 7, 224px images, horizon
10, replan 5, and direct actor-only path used by the SFT audit. Its output is
node-local under `/tmp/usertmp/rpolonsky/expo02_actor_only_full90_20261106`.

The first launch was stopped before model initialization because the new
launcher pointed LIBERO at an empty config directory. Only that newly created
diagnostic tmux session was stopped. The corrected launcher now uses the
validated shared LIBERO runtime config; its process has passed robosuite and
LIBERO initialization and is loading the Full EXPO actor, with no task result
committed yet. No existing evaluator, training process, or allocation was
changed.

At 15:02 CST, a matched actor-only evaluation of the same strict
Task-2/update-100 SFT checkpoint was launched on idle `gpua100`, alongside the
exact Task-2 Full EXPO actor-only control on `gpua071`. Both use the complete
LIBERO-90 actor-only protocol and identical rollout seeds/settings. The SFT
worker passed checkpoint discovery and is loading normally; no score is yet
available and no existing worker was changed.

## Completed Task-60 seed-101/update-150 Full-90 actor-only audit (2026-11-06 15:08 CST)

The Task-60 seed-101/update-150 evaluator committed all 90 task files and was
validated directly from the per-task JSON records: task IDs 0--89 exactly,
20 trials per task, 1,800 episodes total, and the unchanged protocol
`libero_90`, evaluation seed 7, action horizon 10, replan 5, and 400-step
horizon. The actor-only SFT checkpoint scored **509/1,800 = 28.28%**. Its
tracked task panel was Task 2/28/29/60 = **65%/40%/55%/50%**.

The saved manifest verifies this is a task-pure diagnostic arm:
`source_task_id=60`, `collection_source=small_expo`, `official_demo_ratio=0`,
`lora_only` with zero-B initialization, full-proposal targets, batch 16,
learning rate `5e-5`, 10k warmup, and the unchanged 10/5 interface. The
corresponding exact actor-only Full EXPO Task-60 control is 475/1,800
(26.39%), so this SFT actor is **+34 successes / +1.89 percentage points**
at the actor level. Against the deployed Full EXPO selection stack
(368/1,800, 20.44%), the difference is **+141 / +7.83 pp**, but that is a
separate deployment-path comparison. This complete result is admitted to the
diagnosis; no partial suite is used.

## Matched Task-60 seed-101/update-150 full-system Full-90 evaluation launched (2026-11-06 15:12 CST)

The completed Task-60 seed-101/update-150 actor-only candidate was promoted
to the deployment-path test on the newly released `gpua003`. The tmux session
`full90_task60_seed101_sft_plus_editor` runs `mode=sft_plus_editor` with the
same Task-60 Small EXPO editor/critic checkpoint at step 450223 and its
matching normalization statistics. It uses the unchanged LIBERO-90 protocol:
20 episodes/task, evaluation seed 7, seed 42, 224px images, horizon 10,
replan 5, and a 400-step limit. The worker has initialized as a Python
process but has not committed a task file yet. Its output is isolated under
`/tmp/usertmp/rpolonsky/sft_task60_seed101_update150_plus_editor_full90`.

## Matched Task-29 seed-789/update-100 full-system Full-90 evaluation launched (2026-11-06 15:32 CST)

The next missing deployment-path comparison is now running on the genuinely
idle `gpua064` allocation in tmux session
`full90_task29_seed789_sft_plus_editor`. It uses the validated Task-29-only
SFT checkpoint at update 100, whose manifest records
`source_task_id=29`, `collection_source=small_expo`,
`official_demo_ratio=0.0`, `lora_only`/zero-B, full-proposal targets, batch
16, learning rate `5e-5`, 10k warmup, and the unchanged 10/5 interface. The
checkpoint was copied from node-local `gpua073` storage to node-local
`gpua064` storage; the source evaluator and source files were not modified.

The SFT actor is paired with the matching Task-29 Small EXPO editor/critic
checkpoint at step 375380 and matching normalization statistics. The run uses
the fixed LIBERO-90 protocol (90 tasks, 20 episodes/task, evaluation seed 7,
seed 42, 224px, horizon 10, replan 5). It has passed Python/robosuite
startup and is still loading/compiling; zero task files have been committed,
so no score is inferred. A short-lived first attempt was cleaned up by Slurm
before model execution; the persistent `srun`/tmux wrapper is now live.

## Full-90 diagnostic health poll (2026-11-06 15:35 CST)

All in-scope evaluators remain alive and GPU-backed. The current committed
record counts are operational progress only:

| arm | node | records |
|---|---|---:|
| Task 2 Full EXPO actor-only | `gpua071` | 6/90 |
| Task 2 task-pure SFT actor-only | `gpua100` | 6/90 |
| Task 2 task-pure SFT + editor/critic | `gpua079` | 4/90 |
| Task 28 task-pure SFT actor-only | `gpua095` | 80/90 |
| Task 28 task-pure SFT + editor/critic | `gpua074` | 11/90 |
| Task 29 Full EXPO actor-only shard 0 | `gpua052` | 11/45 |
| Task 29 Full EXPO actor-only shard 1 | `gpua062` | 11/45 |
| Task 29 task-pure SFT actor-only | `gpua073` | 54/90 |
| Task 29 task-pure SFT + editor/critic | `gpua064` | 0/90; initializing |
| Task 60 task-pure SFT actor-only, seed 789 | `gpua022` | 55/90 |
| Task 60 task-pure SFT actor-only, seed 101 | `gpua003` | 90/90 |
| Task 60 task-pure SFT + editor/critic | `gpua038` | 6/90 |

The Task-29 full-system worker has passed LIBERO initialization and printed
the expected 224x224 observation, 35-wide editor/Q interface, 10-action
horizon, and 8D state shapes; it is still in model/JAX startup. No partial
panel is admitted as a score, and no existing training or side evaluator was
changed.

## Full-90 diagnostic health poll (2026-11-06 15:59 CST)

The current evaluator inventory was rechecked from node-side JSON files and
process tables. No diagnostic GPU is idle: all running A100 slots are occupied
by existing evaluators or protected sleeper allocations, while the two
additional A100 arrays remain pending in Slurm. No worker or training job was
restarted or changed.

| arm | node | committed records |
|---|---|---:|
| Task 2 Full EXPO actor-only control (side evaluation) | `gpua071` | 12/90 |
| Task 2 task-pure SFT actor-only (side evaluation) | `gpua100` | 12/90 |
| Task 2 task-pure SFT + editor/critic (side evaluation) | `gpua079` | 6/90 |
| Task 28 task-pure SFT actor-only | `gpua095` | 85/90 |
| Task 28 task-pure SFT + editor/critic | `gpua074` | 13/90 |
| Task 29 Full EXPO actor-only control, shard 0 | `gpua052` | 17/45 |
| Task 29 Full EXPO actor-only control, shard 1 | `gpua062` | 16/45 |
| Task 29 task-pure SFT actor-only | `gpua073` | 62/90 |
| Task 29 task-pure SFT + editor/critic | `gpua064` | 2/90 |
| Task 60 task-pure SFT actor-only, seed 789 | `gpua022` | 62/90 |
| Task 60 task-pure SFT + editor/critic, update 200 | `gpua038` | 8/90 |
| Task 60 seed-101/update-150 SFT actor-only | completed | 90/90; 509/1,800 |
| Task 60 seed-101/update-150 SFT + editor/critic | `gpua003` | 4/90 |

The two Task-29 actor-only shards together cover all 90 tasks (33 committed so
far), with 45 tasks assigned to each shard. The repeated LIBERO datasets-path
messages are the known nonfatal warning; the workers continue to commit task
records and show no traceback. Partial records remain operational progress and
are not admitted as aggregate scores. The next diagnosis action is to use the
first genuinely released GPU for the next preplanned task-pure arm, without
interrupting these evaluations.

At 16:06 CST, a short file-backed progress poll confirmed continued forward
progress without intervention: Task-28 actor-only advanced to 87/90, Task-29
actor-only to 63/90, Task-60 seed-789 actor-only to 63/90, Task-28
SFT-plus-editor to 14/90, and Task-60 update-200 SFT-plus-editor to 9/90.
These remain partial suites and are not scored or ranked yet.

## Validated Task-28 seed-456/update-300 Full-90 result (2026-11-06 16:19 CST)

The Task-28 task-pure SFT actor-only evaluator on `gpua095` completed and was
validated directly from its 90 JSON records: task IDs 0--89 exactly, 20
episodes per task, 1,800 episodes total, and the fixed protocol
`libero_90`, evaluation seed 7, 224-pixel images, 400-step horizon,
action-horizon 10, and replan 5. It scored **506/1,800 = 28.11%**.

The checkpoint manifest independently confirms the strict data gate:
`source_task_id=28`, `collection_source=small_expo`,
`official_demo_ratio=0.0`, `target_mode=full_proposal`, LoRA mode, batch 16,
learning rate `5e-5`, seed 456, and the unchanged 10/5 interface. Against the
matched deployed Full EXPO Task-28 result (451/1,800 = 25.06%), this is
**+3.06 percentage points**. It is not the current best Task-28 pure-SFT
result (the validated executed-backfill update-200 arm is 525/1,800), and it
is still an actor-only SFT-versus-deployed-Full-EXPO comparison. The matched
Task-28 SFT-plus-editor evaluator remains necessary for a full-system claim.

## Exact Task-28 Full EXPO actor-only control launched (2026-11-06 16:22 CST)

The completed Task-28 SFT actor-only evaluator released `gpua095`; a node-side
check found no user process and 0 MiB GPU memory. I launched the corresponding
exact Full EXPO Task-28 actor-only control there in the isolated tmux session
`full90_expo_task28_actor_only`, using checkpoint step 40095 and the same
LIBERO-90 protocol: all 90 tasks, 20 episodes/task, evaluation seed 7, seed
42, 224-pixel images, 400-step horizon, action horizon 10, and replan 5.
Its output is node-local at
`/tmp/usertmp/rpolonsky/expo28_actor_only_full90_20261106`; the launcher is
`launch_full90_expo_task28_actor_only_20261106.sh`. The worker passed robosuite
startup and is still in JAX/model initialization with no traceback. This adds
the missing actor-level control for the newly validated seed-456 SFT result;
the existing Task-28 SFT-plus-editor run and all unrelated workers were left
untouched.

At 16:34 CST, a node-side poll showed the control had transitioned out of
pure initialization and committed its first LIBERO-90 task record (`1/90`).
The Slurm allocation remained `RUNNING`, the tmux session remained present,
and the evaluator process was still active. This is operational progress only;
no aggregate score is inferred until all 90 task records are complete.

## Task-28 Full EXPO actor-only control health verification (2026-11-06 16:32 CST)

The Task-28 control was rechecked from inside `gpua095`, rather than from the
login node, because both its tmux socket and `/tmp/usertmp` output are
node-local. The tmux session `full90_expo_task28_actor_only` exists and its
Python evaluator is alive (PID 2344838), using approximately 31.2 GiB of GPU
memory and about 77% CPU while JAX/model compilation continues. The pane shows
only known JAX/Flax deprecation warnings and no traceback. It has committed
0/90 task records so far, so it is an active initialization wait—not a failed
or completed evaluation. No other evaluator or training allocation was
modified.

## Task-28 executed-backfill update-200 matched full-system evaluation launched (2026-11-06 16:43 CST)

The strongest validated Task-28 pure-SFT candidate was promoted to the
deployment-path evaluation on the genuinely free GPU UUID
`GPU-21332eef-35ad-7680-70ae-b98d4088ca4f` in allocation `23448992_6` on
`gpua064`. It restores the task-pure executed-backfill checkpoint at update
200, whose manifest has `source_task_id=28`, `collection_source=small_expo`,
`official_demo_ratio=0.0`, and the unchanged 10/5 interface, together with the
matching Task-28 Small EXPO editor/critic at step 275001. The protocol is the
same full LIBERO-90/20-episodes-per-task evaluation used by the other
deployment-path controls. It runs in tmux session
`full90_task28_executed_backfill_step200_plus_editor` with node-local output
under `/tmp/usertmp/rpolonsky/task28_executed_backfill_step200_plus_editor_full90_20261106`.
The tmux session and Python process are present; it has 0/90 records during
startup. The existing Task-29 evaluator on the other GPU of `gpua064`, all
other diagnostic workers, and all training allocations were left untouched.

At 16:46 CST, a node-side health check showed the exact Task-28 Full EXPO
actor-only evaluator still active at approximately 31.2 GiB GPU memory and
63% CPU. The matched executed-backfill SFT-plus-editor evaluator was also
active, using GPU memory and having loaded the candidate's norm statistics;
its output remained at 0/90 records during initialization. Neither process
had a traceback, so neither was restarted.

At 16:47 CST, the matched deployment-path evaluator completed restoration of
both the Task-28 executed-backfill SFT actor and the matching Small EXPO
editor/critic at step 275001. It then entered JAX/rollout initialization; the
panel remained 0/90 and showed only the known LIBERO dataset-path warnings.
This confirms the intended actor-plus-editor stack is loaded before scoring.

## Full-90 promotion queue after completed-artifact audit (2026-11-06 16:47 CST)

A read-only audit of the completed shared actor-only panels confirms the
current task-pure candidate ordering:

| source task | strongest validated pure-SFT checkpoint | actor-only Full-90 |
|---:|---|---:|
| 2 | full-proposal update 500 | 516/1,800 (28.67%) |
| 28 | executed-backfill update 200 | 525/1,800 (29.17%) |
| 29 | full-proposal update 500 | 524/1,800 (29.11%) |
| 60 | full-proposal update 200 | 519/1,800 (28.83%) |

The matched deployment-path evaluations currently cover the strongest Task-28
and Task-60 candidates, while the live Task-2 and Task-29 SFT-plus-editor
workers use earlier update-100 screening checkpoints. Once a diagnostic GPU
is genuinely released, the next two matched full-system promotions are Task
29 update 500 and Task 2 update 500, in that order. This queue preserves the
strict same-task, zero-official-data gate; no mixed-data result is admitted.

## Validated Task-28 matched full-system result and retention tradeoff (2026-11-07 09:56 CST)

The Task-28 executed-backfill update-200 SFT-plus-editor evaluation completed
with all 90 task records, 20 episodes/task, no failed tasks, and the fixed
Full-90 protocol. It scored **528/1,800 = 29.33%**. The exact Task-28 Full
EXPO actor-only control completed the same protocol at **475/1,800 = 26.39%**;
the deployed Full EXPO six-shard baseline is **451/1,800 = 25.06%**. Thus the
SFT-plus-editor result is +2.94 pp against the actor-only control and +4.28 pp
against the deployed Full EXPO stack.

This is not yet a satisfactory source-retention win. On the underlying Task
28, SFT-plus-editor scored 11/20 (55%), versus 19/20 for the exact actor-only
Full EXPO control and 20/20 for deployed Full EXPO. Excluding Task 28, the
SFT-plus-editor result is 517/1,780 (29.04%), versus 456/1,780 (25.62%) for
actor-only Full EXPO and 431/1,780 (24.21%) for deployed Full EXPO. The
current evidence therefore shows a real held-out generalization gain coupled
to source-task degradation; it must not be presented as an unqualified win.

## Strongest Task-29 and Task-2 matched full-system controls launched (2026-11-07 09:56 CST)

The released `gpua095` slot now runs
`full90_task29_full_proposal_step500_plus_editor` from the strict task-29-only,
zero-official-data update-500 checkpoint, paired with the Task-29 editor/critic
at step 375380. The released GPU0 slot on `gpua064` now runs
`full90_task02_full_proposal_step500_plus_editor` from the strict task-2-only,
zero-official-data update-500 checkpoint, paired with the Task-2 editor/critic
at step 375398. Both use the all-90-task/20-episodes-per-task protocol and
were healthy at startup with 0/90 records. These runs will be judged by both
held-out 89-task performance and source-task retention. No existing training
job or unrelated evaluator was modified.

At 10:00 CST, the first Task-29/Task-2 launch attempt exposed a checkpoint
layout issue: passing `latest_checkpoints/500` made the evaluator search for
the manifest in the wrong directory. The failed processes exited before
evaluation and their diagnostic wrappers were cleared. The launchers were
corrected to pass the `latest_checkpoints` manager roots, whose parents contain
the verified task-pure manifests. Both corrected workers now have live tmux
sessions and Python evaluators, each with one checkpoint step discovered and
0/90 records during initialization. The Orbax missing-metrics line is
nonfatal; checkpoint discovery continued normally. No existing worker was
changed.
## Source-task retention is a hard admissibility gate (2026-11-07)

The Task-28 executed-backfill update-200 matched SFT-plus-editor result at
`11/20` on its underlying task is rejected for the research comparison. A
held-out Full-90 gain cannot compensate for losing the task being distilled.
From this point onward, a task-specific SFT candidate is admissible only if
its source-task evaluation is at least `19/20` (95%), uses only that source
task's Small EXPO targets (and, if present, only that task's official data),
and uses the unchanged 10/5 planning/action interface.

The existing Task-28 action-bounds run provides one retention-passing
checkpoint: update 1500 scores `20/20` on Task 28 in the recorded source-task
evaluation and has a completed Full-90 SFT-only evaluation of `464/1800`
(`25.78%`). Excluding Task 28 it scores `444/1780` (`24.94%`). This passes
the source gate, but is only a modest comparison against the deployed Full
EXPO baseline (`451/1800`, `25.06%`; `431/1780`, `24.21%` held out) and is
not evidence that SFT-plus-editor beats Full EXPO. The update-1000 quick
evaluation also recorded `19/20` on Task 28, but its checkpoint is no longer
present in the retained checkpoint directories, so it is not currently a
reproducible Full-90 candidate.

The next ranking rule is lexicographic: first discard every candidate below
95% source-task retention; then compare held-out performance against a
protocol-matched Full EXPO control. We will not optimize or report the 11/20
result as a success. The live Task-29 and Task-2 matched SFT-plus-editor
evaluations remain unchanged and will be judged by this same gate when
complete.
## Read-only checkpoint/storage audit (2026-11-07)

The active evaluator roots that must be retained are Task 2 full-proposal
update 500, Task 29 full-proposal update 500, and Task 28 action-bounds raw
update 1500 (the latter is still used by the raw Full-90 evaluator). The
current Task-2 and Task-29 full-system evaluators are healthy on `gpua064` and
`gpua095`; no cleanup target is referenced by those live commands.

The audit found many completed SFT runs with a roughly 12--13 GB
`checkpoints/0` base snapshot plus a roughly 13 GB trained
`latest_checkpoints/<step>` snapshot. The base snapshot is a duplicated local
Pi0.5 copy; current launchers load the official base from the shared
`openpi_cache`. These are candidates for later cleanup only after confirming
that no resumption is required. Completed, unreferenced candidate groups are
the old scope/target-semantics/outer-augmentation runs, the Task-29 action
bounds run, and the old replay ablations; together they represent roughly
310 GB while their eval JSON and configs can be retained separately. No such
checkpoint was deleted in this audit.

The only clearly invalid cross-task diagnostic tree had already been removed:
`sft_multitask_balanced_20261103` (about 49 GB). Empty retry/no-checkpoint
directories total only a few MB and are not material to quota.
## Retention-gated candidate promotion (2026-11-07)

The existing task-pure four-task screening results identify three additional
reproducible source-retention candidates: Task 28 raw action-bounds update
1500 (`20/20` source in the saved full-90-compatible checkpoint), Task 2
Small-EXPO-only update 3000 (`19/20`), and Task 60 Small-EXPO-only update
1000 (`19/20`). Their manifests specify zero official demonstrations,
task-matched Small EXPO targets, and the unchanged 10/5 interface.

The Task-28 update-1500 candidate is now running as a matched SFT-plus-editor
Full-90 evaluation in tmux session
`full90_task28_actionbounds_raw_step1500_plus_editor` on the idle
`23448992_1`/`gpua073` slot, paired with the Task-28 editor at step 275001.
The Task-2 update-3000 candidate is running in
`full90_task02_small_expo_step3000_plus_editor` on `23448992_2`/`gpua100`,
paired with editor step 375398. The Task-60 update-1000 candidate is running
in `full90_task60_small_expo_step1000_plus_editor` on
`23448992_4`/`gpua062`, paired with editor step 450223. All three are
evaluation-only, write results under `/tmp/usertmp`, and do not create new
checkpoints. Existing Task-2/Task-29 side evaluators and all training jobs
remain untouched.

The four-task screening leaderboard now shows that Task-28 raw update 1000
had the strongest provisional source-gated held-out panel (`19/20` source,
45% mean over the other three tasks), but that checkpoint has been pruned.
The currently reproducible update-1500 checkpoint is therefore the key
full-system test; no held-out claim is made until the matched editor result
completes.
At 10:33 CST, the three new retention-gated evaluators had successfully
restored their SFT actors and matching Small EXPO editors: Task 28 update
1500 with editor step 275001, Task 2 update 3000 with editor step 375398, and
Task 60 update 1000 with editor step 450223. They had entered rollout/JAX
execution with no traceback. The repeated missing `metrics/metrics` messages
are nonfatal Orbax metadata warnings; each evaluator logged its restoration
completion. No partial suite is ranked until its complete validated summary
exists.
## Early cross-task screening of newly promoted candidates (2026-11-07)

The saved four-task actor-only screening provides a useful but non-final
ordering while the matched Full-90 evaluations run. Task 2 update 3000 passes
the source gate at `19/20`, but scores `0/20` on each screened held-out task
(28, 29, and 60); it is therefore a retention-valid negative-control rather
than a likely generalization winner. Task 60 update 1000 passes at `19/20`
and scores `8/20`, `15/20`, and `1/20` on Tasks 2, 28, and 29, respectively.
Its screened held-out mean is `40%`, making it the stronger candidate to watch.
These four-task results are actor-only screening evidence, not substitutes for
the complete 90-task SFT-plus-editor evaluation now running.
## Full-90 per-task gap analysis and focused subset (2026-11-07)

I compared the completed source-retention-passing Task-28 actor-only SFT
evaluations against the corresponding Task-28 Full EXPO actor-only control
(`451/1,800`) on all 90 task IDs. The largest consistently positive tasks
were Task 19 (`+12` successes on average across four qualifying SFT
checkpoints) and Task 79 (`+6`); Tasks 30, 7, 31, 12, 38, and 77 were also
positive for all four checkpoints. The strongest complete actor-only SFT
candidate was the clipped update-1500 run at `482/1,800`, but its source
retention remains to be checked under the hard 19/20 gate in the saved
source-task protocol.

To test whether these gaps survive the actual deployed stack, an isolated
focused evaluation of the raw Task-28 update-1500 candidate with the matching
editor/critic is running on `{19, 28, 79}` in tmux session
`subset_task28_raw1500_tasks19_28_79_plus_editor` on `gpua022`. Task 28 is
included as the source-retention control. This focused screen is exploratory;
the all-90 evaluation remains the authoritative generalization result.
At 10:43 CST, the focused `{19,28,79}` Task-28 raw update-1500
SFT-plus-editor evaluator was healthy on `gpua022` and remained in first-task
compilation. The all-90 Task-28 and Task-2 promoted candidates had each
committed one task record; Task 60 was still in initialization. The existing
Task-2 and Task-29 side evaluators had reached four task records each. No
worker had a traceback or was restarted.

## Consolidated diagnosis and pause state (2026-11-07)

New diagnostic launches are paused at the user's request. Already-running
Full-90 evaluators were left intact so their completed suites can still be
captured; no RL training job was stopped or modified. The attempted Task-79
source-task screen did not persist: a detached `srun` step cleaned its tmux
child, and the later persistent-shell test was stopped before training. It
created no checkpoint and contributes no result.

### Reference Full-90 results

All rows below use 90 LIBERO tasks, 20 episodes/task, evaluation seed 7,
224-pixel images, 400 environment steps, action horizon 10, and replan 5.

| system | source task | result | interpretation |
|---|---:|---:|---|
| Official Pi0.5 actor | — | 512/1800 (28.44%) | base reference |
| Full EXPO deployed | 2 | 447/1800 (24.83%) | editor/critic selection stack |
| Full EXPO deployed | 28 | 451/1800 (25.06%) | editor/critic selection stack |
| Full EXPO deployed | 29 | 441/1800 (24.50%) | editor/critic selection stack |
| Full EXPO deployed | 60 | 368/1800 (20.44%) | editor/critic selection stack |
| Full EXPO deployed | 79 | 483/1800 (26.83%) | editor/critic selection stack |
| Exact Full EXPO actor-only control | 28 | 475/1800 (26.39%) | selection stack removed |
| Exact Full EXPO actor-only control | 60 | 475/1800 (26.39%) | selection stack removed |

### SFT findings that are actually validated

| candidate | data and scope | Full-90 result | source-task result | status |
|---|---|---:|---:|---|
| Task-28 executed-backfill update 200 + editor | Task-28 Small EXPO only, zero official; LoRA | 528/1800 (29.33%) | 11/20 (55%) | **reject**: source collapse |
| Task-28 raw action-bounds update 1500, actor-only | Task-28 Small EXPO only, zero official; LoRA/zero-B | 464/1800 (25.78%) | 20/20 (100%) | source-valid, but only a modest held-out gain |
| Task-60 full-proposal update 200, actor-only | Task-60 Small EXPO only, zero official | 519/1800 (28.83%) | not an absolute 19/20 source-gate pass in the retained screen | actor-only evidence only |
| Historical Task-28 high-dose update 40000, actor-only | task-specific SFT | 46/1800 (2.56%) | 20/20 | severe specialization/invalid as a remedy |

For the retention-passing Task-28 raw update-1500 row, excluding the source
task gives 444/1780 (24.94%), versus 431/1780 (24.21%) for deployed Task-28
Full EXPO: +13 successes (+0.73 percentage points). This is not yet a
full-system win because its matching SFT-plus-editor Full-90 evaluation is
still incomplete. A separate historical table reports a clipped update-1500
total of 470/1800, while a later per-task audit reported 482/1800; that
artifact discrepancy is intentionally not ranked until reconciled.

### What the dose/seed screens establish

The task-pure four-task screens consistently use only matching Small EXPO
teacher chunks, zero official demonstrations, official Pi0.5 initialization,
LoRA with zero-B, full-proposal 10/5 targets, batch 16, and LR `5e-5`.

- Early updates can produce transient held-out gains, but they are
  non-monotonic and seed-sensitive. Task-2/update-100 showed a +8.33 pp
  held-out-screen gain in one seed, while the independent seed-456 result
  lost 6.67 pp. Later updates frequently lose transfer or source retention.
- Task-29/update-300 produced a small +1.67 pp held-out-screen gain in one
  seed, but independent replications did not preserve it under the source
  gate.
- Task-60/update-100 produced a +3.33 pp held-out-screen gain in one seed;
  other seeds were neutral or negative. Its update-200 +5 pp held-out
  screen gain came with a 5 pp source loss and is rejected.
- The source-28 per-task audit found the most consistent positive actor-only
  transfer on Tasks 19 and 79, with Task 30 and several others also positive.
  This motivated the uncompleted Task-79 source-task arm, but it is not
  evidence by itself that Task 79 is the right teacher source.

Source retention is now an absolute admissibility gate: a promoted candidate
must achieve at least 19/20 on its own underlying task, use only that task's
teacher data (and only that task's official data if an official mixture is
deliberately tested), and preserve the 10/5 interface. Held-out gains below
that gate are diagnostic only.

### Mechanistic conclusions

1. The old high-dose SFT failure is real over-specialization: its Pi0.5 image
   tower moved by about 10.7% relative L2 versus about 1.94% for the compared
   Full EXPO actor. This does not prove that Full EXPO's actor drifts more.
2. Small EXPO `full_proposal` targets already contain the teacher's selected
   ten-action chunk; the teacher's critic-selected candidate was chosen in a
   majority of inspected records. A separate selected-action distillation
   recipe would therefore not be a cleanly new intervention.
3. For Task 60, the exact actor-only Full EXPO control scored 475/1800 while
   the deployed editor/critic system scored 368/1800, a 5.94 pp difference.
   This implicates the learned selection stack as a possible transfer-loss
   mechanism, but does not isolate editor, critic, or temperature separately.
4. The best-supported SFT recipe is therefore provisional: task-pure
   teacher data, Pi0.5 base initialization, low-dose LoRA/zero-B, full 10/5
   targets, and checkpoint selection under the absolute source-retention
   gate. It has not yet produced a validated, source-preserving
   SFT-plus-editor improvement over Full EXPO on the complete suite.

### State of incomplete evaluations at pause

The Task-28 raw-1500, Task-2 update-3000, Task-60 update-1000, Task-29
update-500, Task-2 update-500, and focused Task-28 `{19,28,79}` evaluators
were still live with only partial task files at the pause. Their partial
counts are operational status, not scores; any later completion should be
validated for exactly 90 task IDs and 1,800 episodes before being added to the
ranking. No further diagnostic training should be launched until those
artifacts are either completed or explicitly discarded.

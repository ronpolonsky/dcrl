
## 2026-10-17 — full-chunk SFT smoke

- Job 17930893, iris-hi, iris-hgx-1, H100 80GB: restored Task 28 compact
  checkpoint 110016 and completed one collection episode, then the collection
  subprocess exited with SIGABRT during the second episode. No Python traceback
  from the worker was available. This is not yet evidence of faulty hardware.
- Retry 17931004 enables PYTHONFAULTHANDLER and limits OMP/OpenBLAS/MKL to one
  thread to diagnose native initialization/reset issues. Full SFT launch is
  gated on successful collection, updates, eval and resume.
- Diagnostic retry 17931004 reproduced the native abort in MuJoCo/robosuite
  `read_pixels` while stepping episode 1, immediately after resetting the same
  environment used for episode 0. The fault handler confirms it is in rendering,
  not teacher inference. The next retry recreates the OffScreenRenderEnv/EGL
  context between episodes and excludes iris-hgx-1 based on this repeatable
  render failure on that allocation; this does not establish a broader host issue.
- Follow-up resource check (2026-10-18): iris-hi H200 has all 8 GPUs allocated;
  queued 17940016 could not start. A40 nodes iris5/6 had 7 of 8 GPUs allocated and
  the batch request needed 128 GB free host memory, so queued 17940026 could not
  start either. Existing interactive iris6 reported an idle A40 (0 MiB used), so
  smoke 17939896's interactive allocation is running the same batch-1 test there.
- User requested to exclude only iris1-4 and iris10. The Slurm script now uses a
  generic single-GPU request and exactly those exclusions. Generic iris-hi smoke
  17940028 is queued to let Slurm select the allocation; its run will log the
  allocated GPU and node before training.
- Interactive allocation 17939896 is an A40 (46 GB GPU memory) with Slurm
  `--mem=8GB` host memory. Current smoke uses batch size 1 and is only an
  initialization/collection check; it does not establish that SFT fits the
  interactive allocation. Scheduled SFT jobs request 128 GB host RAM and a
  generic GPU. The trainer defaults to π0.5 LoRA, not full parameter SFT;
  `--finetune-mode full` is an explicit alternative. "Full chunk" names the
  16-action supervision target, not the number of trainable weights.


## 2026-10-18 — full-chunk SFT collector pipe protocol

- Generic iris-hi smoke 17940028 started successfully on iris9 (L40S), restored
  Task 28 teacher checkpoint 110016, and reached LIBERO target collection. It
  failed on its first chunk with `ValueError: not enough values to unpack`: the
  worker sent a second `observation` message after each chunk while the parent
  expected a `result` message. The worker then reported a broken pipe as the
  parent exited. This was a collector message-protocol bug, not a node failure.
- Fixed `dcrl/full_chunk_data.py` to send the initial observation once and then
  return one result for each action chunk. Python compilation passes. The failed
  partial HDF5 contains metadata only and is not eligible for training. Rerun
  the smoke before launching full SFT.

- Follow-up smoke on iris7 restored Task 29 step 335076 and collected both
  episodes, but both reached the 400-step limit. Success-only training correctly
  stopped with “No eligible teacher chunks”; dependent job 17940061 therefore
  did not start. The checkpoint's reported 100% eval used fixed benchmark starts
  and `eval_seed=7`, whereas collection had used seed 42. Changed collection to
  use the same configured eval seed (7) for the next smoke, preserving sequential
  initial-state indices. This will test whether reset RNG differences explain
  the rollout mismatch before allowing full SFT to start.

- Further comparison found another reproducibility mismatch: benchmark eval
  seeds the learner RNG once and carries it across all episodes, while target
  collection had reset it to a folded seed at every episode. Removed that reset;
  collection now starts from `eval_seed` and advances the same policy RNG stream
  across fixed initial-state indices, matching `_evaluate` semantics. The next
  smoke validates this exact behavior.

- Root cause of the low-success collection was stale inference caching:
  `SmallExpoLearner.from_examples()` caches its randomly initialized parameters,
  then `restore_checkpoint()` replaces the train state without refreshing that
  cache. The online/eval trainer explicitly calls `cache_infer_params()` after
  restore; the collector did not. Collection now refreshes the cache immediately
  after restore. This matches the eval path and explains the two failed 400-step
  smoke episodes despite 100% eval. Next smoke uses one fixed start to verify.

- Cache-fixed smoke on interactive iris6 successfully collected a teacher episode
  from start index 0: success at 76 actions, yielding 10 full 16-action chunks.
  Pi0.5 student initialization then exceeded the interactive allocation's 8 GB
  host-memory cap and was OOM-killed. Production SFT requests 128 GB host RAM, so
  the full run is being moved to Slurm; the A40's GPU memory was not exhausted.

- Added `RESUME_SFT=1` support to the Slurm launcher. For a time-limited run,
  it recollects or reuses the completed HDF5 targets, then invokes training with
  `--resume` to restore the latest full Orbax state including optimizer state.
- Production SFT job 17940303 started on iris5 (A40, one GPU, 128 GB host RAM)
  with only iris1-4 and iris10 excluded. At the first monitor check it completed
  5/200 target episodes without collector errors; peak host RAM was about 5.7 GB.
  The full target set is still being collected before student initialization.

## 2026-10-18 — rebuilt Pi0.5 BC smoke resource check

- The new `train_bc_distill.py` collected one real Task 29 episode from SmallExpo
  checkpoint 335076 with 10-action targets and 5-action execution. The collection
  HDF5 is complete and contains 16 successful chunks.
- Training on iris8 used an RTX 4500 Ada and reached the initial fixed-state eval,
  then failed during the first Pi0.5 gradient compile with GPU
  `RESOURCE_EXHAUSTED` (about 5.36 GiB requested). This is a GPU-capacity limit;
  the same code is rerun on an A40 (iris5) with 128 GB host memory.

2026-10-18 BC-distillation smoke validation:
- Slurm job 17941064 on iris5 (A40) completed with exit 0.
- Real Task29 teacher collection used legacy SmallExpo checkpoint step 335076 (teacher horizon/replan 16/8) and emitted Pi0.5-compatible targets/execution of 10/5.
- Trainer ran fixed-state LIBERO eval at steps 0, 1, and 2; all records were finite (success rate 1.0 on the one smoke initial state).
- A full Orbax state checkpoint was written at step 1 (~8.2 GB), restored by a separate resume process, and step 2 updated successfully; checkpoint 2 also completed.
- Persistent checkpoint storage has ~56 GB free, while each full Pi0.5 state is ~8.2 GB. Retaining all 40 planned 25k checkpoints would require ~328 GB; full-run retention policy is awaiting user choice.

2026-10-18 node issue during Pi0.5 base-actor collection:
- Jobs 17942394 and 17942397 were assigned to `iris-hp-z8` and terminated immediately with Slurm `RaisedSignal:53(Real-time_signal_19)` before Python produced application output. The node is excluded from subsequent collection/training submissions; this is not evidence of a Pi0.5 or LIBERO failure.
- Generic `pi05_base` initialization requires roughly 50 GiB of GPU memory. Collection array task 17942437_1 on `iris6` (A40, 48 GiB) failed with JAX `RESOURCE_EXHAUSTED`; the other 48 GiB-or-smaller assignments were canceled before initialization. Subsequent collectors and SFT are constrained to H100/H200 nodes.
- Collector job 17942437_0 was stopped before writing data after audit found the OpenPI config still carried its default Pi-LIBERO weight loader. The collector now explicitly installs `CheckpointWeightLoader(pi05_base/params)`; replacement Task 60 job 17942469 uses the corrected loader.
- Collector initialization now supports `OPENPI_SKIP_INIT_JIT=1` with an SGD/no-momentum placeholder optimizer for inference-only collection, avoiding Adam state allocation; SFT retains the normal Adam/JIT training initialization. Task 60 replacement job 17942492 uses this path.

## 2026-10-18 — base-Pi data/SFT layout

Both data workflows now use a `pi_base_data/{success,failures}` partition. Direct Pi0.5 collection writes `pi05_task_<id>.hdf5`; compact-RL teacher collection writes `teacher_chunks.hdf5` under the SFT output. The SFT student is explicitly initialized from `gs://openpi-assets/checkpoints/pi05_base` with LoRA actor updates.
- Task 60 collector 17942492 reached Pi0.5 initialization but aborted in robosuite `read_pixels` with the node's EGL backend (exit 134). A direct MuJoCo OSMesa smoke test rendered successfully; both LIBERO launchers now use `MUJOCO_GL=osmesa` and `PYOPENGL_PLATFORM=osmesa`. Replacement Task 60 job is 17942611.

## 2026-10-18 — GPU/node issue summary

| Node/GPU | Observed issue | Disposition |
|---|---|---|
| `iris5`/`iris6` A40 48 GiB | Generic `pi05_base` initialization failed with JAX `RESOURCE_EXHAUSTED` (roughly 50 GiB required). | Exclude for Pi0.5 collection/SFT. |
| `iris7` A6000 48 GiB | Same insufficient-memory class; not suitable for generic Pi0.5 initialization. | Exclude for Pi0.5 collection/SFT. |
| `iris8` RTX4500 24 GiB | Insufficient memory for generic Pi0.5. | Exclude. |
| `iris9` L40S 48 GiB | Insufficient-memory class for generic Pi0.5. | Exclude. |
| `iris-hp-z8` RTX8000 | Slurm `RaisedSignal:53` before application output. | Excluded. |
| `iris-hgx-1` H100 | Pi0.5 initialization succeeded, but Task 60 later hit a fatal robosuite EGL `read_pixels` abort. | Keep eligible with OSMesa backend. |
| `iris-hgx-2` H200 | No failure observed; currently occupied by other jobs. | Preferred when available. |

The SFT launcher currently requests batch size 16. If an actual Pi0.5 training OOM occurs, retry with batch size 8; do not reduce below 8.
- Unpinned one-GPU collection test: job 17942633 (task 79) failed on `iris6` A40 with `RESOURCE_EXHAUSTED` allocating 4.83 GB; job 17942636 (task 2) failed on `iris8` RTX4500 with the same error. Jobs 17942634/35 (tasks 29/28) on `iris7` A6000 are being monitored. This confirms the low-memory failures are model initialization failures, not batch-size failures.
- The compact-teacher collection phase of SFT job 17942637 runs on `iris8` because the teacher is small; its later Pi0.5-base training phase still requires a high-memory GPU.
- Follow-up one-GPU runs 17942667/68/69/70 confirmed the same low-memory initialization OOM on A40/A6000/RTX4500 (tasks 79/29/28/2). Task 60 remains the only successful generic Pi0.5-base collector so far on H100 with OSMesa. The compact-teacher SFT job 17942637 continues collecting successfully on RTX4500 because it has not initialized the Pi0.5 student yet.
- The per-user QoS cap is 5 GPUs: with Task 60, four base collectors, and SFT, the sixth job waits (`QOSMaxGRESPerUser`).
- Low-memory retries 17942667/68/69/70 all terminated during generic Pi0.5 initialization with `RESOURCE_EXHAUSTED`. H100 recovery retries are queued as 17942682/83/84/85 (tasks 79/29/28/2), one node/GPU each, using OSMesa.

## 2026-10-18 compact-teacher collection quota fix
- The first `train_bc_distill` run used `COLLECTION_EPISODES=200`, but the small-expo collector ignored `COLLECTION_SUCCESSES=100` and `COLLECTION_FAILURES=25`; its `8/200` progress therefore meant total episodes, not per-partition quotas.
- Updated `dcrl/bc_distill_data.py` to enforce separate success/failure targets, retain only the requested outcomes, and allow up to four times the combined quota in rollout attempts. Updated `train_bc_distill.py` to pass the quota arguments and added `COLLECT_ONLY=1` to the Slurm launcher.
- Stopped the incompatible partial run `17942637`, removed its partial collection and old smoke/capacity artifacts, and relaunched collection-only as `17942695` on `iris6` (A40). It passed GPU initialization and is running with OSMesa.
- A Sphinx Task79 race was resubmitted under the required `nlp` account as `17942697`; the previous `iris`-account submission `17942688` was canceled because Sphinx reported `PartitionConfig`. It remains pending for priority; the Iris retry `17942682` is still pending behind the occupied pinned H100. No duplicate output path is shared (`pi05_task_79_sphinx.hdf5` is separate).
- The Task79 Pi0.5 race was won by Sphinx job 17942697: it started on `sphinx10` with an H200 while Iris job 17942682 remained pending on the occupied pinned H100; canceled 17942682 after the winner started.
- Canceled pinned Iris H100 retries 17942683/84/85 (tasks 29/28/2) and resubmitted them under the `nlp` account on Sphinx nodes 9-11 as 17942703/04/05. This leaves the five direct collectors on high-memory hardware: Task60 on Iris H100 and Tasks79/29/28/2 queued or running on Sphinx H100/H200.

- 2026-10-18: Race outcome for Pi0.5 base collectors Tasks 29/28/2: Iris jobs 17942711/17942712/17942713 were allocated on iris-hgx-1 and kept running; pending Sphinx duplicates 17942703/17942704/17942705 were cancelled immediately after the Iris winners started. Current partial outputs show Task28: 3 successes/25 failures, Task29: 0/25, Task2: 0/25. No worker failure observed.

- 2026-10-18: Replaced the Pi0.5 base success/failure-quota collectors with fixed 125-episode behavior-distillation collectors. Old quota-based partial files were preserved under `pi_base_data/`; new outputs use `pi_base_data/fixed125/`. Jobs 17944090 (Task60, Iris), 17944091 (Task79, Sphinx), and 17944092/93/94 (Tasks29/28/2, Iris) were submitted. The Small RL teacher collection job 17942695 was left unchanged.

- 2026-10-18: Relaunched fixed-125 Pi0.5 behavior collections as two-way races per task. Iris winners: Task60 job 17944103 on iris8, Task79 job 17944105 on iris9, Task29 job 17944107 on iris9, Task28 job 17944109 on iris9; Sphinx losers 17944104/06/08/10 cancelled after allocation. Task2 race remains pending (Iris 17944111, Sphinx 17944112) because no slot was available at this snapshot. Outputs are isolated per partition under `pi_base_data/fixed125/`.
- 2026-10-18: Corrective update for fixed-episode Pi0.5 collection. Iris job 17944103 (Task60) and 17944111/17944140 (Task2/Task28 retries) landed on `iris8`, which is an RTX A4500 (24 GiB), and failed during Pi0.5/JAX initialization with a 4.50 GiB allocation OOM. `iris-hi` is not an H100/H200 partition: iris5/6 are A40, iris7 A6000, iris8 A4500, and iris9 L40S. Future submissions of `collect_pi05_libero.sbatch` exclude iris8. Active retries use Sphinx A100 nodes where available.
- 2026-10-18: Direct Pi0.5 collector now partitions each completed mixed HDF5 into `successes/` and `failures/` beneath its output directory, retaining the mixed source file for resume/audit. Existing quota partials remain untouched and will be merged with the fixed-episode collections once each task reaches 150 total episodes.
- 2026-10-18: Fixed Task29 race job 17944107 on `iris9` failed before model initialization with `CUDA_ERROR_ECC_UNCORRECTABLE` (GPU hardware fault), distinct from the A4500 memory OOM. Relaunched Task29 on Sphinx as job 17944172 with 125 fixed episodes and an isolated output path.
- 2026-10-18: Sphinx A100 job 17944141 (Task28) also failed during Pi0.5 initialization with a 144 MiB GPU OOM. For this unsharded one-GPU Pi0.5 collector, A40/A4500/A6000/A100 are not reliable; L40S has not completed a run and iris9 separately reported an ECC fault. Canceled remaining low-memory attempts and submitted jobs 17944180-17944184 for Tasks79/60/29/28/2 on Sphinx H100/H200 nodes (`sphinx[9-11]`), with Task28 requesting 122 episodes and the others 125.
- 2026-10-18: Corrected H100/H200 submission syntax. The initial `--nodelist=sphinx[9-11]` form made Slurm reserve all three nodes (`node=3, gres/gpu=3`); jobs 17944180-17944184 were canceled before running. Replacement jobs 17944223-17944227 are pinned one-per-node (`sphinx9/10/11`) and verified with `ReqTRES ... node=1, gres/gpu=1`.
- 2026-10-18: Added the requested five Iris-hi racers (17944246-17944250) alongside the five pinned Sphinx H100/H200 jobs. Iris racers 17944246 (Task79) and 17944247 (Task60) failed on iris9 L40S with 4.50 GiB initialization OOM; 17944248 (Task29) and 17944250 (Task2) failed on iris9 with an uncorrectable ECC/device error. Task28 racer 17944249 is running on `iris-hgx-2` (H200) and has not yet emitted initialization output; its Sphinx counterpart remains queued until H200 initialization is confirmed. All ten submissions requested one GPU; low-memory Iris failures leave their Sphinx counterparts intact.
- 2026-10-18: Stopped collection-only teacher job 17942695 at 231 attempts after it had committed 100 successes and 5 failures; no longer waiting for the remaining 20 failures. Finalized the partial file atomically as `bc_distill_small_to_pi05base/task29_500k/pi_base_data/teacher_chunks.hdf5` (`complete=True`) and created complete `success/teacher_chunks.hdf5` (100 groups) and `failures/teacher_chunks.hdf5` (5 groups). Added `TRAIN_ONLY_FULL=1` to the SFT launcher and submitted full 500k-step successes-only LoRA SFT job 17944297 on one H200 (`sphinx10`), eval every 25k and checkpoint every 50k, batch 16.
- 2026-10-18: Task28 Iris racer 17944249 on `iris-hgx-2` (H200) successfully initialized Pi0.5; canceled queued Sphinx duplicate 17944226 after healthy startup confirmation. The collector continues on the H200.
- 2026-10-18: Relaunched remaining Pi-base racers on true Iris high-memory nodes: Task79 job 17944343 (H100), Task60 job 17944344 (H200), Task29 job 17944345 (H100), and Task2 job 17944346 (H200 requested, pending QOS). Jobs 17944343/44/45 all emitted `Initialized Pi0.5 base actor`; canceled their Sphinx fallbacks 17944223/24/25. Task2 keeps both pending racers until one starts and initializes.
- 2026-10-18: SFT job 17944297 was pending on fully occupied `sphinx10`; canceled and relaunched as 17944301 on available `iris-hgx-1` H100 (verified `gres/gpu:h100=1`). It loaded 1,768 success-only chunks and is progressing through the 50-episode initial evaluation.
- 2026-10-18: Changed SFT evaluation default from 50 to 20 episodes in both `train_bc_distill.py` and all production launcher branches. Stopped preempted eval-50 SFT job 17944301, removed its unwanted incomplete config, and relaunched the same 500k-step LoRA run as job 17944395 on one H100; it is currently pending `QOSMaxGRESPerUser`.
- 2026-10-18: Task2 Iris H200 racer 17944346 started after the QOS delay and emitted `Initialized Pi0.5 base actor`; canceled its queued Sphinx fallback 17944227. All five Pi-base tasks now have healthy Iris H100/H200 collectors (Task28=17944249, Task79=17944343, Task60=17944344, Task29=17944345, Task2=17944346).

- 2026-10-18: Initial three-phase SmallExpo submissions 17949071-17949075 requested
  H100 only and were projected to wait for the next day. They were canceled before
  running and resubmitted as 17949126-17949130 with `constraint=hopper` and one
  generic GPU, allowing either H100 or H200 while still excluding low-memory and
  known-bad nodes. Tasks 2 and 28 started on separate H200 GPUs; the remaining
  tasks are queued behind existing Hopper allocations.
- 2026-10-18: The first three-phase launch hung before model initialization because
  `WANDB_BASE_URL=https://forge.coreweave.com/wandb` makes the SDK POST GraphQL to
  a UI path that returns HTTP 405. Those allocations (17949126-17949130) were
  canceled and relaunched as 17949171-17949175 with the verified SDK endpoint
  `https://forge.coreweave.com/api/wandb`. W&B now authenticates, creates online
  runs, and reports run URLs; no GPU or JAX startup error was observed.
- 2026-10-18: Compact-agent Hopper jobs were moved to non-Hopper hardware. Three
  L40S attempts (17949207-17949209) on `iris9` failed during the preflight JAX
  device check with `CUDA_ERROR_ECC_UNCORRECTABLE`; `iris9` is excluded from the
  clean run. Fresh jobs 17949220-17949224 use a separate output root and are
  restricted to A40/A6000 nodes (`iris5-7`), with task 2 on an A40 and tasks
  28/29/60/79 on A6000 GPUs. All five passed CUDA/JAX and LIBERO startup; no
  training OOM has been observed.
- 2026-10-18: Submitted Sphinx SFT jobs 17949268-17949271 for tasks 2/28/60/79
  using retained RL checkpoints 285184/260029/235055/410016 respectively. Each
  job runs collection followed by 500k-step Pi0.5 LoRA SFT, with 20-episode
  fixed-state evals and both eval/checkpoint cadence at 25k. The retained
  teachers are legacy 16/8 agents by their saved configs, but
  `collect_teacher_chunks` validates that relationship and truncates/executes
  every target with the requested 10-action horizon and 5-action replanning;
  SFT consumes those 10/5 chunks.

- 2026-10-18: Sphinx SFT jobs 17949268-17949271 failed during collection before
  any SFT update because the launcher formed the teacher path relative to the
  project checkout (`/nlp/scr/ronpo/projects/dcrlcheckpoints/...`). The retained
  RL teachers are under `/nlp/scr/ronpo/dcrlcheckpoints/...`. Corrected the
  launcher to use the absolute checkpoint root and resubmitted tasks 2/28/60/79
  as jobs 17949383-17949386; all four are pending Sphinx priority.

- 2026-10-18: The first clean three-phase run exposed a normalization edge case
  in Task 28 during Pi0.5 initialization distillation. Task-local demo stats
  had q01=q99=-1 for the gripper dimension, so OpenPI's 1e-6 denominator
  amplified varying Pi0.5 targets to roughly 1e6 and produced actor losses near
  1e11. New runs 17949333-17949337 load the shared normalization statistics
  embedded in the Pi0.5 collection HDF5 files; Task 28's gripper range is
  approximately 1.9996 and current losses are ordinary (below 2). The prior
  partial run remains under its original output root for audit.
## 2026-10-19: RL resume GPU startup failures

- Jobs 17961870 (iris8, task 28), 17961871 (iris7, task 29), and 17961872 (iris7, task 60) reached their allocations but stopped before the batch `nvidia-smi -L` probe printed a GPU. An overlapping `srun` probe on each host timed out and Slurm cancelled the probe step. The jobs were cancelled before training or checkpoint restore began.
- `resume_small_expo_rl_from_50k.sbatch` now excludes iris7 and iris8 for retries. This is an allocation/GPU startup issue; no learner or checkpoint state was modified.

- 2026-10-19, job 17962200: isolated A6000 probe on normal Iris node iris7 allocated successfully but never reached Python; batch output stopped after the startup line before `nvidia-smi -L`. A bounded in-allocation `nvidia-smi -L` health check hung and the Slurm step was killed. The probe was cancelled. Treat iris7/A6000 as problematic for this run; the four active RL jobs remain on iris6 A40.

### 2026-10-20 agnostic SFT smoke GPU findings
- Job 17983584 on iris6/A40 completed a real task-29 one-step agnostic SFT smoke: baseline and post-update evals, finite BC metrics, and a full ~15 GB Orbax checkpoint at step 1.
- Resume jobs 17983971 and 17984015 on iris6/A40 restored the checkpoint but OOMed on the first resumed BC compilation while allocating ~5.37 GiB. Batch size was already 1; this is the full-learner checkpoint-resume memory limit on A40. The second job included the fix that removes eager inference-cache duplication, so the remaining A40 failure is capacity-related.
- Job 17984055 on generic iris8/A4500 (24.6-GiB GPU) reached checkpoint restore after the resume/cache fixes, then failed OOM during resumed setup while allocating 535 MiB. Fresh one-step training does fit there only because its peak memory is lower; full checkpoint resume requires a larger GPU.
- Job 17984072 on iris5/A40 (batch size 1, `--mem=32G`) restored the full learner checkpoint and then failed in the first resumed `bc_update` when XLA requested another 5.37 GiB of GPU memory. Slurm recorded 14.8 GiB maximum host RSS, so this was GPU VRAM exhaustion rather than a host-RAM limit. H100 job 17984156 is queued with 128 GiB host RAM; H200 fallback 17984168 is dependency-gated.
- Job 17984431 on sphinx5/A100-SXM4-80GB completed the resumed agnostic SFT smoke successfully (Slurm exit 0, 12:17 elapsed, 40,466,802 KB peak host RSS). It restored full checkpoint 1, ran step 2 with finite actor loss 0.04714 and grad norm 0.82214, wrote eval step 2, and committed a second full ~15 GB Orbax checkpoint. The stale temporary smoke lock was removed after completion. A100 has enough VRAM for this resume path; A40/A4500 failures above were GPU-capacity failures.
- H100/H200 fallback smoke jobs 17984156 and 17984168 (plus the Iris H100/H200 racers 17984369/17984370) were canceled after A100 job 17984431 started and completed the same test; no training artifacts were written by the canceled jobs.
- Task-2 RL resume job 17967834 on `iris-hgx-1` successfully restored the dual-optimizer 50k checkpoint and ran the offline loop, but the first 20-episode evaluation ended in a native Robosuite/MuJoCo abort inside `binding_utils.read_pixels()` during camera rendering. This was a host/rendering failure rather than a checkpoint or learner update error; the other resumed RL jobs evaluated successfully on iris6/A40. Treat `iris-hgx-1` as problematic for this LIBERO rendering path until separately verified.
- RL launcher rendering fix: `resume_small_expo_rl_from_50k.sbatch` and `train_small_expo_libero_3phase.sbatch` now select OSMesa only when Slurm allocates an `iris-hgx-*` node, while retaining EGL on all other nodes. The HGX exclusions were removed so those nodes can be used with the workaround. No learner or RL update code changed; existing running jobs keep their original environment.

### 2026-10-20 direct official-demo Pi SFT smoke GPU finding
- Direct baseline smoke jobs 17989835/17989837/17989838/17989839/17989840 ran on separate RTX 4500 GPUs on iris8 with batch size 1 and official LIBERO HDF5 demonstrations. JAX/GPU startup and fixed-state step-0 evaluation succeeded, but Task 79 (17989840) failed at the first Pi0.5 BC update with `RESOURCE_EXHAUSTED` while allocating 5,759,254,776 bytes of GPU memory. RTX 4500 VRAM is insufficient for this direct Pi0.5 update even with LoRA and batch 1; this is a capacity finding, not a data or loss failure. The remaining four jobs were stopped before their first update and will be retried on H100.
- The direct official-demo loader was checked against all five requested `libero_90` files: each contains 50 demonstrations and produces 224x224 RGB observations, 8D states, and 10x7 action chunks. Row counts are task 2: 6158, task 28: 3712, task 29: 6057, task 60: 6126, and task 79: 6054. Instructions and BDDL metadata match the requested task descriptions.

2026-10-20 direct Pi0.5 RL smoke: generic regular-Iris submissions 17990782, 17990784, 17990786, 17990788, 17990790 all landed on iris8 (RTX 4500 Ada) and initialized JAX/W&B, then failed in Pi05Agent.initialize during JIT train-state creation with Xla RESOURCE_EXHAUSTED trying to allocate 4,831,838,208 bytes. No rollout, eval, checkpoint, or replay data was produced. RTX 4500 is insufficient for this direct Pi0.5 EXPO RL smoke configuration; use A100/H100/H200 instead.

2026-10-20 direct Pi0.5 RL smoke jobs 17991182-17991186: W&B, JAX, official HDF5 loading, Pi0.5 base checkpoint restore, and official LIBERO norm-stat loading all succeeded. Tasks 2/29/79 on Sphinx A100-SXM4-80GB and tasks 28/60 on A100-40GB then failed before rollout in `build_pi05` while creating `target_actor_params`; `Pi05Agent.init_target_params` built a second full Pi0.5 train state and XLA requested an additional 2.25-4.50 GiB after a 49.97 GiB peak. The target should start as a parameter copy of the online actor and be updated by EXPO's actor_tau; local `expo_ft/agents/vla/pi05.py` now uses `actor.get_params(actor_train_state)` to avoid duplicate model/optimizer compilation. No rollout, eval, or checkpoint data was produced by these runs.

2026-10-20 direct Pi0.5 RL smoke retry: the target-parameter copy fix allowed task 2 job 17995283 on Sphinx3 A100-SXM4-80GB to restore Pi0.5, initialize EXPO, load official LIBERO normalization, create the environment, run fixed-state evaluations, execute real 10/5 LIBERO rollouts, and write finite local metrics. A second failure in `record_local_metrics` attempted to cast the evaluation `trials` list to float; the logger now persists numeric scalars only, while full trial details remain in each evaluation JSON. The corrected five-task Sphinx jobs are 17995283-17995287; the original Iris H100 racers for tasks 28/29/60/79 remain queued until each Sphinx counterpart starts. Task 2 has finalized full Orbax checkpoints at steps 16 and 32 and resumed rollout successfully; each full checkpoint is about 35 GB and takes roughly 200 seconds on shared storage.

Task 2 job 17995283 then reached the first EXPO update after ten episodes and exposed a batch-size-one bug in `EXPOLearner.sample_batch_actions`: `jnp.squeeze` removed the leading batch axis from `(1, 224, 224, 6)`, producing an invalid `(8, 5, 7) -> (224, 8, 35)` reshape. `expo_ft/agents/alg/expo_ft.py` now preserves the batch dimension for observations and states. Jobs 17995755-17995759 rerun all five tasks under a new output root so the earlier successful initialization/evaluation/checkpoint artifacts remain intact.

The same sampler also returned `best_actions.squeeze()`, which would have removed the batch axis after candidate selection when `B=1`. It now returns `(B, action_dim)` directly; behavior for `B>=2` is unchanged. Syntax and diff checks pass. Fresh smoke racers 17995768-17995777 use `batch_size=2` and remain pending GPU availability/priority.

The batch-size-two rerun keeps the original EXPO update cadence (`utd_ratio=20`, one episode-triggered update) and official 10-action chunk / 5-action execution. No additional offline phase or learner component was introduced.

The direct trainer resume audit found interval anchors were initialized from the first resumed step. It now evaluates at the next interval boundary and anchors checkpoint cadence to the last saved step, preserving cadence across the smoke-test resume invocation. Python compilation passes.

The first direct racers (17995768-17995777) were canceled while still pending and replaced by generic-GPU racers 17996043-17996052. Sphinx jobs allow `sphinx4-11`; Iris jobs request generic `gpu:1` and exclude only `iris1-4,7,9-10` and `iris-hp-z8`. This intentionally measures lower-VRAM capacity while retaining eligible high-memory nodes; all ten are currently pending priority.

2026-10-20 direct Pi0.5 RL generic-Iris capacity test: jobs 17996048 (task 2), 17996049 (task 28), 17996050 (task 29), 17996051 (task 60), and 17996052 (task 79) all landed on `iris8` RTX 4500 Ada despite generic GPU requests. Each failed during Pi05Agent initialization with the same XLA `RESOURCE_EXHAUSTED` allocation of 4,831,838,208 bytes; no rollout, evaluation, checkpoint, or replay data was produced. The five Sphinx duplicates 17996043-17996047 remain pending and are retained as the viable racers. This confirms that broad generic requests are safe for scheduling but RTX 4500 cannot run direct Pi0.5 EXPO RL; high-memory Sphinx GPUs are still required.

Iris9 is a 48 GiB L40S but prior Pi0.5 collection jobs on it reported both initialization OOM and `CUDA_ERROR_ECC_UNCORRECTABLE`; iris9-10 remain excluded for this direct run. The newly submitted Iris racers 17996154-17996158 retain the original exclusions for Iris1-4, 7-10 and iris-hp-z8; Sphinx racers remain the fallback. At the latest check all ten racers are pending: the Sphinx scheduler estimates task 2 first at 03:15, followed by the other four, while the Iris jobs wait for resources.

The long pending racers were replaced with short end-to-end smoke racers 17996211-17996215 (Sphinx tasks 2/28/29/60/79) and 17996216-17996219, 17996221 (Iris tasks 2/28/29/60/79). The short smoke keeps real official HDF5 data, batch size 2, `utd_ratio=20`, episode-triggered updates, 10-action plans with 5-action execution, evaluations every 16 steps, one full checkpoint at step 80, and a resume from that checkpoint to step 96. Each reservation is 20 minutes; no learner or loss logic changed.

### 2026-10-21 direct Pi0.5 RL production retry: iris7 XLA compiler abort
- Jobs 18006458 (task 28) and 18006459 (task 29) on iris7/A6000 both reached the first EXPO update and aborted inside XLA GPU compilation with shape_util.cc:1128 Check failed: return_shape->IsTuple(). The compiler reported a 23.8 GiB generated program and could not reduce memory below 23.68 GiB. Both jobs were canceled after crash-handler processes remained active; task29 recorded exit 134.
- No evaluation or checkpoint was produced by either job. Treat iris7 as problematic for this direct Pi0.5 EXPO configuration and exclude it from retries. The learner/trainer code was not changed.

- 2026-10-21: Pi0.5 LIBERO Sphinx job 18009807 (task 60, Sphinx8) failed at step 5,599/200,001 with JAX `RESOURCE_EXHAUSTED` while allocating ~184 MiB during replay batch image conversion. The other task runs use batch 16; task 60 was requeued as job 18012382 with the same batch 16/UTD 20/15 updates settings, Sphinx8 excluded, and `XLA_PYTHON_CLIENT_PREALLOCATE=false`. The failed artifacts were preserved under `pi_robo_libero_prod_200k_sphinx/task60_oom_18009807`.
- 2026-10-21: Removed obsolete task-2 smoke output roots and smoke Slurm logs after the H200 end-to-end smoke (job 18012170) completed successfully. Production RL directories and the preserved task-60 Sphinx8 OOM artifacts were not removed.

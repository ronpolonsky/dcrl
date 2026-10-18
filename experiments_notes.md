
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

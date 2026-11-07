# gpue05 full EXPO runtime debugging (2026-11-01)

Research invariants: official pi05_libero weights, EXPOLearner, all task demos
(`num_data=0`), offline_ratio=0.5, horizon/replan 10/5, render 256x256 then the
original 224 observation adapter. Batch 16, UTD 20, num_updates 15.
Shared project trainer/environment files have not been edited by these tests.

Native gdb reproduction (`pi05_native_debug_task29.log`) caught SIGABRT in
`libnvidia-eglcore.so.595.71.05`, called by MuJoCo 3.8.1 `mjr_readPixels`.
The initial claim that CUDA/physical EGL remapping is the root cause is NOT
proven and has now been ruled out by UUID. CUDA and OpenGL both report
`adb553ae-0dad-02ab-a1db-c64a86ae9511` on allocation 23392681.
A physical-device override fails, but local device 0 successfully
renders through NVIDIA H200. All 251 recorded policy actions are finite and
replay successfully without the learner on the same allocation.

Unsuccessful full-learner experiments:

- Explicitly bind the environment's GL context on every render.
- Separate LIBERO rendering into another process on the same assigned GPU.
- Disable NVIDIA threaded optimizations and call glFinish around readback.
- Also disable NVIDIA shader disk caching.
- RGBA transfer alone: task29 passed initial evaluation, then aborted during
  training at about env step 121; task79 aborted during initial evaluation.
- RGBA transfer plus explicit context binding also aborted for tasks29/79.
- Pixel-buffer transfer (PBO) also aborted on tasks29/79, in glGetBufferSubData.
- Both `XLA_PYTHON_CLIENT_ALLOCATOR=platform` and
  `TF_GPU_ALLOCATOR=cuda_malloc_async` still aborted with the original trainer.
- Disabling the JAX persistent compilation cache still aborted on task29.
- The known-working task2 also aborted on gpue05 using the original trainer;
  its protected gpue04 production run remains healthy and unchanged.

Read-only resource checks after these failures found: 96 GiB memory limits,
peak under 40 GiB, zero cgroup OOM/max events, unlimited memlock, ample host
RAM and shared memory, and NVIDIA recovery action None. The loaded NVIDIA
EGL library hashes on healthy gpue03 and failing gpue05 are identical.

Failed candidate: `train_pi_robo_libero_pbo_readback.py` explicitly binds
each environment's GL context and uses an OpenGL pixel-pack buffer to stage
the readback. It follows MuJoCo's multisample resolve/framebuffer selection.
No renderer, physics,
camera, resolution, image transform, policy, action, or data changes.
`verify_rgba_readback.py` passed exact RGB equality for 251 states x 2 cameras
when reading the SAME framebuffer, without rerendering between captures.
The PBO candidate also passed the same check (`VERIFY_PBO=1`), bit-identical
to the original RGB output for all 502 captures.
Separate renders/constructions are not suitable for a bitwise comparison.
Environment shutdown emitted EGL destructor warnings after the passing check.

Previous isolated tmux smoke sessions (all now terminal; task60 also aborted):

| Task | Allocation | Session | Log under /work/hdd/bgqe/rpolonsky |
| --- | --- | --- | --- |
| 29 | 23392681 | pi05-pbo-task29-h200 | pi05_pbo_task29.log |
| 60 | 23392680 | pi05-rgba-task60-h200 | pi05_rgba_task60.log |
| 79 | 23324737_5 | pi05-pbo-task79-h200 | pi05_pbo_task79.log |

Smoke-only cadence: max_steps=800, max_episode_steps=400, eval every 400,
one eval episode; checkpoints and buffer saving disabled. Production would
restore the requested 200k/20k/20-episode cadence only after verification.
Outputs are node-local /tmp/rpolonsky-egl-context-task*/output.
The candidate is NOT validated for training until evaluation, real updates,
and subsequent evaluation complete. Checkpoint writing is not tested here.

Completed diagnostic: pi05-driver-strace-task02-gpue05 on allocation 23392681.
Original logged trainer, unchanged model/data/rendering, failed syscalls only.
Logs under /work/hdd/bgqe/rpolonsky/pi05_driver_strace_task02*.log.
It also aborted. No host memory exhaustion, quota, or file-limit error was
identified at the failure. Enumeration denials on unallocated /dev/nvidia*
devices are expected and do not establish a cause.
No successful production relaunch for tasks29/60/79 yet.

Successful rollout test: separate the GPUs used for learning and rendering. Original
EXPOLearner on allocation 23324737_5 (physical GPU1), original LiberoTaskEnv
workers on allocation 23392680 (physical GPU0), authenticated loopback IPC.
Session pi05-separate-render-task29-h200, log pi05_separate_render_task29.log.
Learner is unsharded. No render monkey-patches are active in this wrapper.
The remote boundary passed shape/dtype, exact executed actions, reset,
benchmark episode-index reset, and close checks. The nested renderer srun
needs --cpu-bind=none and --cpus-per-task=1 to avoid inherited CPU bindings.
The worker reports renderer UUID f991e81e-9a1c-1d45-8cff-112770988a9a.
No shared trainer/environment/learner files were changed.

This completed all 801 loop iterations (max_steps=800), initial evaluation,
and a later evaluation at step771 with no crash. Both one-episode evaluations
had success0/1. This is RENDER/ROLLOUT evidence only: the original trainer's
`finish_episode` gates updates on ep_count>=10, so 800 steps did not exercise
learning. Do not describe this as a verified learner-update fix.

Current update-validation tests use max_steps4500, max_episode_steps400,
the original 10-episode update gate, batch16/UTD20/num_updates15 unchanged:

| Task | Learner allocation | Session | Log |
| --- | --- | --- | --- |
| 29 | 23324737_5 | pi05-render-update-task29-h200 | pi05_render_update_task29.log |
| 79 | 23392681 | pi05-render-update-task79-h200 | pi05_render_update_task79.log |

Both use 23392680 as a shared renderer-only allocation. Their /tmp roots are
`/tmp/rpolonsky-render-update-task29` and `...task79`, isolated from previous
smoke outputs. Both checkpoint and buffer writes remain disabled. Nested
renderer steps use TMPDIR=/tmp because Slurm allocation-local /tmp paths
from a learner are not visible inside the rendering allocation.

The H200-renderer parallel update tests failed at initial evaluation with
Pi0.5's image max>1 assertion (black images), before any gradient update.
Do not use that configuration for production.

Current route (2026-11-01 ~08:10): use previously idle A100 allocation 23380729
on gpua019 solely for original NVIDIA EGL rendering. All three learners stay
on separate H200 allocations on gpue05. Authenticated TCP carries unchanged
arrays; RENDER_RPC_HOST=gpue05.delta.ncsa.illinois.edu. A standalone cross-node
test passed shape/dtype, nonblack frames, exact actions, and clean shutdown.
The A100 allocation expires 2026-11-02T11:26:04 CST (~27 hours at test start).

| Task | Learner allocation | Session | Log |
| --- | --- | --- | --- |
| 29 | 23324737 | pi05-a100-render-smoke-task29-h200 | pi05_a100_render_task29.log |
| 60 | 23392680 | pi05-a100-render-smoke-task60-h200 | pi05_a100_render_task60.log |
| 79 | 23392681 | pi05-a100-render-smoke-task79-h200 | pi05_a100_render_task79.log |

Separate temp roots /tmp/rpolonsky-a100-render-task{29,60,79}; max4500,
batch16/UTD20/num_updates15, original max episode400 and 10-episode gate.
All cameras at render256 -> original224 adapter, all offline demos, ratio0.5.
All three completed their first 15-call learner-update cycle and resumed
rollout (verified 2026-11-01 ~08:25 CST). Cold compilation/autotuning took
roughly 10 minutes for the first call, with further compilation in calls2/3;
subsequent calls take approximately 5.9 seconds. Tasks29/60 have logged finite
actor/critic gradient norms and losses, editor loss, and temperature loss;
next_q_nan_ratio=0. Their next update cycles have started. Task79 also completed
15/15 and resumed rollout beyond step3199. End-to-end smoke completion,
post-update evaluation, and persistent checkpoint storage remain to verify.
Checkpoint writes are disabled. Original seven runs remain protected.

Protected: original Task2/28 full-run sessions and all five small-RL sessions;
no Slurm allocations cancelled. Removed nine failed-test core dumps (~30 GiB
allocated) and old failed/smoke checkpoint metadata directories. Official
weights and live-run checkpoints were preserved. All diagnostics now disable
core dumps. Shared /projects and /work report no available capacity via df;
do not assume production checkpoint storage is solved by this runtime work.
Quota check: project 20951 /projects is ~523 GiB used, 500 GiB soft / 550 GiB
hard with ~6 days grace. /work has ~21788 GiB used, 20000 GiB soft / 22000 GiB hard,
with ~23 hours grace. /work/hdd and /work/nvme share that project quota.

## Relaunch preparation (2026-11-01)

`launch_full_expo_remote_render.sh` is a separate copy of the remaining-task
launcher. Its training/evaluation/checkpoint CLI flags compare identically:
200000 environment steps, batch16, fsdp1 (unsharded), UTD20, num_updates15
per episode trigger after the original ten-episode gate, horizon10/replan5,
all demos (num_data0), offline_ratio0.5, eval20 every20000 steps, eval seed7,
model and replay checkpointing every20000 steps, max_to_keep1.
It adds only the renderer wrapper/environment, logging/thread settings,
core-dump suppression, and refusal to overwrite numeric saved checkpoints.
Original launchers, trainer, environment, learner and config are untouched.
Supply RENDER_JOB_ID=23380729 and RENDER_RPC_HOST=gpue05.delta.ncsa.illinois.edu.

The runtime imports `/projects/bgqe/rpolonsky/dcrl/expo_ft` (a real directory,
not a symlink); do not confuse it with the separate third_party/expo-ft source
copy. The Python interpreter is the third_party/expo-ft/.venv interpreter.
`inspect_full_expo_smoke.py` reads actual metrics/evaluations, checks finite
actor/critic/editor/temperature metrics and zero Q NaN ratios, and optionally
requires the final4500 step plus at least one evaluation after learning.
All three had six completed15-call cycles, finite metrics and post-update
evaluations by ~08:34. These tests deliberately do not write checkpoints.

SHA256 verified all three copies (dcrl/datasets, /projects/.../libero_demos,
and /u/.../libero_demos) are byte-identical for each task:

- Task29: 96f36383107129d5e802923f7dd6b96870c9d58b05c2739c2f85cf8d9295c70c
- Task60: 1c6bd7024ecbac964971669e18425f61da135663b0b2a7ade58c02dd5bce7e93
- Task79: 96ce64d79589b52e4ab8ab05633e75dced5cf0acc691d78f9f7a8473ffce82b4

Latest quota check08:25: /work project usage22910544992KiB,
hard23068672000KiB (~151GiB headroom), soft-quota grace20h20m58s.
The two original full runs' checkpoints and all small-run checkpoints remain
untouched. max_to_keep1 applies to model snapshots, not cumulative replay
transition files. Remaining capacity and the grace deadline both matter.

Protected source hashes recorded before relaunch:

- train_pi_robo_libero.py: cbeec9c63fae17039c61cf7c27c4abcd54d492df1a0df66833fefeca47f57a2b
- dcrl/libero_env.py: bca8c9166df0ce6698dad25da3b4f0092af54b7e2b4ff7120a7716ab0a59d6f6
- configs/model/expo_ft_pi_libero_config.py: d646e4d46b556ab5cb553ee777c11ac7084846b22196b5d044f7ba6f6f18ab05
- original launch_unsharded_h200.sh: e12547a7f281ef24e111a127bc6a4bb02c8d88340d294ae2d712d352962ba876
- train_pi_robo_libero_logged.py: 952b3f7cc3bf45e5b410ada2751822afbdf36558d7824ed2f47ab4588f6e7d54

## Smoke results and production relaunch (~08:42 CST)

All three smoke loops reached step4500, with finite actor/critic gradients
and losses, editor loss, temperature loss and zero reported Q NaN ratios:

| Task | Completed15-call cycles | Agent.update calls | Last actor_state_step metric | Post-learning evaluations |
| --- | --- | --- | --- | --- |
|29|10|150|149|7|
|60|9|135|134|6|
|79|9|135|134|4|

Reports are `logs/smoke_task{29,60,79}_report.json`; evaluation JSONs are in
`logs/smoke_task{29,60,79}_evals`. The checkpoint-free allocation-local temp
roots `/tmp/rpolonsky-a100-render-task{29,60,79}` were removed only after
completion, learner GPU release, and archiving these results.

The old renderer subprocesses acknowledged env.close but sometimes hung in
interpreter teardown, causing their owning srun to terminate that step.
No allocation was cancelled. The isolated wrapper now exits its worker with
os._exit(0) after explicit env.close, socket closure and stdout/stderr flush,
avoiding repeated EGL destructors. A separate original-env reset/close check
returned nonblack frames, renderer exit0 and close time0.25seconds (see
renderer_close_check.log). No learner/training behavior changed for this fix.

Fresh200k production runs were launched after verifying all three assigned
H200 GPUs had no compute processes. Launch manifest:
`full_expo_remote_render_launch.json`. Sessions and online W&B runs:

|Task|tmux|W&B run ID|
|---|---|---|
|29|pi05-unsharded-b16-task29-h200|bdfj1cik|
|60|pi05-unsharded-b16-task60-h200|6lsul0nd|
|79|pi05-unsharded-b16-task79-h200|n52791qx|

Logs: `logs/full_expo_task{29,60,79}.log`. W&B reports successful login and
online syncing to https://forge.coreweave.com/api/wandb/distill_rl/dcrl .
Output parent remains
`/work/hdd/bgqe/rpolonsky/dcrlcheckpoints/pi_robo_libero_unsharded_h200_b16_200k`;
each task writes to its own `pi_robo_libero_unsharded_h200_b16_taskXX` directory.
Their first new production updates/checkpoints are not yet verified here.

Original source/config/launcher/logging-wrapper hashes matched after launch.
All11 running allocations remained running, including all five small-RL jobs
and the two original full-EXPO jobs. Small-RL task60 is alive and CPU-active,
but its metrics have been stale at online250095/global300095 since ~08:14;
do not claim healthy forward progress for that protected process without
further evidence. It has not been modified or restarted. At ~08:48 task60
resumed progress on its own to online251872, with fresh metrics for all five
small runs. The earlier stale interval ended without intervention.

Production data checks printed actual loaded lengths29=6057,60=6126,79=6054,
all max_episodes0. All three restored official pi05_libero weights, created
their original training/evaluation environments and entered initial evaluation.
At08:49 all three production learner PIDs3935977(task29),3935978(task60),
3936602(task79) were live, each using17017MiB; sampled GPU activity confirmed
evaluation computation. No new production gradient update or checkpoint has
yet been observed. The active goal remains open for that verification and
the unresolved persistent-storage grace deadline. A question about alternate
persistent quota or extending bgqe quota is pending with the user.

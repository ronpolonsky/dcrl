# SmallExpoLearner: synchronous EXPO-FT on LIBERO

The trainer is [`train_libero.py`](train_libero.py). It follows EXPO-FT's
[`train_pi_robo.py`](https://github.com/pd-perry/expo-ft/blob/main/train_pi_robo.py)
control loop and calls the learner's existing `sample_actions()` and `update()`
APIs. Each run trains and evaluates one LIBERO task.

## Run

Install the EXPO/OpenPI environment plus LIBERO, its BDDL/initial-state assets,
and the LIBERO demonstration HDF5 files. This environment currently lacks
LIBERO, so actual simulator rollouts have not yet been run here.

```bash
python -m dcrl.train_libero --suite libero_90 --task-id 60 \
  --demos-root datasets \
  --num-offline-steps 2000 --offline-eval-interval 250 \
  --num-steps 10000 --output-dir runs/small_expo_libero
```

Use `--task-id` to select one zero-based task from the suite. The default is
LIBERO-90 task 60; tasks 79, 29, 28, and 2 can be run separately. Task names
come from LIBERO's benchmark API. The default update schedule matches
upstream's synchronous script: one update after each episode, 64 examples per
critic minibatch, UTD ratio 20, and a 10-completed-episode warmup. Switch to
`--update-type step` or `batch` if needed. `--offline-ratio 0` seeds all
demonstrations into online replay, as upstream does; a positive ratio keeps a
separate offline replay and mixes its critic and actor examples. `--num-demos`
limits the number seeded into replay; 0 uses all.

`--num-offline-steps` runs a first `tqdm` loop of complete EXPO learner updates
on demonstrations before the online action loop. Each offline update samples
`batch_size * utd_ratio` critic examples and `batch_size` successful actor
examples from the demo replay, then calls the same `agent.update()` used online.
This stage is optional (default 0) and is an addition to upstream EXPO-FT's
synchronous trainer. The separate `--warmup-episodes 10` still applies once
online collection starts; it collects episodes before online updates.

`--offline-eval-interval` evaluates every N offline updates and at the last
offline update (default 500, 0 disables). `--eval-every` controls online
evaluation in environment actions (default 1000, 0 disables). Both use a
separate LIBERO environment. Its reset and camera defaults match OpenPI's
official π₀.₅ LIBERO client: seed 7, the first 50 benchmark initial states, 10 settling
steps with action `[0,0,0,0,0,0,-1]`, 256-pixel simulator cameras, 180° image
rotation and `resize_with_pad` to 224, and the suite-specific action limit
(400 for LIBERO-90). Evaluation uses EXPO's 8-action replan interval and sends
policy actions directly to LIBERO without a client-side clip. This differs from
OpenPI π₀.₅ evaluation, which replans every 5 actions. Each evaluation resets
the simulator and policy sampler seeds and repeats the same first 50 initial
states, so checkpoint results are comparable. The evaluator reports success rate and mean
policy actions to termination. Use `--eval-episodes` for a shorter diagnostic
run or 0 to disable evaluation entirely; `--eval-seed` overrides the
evaluation seed.

Training now uses the same camera preprocessing and 10 settling actions, and
its default episode limit is also suite-specific (400 for LIBERO-90).
Training and evaluation both execute 8 actions per sampled chunk, matching
EXPO's 8-step replay and critic targets. Training advances through the task
initial-state list using the training seed; evaluation repeatedly uses the
first 50 states with seed 7. Training updates augment replay images, while
evaluation does not. Raw HDF5 demos receive the same image transform, but were
captured at 128 pixels; live training and evaluation render at 256 pixels.

Checkpoints contain the full learner state, including actor, editor, critic,
target networks, temperature, optimizer states and RNG, plus replay rows and
sampling RNG and phase counters. Offline
checkpoints are written every `--checkpoint-every` updates and at the offline
phase boundary. Online checkpoints are written at episode boundaries after the
requested action interval and at the end. `--resume` restores a matching model
and replay snapshot and resumes the relevant phase; `--overwrite` starts fresh.

## EXPO-FT behavior retained

- The base actor samples 8 action chunks of horizon 16. The edit policy makes
  8 more candidates by editing the first 8 actions of base chunks. The target
  critic ensemble scores all 16; the highest-Q candidate is selected.
- The selected chunk executes one action at a time for 8 steps, then the
  actor replans. Replay stores the 7D policy action sent to LIBERO.
- Replay fills each stored action chunk with future executed actions within the
  same episode and repeats the final available action in its unused tail.
- A critic sample starts at time `t`, takes the next observation at `t+8`,
  sums rewards over 8 steps with powers of `gamma=0.99` and preceding
  continuation masks, and sets `valids` to the product of the first 7 masks.
  The learner bootstraps with `gamma**8 * masks` and the minimum of 2
  randomly chosen target critics from the 10-critic ensemble.
- Every learner update augments current/next camera images, runs UTD critic
  updates, then flow-matching base-actor training on successful episodes,
  then the edit-policy Q/entropy objective and its learned temperature.
  Critic and actor target parameters use the learner's upstream soft updates.
- States/actions use OpenPI's fixed quantile normalization. The first run
  computes `norm_stats.json` using OpenPI `RunningStats` over the full task
  demonstration file; subsequent runs load it. Generated actions are
  unnormalized before execution.

## LIBERO-specific interfaces

LIBERO supplies sparse task success and seven-dimensional actions. The
environment's success or episode time limit ends an episode; either sets the
continuation mask to zero. The reward belongs to the action just executed.
There is no human-intervention control path. The small actor uses a compact
vision/flow network and deterministic 8192-bucket text hash, rather than
Pi0.5 weights or PaliGemma token embeddings. LIBERO images are rotated 180°,
resized with padding, and scaled to `[-1, 1]` before entering the learner.

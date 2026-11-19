# OCBench Pi0.5 initialization experiment

The initial sweep uses `ocbench_lite` tasks 0–26. Task 27 is Bowling, which uses normalized return and has no official binary success label. The mixed policy is evaluated on all 28 lite suite IDs, including Bowling and tasks below the standalone cutoff. The standard and lite environments use different control frequencies and episode horizons. Keep the suite fixed when comparing runs.

## Fixed protocol

| Setting | Value |
|---|---|
| Official dataset | `seohongpark/ocbench` at `fd771640f7a47879d8b06657fe05389641b5bff9` |
| Episode selection | First 200 successful training episodes per task, in filename and episode order |
| Student initialization | `gs://openpi-assets/checkpoints/pi05_base/params` for every run |
| Inputs | Native visual CPU environment, front and wrist RGB at 224×224, seven proprioceptive values, task prompt |
| Actions | Ten 7D action targets per chunk, replan after five actions |
| Training | LoRA only, zero initial LoRA-B, seed 42, batch size 4, learning rate `5e-5`, 200 warmup steps |
| Standalone screen | 2,000 updates, 20 rollouts at the task horizon, evaluation seed 100000, no model checkpoint |
| Gate | Include tasks with at least 10% standalone success |
| Mixture | Reuse the exact selected archives, sample task IDs uniformly, 2,000 updates per included task in aggregate |
| Mixed checkpoint | Save the final checkpoint only and evaluate it over the entire suite with the same reset seeds |

Each `.hdf5` archive contains the original actions and the simulation snapshots used at replanning time. Images render from these snapshots when sampled. The adjacent `.json` records every source shard, original episode number and transition range, plus the release-manifest and archive SHA-256. Identical archive copies therefore preserve the episode selection across clusters. GPU kernels and renderer drivers can still produce numerical differences across hardware and software versions.

The Delta experiment directory is `/projects/bgqe/rpolonsky/ocbench_pi05_sft_20261118`. `data/ocbench_lite` holds the selected archives and indices. `pi05_base_gcs_objects.json` records the public GCS object generations, sizes, and checksums. The full official base is cached on each assigned compute node. The archives, normalization, and evaluation records live on shared storage.

## Reproducing on another cluster

Use OCBench checkout `ab6a2c2602b9ac2196e786aea7b7fab2fe779d98`, OpenPI checkout `46407a41183b037313a383ff679683f2773b766d`, and the DCRL OCBench integration. The Delta environment used JAX and jaxlib 0.5.3, Flax 0.10.2, Orbax Checkpoint 0.11.13, NumPy 1.26.4, MuJoCo 3.8.1, dm-control 1.0.41, and h5py 3.16.0. Set `PYTHONPATH` to include DCRL, OCBench, the OpenPI `src` directory, and OpenPI's client `src` directory. Use `MUJOCO_GL=egl`, `PYOPENGL_PLATFORM=egl`, and set `OPENPI_DATA_HOME` and `XDG_CACHE_HOME` to writable caches. Transfer the selected archives and their `.json` files, or recreate an archive with the pinned release.

```bash
suite=ocbench_lite
task=0
export OPENPI_DATA_HOME=/path/to/local/openpi-cache
export XDG_CACHE_HOME=/path/to/local/cache
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.95
base=$(python -c 'from openpi.shared.download import maybe_download;print(maybe_download("gs://openpi-assets/checkpoints/pi05_base/params").parent)')
env_id=$(python -c 'from dcrl.ocbench_env import environment_id;import sys;print(environment_id(sys.argv[1],int(sys.argv[2])))' "$suite" "$task")

python scripts/prepare_ocbench_sft_data.py --suite "$suite" --task-id "$task" \
  --episodes 200 --cache "$raw_release_cache" --output-dir "$experiment/data/$suite"
python -m dcrl.collect_ocbench_demos --stats-from "$experiment/data/$suite/$env_id.hdf5" \
  --norm-stats-dir "$experiment/stats/$suite/task_$task"
python -m dcrl.train_pi_sft_agnostic --phase train --collection-source benchmark_demos \
  --source-suite "$suite" --source-task-id "$task" --official-demo-root "$experiment/data/$suite" \
  --student-checkpoint "$base" --norm-stats "$experiment/stats/$suite/task_$task/norm_stats.json" \
  --output-dir "$experiment/screens/$suite/task_$task" --num-sft-steps 2000 --batch-size 4 \
  --warmup-steps 200 --lr-decay-steps 2000 --learning-rate 0.00005 --zero-lora-b \
  --seed 42 --eval-seed 100000 --eval-episodes 20 --eval-every 2000 --skip-initial-eval \
  --checkpoint-policy none --wandb-mode disabled
```

Run `python scripts/summarize_ocbench_sft.py "$experiment" --suite "$suite" --write-selection` after all 27 screens. For the mixed run, fit one normalization from the selected archives with `--stats-from`, pass the surviving task IDs to `--benchmark-demo-task-ids`, set `--source-task-id` to the first of those IDs, and use `--checkpoint-policy final`. Evaluate the numeric mixed checkpoint with `full_90_libero_evals.py --mode sft_pi --suite "$suite" --task-ids all --eval-seed 100000 --episodes 20` and pass its output directory to `scripts/summarize_ocbench_sft.py --final-evals`.

## Full-data follow-up

The follow-up uses the same pinned release and Pi0.5 base. Pass `--episodes 0 --replan-steps 3` to the preparation script to include all successful training episodes for each task. Bowling uses all its training episodes and reports normalized return separately. Training and evaluation use three-action chunks and replan every three actions. Training draws chunks with replacement from the selected episodes using seed 42. Each rollout evaluation uses 20 episodes at the task's registered horizon and seed 100000.

On Delta, launch each task with `launch_trial.sh ALLOCATION TASK 0 UPDATES EVAL_INTERVAL` under `/projects/bgqe/rpolonsky/ocbench_pi05_sft_tuning_20261118`. The launcher attaches a detached tmux session to an idle sleeper. The dataset and generated observation archive live in node-local scratch. Shared outputs under `full_release/task_TASK` include the ordered source episode list, dataset revision, normalization statistics, training metrics and eval records. These runs use W&B project `distill_rl/sft_checkpoint` with model checkpoint writes disabled.

After all 27 success-rate screens finish, run `python summarize_trials.py --full-release --freeze-selection` to freeze the exact episode selections for tasks above zero. The full-data mixture launcher rebuilds those archives in scratch, verifies that the ordered source episode lists and release metadata match, fits one set of normalization statistics from the mixture, and samples task IDs uniformly. It takes its update budget as `run_mixture.sh --full-release UPDATES`.

### Snapshot and action alignment

An audit of the pilot archives for tasks 0, 2 and 8 verified that the sampler returns the three actions beginning at the stored pre-action state, with edge padding for a short final chunk. Four sampled rows per task matched the archived actions and the proprioception recomputed from the rendered state. Replaying the final recorded chunk from its snapshot in the CPU environment reached success on the first 20 archived episodes for each task.

Open-loop replay from the first snapshot reached success on 20/20, 15/20 and 19/20 episodes respectively. These are checks of archived expert actions, separate from the trained policy's closed-loop rollout scores. The difference between final-chunk and full-trajectory replay is consistent with state drift accumulating over open-loop replay.

### Task 0 action normalization comparison

The standalone task-0 Pi0.5 run had 9,670 successful episodes and 238,638 three-action chunks. Its 12,000-update evaluation scored 0/20 after 48,000 chunk draws, of which 43,482 were distinct. The median training flow loss decreased from 0.0819 in the first 250 updates to 0.0245 in the 250 updates preceding the eval. The [paper's state-based Lite results](https://arxiv.org/html/2610.07056v1#A4.T6) and [visual experiments](https://arxiv.org/html/2610.07056v1#A3) use different policy inputs and training budgets.

At 24,000 updates, the full-data policy solved seed 100005 in 27 steps for 1/20. The shortest-500 pilot also solved that seed at 3,000 updates, in 26 steps. The cube began 4.5 cm from the goal on this reset, the closest of the 20 resets and just beyond the 4 cm success radius. The median starting distance was 24.4 cm. A zero-action control solved 0/20 on the same resets. This first success demonstrates progress on a favorable reset and leaves performance on the other resets open.

The existing Pi0.5 quantile transform mapped axis 5 to a range of about -51.77 to 25.13. The raw 1st and 99th percentiles for that axis were approximately -0.0015 and 0.0019. Reconstructing the training batches from the saved seed matched the two largest recent loss spikes, 12.82 and 3.14, to samples with normalized action magnitudes of 51.77 and 29.35. The published [OCBench action normalizer](https://github.com/seohongpark/ocbench/blob/ab6a2c2602b9ac2196e786aea7b7fab2fe779d98/impls/envs/env_utils.py#L11-L62) clips normalized joint actions to ±5 and keeps gripper commands in their native ±1 range.

The same narrow joint-action range produced normalized magnitudes up to 103 and 374 in the full-data archives for tasks 21 and 22. The audit used each task's recorded split and normalization. A CPU test of the comparison configuration confirmed the clipping on training targets and sampled outputs, along with agreement between the training and inference RGB, state and prompt transforms.

Launch the matched control with `launch_trial.sh ALLOCATION 0 0 64000 12000 --reference-actions`. It verifies the ordered demo selection against the full-data baseline. It keeps the same Pi0.5 base, batch size, learning rate, three-action chunks and eval seeds, and applies the reference limits to normalized training targets and to outputs before de-normalization. It has a separate tmux session and writes under `full_release_reference_actions/task_0`. Compare rollout success at matched update counts to measure the effect of the action normalization.

New task-0 rollout records include the initial and minimum cube-to-goal distance, maximum cube displacement, and maximum lift in meters. This shows whether a run is learning to interact with the cube before it solves complete episodes. The success criterion and rollout seeds are identical to the earlier evaluations.

### Upstream code comparison

The installed OCBench source and GitHub master matched at `ab6a2c2602b9ac2196e786aea7b7fab2fe779d98` during this audit. On the released task-0 Lite shard, the upstream success filter and the local terminal-reward filter select the same 9,670 episode IDs. Both cap chunks at the episode boundary and repeat the terminal action for padding. The upstream dense-state loader can start a chunk at any of the 706,281 selected control transitions. Our visual archives render starts at steps 0, 3, 6 and so on, for 238,638 eligible chunk starts. Both eval paths execute the full three-action chunk for Lite.

Upstream fits the action percentiles to the full training shard before filtering for success. On task 0, the maximum arm percentile change from using successful episodes was 3.4% of that joint's full-data percentile range. The reference-action comparison preserves the existing arm statistics and tests clipping and the native gripper range together.

The upstream visual environment provides front, side and wrist images and hides the placement markers. Our LoRA runs condition on front and wrist images and proprioception. The local task-0 prompt refers to a marked target even though the target marker is fully transparent. The task has a fixed goal near the center of the workspace. Correcting the prompt and increasing the capture density are two separate comparisons that could reuse the same demonstration IDs.

The upstream CPU scripted controller solved 20/20 of our task-0 reset seeds at the same 200-step horizon. Passing that controller through our actual visual wrapper and SFT scorer on the first two seeds also returned success in the same 42 and 52 steps, using one-action replanning for the script. The learned actor uses the three-action replanning interval. The actor-only eval reads the current Pi0.5 training state. The learner clears its inference cache after each supervised update.

The task-0 prompt comparison uses `launch_trial.sh ALLOCATION 0 0 24000 12000 --reference-actions-goal-prompt`. It uses the same full demo split and action transforms as the normalization comparison. Training and evaluation both condition on “Pick up the cube and place it at the center of the workspace.” Evaluate its 12k and 24k results against the normalization comparison at those update counts.

### Task 8 as an alternate first task

Task 8 is OCBench's `chamber-lite-easy-task1`, opening both the drawer and window. It has the highest reported flow-BC success among the Lite rows in [Table 6](https://arxiv.org/html/2610.07056v1#A4.T6), 78% with a standard deviation of 9 points across the two training seeds. The published command uses state observations, successful demonstrations, three-action chunks, two million optimizer updates and batch size 1,024. The released train shard has 10,000 episodes, of which 9,590 satisfy the success filter. The local scripted CPU control scored 20/20 on reset seeds 100000–100019, with a median of 157 steps and a 500-step horizon. The previous visual Pi0.5 pilot with 1,000 demos scored 0/20 at 3,000 and 6,000 updates.

The task-8 full-data run uses `launch_trial.sh 24037145_1 8 0 160000 16000 --reference-actions`. It uses all 9,590 successful official demonstrations, the Pi0.5 base, batch size 4, and 20 rollout episodes every 16,000 updates. The run uses the reference action bounds and the benchmark prompt “Open the drawer and the window.” Its config, data manifest and eval records are under `full_release_reference_actions/task_8`. [Task-8 W&B run](https://wandb.ai/distill_rl/sft_checkpoint/runs/xlajrdnb)

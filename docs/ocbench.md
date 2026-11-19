# Running DCRL on OCBench

Use the existing trainers with `--suite ocbench` or `--source-suite ocbench`. The lite variant is `ocbench_lite`. The [task list](../dcrl/ocbench_env.py) maps DCRL's zero-based IDs to the upstream environment IDs. Suite evaluation uses each task's own horizon and the same reset seeds for each policy. Record the selected suite and backend when reporting results.

Install `ocbench==1.0.2` in the existing DCRL Python environment. Use `MUJOCO_GL=egl` and `PYOPENGL_PLATFORM=egl` for headless runs on Delta. The adapter uses OCBench's visual MuJoCo environment and the learner receives front RGB, wrist RGB, six arm joint positions and gripper opening. The recorded demonstrations also preserve the side RGB view. The task prompts are supplied by DCRL. The simulator handles seven normalized joint/gripper delta controls and reports success, termination and time-limit truncation. All runs in a comparison should use the same observation channels, demonstration split and control settings.

## Demonstrations and initial Pi0.5 policy

Set `suite`, `task`, `demos`, `assets`, `base_weights`, and `base_run` to your run configuration. `base_weights` is the directory containing the general OpenPI `pi05_base/params` checkpoint. The paths below assume `assets` is a common directory for OpenPI assets.

```bash
stats="$assets/dcrl/$suite"
python -m dcrl.collect_ocbench_demos --suite "$suite" --task-id "$task" \
  --episodes 100 --seed 42 --output-dir "$demos" --norm-stats-dir "$stats"

python -m dcrl.train_pi_sft_agnostic --phase train --collection-source benchmark_demos \
  --source-suite "$suite" --source-task-id "$task" --official-demo-root "$demos" \
  --student-checkpoint "$base_weights" --norm-stats "$stats/norm_stats.json" \
  --output-dir "$base_run" --num-sft-steps 2000 --eval-every 250 --eval-seed 100000 --zero-lora-b
```

The collector uses OCBench scripted policies and records RGB and proprioception before every action. The training replay expects dense frames and aligned proprioception. The released sparse visual datasets need conversion before they can be used by these trainers. Use training demonstrations to fit normalization and reuse the exact same stats throughout the comparison.

For a shared starting policy across tasks, collect each training task into the same demonstration directory, then run `python -m dcrl.collect_ocbench_demos --stats-from "$demos"/*.hdf5 --norm-stats-dir "$stats"`. Add `--benchmark-demo-task-ids all` or a comma-separated task list to the initial SFT command. `--source-task-id` selects the task for periodic training evaluation. Evaluate the fitted starter across the full suite before using it as the common initialization for task-specific RL.

## Compact RL and distillation

Set `adapted_base` to the selected numeric checkpoint directory under `$base_run/checkpoints` and `distill_file` to a new HDF5 path. The same offline and online trainer also runs without the initial distillation phase by setting `--init-distill-steps 0`.

```bash
python -m dcrl.collect_pi05_libero --suite "$suite" --task-id "$task" \
  --checkpoint "$adapted_base" --norm-stats "$stats/norm_stats.json" \
  --episodes 100 --output "$distill_file"

python -m dcrl.train_libero --suite "$suite" --task-id "$task" \
  --demos-root "$demos" --norm-stats-dir "$stats" \
  --distill-data-path "$distill_file" --init-distill-steps 50000 \
  --offline-rl-steps 50000 --online-rl-steps 200000 --offline-ratio 0.5 \
  --output-dir "$small_run" --eval-seed 100000
```

For learner-agnostic SFT from the compact teacher, use `--collection-source small_expo` with `--teacher-dir` and `--teacher-step`. For edited Pi0.5 proposals, use `--collection-source pi05_small_expo_editor` with `--small-expo-checkpoint` and `--pi05-checkpoint "$adapted_base"`. In both cases pass `--source-suite "$suite" --source-task-id "$task" --student-checkpoint "$adapted_base" --norm-stats "$stats/norm_stats.json"` and select a fresh output directory.

The success-filtered RL teacher pipeline applies to the 27 tasks with binary success. Bowling can participate in base adaptation using all scripted demonstrations. Task-specific Bowling RL and teacher selection need a return-based selection rule.

## Full EXPO and evaluation

```bash
python train_pi_robo_libero.py --suite "$suite" --task_id "$task" \
  --demos_root "$demos" --output_dir "$full_run" \
  --config configs/model/expo_ft_pi_ocbench_config.py \
  --config.pi05_weight_loader_path "$adapted_base/params" \
  --config.pi05_assets_dir "$assets" --config.pi05_asset_id "dcrl/$suite" \
  --offline_ratio 0.5 --replay_capacity 50000 \
  --eval_interval 20000 --eval_seed 100000 --checkpoint_model
```

The 40 GB A100 Full EXPO smoke test used `--batch_size 1 --utd_ratio 1 --offline_ratio 0`. The resume test used `XLA_PYTHON_CLIENT_ALLOCATOR=default`, `XLA_PYTHON_CLIENT_MEM_FRACTION=.98`, and `XLA_PYTHON_CLIENT_PREALLOCATE=false` on an otherwise idle GPU. The example above uses a larger research configuration. Check VRAM and host replay memory before scaling batch size or replay capacity.

Choose evaluation seeds outside the demonstration and online rollout seed ranges. Use `python full_90_libero_evals.py --suite "$suite" --task-ids all --mode MODE --checkpoint PATH --output-dir OUTPUT --eval-seed SEED`, with the same seeds across policies. Run config and checkpoint layouts remain the same for SFT and Small EXPO.

| Mode | Checkpoint and additional flags |
|---|---|
| `official_pi05` | Benchmark-adapted Pi0.5 numeric checkpoint and `--assets-dir "$assets"` |
| `compact` | Small EXPO task directory or numeric checkpoint |
| `sft_pi` | Learner-agnostic SFT run or numeric checkpoint |
| `sft_plus_editor` | SFT run and `--editor-checkpoint` pointing to a Small EXPO checkpoint |
| `direct_rl_pi` | Full EXPO run with `--assets-dir "$assets" --pi05-params "$adapted_base/params"` |

The same `--shard-index` and `--num-shards` options work for a suite split. To aggregate per-task outputs, pass `--suite "$suite"` to `scripts/merge_full_90_libero_evals.py`.

## Comparing evaluation with the OCBench paper

The per-task OCBench evaluator reports success rate and throughput. Throughput is the success rate multiplied by the episode horizon and divided by the mean length of all episodes. Bowling is scored by normalized return (100 times mean episode return) and omitted from mean success rate. Use `--episodes 500` and the registered horizon for a paper-style per-checkpoint evaluation. Tables 6 and 7 report state-based MJWarp BC baselines across two seeds and the last five checkpoints of 2M-update runs. The published baseline commands execute 25-action chunks before replanning. The DCRL visual runs predict 10 actions and execute 5 before replanning. [OCBench paper](https://arxiv.org/html/2610.07056v1) · [Upstream baseline commands](https://github.com/seohongpark/ocbench/blob/ab6a2c2602b9ac2196e786aea7b7fab2fe779d98/impls/commands.sh)

Run `python scripts/check_ocbench_eval.py --upstream-checkout /path/to/ocbench --suite ocbench --episodes 4` to compare the upstream evaluator and DCRL on identical simulator seeds and actions from the official block controller and a zero-action policy. The check asserts agreement on success, episode length, throughput, and success-weighted speed. Repeat with `--suite ocbench_lite` to check the shorter horizon or add `--task-id 27` to check bowling return.

Run `python scripts/check_ocbench_eval.py --table5-shard /path/to/000.npz` on the released `lite/block-lite-single-task1-v0/000.npz` training shard. This checks the reported episode and transition counts and recomputes the table's success rate, mean episode length, and throughput. [Pinned dataset shard](https://huggingface.co/datasets/seohongpark/ocbench/blob/fd771640f7a47879d8b06657fe05389641b5bff9/lite/block-lite-single-task1-v0/000.npz)

| Lite block scripted data | Recomputed from 10,000 episodes | Table 5 |
|---|---:|---:|
| Success rate | 96.7% | 97% |
| Mean episode length | 77.2281 | 77.2 |
| Throughput | 2.50427 | 2.50 |

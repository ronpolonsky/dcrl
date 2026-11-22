# Pi0.5 OCBench launchers

These scripts launch the Pi0.5 standalone OCBench SFT protocol used in the
long-running sweep. They keep experiment outputs outside the Git repository.

The default protocol is:

- \`ocbench_lite\` official release data, using all eligible successful episodes;
- Pi0.5 base checkpoint \`gs://openpi-assets/checkpoints/pi05_base\`;
- standard LoRA (\`lora_only\`), batch size 4, learning rate \`5e-5\`;
- 10-action prediction horizon, execute 5 actions, then replan;
- 500,000 SFT updates, evaluation every 50,000 updates with 50 episodes;
- W&B project \`distill_rl/sft_checkpoint\`;
- no checkpoints in the long JSON-only sweep unless \`CHECKPOINT_POLICY\` is
  explicitly changed.

## One task

From the repository checkout, with a running sleeper allocation:

\`\`\`bash
scripts/ocbench/launch_pi05_standalone_long_tmux.sh JOBID:TASK
\`\`\`

The task runs in its own tmux session. \`JOBID\` may be a concrete Slurm
allocation or an array element such as \`24117934_5\`.

## Automatic dispatch

\`\`\`bash
tmux new-session -d -s ocbench-pi05-dispatch \\
  'scripts/ocbench/dispatch_pi05_standalone_long.sh'
\`\`\`

The dispatcher uses one idle running \`gpu-sleep-ocbench*\` allocation per task
and creates one tmux session per task. Set \`OCBENCH_TASK_QUEUE\` to a
space-separated list to control the queue; otherwise it starts at task 10,
which is the remaining queue for the current sweep after tasks 0, 1, 2, 3, 4,
5, 6, 7, 8, and 9.

## Paths

The scripts derive the repository root and project root automatically. Override
these when using another cluster layout:

\`\`\`bash
export OCBENCH_PROJECT_ROOT=/projects/bgqe/rpolonsky
export OCBENCH_RUN_ROOT=/projects/bgqe/rpolonsky/ocbench_pi05_sft_tuning_20261118
export OCBENCH_SCRATCH=/tmp/\$USER-ocbench-pi05-sft
export OCBENCH_ENV_FILE=/work/hdd/bgqe/rpolonsky/.env
\`\`\`

\`OCBENCH_RUN_ROOT\` contains logs and results; it is not part of the Git
checkout. The scripts never stop or modify unrelated Slurm jobs.

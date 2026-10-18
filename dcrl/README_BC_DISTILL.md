# SmallExpo RL to Pi0.5 BC distillation

`dcrl.train_bc_distill` is a standalone distillation path. It loads a frozen
SmallExpo RL checkpoint, collects one complete action chunk per replanning
observation, and updates only the large Pi0.5 base actor.

The default interface matches official Pi0.5 LIBERO: predict 10 actions,
execute the first 5, then replan. Each SFT example is an observation plus the
full 10-action target chunk. `executed_prefix` is retained for rollout audit;
it is not a one-step loss mask. Success chunks are used by default; pass
`--include-failures` to mix failed episodes.

For a legacy SmallExpo checkpoint trained with 16/8 internal shapes, collection
restores that checkpoint unchanged, scores candidates with its trained shapes,
stores the first 10 actions, and executes 5. The RL checkpoint and RL trainer
are never modified. A newly trained 10/5 RL checkpoint can be used directly.

The train loop runs `--num-sft-steps`, logs each update to `metrics.jsonl`,
runs fixed-initial-state LIBERO evaluations every `--eval-every` updates, and
writes a full Orbax `state` checkpoint after each `--checkpoint-every` update.
The full state includes actor parameters, optimizer state, step, and EMA when
full finetuning is selected. `--resume --phase train` restores the latest
checkpoint and verifies the immutable run manifest. `PiExpoLearner.bc_update`
is the corresponding shared learner API; it updates only the base actor and
its target copy, leaving critic, editor, and temperature states unchanged.

Official-style defaults are 10/5, batch 256, learning rate `5e-5`, 10,000 warmup
steps, 1,000,000-step decay horizon, 1,000,000 SFT updates, and 50 eval trials.
The default finetune mode remains LoRA for the available interactive GPU; pass
`--finetune-mode full` for the full Pi0.5 update.

Example:

```bash
python -m dcrl.train_bc_distill --phase all \
  --teacher-dir /nlp/scr/ronpo/dcrlcheckpoints/small_expo_learner/libero_90/task_29 \
  --teacher-step 335076 \
  --output-dir /nlp/scr/ronpo/dcrlcheckpoints/bc_distill/task29
```

For the small end-to-end smoke run, use `SMOKE=1 sbatch scripts/train_bc_distill.sbatch`.

# Frozen compact teacher → pi0.5 full-chunk SFT

`python -m dcrl.train_full_chunk_sft` collects full selected teacher chunks,
then trains the PiExpoLearner's Pi05Agent actor backend. Teacher collection runs
in a subprocess, freeing its GPU memory before student initialization.

The teacher checkpoint is explicit (`--teacher-step` is its global checkpoint
step, including offline updates). It remains frozen; only sampling RNG advances.
Every replan stores the starting observation, full selected chunk in environment
units, executed-prefix length and episode outcome in compressed HDF5. Failed
episodes are retained. SFT defaults to successful episodes; `--include-failures`
opts into all collected chunks. Zero successful chunks is an error.
Interrupted collection resumes from the partial HDF5 and restores the teacher's
post-episode sampling RNG, so later targets keep the same continuous stream.
After collection, the mixed source is partitioned into
`data/success/teacher_chunks.hdf5` and `data/failures/teacher_chunks.hdf5`.
Training reads the success partition by default and reads both partitions when
`--include-failures` is requested.

The student keeps its pretrained LIBERO normalization assets. Raw chunks pass
through the student's input transforms and normalization exactly once. Prompts
are tokenized by the student's own tokenizer, not the compact vocabulary. Images
are already upright from the shared LIBERO adapter and are not rotated again.
All 16 teacher action positions are supervised, including the unexecuted suffix.
The teacher editor/critic operate on the 8-action prefix; suffix targets are the
corresponding base actor predictions. Evaluation executes 8 actions per replan.

Default adaptation is LoRA (both language/vision backbone's supported Gemma
adapters and action expert); `--finetune-mode full` enables full finetuning.
Training calls the same Pi05Agent.train_step used by PiExpoLearner.update_actor.
The student editor/critic are not updated in this actor-distillation stage.
Evaluation uses the student actor alone, official suite horizons, 256px rendering,
ten settling actions, and the same fixed initial states at each evaluation.

Checkpoints retain full actor parameters, optimizer state, step and EMA where
applicable. Sampling and training RNG are deterministic functions of seed and
step. Student statistics and immutable run configuration are saved alongside
checkpoints. Resume uses `--resume --phase train`; the SFT step budget may extend.
Every checkpoint is retained. Per-update metrics are in `metrics.jsonl`; individual
evaluation trial outcomes are in `evals/`. This trainer currently logs locally.

Example (check the teacher checkpoint exists before running):

```bash
python -m dcrl.train_full_chunk_sft \
  --teacher-dir /nlp/scr/ronpo/dcrlcheckpoints/small_expo_learner/libero_90/task_28 \
  --teacher-step 110016 \
  --output-dir /nlp/scr/ronpo/dcrlcheckpoints/full_chunk_sft/task28 \
  --collection-episodes 200 --num-sft-steps 10000 --batch-size 8 \
  --eval-every 1000 --eval-episodes 20 --checkpoint-every 1000
```

The Slurm script `scripts/train_full_chunk_sft.sbatch` defaults to a two-update
smoke followed by resumption to step three. For a full experiment set
`SMOKE_RESUME=0`, `COLLECTION_EPISODES=200`, `SFT_STEPS=10000`, `BATCH_SIZE=8`,
`EVAL_EPISODES=20`, `EVAL_EVERY=1000`, `CHECKPOINT_EVERY=1000`, and a fresh
`SFT_OUTPUT_DIR`. Run smoke validation before submitting the full experiment.

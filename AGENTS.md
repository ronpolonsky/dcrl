# Local execution notes

- On Delta interactive/GPU nodes, use `/projects/bgqe/rpolonsky/dcrl` as the active runtime checkout when possible.
- Keep large checkpoints, OpenPI caches, evaluation outputs, and temporary runtime files under `/projects/bgqe/rpolonsky/dcrlcheckpoints` or `/projects/bgqe/rpolonsky/dcrl/results`; use node-local `/tmp` for fast staging when needed.
- Treat `/work/hdd/bgqe/rpolonsky` as source/archive storage only. Check `df -h` before using it for model loading or large writes; it can be full even when paths remain visible.
- If source changes are made in the `/work` checkout, sync the relevant code to the `/projects` runtime checkout before launching GPU jobs.

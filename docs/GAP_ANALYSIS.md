# Known limitations

## Reproducing experiments

- ASVspoof LA/PA datasets, reference recordings, generated audio, tensor caches, and XTTS weights are external assets.
- Some notebooks and research scripts use server-specific paths. Update those paths before running them locally.
- There is no complete research dependency lockfile. The web requirements cover the inference service, with XTTS dependencies listed separately.
- AudioMamba is mentioned in earlier experiments but its implementation is absent. The model factory rejects that option.
- Saved benchmarks use an RTX A5000. They do not establish CPU latency or HTTP response time.

## Application state and deployment

- Upload metadata and result history reset when the process restarts.
- Audio files are stored on local disk. A host with ephemeral storage can lose them on restart or redeployment.
- Generation runs within the application rather than through a separate job queue.
- Classification requires both AASIST checkpoints to load.
- The frontend fetches Lucide icons from a CDN, so icons need a network connection unless bundled locally.
- The app has no authentication or request rate limiting.

## Evaluation

Default web thresholds are `0.420` and `0.118`. Record any threshold changes alongside the checkpoint and evaluation dataset used to select them.

The existing tests check repository files, configuration strings, and optional dependency separation. They do not test uploads, inference, generation, concurrency, or recovery after restart.

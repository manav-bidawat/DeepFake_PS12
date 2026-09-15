# Remaining work

## Application

- Persist upload metadata, jobs, and classification history across restarts.
- Add duration checks and content validation for uploaded audio.
- Add API tests for uploads, health reporting, missing checkpoints, and classification errors.
- Bundle frontend icons for offline use.

## Experiments

- Replace server-specific paths with configuration or CLI arguments.
- Record research dependencies and dataset setup instructions.
- Tie each evaluation result to its checkpoint, threshold, and dataset split.
- Measure CPU inference time and complete request latency.

## Hosted generation

- Move generation to a worker with a job queue.
- Store uploads and generated audio outside the application container.
- Add request limits, access controls, and monitoring before allowing unrestricted public use.

These items describe work still to do; they are not implemented features.

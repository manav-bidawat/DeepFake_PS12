#!/usr/bin/env bash
set -euo pipefail

# Build manifests once.
python -m src.data.make_manifests --data-root . --out-dir data/manifests --verify-exists

# Shared preprocessing config for Task 1.
PREPROC_ARGS=(
  --trim-silence
  --pre-emphasis
  --pre-emphasis-coef 0.97
)

# Run 1: train on LA, evaluate on LA (in-domain) and PA (cross-domain).
python -m src.train \
  --data-root . \
  --train-manifest data/manifests/la_train.csv \
  --val-manifest data/manifests/la_dev.csv \
  --output-dir models/run1/specrnet \
  --model specrnet \
  "${PREPROC_ARGS[@]}"

python -m src.evaluate \
  --data-root . \
  --manifest data/manifests/la_eval.csv \
  --checkpoint models/run1/specrnet/best.pt \
  --output-dir reports/run1/specrnet/in_domain \
  "${PREPROC_ARGS[@]}"

python -m src.evaluate \
  --data-root . \
  --manifest data/manifests/pa_eval.csv \
  --checkpoint models/run1/specrnet/best.pt \
  --output-dir reports/run1/specrnet/cross_domain \
  "${PREPROC_ARGS[@]}"

# Run 2: train on PA, evaluate on PA (in-domain) and LA (cross-domain).
python -m src.train \
  --data-root . \
  --train-manifest data/manifests/pa_train.csv \
  --val-manifest data/manifests/pa_dev.csv \
  --output-dir models/run2/specrnet \
  --model specrnet \
  "${PREPROC_ARGS[@]}"

python -m src.evaluate \
  --data-root . \
  --manifest data/manifests/pa_eval.csv \
  --checkpoint models/run2/specrnet/best.pt \
  --output-dir reports/run2/specrnet/in_domain \
  "${PREPROC_ARGS[@]}"

python -m src.evaluate \
  --data-root . \
  --manifest data/manifests/la_eval.csv \
  --checkpoint models/run2/specrnet/best.pt \
  --output-dir reports/run2/specrnet/cross_domain \
  "${PREPROC_ARGS[@]}"

# Benchmark 5-second clip profile for each run.
python -m src.benchmark \
  --data-root . \
  --manifest data/manifests/la_eval.csv \
  --checkpoint models/run1/specrnet/best.pt \
  --output-json reports/run1/specrnet/benchmark.json

python -m src.benchmark \
  --data-root . \
  --manifest data/manifests/pa_eval.csv \
  --checkpoint models/run2/specrnet/best.pt \
  --output-json reports/run2/specrnet/benchmark.json

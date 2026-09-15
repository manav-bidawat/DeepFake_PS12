#!/usr/bin/env bash
set -euo pipefail

# Build manifests once.
python -m src.data.make_manifests --data-root . --out-dir data/manifests --verify-exists

# RawNet-specific training settings.
TRAIN_ARGS=(
  --trim-silence
  --pre-emphasis
  --pre-emphasis-coef 0.97
  --augment
  --balance-data
  --amp
)

# Run 1: train on LA, evaluate on LA (in-domain) and PA (cross-domain).
python -m src.train \
  --data-root . \
  --train-manifest data/manifests/la_train.csv \
  --val-manifest data/manifests/la_dev.csv \
  --output-dir models/run1/rawnet \
  --model rawnet \
  --batch-size 64 \
  --num-workers 8 \
  "${TRAIN_ARGS[@]}"

python -m src.evaluate \
  --data-root . \
  --manifest data/manifests/la_eval.csv \
  --checkpoint models/run1/rawnet/best.pt \
  --output-dir reports/run1/rawnet/in_domain \
  --trim-silence \
  --pre-emphasis \
  --pre-emphasis-coef 0.97

python -m src.evaluate \
  --data-root . \
  --manifest data/manifests/pa_eval.csv \
  --checkpoint models/run1/rawnet/best.pt \
  --output-dir reports/run1/rawnet/cross_domain \
  --trim-silence \
  --pre-emphasis \
  --pre-emphasis-coef 0.97

# Run 2: train on PA, evaluate on PA (in-domain) and LA (cross-domain).
python -m src.train \
  --data-root . \
  --train-manifest data/manifests/pa_train.csv \
  --val-manifest data/manifests/pa_dev.csv \
  --output-dir models/run2/rawnet \
  --model rawnet \
  --batch-size 64 \
  --num-workers 8 \
  "${TRAIN_ARGS[@]}"

python -m src.evaluate \
  --data-root . \
  --manifest data/manifests/pa_eval.csv \
  --checkpoint models/run2/rawnet/best.pt \
  --output-dir reports/run2/rawnet/in_domain \
  --trim-silence \
  --pre-emphasis \
  --pre-emphasis-coef 0.97

python -m src.evaluate \
  --data-root . \
  --manifest data/manifests/la_eval.csv \
  --checkpoint models/run2/rawnet/best.pt \
  --output-dir reports/run2/rawnet/cross_domain \
  --trim-silence \
  --pre-emphasis \
  --pre-emphasis-coef 0.97

# Benchmark 5-second clip profile for each run.
python -m src.benchmark \
  --data-root . \
  --manifest data/manifests/la_eval.csv \
  --checkpoint models/run1/rawnet/best.pt \
  --output-json reports/run1/rawnet/benchmark.json

python -m src.benchmark \
  --data-root . \
  --manifest data/manifests/pa_eval.csv \
  --checkpoint models/run2/rawnet/best.pt \
  --output-json reports/run2/rawnet/benchmark.json
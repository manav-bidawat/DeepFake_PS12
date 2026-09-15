#!/usr/bin/env bash
set -euo pipefail

# 1) Build manifests and run mapping files.
python -m src.data.make_manifests --data-root . --out-dir data/manifests --verify-exists

# 2) Run 1 (Set A=LA train/dev, test on LA + PA).
python -m src.train \
  --data-root . \
  --train-manifest data/manifests/la_train.csv \
  --val-manifest data/manifests/la_dev.csv \
  --output-dir models/run1/cnn \
  --model cnn

python -m src.train \
  --data-root . \
  --train-manifest data/manifests/la_train.csv \
  --val-manifest data/manifests/la_dev.csv \
  --output-dir models/run1/crnn \
  --model crnn

python -m src.evaluate \
  --data-root . \
  --manifest data/manifests/la_eval.csv \
  --checkpoint models/run1/cnn/best.pt \
  --output-dir reports/run1/cnn/in_domain

python -m src.evaluate \
  --data-root . \
  --manifest data/manifests/pa_eval.csv \
  --checkpoint models/run1/cnn/best.pt \
  --output-dir reports/run1/cnn/cross_domain

python -m src.evaluate \
  --data-root . \
  --manifest data/manifests/la_eval.csv \
  --checkpoint models/run1/crnn/best.pt \
  --output-dir reports/run1/crnn/in_domain

python -m src.evaluate \
  --data-root . \
  --manifest data/manifests/pa_eval.csv \
  --checkpoint models/run1/crnn/best.pt \
  --output-dir reports/run1/crnn/cross_domain

# 3) Run 2 (Set B=PA train/dev, test on PA + LA).
python -m src.train \
  --data-root . \
  --train-manifest data/manifests/pa_train.csv \
  --val-manifest data/manifests/pa_dev.csv \
  --output-dir models/run2/cnn \
  --model cnn

python -m src.train \
  --data-root . \
  --train-manifest data/manifests/pa_train.csv \
  --val-manifest data/manifests/pa_dev.csv \
  --output-dir models/run2/crnn \
  --model crnn

python -m src.evaluate \
  --data-root . \
  --manifest data/manifests/pa_eval.csv \
  --checkpoint models/run2/cnn/best.pt \
  --output-dir reports/run2/cnn/in_domain

python -m src.evaluate \
  --data-root . \
  --manifest data/manifests/la_eval.csv \
  --checkpoint models/run2/cnn/best.pt \
  --output-dir reports/run2/cnn/cross_domain

python -m src.evaluate \
  --data-root . \
  --manifest data/manifests/pa_eval.csv \
  --checkpoint models/run2/crnn/best.pt \
  --output-dir reports/run2/crnn/in_domain

python -m src.evaluate \
  --data-root . \
  --manifest data/manifests/la_eval.csv \
  --checkpoint models/run2/crnn/best.pt \
  --output-dir reports/run2/crnn/cross_domain

# 4) Benchmark one 5-second clip per model.
python -m src.benchmark \
  --data-root . \
  --manifest data/manifests/la_eval.csv \
  --checkpoint models/run1/cnn/best.pt \
  --output-json reports/run1/cnn/benchmark.json

python -m src.benchmark \
  --data-root . \
  --manifest data/manifests/la_eval.csv \
  --checkpoint models/run1/crnn/best.pt \
  --output-json reports/run1/crnn/benchmark.json

python -m src.benchmark \
  --data-root . \
  --manifest data/manifests/pa_eval.csv \
  --checkpoint models/run2/cnn/best.pt \
  --output-json reports/run2/cnn/benchmark.json

python -m src.benchmark \
  --data-root . \
  --manifest data/manifests/pa_eval.csv \
  --checkpoint models/run2/crnn/best.pt \
  --output-json reports/run2/crnn/benchmark.json

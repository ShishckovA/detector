#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

python -m training.train_face_classifier \
  --dataset-dir data/face_dataset \
  --run-name face_efficientnet_b0_more_inter_10 \
  --weights imagenet \
  --val-size 0.15 \
  --device auto \
  --lr 5e-5 \
  --dataset-weight inter-escort=10

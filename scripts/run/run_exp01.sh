#!/usr/bin/env bash
set -euo pipefail

python scripts/train/train_baseline.py --config configs/experiments/exp_01_baseline_uq.yaml

echo "Exp01 scaffold finished. Next: implement real dataloader/inference/UQ evaluation pipeline."

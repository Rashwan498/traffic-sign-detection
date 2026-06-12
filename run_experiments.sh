#!/usr/bin/env bash
# Run all remaining experiments back-to-back, after the baseline training
# completes. This script:
#   1. Waits for the baseline training to finish (if still running)
#   2. Runs E1 (augmentation)        -> results/aug/
#   3. Runs E2 (EfficientNet-B0)     -> results/transfer/
#   4. Evaluates all three on test set
#   5. Runs robustness sweep on the best model
#   6. Runs Phase 1 classical rerun
#   7. Exports the best model to ONNX + CoreML
#
# Usage:
#     nohup ./run_experiments.sh > results/run_all.log 2>&1 &
#     echo $! > results/.run_all.pid
#
# Pause:  kill -INT $(cat results/.run_all.pid)
# Resume: re-run the same command (every step is idempotent / resumable).

set -e
cd "$(dirname "$0")"

PY=.venv/bin/python

echo "=== Waiting for baseline training to finish (if still running) ==="
if [ -f results/.baseline.pid ]; then
  PID=$(cat results/.baseline.pid)
  if ps -p $PID > /dev/null 2>&1; then
    echo "baseline PID $PID still running -- waiting..."
    while ps -p $PID > /dev/null 2>&1; do sleep 30; done
    echo "baseline finished."
  fi
fi

echo
echo "=== E1: augmentation experiment ==="
if [ -f results/aug/best.pt ]; then
  echo "  results/aug/best.pt already exists -- skipping."
else
  $PY -m src.train --name aug --arch baseline_cnn --augment --workers 2
fi

echo
echo "=== E2: transfer-learning experiment (EfficientNet-B0) ==="
if [ -f results/transfer/best.pt ]; then
  echo "  results/transfer/best.pt already exists -- skipping."
else
  $PY -m src.train --name transfer --arch efficientnet_b0 --augment --workers 2 --lr 1e-3
fi

echo
echo "=== Evaluation: baseline / aug / transfer on test set ==="
for run in baseline aug transfer; do
  $PY -m src.evaluate --ckpt results/$run/best.pt --split test
done

echo
echo "=== Robustness sweep (on best model) ==="
# Pick best macro-F1 across the three runs
BEST=$($PY -c "
import json, glob, os
best = None
for run in ['baseline','aug','transfer']:
    p = f'results/{run}/test_metrics.json'
    if os.path.exists(p):
        m = json.load(open(p))
        if best is None or m['macro_f1'] > best[0]: best = (m['macro_f1'], run)
print(best[1] if best else 'transfer')
")
echo "best run: $BEST"
$PY -m src.robustness --ckpt results/$BEST/best.pt

echo
echo "=== Phase 1 classical rerun ==="
$PY -m src.phase1_rerun --tune

echo
echo "=== Export best model (ONNX + CoreML) ==="
$PY -m src.deploy.export --ckpt results/$BEST/best.pt
# Stage the chosen checkpoint for the docker build context
cp results/$BEST/best.pt results/best_model.pt
echo "Staged results/best_model.pt for Docker build."

echo
echo "=== ALL DONE ==="
date

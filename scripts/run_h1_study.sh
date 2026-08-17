#!/usr/bin/env bash
# H1 experiment launcher (smoke / full).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

TAG="${TAG:-h1_v1}"
DATASETS="${DATASETS:-ml anime myket kuairec}"

# Smoke (fast):
#   SMOKE=1 ./scripts/run_h1_study.sh
if [[ "${SMOKE:-0}" == "1" ]]; then
  python -m training.run_h1_study \
    --datasets ml \
    --run-tag "${TAG}_smoke" \
    --train-sizes 5000 \
    --target-rand-ctrs 0.08 \
    --q-errors 0.0 1.0 \
    --logging-mixes 0.0 0.3 \
    --noise-levels high \
    --val-sizes 50000 \
    --seeds 0 \
    --n-trials 3 \
    --policy-losses sndr \
    --qhat-user-chunk 10000 \
    --qhat-action-chunk 10000 \
    --max-workers 2 \
    --slim \
    --n-rand-ctr-samples 5000
  python -m training.analyze_h1_study --root "artifacts/h1_study/run_${TAG}_smoke"
  exit 0
fi

# shellcheck disable=SC2086
python -m training.run_h1_study \
  --datasets $DATASETS \
  --run-tag "$TAG" \
  --train-sizes 5000 25000 100000 250000 400000 \
  --target-rand-ctrs 0.02 0.08 0.18 \
  --q-errors 0.0 0.25 0.5 0.75 1.0 \
  --logging-mixes 0.0 0.3 \
  --noise-levels low medium high extreme brutal \
  --val-sizes 50000 100000 200000 \
  --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 \
  --n-trials 15 \
  --policy-losses sndr \
  --policy-temperature 2.0 \
  --qhat-user-chunk 10000 \
  --qhat-action-chunk 10000 \
  --num-gpus 2 \
  --workers-per-gpu 16 \
  --require-cuda \
  --slim

python -m training.analyze_h1_study --root "artifacts/h1_study/run_${TAG}"

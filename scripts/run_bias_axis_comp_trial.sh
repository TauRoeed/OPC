#!/usr/bin/env bash
# Big axis × component bias trial.
# Usage:
#   ./scripts/run_bias_axis_comp_trial.sh
#   IMAGE=opc:gpu ./scripts/run_bias_axis_comp_trial.sh          # docker + all GPUs
#   NUM_GPUS=2 MAX_WORKERS=16 ./scripts/run_bias_axis_comp_trial.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

RUN_TAG="${RUN_TAG:-bias_axis_comp_l5_t20_s5}"
NUM_GPUS="${NUM_GPUS:-2}"
MAX_WORKERS="${MAX_WORKERS:-16}"
SHM="${SHM:-128g}"
IMAGE="${IMAGE:-}"
LOG_DIR="${LOG_DIR:-artifacts/full_study/ablation_logs}"
mkdir -p "$LOG_DIR"
LOGFILE="$LOG_DIR/${RUN_TAG}.log"

ARGS=(
  -m training.run_full_study_parallel
  --datasets ml
  --noise-axes context action
  --noise-components linear cluster metadata
  --noise-levels low medium high extreme brutal
  --ctr-levels 0.05
  --train-sizes 5000 25000 50000 100000
  --seeds 0 1 2 3 4
  --n-trials 20
  --num-gpus "$NUM_GPUS"
  --max-workers "$MAX_WORKERS"
  --require-cuda
  --skip-completed
  --emb-dir BPR/embeddings
  --out-dir artifacts/full_study
  --run-tag "$RUN_TAG"
)

echo "[$(date -Is)] starting $RUN_TAG"
echo "  grid: 2 axes × 3 components × 5 levels × 5 seeds = 150 conditions"
echo "  num_gpus=$NUM_GPUS max_workers=$MAX_WORKERS"
echo "  log: $LOGFILE"

if [[ -n "$IMAGE" ]]; then
  echo "  docker image=$IMAGE"
  nohup docker run --rm \
    --gpus all \
    --shm-size="$SHM" \
    -v "$ROOT:/app" -w /app \
    "$IMAGE" \
    "${ARGS[@]}" \
    >"$LOGFILE" 2>&1 &
else
  nohup python "${ARGS[@]}" >"$LOGFILE" 2>&1 &
fi

echo $! >"$LOG_DIR/${RUN_TAG}.pid"
echo "[$(date -Is)] pid=$(cat "$LOG_DIR/${RUN_TAG}.pid")"
echo "  tail -f $LOGFILE"
